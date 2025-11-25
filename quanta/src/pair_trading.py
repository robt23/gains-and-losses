import pandas as pd
from statsmodels.tsa.stattools import coint
from itertools import combinations
import statsmodels.api as sm
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


def cointegration_filter(training_df):
    """
    Finds statistically significant cointegrated ETF pairs in the training sample
    using the Engle–Granger two-step cointegration test within predefined sectors.

    For each sector, this function tests all pairwise combinations of ETF close 
    price series, runs the Engle–Granger test, and records the ADF statistic and 
    p-value. It then filters pairs with p-values below a chosen significance 
    level and saves both the full test results and the significant subset to CSV.

    Args:
        training_df (Pandas DataFrame):
            DataFrame containing historical price data for the training period.
            Must include columns of the form "Close_<TICKER>" for each ETF 
            defined in the sector universe.

    Returns:
        Pandas DataFrame:
            A DataFrame of ETF pairs that exhibit statistically significant
            cointegration (e.g., p-value < 0.05). Columns include:
                - "sector": Sector name
                - "pair": String label "ETF1 & ETF2"
                - "ETF1", "ETF2": Individual ETF tickers
                - "adf_stat": ADF test statistic from the cointegration test
                - "p_value": Associated p-value

            This result is intended to be passed into `calculate_spreads()` for
            further spread and strategy analysis.

    Side Effects:
        Writes two CSV files to disk:
            - "../data/all_cointegration_results.csv":
                  All tested ETF pairs with their ADF statistics and p-values.
            - "../data/significant_cointegrated_pairs.csv":
                  Only pairs that pass the p-value significance filter.
    """


    # Define sectors and tickers
    sectors = {
        'Consumer Discretionary': ["XLY", "IYC", "VCR", "XHB"],
        'Consumer Staples':       ["XLP", "IYK", "VDC"],
        'Energy':                 ["XLE", "IYE", "VDE", "XOP", "AMLP", "OIH"],
        'Financials':             ["XLF", "IYF", "VFH", "KBE", "KRE"],
        'Health Care':            ["XLV", "IYH", "IBB", "XBI", "VHT"],
        'Industrials':            ["XLI", "IYJ", "VIS"],
        'Materials':              ["XLB", "IYM", "VAW", "GDX", "GDXJ"],
        'Information Technology': ["XLK", "IYW", "VGT", "FDN", "IGV"],
        'Communication Services': ["IYZ", "VOX", "XLC"],
        'Utilities':              ["XLU", "IDU", "VPU"],
        'Real Estate':            ["RWR", "XLRE", "VNQ"]
    }

    # Run Engle–Granger cointegration test on every pair within each sector
    results = []
    for sector, tickers in sectors.items():
        # Build the column names for close prices
        close_cols = [f"Close_{t}" for t in tickers if f"Close_{t}" in training_df.columns]
        
        # Test all combinations of two ETFs in this sector
        for col1, col2 in combinations(close_cols, 2):
            data = training_df[[col1, col2]].dropna()
            if len(data) < 30:
                # skip pairs with too few overlapping points
                continue
            
            # Record statistics
            score, pvalue, _ = coint(data[col1], data[col2])
            results.append({
                'sector':   sector,
                'pair':     f"{col1.replace('Close_', '')} & {col2.replace('Close_', '')}",
                'ETF1': f"{col1.replace('Close_', '')}",
                'ETF2': f"{col2.replace('Close_', '')}",
                'adf_stat': score,
                'p_value':  pvalue
            })

    # Compile results into a DataFrame
    results_df = pd.DataFrame(results).sort_values(['sector', 'p_value'])
    results_df = results_df.reset_index(drop=True)


    # Keep likely cointegrated pairs (p < 0.05)
    sig = results_df[results_df['p_value'] < 0.05]
    sig = sig.reset_index(drop=True)

    # Save to CSV
    results_df.to_csv('../data/all_cointegration_results.csv', index=True)
    sig.to_csv('../data/significant_cointegrated_pairs.csv', index=True)
    
    return sig


def calculate_spreads(backtesting_df, sig):
    """
    Calculates the hedge ratio, log-price spread, and mean-reversion half-life 
    for each statistically significant cointegrated ETF pair. 

    This function fits an OLS regression between the log prices of ETF1 and ETF2
    to estimate the hedge ratio (beta), constructs the stationary spread series,
    and computes the mean-reversion half-life using an Ornstein–Uhlenbeck (OU)
    approximation based on the spread's first-order autoregressive dynamics.

    Args:
        backtesting_df (Pandas DataFrame): 
            DataFrame containing price history (Close_ETF) for the ETFs during 
            the backtesting period. Must include the same tickers found in 
            the training set used for the cointegration test.

        sig (Pandas DataFrame): 
            Output from `cointegration_filter()`, containing pairs of ETFs 
            that exhibit statistically significant cointegration based on the 
            Engle–Granger test.

    Returns:
        List[Dict]: A list of dictionaries, one per ETF pair, where each dict 
        contains:
            - "ETF1", "ETF2": Ticker symbols of the pair
            - "alpha": Regression intercept
            - "hedge_ratio": OLS slope coefficient (beta)
            - "half_life": Estimated mean-reversion half-life of the spread
            - "spread_prices": DataFrame with Close prices and computed spread
              (columns: Close_ETF1, Close_ETF2, spread)

        This output is intended as the input to `generate_signals()`.
    """

    # Get pairs
    pairs = sig['pair'].str.split(' & ').apply(tuple).tolist()
    results = []

    # Iterate through pairs and calculate spread, half-life
    for etf1, etf2 in pairs:
        col1 = f"Close_{etf1}"
        col2 = f"Close_{etf2}"

        if col1 not in backtesting_df.columns or col2 not in backtesting_df.columns:
            print(f"Missing columns for pair: {etf1}, {etf2}")
            continue

        pair_df = backtesting_df[[col1, col2]].dropna().copy()

        # OLS: log(P1) ~ log(P2) to find hedge ratio
        log1 = np.log(pair_df[col1])
        log2 = np.log(pair_df[col2])
        X = sm.add_constant(log2)
        y = log1
        model = sm.OLS(y, X).fit()
        alpha, beta = model.params  # beta = hedge ratio

        # Construct spread
        spread = y - (alpha + beta * log2)
        pair_df["spread"] = spread

        # Half-life of mean reversion
        spread_lag = spread.shift(1)
        spread_ret = spread - spread_lag

        spread_lag = spread_lag.dropna()
        spread_ret = spread_ret.dropna()

        # Calculate half life
        if len(spread_lag) > 0:
            hl_model = sm.OLS(spread_ret, sm.add_constant(spread_lag)).fit()
            if len(hl_model.params) > 1:
                phi = hl_model.params.iloc[1]
            else:
                phi = np.nan
            half_life = -np.log(2) / phi if phi < 0 else np.nan
        else:
            half_life = np.nan

        # Record statistics
        results.append({
            "ETF1": etf1,
            "ETF2": etf2,
            "alpha": alpha,
            "hedge_ratio": beta,
            "half_life": half_life,
            "spread_prices": pair_df  # includes Close_ETF1, Close_ETF2, spread
        })
        
        # def spread_graphs():
        #     for r in results:
        #         print(f"{r['ETF1']} & {r['ETF2']} → hedge ratio: {r['hedge_ratio']:.4f}")
        #         print(r['spread_prices'].head(), "\n")
            
        #         sns.lineplot(data=r['spread_prices']['spread'])
        #         plt.axhline(y=r['mean'], color='red', linestyle='--', linewidth=2)
        #         plt.xlabel('Date')
        #         plt.ylabel('Spread')
        #         plt.title(f"{r['ETF1']} & {r['ETF2']} Spread Over Time")
        #         plt.show()
        
        # spread_graphs()
    
    return results


def generate_signals(spreads, entry_z=2.0, exit_z=0.5, default_window=60):
    """
    Generates trading signals, held positions, and strategy returns for each 
    cointegrated ETF pair using a z-score mean-reversion strategy.

    This function applies rolling z-score normalization to the spread, identifies 
    long/short entry points when the spread deviates beyond specified thresholds, 
    and tracks the held trading position over time. It computes theoretical 
    pair returns, position-weighted strategy returns, and cumulative returns for 
    each pair.

    Args:
        spreads (List[Dict]): 
            Output from `calculate_spreads()`. Each element must contain:
                - "ETF1", "ETF2": Tickers of the cointegrated pair
                - "hedge_ratio": Hedge ratio (beta) from log-price regression
                - "half_life": Estimated mean-reversion half-life of the spread
                - "spread_prices": DataFrame with Close prices and spread values

        entry_z (float, default=2.0): 
            Z-score threshold for opening a trade.
                - zscore >  +entry_z → short the spread
                - zscore <  -entry_z → long the spread

        exit_z (float, default=0.5): 
            Z-score threshold for closing an existing position. 
            When |zscore| falls below this level, the strategy exits to flat.

        default_window (int, default=60): 
            Rolling window length for computing the spread's mean and 
            standard deviation. If a valid spread half-life exists and is 
            greater than 5 trading periods, it replaces this default.

    Returns:
        Dict[str, Pandas DataFrame]: 
            A dictionary keyed by "ETF1 & ETF2". Each value is a DataFrame 
            containing:
                - Close prices for both ETFs
                - "spread": Log-price spread
                - "mu", "sd": Rolling mean and standard deviation of the spread
                - "zscore": Standardized spread
                - "signal": Raw entry signals (impulse long/short)
                - "position": Held position over time (long/short/flat)
                - "pair_return": Theoretical spread return 
                - "strategy_return": Return after applying held position
                - "cum_return": Geometrically compounded cumulative return

    Notes:
        - `signal` represents instantaneous trade-entry impulses only.  
          `position` represents the actual held trade across time.

        - The strategy enters only when flat and exits only when the spread 
          reverts inside the exit threshold.

        - One-period lag (`position.shift(1)`) is applied to avoid look-ahead 
          bias when computing strategy returns.

        - Cumulative returns are computed as:
              cum_return = (1 + strategy_return).cumprod() - 1
    """


    trades = {}

    # Iterate through each spread DataFrame
    for sp in spreads:
        etf1, etf2 = sp["ETF1"], sp["ETF2"]
        hr = sp["hedge_ratio"]
        df = sp["spread_prices"].copy()
        half_life = sp.get("half_life", np.nan)

        col1 = f"Close_{etf1}"
        col2 = f"Close_{etf2}"

        # Choose rolling window (use half-life if reasonable)
        if np.isfinite(half_life) and half_life > 5:
            window = int(round(half_life))
        else:
            window = default_window

        # Rolling z-score on spread
        df["mu"] = df["spread"].rolling(window).mean()
        df["sd"] = df["spread"].rolling(window).std()
        df["zscore"] = (df["spread"] - df["mu"]) / df["sd"]

        # Raw entry signals
        df["signal"] = 0
        df.loc[df["zscore"] >  entry_z, "signal"] = -1   # short spread
        df.loc[df["zscore"] < -entry_z, "signal"] =  1   # long spread

        # Turn discrete signals into held positions until exit;  1/-1 after 0 means enter, subsequent 1/-1's mean hold, 0 after 1/-1 means exit
        df["position"] = 0
        current_position = 0
        

        for i in range(len(df)):
            z = df["zscore"].iloc[i]
            s = df["signal"].iloc[i]

            # Entry
            if current_position == 0 and s != 0:
                current_position = s
            # Exit
            elif current_position != 0 and abs(z) < exit_z:
                current_position = 0

            df.iat[i, df.columns.get_loc("position")] = current_position

        # Leg returns
        df["r1"] = np.log(df[col1]).diff()
        df["r2"] = np.log(df[col2]).diff()

        # Pair return (before applying position)
        df["pair_return"] = df["r1"] - hr * df["r2"]

        # Strategy return = position * pair_return (use yesterday's position)
        df["strategy_return"] = df["position"].shift(1) * df["pair_return"]

        # Clean NaNs for cumulative metrics
        df["strategy_return"] = df["strategy_return"].fillna(0.0)

        # Cumulative return curve
        df["cum_return"] = (1 + df["strategy_return"]).cumprod() - 1

        trades[f"{etf1} & {etf2}"] = df[[
            col1, col2,
            "spread", "mu", "sd", "zscore", "signal",
            "position", "pair_return", "strategy_return", "cum_return"
        ]]


    return trades


def performance_metrics(trades, trading_days=252):
    """
    Computes performance statistics for each cointegrated ETF pair's trading
    strategy, including cumulative return, annualized volatility, Sharpe ratio,
    and maximum drawdown.

    This function evaluates the time series of strategy returns generated by
    `generate_signals()` and summarizes key risk and performance metrics commonly
    used in quantitative finance. Each pair's results are collected into a
    summary table and ranked by Sharpe ratio.

    Args:
        trades (Dict[str, Pandas DataFrame]):
            Output from `generate_signals()`. Keys are pair labels such as 
            "XLY & VCR". Each DataFrame must contain:
                - "strategy_return": daily position-weighted PnL
                - "cum_return": cumulative strategy return
                Additional columns (prices, spread, zscore, etc.) are optional.

        trading_days (int, default=252):
            Number of trading days in a year. Used to annualize mean return
            and volatility.

    Returns:
        Pandas DataFrame:
            Summary table where each row corresponds to an ETF pair, with:
                - "pair": pair label
                - "cumulative_return": total return over the full backtest
                - "annualized_vol": annualized standard deviation of returns
                - "sharpe": Sharpe ratio (annualized return / annualized vol)
                - "max_drawdown": worst peak-to-trough drawdown of equity curve

            The returned DataFrame is sorted in descending order of Sharpe ratio.
    """

    summary = []

    for pair, df in trades.items():
        r = df["strategy_return"]
        cum_ret = df["cum_return"].iloc[-1]
        
        # Annualized volatility
        vol = r.std() * np.sqrt(trading_days)
        
        # Sharpe ratio
        sharpe = r.mean() * trading_days / vol if vol > 0 else np.nan

        # Max drawdown
        cum_equity = (1 + r).cumprod()
        peak = cum_equity.cummax()
        dd = (cum_equity - peak) / peak
        max_dd = dd.min()

        # Record statistics
        summary.append({
            "pair": pair,
            "cumulative_return": cum_ret,
            "annualized_vol": vol,
            "sharpe": sharpe,
            "max_drawdown": max_dd
        })

    return pd.DataFrame(summary).sort_values("sharpe", ascending=False)



def plot_pair(trades, pair, entry_z=2.0, exit_z=0.5):
    """
    Creates a 2x2 window-pane layout visualization for a given pair.
    Panels:
      (1,1) Normalized ETF Prices
      (1,2) Spread + rolling mean + entry/exit bands + markers
      (2,1) Z-Score + thresholds + Position
      (2,2) Equity Curve + Drawdown
    """

    if pair not in trades:
        raise ValueError(f"Pair '{pair}' not found in trades dictionary.")

    df = trades[pair].copy()
    etf1, etf2 = pair.split(" & ")

    col1 = f"Close_{etf1}"
    col2 = f"Close_{etf2}"

    # Normalize prices for Panel 1
    price1_norm = df[col1] / df[col1].iloc[0]
    price2_norm = df[col2] / df[col2].iloc[0]

    # Rolling spread bands
    upper_entry = df["mu"] + entry_z * df["sd"]
    lower_entry = df["mu"] - entry_z * df["sd"]
    upper_exit = df["mu"] + exit_z * df["sd"]
    lower_exit = df["mu"] - exit_z * df["sd"]

    # Entry/Exit locations
    long_entries = df[(df["signal"] == 1) & (df["position"].shift(1) == 0)]
    short_entries = df[(df["signal"] == -1) & (df["position"].shift(1) == 0)]
    exits = df[(df["position"] == 0) & (df["position"].shift(1) != 0)]

    # Equity curve and drawdown
    equity = (1 + df["strategy_return"]).cumprod()
    peak = equity.cummax()
    drawdown = (equity - peak) / peak

    # ---- Create 2x2 layout ----
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle(f"Pairs Trading Analysis: {pair}", fontsize=18, y=0.98)

    ax11, ax12, ax21, ax22 = axes.flatten()

    # ---------------- Panel (1,1): Prices ----------------
    sns.lineplot(x=df.index, y=price1_norm, ax=ax11, label=etf1)
    sns.lineplot(x=df.index, y=price2_norm, ax=ax11, label=etf2)
    ax11.set_title("Normalized Prices")
    ax11.set_ylabel("Normalized")
    ax11.legend(loc="upper left")

    # ---------------- Panel (1,2): Spread Bands ----------------
    sns.lineplot(x=df.index, y=df["spread"], ax=ax12, label="Spread", linewidth=1.2)
    ax12.plot(df.index, df["mu"], linestyle="--", label="Rolling Mean")
    ax12.plot(df.index, upper_entry, linestyle=":", label=f"+{entry_z}σ (entry)")
    ax12.plot(df.index, lower_entry, linestyle=":", label=f"-{entry_z}σ (entry)")
    ax12.plot(df.index, upper_exit, linestyle="-.", label=f"+{exit_z}σ (exit)")
    ax12.plot(df.index, lower_exit, linestyle="-.", label=f"-{exit_z}σ (exit)")

    # Markers
    ax12.scatter(long_entries.index, long_entries["spread"], marker="^", s=45, color="green", label="Long Entry")
    ax12.scatter(short_entries.index, short_entries["spread"], marker="v", s=45, color="red", label="Short Entry")
    ax12.scatter(exits.index, exits["spread"], marker="o", s=40, color="orange", label="Exit")

    ax12.set_title("Spread with Rolling Bands")
    ax12.set_ylabel("Spread")
    ax12.legend(loc="upper left")

    # ---------------- Panel (2,1): Z-score + position ----------------
    sns.lineplot(x=df.index, y=df["zscore"], ax=ax21, label="Z-Score")
    ax21.axhline(entry_z, linestyle="--", color="red", linewidth=0.8)
    ax21.axhline(-entry_z, linestyle="--", color="red", linewidth=0.8)
    ax21.axhline(exit_z, linestyle=":", color="orange", linewidth=0.8)
    ax21.axhline(-exit_z, linestyle=":", color="orange", linewidth=0.8)
    ax21.axhline(0, linestyle="-", color="black", linewidth=0.5)
    ax21.set_title("Z-Score and Position")
    ax21.set_ylabel("Z-Score")

    # Position overlay
    ax21b = ax21.twinx()
    ax21b.step(df.index, df["position"], where="post", color="purple", alpha=0.6, label="Position")
    ax21b.set_ylim(-1.2, 1.2)
    ax21b.set_yticks([-1, 0, 1])
    ax21b.set_ylabel("Position")

    # ---------------- Panel (2,2): Equity & Drawdown ----------------
    sns.lineplot(x=df.index, y=equity, ax=ax22, label="Equity")
    ax22.set_title("Equity Curve and Drawdown")
    ax22.set_ylabel("Equity Index")

    ax22b = ax22.twinx()
    ax22b.fill_between(df.index, drawdown, 0, color="red", alpha=0.2, label="Drawdown")
    ax22b.set_ylabel("Drawdown")
    ax22b.set_ylim(drawdown.min() * 1.05, 0)

    # Tight layout
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()




def main():
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    training_df = pd.read_csv('../data/training.csv', index_col=0, parse_dates=True)
    backtesting_df = pd.read_csv('../data/backtesting.csv', index_col=0, parse_dates=True)
    sig = cointegration_filter(training_df)
    spreads = calculate_spreads(backtesting_df, sig)
    trades = generate_signals(spreads)
    perf = performance_metrics(trades)
    
    sns.set_theme(style="whitegrid")

    pair = "XLY & VCR"
    plot_pair(trades, pair, entry_z=2.0, exit_z=0.5)
    
    

if __name__ == "__main__":
    main()
