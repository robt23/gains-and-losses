import pandas as pd
from statsmodels.tsa.stattools import coint
from itertools import combinations
import statsmodels.api as sm
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
import sys


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

    results = []

    # Run Engle–Granger cointegration test on every pair within each sector
    for sector, tickers in sectors.items():
        close_cols = [f"Close_{t}" for t in tickers if f"Close_{t}" in training_df.columns]

        for col1, col2 in combinations(close_cols, 2):
            data = training_df[[col1, col2]].dropna()
            if len(data) < 30:
                # skip pairs with too few overlapping points
                continue

            score, pvalue, _ = coint(data[col1], data[col2])
            etf1 = col1.replace("Close_", "")
            etf2 = col2.replace("Close_", "")

            results.append({
                "sector":   sector,
                "pair":     f"{etf1} & {etf2}",
                "ETF1":     etf1,
                "ETF2":     etf2,
                "adf_stat": score,
                "p_value":  pvalue,
            })

    # Compile results into a DataFrame
    results_df = pd.DataFrame(results).sort_values(["sector", "p_value"]).reset_index(drop=True)

    # Keep likely cointegrated pairs (p < 0.05)
    sig = results_df[results_df["p_value"] < 0.05].reset_index(drop=True)

    # Save to CSV
    # results_df.to_csv("../data/all_cointegration_results.csv", index=True)
    # sig.to_csv("../data/significant_cointegrated_pairs.csv", index=True)

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

    pairs = sig["pair"].str.split(" & ").apply(tuple).tolist()
    results = []

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

        if len(spread_lag) > 0:
            hl_model = sm.OLS(spread_ret, sm.add_constant(spread_lag)).fit()
            # params: [const, phi]
            phi = hl_model.params.iloc[1] if len(hl_model.params) > 1 else np.nan
            half_life = -np.log(2) / phi if (phi is not np.nan and phi < 0) else np.nan
        else:
            half_life = np.nan

        results.append({
            "ETF1": etf1,
            "ETF2": etf2,
            "alpha": alpha,
            "hedge_ratio": beta,
            "half_life": half_life,
            "spread_prices": pair_df,  # includes Close_ETF1, Close_ETF2, spread
        })

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
    """

    trades = {}

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
        df.loc[df["zscore"] > entry_z, "signal"] = -1   # short spread
        df.loc[df["zscore"] < -entry_z, "signal"] = 1   # long spread

        # Convert impulses -> held positions
        df["position"] = 0
        current_position = 0

        for i in range(len(df)):
            z = df["zscore"].iloc[i]
            s = df["signal"].iloc[i]

            if current_position == 0 and s != 0:
                # Enter new position
                current_position = s
            elif current_position != 0 and abs(z) < exit_z:
                # Exit existing position
                current_position = 0

            df.iat[i, df.columns.get_loc("position")] = current_position

        # Leg returns
        df["r1"] = np.log(df[col1]).diff()
        df["r2"] = np.log(df[col2]).diff()

        # Pair return (spread return before position)
        df["pair_return"] = df["r1"] - hr * df["r2"]

        # Strategy return with 1-bar lag on position
        df["strategy_return"] = df["position"].shift(1) * df["pair_return"]
        df["strategy_return"] = df["strategy_return"].fillna(0.0)

        # Cumulative return
        df["cum_return"] = (1 + df["strategy_return"]).cumprod() - 1

        trades[f"{etf1} & {etf2}"] = df[
            [col1, col2, "spread", "mu", "sd", "zscore",
             "signal", "position", "pair_return", "strategy_return", "cum_return"]
        ]

    return trades


def extract_trades(trades):
    """
    Convert position time-series into discrete trade records.

    A trade is defined as a continuous period where position ≠ 0. This function
    detects entries, exits, and sign flips, and records holding period and
    compounded returns for each completed trade.

    Args:
        trades (dict[str, pd.DataFrame]):
            Output from `generate_signals()`, containing per-pair backtest data.

    Returns:
        pd.DataFrame:
            One row per trade with columns:
            pair, side, entry_time, exit_time, holding_period, entry_z,
            exit_z, entry_spread, exit_spread, trade_return.
            Empty DataFrame if no trades occurred.
    """
    all_trades = []

    for pair, df in trades.items():
        df = df.copy()
        pos = df["position"].values
        idx = df.index

        current_side = 0        # +1 (long spread), -1 (short spread), 0 (flat)
        entry_i = None          # index into df for entry row

        for i in range(len(df)):
            p = pos[i]

            # Case 1: flat → possible new entry
            if current_side == 0:
                if p != 0:
                    current_side = p
                    entry_i = i

            # Case 2: already in a trade
            else:
                close_trade = False
                new_entry_after_flip = False

                if p == 0:
                    close_trade = True
                elif np.sign(p) != np.sign(current_side):
                    # flip sign = close + immediately re-enter
                    close_trade = True
                    new_entry_after_flip = True

                if close_trade and entry_i is not None:
                    exit_i = i

                    seg = df.iloc[entry_i:exit_i + 1]
                    strat = seg["strategy_return"]
                    trade_ret = (1.0 + strat).prod() - 1.0
                    holding_period = len(seg)

                    entry_row = df.iloc[entry_i]
                    exit_row = df.iloc[exit_i]

                    all_trades.append({
                        "pair": pair,
                        "side": "long" if current_side > 0 else "short",
                        "entry_time": idx[entry_i],
                        "exit_time": idx[exit_i],
                        "holding_period": holding_period,
                        "entry_z": entry_row.get("zscore", np.nan),
                        "exit_z": exit_row.get("zscore", np.nan),
                        "entry_spread": entry_row.get("spread", np.nan),
                        "exit_spread": exit_row.get("spread", np.nan),
                        "trade_return": trade_ret,
                    })

                    if new_entry_after_flip:
                        current_side = p
                        entry_i = i
                    else:
                        current_side = 0
                        entry_i = None

    if not all_trades:
        return pd.DataFrame(columns=[
            "pair", "side", "entry_time", "exit_time", "holding_period",
            "entry_z", "exit_z", "entry_spread", "exit_spread", "trade_return"
        ])

    trade_log = pd.DataFrame(all_trades)
    trade_log = trade_log.sort_values(["pair", "entry_time"]).reset_index(drop=True)
    return trade_log


def summarize_trades(trade_log, min_trades=5):
    """
    Summarize trade-level performance for each ETF pair.

    Computes win rate, expectancy, and distribution statistics for trade
    returns. Filters to “good pairs” meeting minimum performance criteria.

    Args:
        trade_log (pd.DataFrame):
            Output of `extract_trades()`.

        min_trades (int):
            Minimum number of completed trades required to include the pair.

    Returns:
        pd.DataFrame:
            A table of pairs with:
            n_trades, win_rate, avg_trade_return, med_trade_return,
            std_trade_return, expectancy, avg_holding.
    """

    if trade_log.empty:
        return trade_log

    grp = trade_log.groupby("pair")

    summary = grp["trade_return"].agg(
        n_trades="count",
        win_rate=lambda x: (x > 0).mean(),
        avg_trade_return="mean",
        med_trade_return="median",
        std_trade_return="std",
    )

    summary["expectancy"] = summary["avg_trade_return"]
    summary["avg_holding"] = grp["holding_period"].mean()

    summary = summary.sort_values("expectancy", ascending=False)

    good_pairs = summary[
        (summary["expectancy"] > 0) &
        (summary["win_rate"] > 0.55) &
        (summary["n_trades"] >= min_trades)
    ]

    return good_pairs


def portfolio_metrics(trades, trading_days=252):
    """
    Compute portfolio-level performance assuming equal weighting of all pairs.

    Merges all pair return streams, averages them to form a portfolio return
    series, and computes Sharpe ratio, maximum drawdown, and total cumulative
    return.

    Args:
        trades (dict[str, pd.DataFrame]):
            Output from `generate_signals()`. Each DataFrame must include
            a 'strategy_return' column.

        trading_days (int):
            Number of trading days used for annualization.

    Returns:
        tuple:
            (sharpe, max_drawdown, cumulative_return)
    """

    if not trades:
        return np.nan, np.nan, np.nan

    returns = [df["strategy_return"] for df in trades.values()]
    merged = pd.concat(returns, axis=1).fillna(0.0)
    port_r = merged.mean(axis=1)

    vol = port_r.std() * np.sqrt(trading_days)
    sharpe = port_r.mean() * trading_days / vol if vol > 0 else np.nan

    equity = (1 + port_r).cumprod()
    peak = equity.cummax()
    dd = (equity - peak) / peak
    max_dd = dd.min()
    cum_ret = equity.iloc[-1] - 1.0

    return sharpe, max_dd, cum_ret


def plot_pair(trades, pair, entry_z, exit_z, window):
    """
    Plot a 4-panel diagnostic view for a single ETF pair.

    Panels include:
        (1) Normalized ETF prices
        (2) Spread with rolling mean and entry/exit bands
        (3) Z-score with position overlay
        (4) Equity curve with drawdown

    Args:
        trades (dict[str, pd.DataFrame]):
            Output from `generate_signals()`.

        pair (str):
            Pair label in the form "ETF1 & ETF2".

        entry_z (float):
            Entry z-score threshold, shown on the plot.

        exit_z (float):
            Exit z-score threshold.

    Returns:
        None. Displays a matplotlib figure.
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

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle(f"Pairs Trading Analysis: {pair}, entry_z {entry_z}, exit_z {exit_z}, window {window}", fontsize=18, y=0.98)
    ax11, ax12, ax21, ax22 = axes.flatten()

    # (1,1) Normalized prices
    sns.lineplot(x=df.index, y=price1_norm, ax=ax11, label=etf1)
    sns.lineplot(x=df.index, y=price2_norm, ax=ax11, label=etf2)
    ax11.set_title("Normalized Prices")
    ax11.set_ylabel("Normalized")
    ax11.legend(loc="upper left")

    # (1,2) Spread + bands
    sns.lineplot(x=df.index, y=df["spread"], ax=ax12, label="Spread", linewidth=1.2)
    ax12.plot(df.index, df["mu"], linestyle="--", label="Rolling Mean")
    ax12.plot(df.index, upper_entry, linestyle=":", label=f"+{entry_z}σ (entry)")
    ax12.plot(df.index, lower_entry, linestyle=":", label=f"-{entry_z}σ (entry)")
    ax12.plot(df.index, upper_exit, linestyle="-.", label=f"+{exit_z}σ (exit)")
    ax12.plot(df.index, lower_exit, linestyle="-.", label=f"-{exit_z}σ (exit)")

    ax12.scatter(long_entries.index, long_entries["spread"], marker="^", s=45,
                 color="green", label="Long Entry")
    ax12.scatter(short_entries.index, short_entries["spread"], marker="v", s=45,
                 color="red", label="Short Entry")
    ax12.scatter(exits.index, exits["spread"], marker="o", s=40,
                 color="orange", label="Exit")

    ax12.set_title("Spread with Rolling Bands")
    ax12.set_ylabel("Spread")
    ax12.legend(loc="upper left")

    # (2,1) Z-score + position
    sns.lineplot(x=df.index, y=df["zscore"], ax=ax21, label="Z-Score")
    ax21.axhline(entry_z, linestyle="--", color="red", linewidth=0.8)
    ax21.axhline(-entry_z, linestyle="--", color="red", linewidth=0.8)
    ax21.axhline(exit_z, linestyle=":", color="orange", linewidth=0.8)
    ax21.axhline(-exit_z, linestyle=":", color="orange", linewidth=0.8)
    ax21.axhline(0, linestyle="-", color="black", linewidth=0.5)
    ax21.set_title("Z-Score and Position")
    ax21.set_ylabel("Z-Score")

    ax21b = ax21.twinx()
    ax21b.step(df.index, df["position"], where="post", color="purple",
               alpha=0.6, label="Position")
    ax21b.set_ylim(-1.2, 1.2)
    ax21b.set_yticks([-1, 0, 1])
    ax21b.set_ylabel("Position")

    # (2,2) Equity & drawdown
    sns.lineplot(x=df.index, y=equity, ax=ax22, label="Equity")
    ax22.set_title("Equity Curve and Drawdown")
    ax22.set_ylabel("Equity Index")

    ax22b = ax22.twinx()
    ax22b.fill_between(df.index, drawdown, 0, color="red", alpha=0.2, label="Drawdown")
    ax22b.set_ylabel("Drawdown")
    ax22b.set_ylim(drawdown.min() * 1.05, 0)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    return fig

def run_backtest(training_df, backtesting_df,
                 entry_z=2.0, exit_z=0.5, default_window=60,
                 trading_days=252):
    """
    Execute a full backtest pipeline for a single parameter configuration.

    Runs: cointegration → spread modeling → signal generation → portfolio
    evaluation → trade extraction.

    Args:
        training_df (pd.DataFrame):
            Training data for cointegration estimation.

        backtesting_df (pd.DataFrame):
            Out-of-sample data for trading evaluation.

        entry_z, exit_z (float):
            Z-score thresholds for the strategy.

        default_window (int):
            Default rolling window for z-scores.

        trading_days (int):
            Annualization constant.

    Returns:
        tuple:
            perf      : (Sharpe, max_drawdown, cumulative_return)
            trade_log : DataFrame of trades
            trades    : Dict of per-pair time series
    """
    sig = cointegration_filter(training_df)
    spreads = calculate_spreads(backtesting_df, sig)
    trades = generate_signals(spreads, entry_z=entry_z,
                              exit_z=exit_z,
                              default_window=default_window)
    perf = portfolio_metrics(trades, trading_days=trading_days)
    trade_log = extract_trades(trades)
    return perf, trade_log, trades


def make_walk_forward_splits(df, train_years=3, test_years=1, step_years=1):
    """
    Generate chronological train/test splits for walk-forward analysis.

    Uses year-based offsets to roll training windows forward and create
    non-overlapping test periods.

    Args:
        df (pd.DataFrame):
            Full price DataFrame indexed by datetime.

        train_years (int):
            Length of training window.

        test_years (int):
            Length of test window.

        step_years (int):
            Step size before generating the next split.

    Returns:
        list[tuple]:
            List of (train_start, train_end, test_start, test_end) timestamps.
    """

    dates = df.index.sort_values()
    start_date = dates.min()
    end_date = dates.max()

    splits = []
    current_train_start = start_date

    while True:
        train_end = current_train_start + pd.DateOffset(years=train_years) - pd.DateOffset(days=1)
        test_start = train_end + pd.DateOffset(days=1)
        test_end = test_start + pd.DateOffset(years=test_years) - pd.DateOffset(days=1)

        if test_end > end_date:
            break

        splits.append((current_train_start, train_end, test_start, test_end))
        current_train_start = current_train_start + pd.DateOffset(years=step_years)

    return splits


def walk_forward_backtest(all_prices_df,
                          train_years=3, test_years=1, step_years=1,
                          entry_z=2.0, exit_z=0.5, default_window=60,
                          trading_days=252):
    """
    Perform multi-period walk-forward backtests across rolling windows.

    For each train/test split, runs the full backtest pipeline and records
    portfolio-level performance.

    Args:
        all_prices_df (pd.DataFrame):
            Complete price history indexed by date.

        train_years, test_years, step_years (int):
            Window lengths and step size for walk-forward splitting.

        entry_z, exit_z (float):
            Z-score thresholds for the strategy.

        default_window (int):
            Rolling window length for z-score estimation.

        trading_days (int):
            Annualization constant.

    Returns:
        pd.DataFrame:
            Fold-level results sorted by test_start date.
    """

    splits = make_walk_forward_splits(all_prices_df, train_years, test_years, step_years)
    records = []

    for (train_start, train_end, test_start, test_end) in splits:
        train_df = all_prices_df.loc[train_start:train_end]
        test_df = all_prices_df.loc[test_start:test_end]

        print(f"Fold: train {train_start.date()}–{train_end.date()}, "
              f"test {test_start.date()}–{test_end.date()}")

        (port_sharpe, port_max_dd, port_cum_ret), trade_log, trades = run_backtest(
            train_df,
            test_df,
            entry_z=entry_z,
            exit_z=exit_z,
            default_window=default_window,
            trading_days=trading_days,
        )

        n_trades = len(trade_log)

        records.append({
            "train_start": train_start,
            "train_end": train_end,
            "test_start": test_start,
            "test_end": test_end,
            "portfolio_sharpe": port_sharpe,
            "portfolio_max_dd": port_max_dd,
            "portfolio_cum_return": port_cum_ret,
            "n_trades": n_trades,
        })

    return pd.DataFrame(records).sort_values("test_start")


def walk_forward_parameter_sweep(all_prices_df,
                                 train_years=3, test_years=1, step_years=1,
                                 entry_grid=(1.5, 2.0, 2.5),
                                 exit_grid=(0.25, 0.5, 0.75),
                                 window_grid=(30, 60, 90),
                                 trading_days=252):
    """
    Combine walk-forward evaluation with a grid search of strategy parameters.

    For every train/test split and every combination of entry/exit thresholds
    and rolling window, runs a full backtest and records the results.

    Returns one row per (split, parameter set).
    """
    splits = make_walk_forward_splits(all_prices_df, train_years, test_years, step_years)
    records = []
    
    plot_dir = Path("trade_plots")
    plot_dir.mkdir(exist_ok=True)
    log_dir = Path("trade_logs")
    log_dir.mkdir(exist_ok=True)
    summary_dir = Path("trade_summaries")
    summary_dir.mkdir(exist_ok=True)

    for fold_idx, (train_start, train_end, test_start, test_end) in enumerate(splits, start=1):
        train_df = all_prices_df.loc[train_start:train_end]
        test_df = all_prices_df.loc[test_start:test_end]

        print(f"Fold {fold_idx}: train {train_start.date()}–{train_end.date()}, "
              f"test {test_start.date()}–{test_end.date()}")

        for entry_z in entry_grid:
            for exit_z in exit_grid:
                if exit_z >= entry_z:
                    continue

                for window in window_grid:
                    (port_sharpe, port_max_dd, port_cum_ret), trade_log, trades = run_backtest(
                        train_df,
                        test_df,
                        entry_z=entry_z,
                        exit_z=exit_z,
                        default_window=window,
                        trading_days=trading_days,
                    )
                    
                    for pair in trades:
                        plot_path = plot_dir / (f"fold{fold_idx}_entry{entry_z}_exit{exit_z}_window{window}_{pair}.png")
                        fig = plot_pair(trades, pair, entry_z, exit_z, window)
                        fig.savefig(plot_path, dpi=300, bbox_inches="tight")
                        plt.close(fig)
                    
                    log_path = log_dir / (f"fold{fold_idx}_entry{entry_z}_exit{exit_z}_window{window}_log.csv")
                    trade_log.to_csv(log_path, index=False)

                    trade_summary = summarize_trades(trade_log)
                    summary_path = summary_dir / (
                        f"fold{fold_idx}_entry{entry_z}_exit{exit_z}_window{window}_summary.csv"
                    )
                    trade_summary.to_csv(summary_path, index=False)

                    if trade_log.empty:
                        n_trades = 0
                        win_rate = np.nan
                        avg_trade_return = np.nan
                    else:
                        n_trades = len(trade_log)
                        win_rate = (trade_log["trade_return"] > 0).mean()
                        avg_trade_return = trade_log["trade_return"].mean()

                    records.append({
                        "fold": fold_idx,
                        "train_start": train_start,
                        "train_end": train_end,
                        "test_start": test_start,
                        "test_end": test_end,
                        "entry_z": entry_z,
                        "exit_z": exit_z,
                        "window": window,
                        "portfolio_sharpe": port_sharpe,
                        "portfolio_max_dd": port_max_dd,
                        "portfolio_cum_return": port_cum_ret,
                        "n_trades": n_trades,
                        "win_rate": win_rate,
                        "avg_trade_return": avg_trade_return,
                    })

                    print(
                        f"  Done params entry_z={entry_z}, exit_z={exit_z}, window={window}"
                    )

    results = pd.DataFrame(records)
    if results.empty:
        return results

    return results.sort_values(["test_start", "entry_z", "exit_z", "window"])


def main():
    all_prices_df = pd.read_csv("../data/all_prices.csv", index_col=0, parse_dates=True)

    wf_results = walk_forward_parameter_sweep(
        all_prices_df,
        train_years=3,
        test_years=1,
        step_years=1,
        entry_grid=(1.5, 2.0, 2.5),
        exit_grid=(0.25, 0.5, 0.75),
        window_grid=(30, 60, 90),
    )
    wf_results.to_csv("walk_forward_results.csv", index=False)
    
    sys.exit()


if __name__ == "__main__":
    main()
