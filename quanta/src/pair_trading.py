import pandas as pd
from statsmodels.tsa.stattools import coint
from itertools import combinations
import statsmodels.api as sm
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


def cointegration_filter(training_df):
    """
    Finds cointegrated ETFs from training data via Engle-Granger.

    Args:
        price_df (Pandas DataFrame): Contains open/close and high/low prices of ETFs in specific industries (specified below).

    Returns:
        Pandas DataFrame: Likely cointegrated ETFs. 
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
    results_df = results_df.reset_index()
    results_df = results_df.drop("index", axis=1)


    # Keep likely cointegrated pairs (e.g. p < 0.05):
    sig = results_df[results_df['p_value'] < 0.05]
    sig = sig.reset_index()
    sig = sig.drop("index", axis=1)

    # Save to CSV:
    results_df.to_csv('../data/all_cointegration_results.csv', index=True)
    sig.to_csv('../data/significant_cointegrated_pairs.csv', index=True)
    
    return sig


def calculate_spreads(backtesting_df, sig):
    """Calculate hedging ratio, spread between statistically significantly cointegrated ETFs.

    Args:
        backtesting_df (Pandas DataFrame): Contains open/close and high/low prices of the same ETFs as training data. More recent dates than training data.

    Returns:
        Pandas List[Dict]: Contains various metrics between cointegrated ETFs (e.g. hedge ratio, spread, mean, SD).
    """
    pairs = sig['pair'].str.split(' & ').apply(tuple).tolist()
    
    results = []
    for etf1, etf2 in pairs:
        col1 = f"Close_{etf1}"
        col2 = f"Close_{etf2}"
        
        if col1 not in backtesting_df.columns or col2 not in backtesting_df.columns:
            print(f"Missing columns for pair: {etf1}, {etf2}")
            continue
        
        pair_df = backtesting_df[[col1, col2]].dropna()
        
        # Run OLS: ETF1 ~ ETF2
        X = sm.add_constant(np.log(pair_df[col2]))
        y = np.log(pair_df[col1])
        model = sm.OLS(y, X).fit()
        alpha, hedge_ratio = model.params
        
        spread = y - (alpha + hedge_ratio * np.log(pair_df[col2]))
        pair_df["spread"] = spread

        # Statistic metrics of spread
        mean = np.mean(spread)
        sd = np.std(spread, ddof=1)
        
        results.append({
            "ETF1": etf1,
            "ETF2": etf2,
            "hedge_ratio": hedge_ratio,
            "spread_prices": pair_df,
            "mean": mean,
            "SD": sd
        })
        
        def spread_graphs():
            for r in results:
                print(f"{r['ETF1']} & {r['ETF2']} → hedge ratio: {r['hedge_ratio']:.4f}")
                print(r['spread_prices'].head(), "\n")
            
                sns.lineplot(data=r['spread_prices']['spread'])
                plt.axhline(y=r['mean'], color='red', linestyle='--', linewidth=2)
                plt.xlabel('Date')
                plt.ylabel('Spread')
                plt.title(f"{r['ETF1']} & {r['ETF2']} Spread Over Time")
                plt.show()
        
        # spread_graphs()
    
    return results


def generate_signals(spreads, entry_z=2.0, exit_z=0.5):
    """
    Extended version with:
    - Entry when |zscore| > entry_z
    - Exit when |zscore| < exit_z
    - Position held until exit condition
    - PnL tracking
    """
    trades = {}

    for sp in spreads:
        etf1, etf2 = sp["ETF1"], sp["ETF2"]
        hr, df = sp["hedge_ratio"], sp["spread_prices"].copy()
        mu, sd = sp["mean"], sp["SD"]

        # Compute z-score with window, avoid look-ahead bias
        window = 60  
        df["mu"] = df["spread"].rolling(window).mean()
        df["sd"] = df["spread"].rolling(window).std()
        df["zscore"] = (df["spread"] - df["mu"]) / df["sd"]

        # Signal logic
        df["signal"] = 0
        df.loc[df["zscore"] >  entry_z, "signal"] = -1   # Short spread
        df.loc[df["zscore"] < -entry_z, "signal"] =  1   # Long spread

        # Position holding until exit
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

            df.at[df.index[i], "position"] = current_position

        # Spread return
        df["spread_return"] = df["spread"].pct_change()
        df["pnl"] = df["position"].shift(1) * df["spread_return"]

        # Save full record
        trades[f"{etf1} & {etf2}"] = df[[
            "spread", "zscore", "signal", "position", "spread_return", "pnl"
        ]]
        
    for pair in trades:
        print(trades[pair])
                
    return trades




def main():
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    training_df = pd.read_csv('../data/training.csv', index_col=0, parse_dates=True)
    backtesting_df = pd.read_csv('../data/backtesting.csv', index_col=0, parse_dates=True)
    sig = cointegration_filter(training_df)
    spreads = calculate_spreads(backtesting_df, sig)
    trades = generate_signals(spreads)
    
    
    

if __name__ == "__main__":
    main()
