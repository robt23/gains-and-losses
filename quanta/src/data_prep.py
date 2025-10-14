import pandas as pd
import yfinance as yf
from pandas_datareader import data as pdr
import pandas_market_calendars as mcal
import holidays

# New np changes NaN casing
import numpy as np
if not hasattr(np, "NaN"):
    np.NaN = np.nan


import pandas_ta as ta


def load_ohlcv(etfs, start_date, end_date):
    """
    Download OHLCV for given ETFs between start_date and end_date.
    Returns a DataFrame with columns like 'Open_SPY', 'High_SPY', ...
    """
    # Download from Yahoo Finance
    df = yf.download(etfs, start=start_date, end=end_date, auto_adjust=False)
    # Stack ticker level into columns
    df = df.stack(level=1).rename_axis(['Date','Ticker']).reset_index()
    df = df.pivot(index='Date', columns='Ticker', values=['Open','High','Low','Close','Volume','Adj Close'])
    # Flatten MultiIndex columns
    df.columns = [f"{col[0]}_{col[1]}" for col in df.columns]
    df.sort_index(inplace=True)
    return df


def load_macro_features(series_map, start_date, end_date):
    """
    Pull macro series from FRED.
    series_map: dict of {fred_series_code: desired_column_name}
    """
    fred_codes = list(series_map.keys())
    df = pdr.DataReader(fred_codes, 'fred', start_date, end_date)
    df.rename(columns=series_map, inplace=True)
    return df


def add_technical_indicators(df, etfs):
    """
    For each ticker in etfs, compute a handful of common indicators:
      - SMA50, EMA20, RSI14, Rolling Std (20)
    Appends new columns to df in-place and returns df.
    """
    for ticker in etfs:
        close = df[f"Close_{ticker}"]
        df[f"SMA50_{ticker}"] = close.rolling(window=50).mean()
        df[f"EMA20_{ticker}"] = close.ewm(span=20, adjust=False).mean()
        df[f"RSI14_{ticker}"] = ta.rsi(close, length=14)
        df[f"STD20_{ticker}"] = close.rolling(window=20).std()
    return df


def add_calendar_flags(df, market_calendar='NYSE'):
    """
    Add calendar-based features:
      - day_of_week (0=Mon,...)
      - month
      - is_holiday (US federal)
      - is_trading_day (per market calendar)
    Returns df with new columns.
    """
    # Ensure index is DatetimeIndex
    df = df.copy()
    df.index = pd.to_datetime(df.index)

    # Calendar features
    df['day_of_week'] = df.index.dayofweek
    df['month'] = df.index.month

    # US federal holidays
    us_hols = holidays.US()
    df['is_holiday'] = df.index.map(lambda d: 1 if d in us_hols else 0)

    # Trading days per exchange
    cal = mcal.get_calendar(market_calendar)
    schedule = cal.schedule(start_date=df.index.min(), end_date=df.index.max())
    trading_days = schedule.index
    df['is_trading_day'] = df.index.isin(trading_days).astype(int)

    return df


def prepare_master_dataframe(etfs, start_date, end_date, fred_map):
    """
    Load prices, macro, compute indicators and flags into one DataFrame.
    - etfs: list of string tickers, e.g. ['SPY','VOO','IVV']
    - fred_map: dict {'VIXCLS':'VIX','DGS10':'yield10Y'}
    Returns cleaned DataFrame ready for modeling.
    """
    # 1. Price data
    df_prices = load_ohlcv(etfs, start_date, end_date)

    # 2. Macro factors
    df_macro = load_macro_features(fred_map, start_date, end_date)

    # 3. Merge on date
    df = df_prices.join(df_macro, how='left')

    # 4. Technical indicators
    df = add_technical_indicators(df, etfs)

    # 5. Calendar & event flags
    df = add_calendar_flags(df)

    # 6. Clean: forward-fill then drop remaining NaNs
    df = df.sort_index().ffill().dropna()

    return df
