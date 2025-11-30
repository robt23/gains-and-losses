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
    Download daily OHLCV data for a list of ETFs from Yahoo Finance.

    Data is pulled between the specified start and end dates and returned in a
    wide format with columns such as 'Open_SPY', 'High_SPY', etc.

    Args:
        etfs (list[str]):
            List of ETF tickers to download.

        start_date (str or datetime-like):
            Start date for the download (inclusive).

        end_date (str or datetime-like):
            End date for the download (exclusive or inclusive per yfinance).

    Returns:
        pd.DataFrame:
            OHLCV data indexed by date, with one column per (field, ticker).
    """
    df = yf.download(etfs, start=start_date, end=end_date, auto_adjust=False)

    # Stack ticker level into columns
    df = df.stack(level=1).rename_axis(['Date', 'Ticker']).reset_index()

    df = df.pivot(
        index='Date',
        columns='Ticker',
        values=['Open', 'High', 'Low', 'Close', 'Volume', 'Adj Close']
    )

    # Flatten MultiIndex columns
    df.columns = [f"{col[0]}_{col[1]}" for col in df.columns]
    df.sort_index(inplace=True)
    return df


def load_macro_features(series_map, start_date, end_date):
    """
    Load macroeconomic time series from FRED.

    Uses pandas_datareader to pull one or more FRED series and renames them
    according to the provided mapping.

    Args:
        series_map (dict[str, str]):
            Mapping from FRED series codes to desired column names,
            e.g. {"VIXCLS": "VIX", "DGS10": "yield10Y"}.

        start_date (str or datetime-like):
            Start date for the series.

        end_date (str or datetime-like):
            End date for the series.

    Returns:
        pd.DataFrame:
            DataFrame indexed by date with one column per macro feature.
    """
    fred_codes = list(series_map.keys())
    df = pdr.DataReader(fred_codes, "fred", start_date, end_date)
    df.rename(columns=series_map, inplace=True)
    return df


def add_technical_indicators(df, etfs):
    """
    Compute basic technical indicators for each ETF and append them to df.

    Indicators:
        - SMA50: 50-day simple moving average of Close
        - EMA20: 20-day exponential moving average of Close
        - RSI14: 14-period relative strength index
        - STD20: 20-day rolling standard deviation of Close

    Args:
        df (pd.DataFrame):
            DataFrame containing at least 'Close_<ticker>' for each ETF.

        etfs (list[str]):
            List of ETF tickers for which to compute indicators.

    Returns:
        pd.DataFrame:
            The original DataFrame with new indicator columns added.
    """
    for ticker in etfs:
        close = df[f"Close_{ticker}"]
        df[f"SMA50_{ticker}"] = close.rolling(window=50).mean()
        df[f"EMA20_{ticker}"] = close.ewm(span=20, adjust=False).mean()
        df[f"RSI14_{ticker}"] = ta.rsi(close, length=14)
        df[f"STD20_{ticker}"] = close.rolling(window=20).std()
    return df


def add_calendar_flags(df, market_calendar="NYSE"):
    """
    Add calendar- and event-based features to the DataFrame.

    Features:
        - day_of_week: integer (0=Monday,...,4=Friday, etc.)
        - month: calendar month (1–12)
        - is_holiday: 1 if date is a US federal holiday, else 0
        - is_trading_day: 1 if date is an exchange trading day, else 0

    Args:
        df (pd.DataFrame):
            DataFrame indexed by date (or convertible to DatetimeIndex).

        market_calendar (str):
            Exchange calendar name used by pandas_market_calendars,
            e.g. 'NYSE'.

    Returns:
        pd.DataFrame:
            Copy of df with calendar flag columns added.
    """
    df = df.copy()
    df.index = pd.to_datetime(df.index)

    # Calendar features
    df["day_of_week"] = df.index.dayofweek
    df["month"] = df.index.month

    # US federal holidays
    us_hols = holidays.US()
    df["is_holiday"] = df.index.map(lambda d: 1 if d in us_hols else 0)

    # Trading days per exchange
    cal = mcal.get_calendar(market_calendar)
    schedule = cal.schedule(start_date=df.index.min(), end_date=df.index.max())
    trading_days = schedule.index
    df["is_trading_day"] = df.index.isin(trading_days).astype(int)

    return df


def prepare_master_dataframe(etfs, start_date, end_date, fred_map):
    """
    Build a master feature DataFrame for modeling from ETF and macro data.

    This function orchestrates price loading, macro loading, technical
    indicator computation, and calendar flag generation, and then performs
    basic cleaning (forward-fill and NaN dropping on core price columns).

    Args:
        etfs (list[str]):
            List of ETF tickers, e.g. ['SPY', 'VOO', 'IVV'].

        start_date (str or datetime-like):
            Start date for all data sources.

        end_date (str or datetime-like):
            End date for all data sources.

        fred_map (dict[str, str]):
            Mapping of FRED series codes to desired macro column names.

    Returns:
        pd.DataFrame:
            Cleaned and enriched feature DataFrame indexed by date.
    """
    # 1. Price data
    df_prices = load_ohlcv(etfs, start_date, end_date)

    # 2. Macro factors
    df_macro = load_macro_features(fred_map, start_date, end_date)

    # 3. Merge on date
    df = df_prices.join(df_macro, how="left")

    # 4. Technical indicators
    df = add_technical_indicators(df, etfs)

    # 5. Calendar & event flags
    df = add_calendar_flags(df)

    # 6. Clean:
    #    - sort by date
    #    - forward-fill all columns
    #    - only require core Close_ columns to be present
    df = df.sort_index().ffill()

    price_cols = [f"Close_{t}" for t in etfs]
    df = df.dropna(subset=price_cols)

    return df

