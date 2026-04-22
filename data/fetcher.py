"""
Market data fetching via yfinance.
Provides daily OHLCV and intraday minute-level bars.
"""

import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta


def fetch_daily(ticker: str, lookback_days: int = 60) -> pd.DataFrame:
    """Fetch daily OHLCV bars for the past `lookback_days` calendar days."""
    end = datetime.today()
    start = end - timedelta(days=lookback_days)
    df = yf.download(ticker, start=start.strftime("%Y-%m-%d"),
                     end=end.strftime("%Y-%m-%d"), interval="1d",
                     auto_adjust=True, progress=False)
    df.index = pd.to_datetime(df.index)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df


def fetch_intraday(ticker: str, interval: str = "5m", lookback_days: int = 59) -> pd.DataFrame:
    """
    Fetch intraday OHLCV bars.
    yfinance supports up to 60 days of history for intervals <= 1h.
    interval: '1m','2m','5m','15m','30m','60m'
    """
    end = datetime.today()
    start = end - timedelta(days=lookback_days)
    df = yf.download(ticker, start=start.strftime("%Y-%m-%d"),
                     end=end.strftime("%Y-%m-%d"), interval=interval,
                     auto_adjust=True, progress=False)
    df.index = pd.to_datetime(df.index)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    if df.index.tz is None:
        df.index = df.index.tz_localize("America/New_York")
    else:
        df.index = df.index.tz_convert("America/New_York")
    return df


def get_previous_day_open(daily_df: pd.DataFrame, date: pd.Timestamp) -> float:
    """
    Return the opening price of the trading day immediately before `date`.
    `date` should be a date with or without timezone.
    """
    date_only = pd.Timestamp(date).normalize().tz_localize(None)
    idx = daily_df.index.tz_localize(None) if daily_df.index.tz else daily_df.index
    past = daily_df[idx.normalize() < date_only]
    if past.empty:
        raise ValueError(f"No trading day found before {date_only.date()}")
    prev_open = float(past.iloc[-1]["Open"])
    return prev_open


def split_by_day(intraday_df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Split an intraday DataFrame into {date_str: day_df} mapping."""
    groups = {}
    for date, group in intraday_df.groupby(intraday_df.index.normalize()):
        groups[str(date.date())] = group.copy()
    return groups
