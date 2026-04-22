"""
Technical indicator helpers used by strategies.
"""

import numpy as np
import pandas as pd


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def average_volume(volume: pd.Series, window: int = 20) -> pd.Series:
    return volume.rolling(window=window, min_periods=1).mean()


def vwap(df: pd.DataFrame) -> pd.Series:
    """Session VWAP — resets each day if df spans multiple days."""
    tp = (df["High"] + df["Low"] + df["Close"]) / 3
    cum_vol = df["Volume"].cumsum()
    cum_tp_vol = (tp * df["Volume"]).cumsum()
    return cum_tp_vol / cum_vol
