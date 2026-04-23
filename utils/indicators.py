"""
Technical indicator helpers used by strategies.
"""

import math
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


def bollinger_bands(
    series: pd.Series,
    period: int = 20,
    std_dev: float = 2.0,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """
    Compute Bollinger Bands.

    Returns (middle, upper, lower) where:
      middle = rolling SMA(period)
      upper  = middle + std_dev × rolling_std
      lower  = middle − std_dev × rolling_std
    """
    sma = series.rolling(window=period, min_periods=period).mean()
    std = series.rolling(window=period, min_periods=period).std(ddof=1)
    return sma, sma + std_dev * std, sma - std_dev * std


def historical_volatility(prices: pd.Series, window: int = 20) -> pd.Series:
    """
    Annualised historical volatility from log returns.
    Returns NaN for the first `window` rows.
    """
    log_ret = np.log(prices / prices.shift(1))
    return log_ret.rolling(window=window, min_periods=window).std(ddof=1) * math.sqrt(252)


def beta(asset_returns: pd.Series, market_returns: pd.Series) -> float:
    """
    OLS beta of `asset_returns` relative to `market_returns`.
    Both series should be simple (not log) daily returns.
    Returns 1.0 if there is insufficient data.
    """
    df = pd.concat([asset_returns, market_returns], axis=1).dropna()
    if len(df) < 20:
        return 1.0
    cov_matrix = np.cov(df.values.T)
    market_var = cov_matrix[1, 1]
    return float(cov_matrix[0, 1] / market_var) if market_var != 0 else 1.0


def momentum_score(prices: pd.Series, skip_months: int = 1, lookback_months: int = 12) -> float:
    """
    12-1 month price momentum: total return from (lookback_months + skip_months)
    ago to skip_months ago.  Skipping the most recent month avoids the
    short-term reversal effect documented in academic literature.

    Assumes ~21 trading days per calendar month.
    Returns 0.0 if there is insufficient history.
    """
    bars_per_month = 21
    n_start = (lookback_months + skip_months) * bars_per_month
    n_end   = skip_months * bars_per_month

    if len(prices) < n_start:
        return 0.0

    p_start = float(prices.iloc[-n_start])
    p_end   = float(prices.iloc[-n_end])
    return (p_end - p_start) / p_start if p_start != 0 else 0.0


# ------------------------------------------------------------------
# Black-Scholes helpers (used by theta_decay strategy)
# ------------------------------------------------------------------

def _norm_cdf(x: float) -> float:
    """Standard normal CDF via math.erf — no scipy dependency."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _norm_pdf(x: float) -> float:
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


def bsm_put_price(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """
    Black-Scholes European put price.

    S     : current stock price
    K     : strike price
    T     : time to expiry in years  (e.g. 35/365)
    r     : risk-free rate (e.g. 0.05)
    sigma : annualised implied/historical volatility (e.g. 0.25)
    """
    if T <= 0:
        return max(K - S, 0.0)
    if sigma <= 0:
        return max(K * math.exp(-r * T) - S, 0.0)

    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    return K * math.exp(-r * T) * _norm_cdf(-d2) - S * _norm_cdf(-d1)


def bsm_put_delta(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """Put delta — always in [−1, 0]."""
    if T <= 0 or sigma <= 0:
        return -1.0 if S < K else 0.0
    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    return _norm_cdf(d1) - 1.0


def find_put_strike_for_delta(
    S: float,
    T: float,
    sigma: float,
    r: float = 0.05,
    target_delta: float = -0.30,
) -> float:
    """
    Binary-search for the put strike that yields approximately `target_delta`.

    target_delta should be negative (e.g. −0.30 for a 30-delta put).
    Searches between 50% OTM (K = 0.5×S) and ATM (K = S).
    """
    low, high = 0.5 * S, S
    for _ in range(60):
        mid = (low + high) / 2.0
        delta = bsm_put_delta(S, mid, T, r, sigma)
        if delta > target_delta:
            # delta too close to 0 (too OTM) — raise strike
            low = mid
        else:
            # delta too negative (too ITM) — lower strike
            high = mid
    return round((low + high) / 2.0, 2)
