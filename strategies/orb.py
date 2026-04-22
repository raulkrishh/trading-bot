"""
Opening Range Breakout (ORB) — Intraday Strategy
=================================================

Concept
-------
The **opening range** is defined by the High and Low of the first N 5-minute
candles (default: 3 candles = 15 minutes, 09:30–09:44 ET).  When a subsequent
candle *closes* outside that range with confirming indicators (VWAP side, EMA
trend, RSI momentum, volume surge), the strategy enters in the direction of the
breakout and targets a measured move equal to the full range height.

Signal Logic
------------
Long setup (upside breakout):
  1. Candle closes above OR_High.
  2. Close is above session VWAP.
  3. 20-EMA is rising (current bar EMA > previous bar EMA).
  4. RSI (14) > rsi_long_min (default 55).
  5. Bar volume > vol_multiplier × 20-bar rolling average.

Short setup (downside breakout):
  1. Candle closes below OR_Low.
  2. Close is below session VWAP.
  3. 20-EMA is falling (current bar EMA < previous bar EMA).
  4. RSI (14) < rsi_short_max (default 45).
  5. Same volume filter.

Exit Logic
----------
- Take Profit : entry ± range_height (measured move).
- Stop Loss   : VWAP at entry time (not the candle low/high).
- EOD cutoff  : any open position is closed at `eod_exit_time` (default 15:30 ET).

Time Filter
-----------
- Opening range forms during the first `or_candles` bars after market open.
- New entries are accepted only up to `trade_end_time` (default 11:00 ET) —
  after that, volume and volatility drop and the setup degrades.
- Only one trade per session (first valid signal wins).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal
import pandas as pd
import numpy as np

from utils.indicators import rsi as calc_rsi, average_volume, ema as calc_ema, vwap as calc_vwap


Side = Literal["long", "short"]


@dataclass
class ORBTrade:
    side: Side
    entry_price: float
    entry_time: pd.Timestamp
    or_high: float
    or_low: float
    range_height: float
    sl_price: float
    tp_price: float
    exit_price: float | None = None
    exit_time: pd.Timestamp | None = None
    exit_reason: str = ""
    shares: int = 1

    @property
    def pnl(self) -> float:
        if self.exit_price is None:
            return 0.0
        if self.side == "long":
            return (self.exit_price - self.entry_price) * self.shares
        return (self.entry_price - self.exit_price) * self.shares

    @property
    def pnl_pct(self) -> float:
        if self.exit_price is None:
            return 0.0
        raw = (self.exit_price - self.entry_price) / self.entry_price
        return (raw if self.side == "long" else -raw) * 100

    def to_dict(self) -> dict:
        return {
            "side": self.side,
            "entry_time": self.entry_time,
            "entry_price": self.entry_price,
            "exit_time": self.exit_time,
            "exit_price": self.exit_price,
            "exit_reason": self.exit_reason,
            "or_high": self.or_high,
            "or_low": self.or_low,
            "range_height": self.range_height,
            "sl_price": self.sl_price,
            "tp_price": self.tp_price,
            "pnl": self.pnl,
            "pnl_pct": self.pnl_pct,
        }


@dataclass
class ORBConfig:
    # --- Opening range ---
    or_candles: int = 3
    """Number of bars that define the opening range (3 × 5m = 15 min)."""

    # --- Confirmation filters ---
    rsi_long_min: float = 55.0
    """RSI must be above this to take a long breakout."""

    rsi_short_max: float = 45.0
    """RSI must be below this to take a short breakout."""

    ema_period: int = 20
    """Period for the EMA trend filter."""

    vol_multiplier: float = 2.0
    """Bar volume must exceed this multiple of the rolling average."""

    vol_avg_window: int = 20
    """Rolling window (bars) for average volume computation."""

    # --- Time filters ---
    trade_end_time: str = "11:00"
    """No new entries after this time (ET)."""

    eod_exit_time: str = "15:30"
    """Force-close any open position at or after this time (ET)."""

    # --- Position sizing ---
    shares_per_trade: int = 100
    """Fixed share size per trade."""


class ORBStrategy:
    """
    Runs the Opening Range Breakout strategy bar-by-bar on a single day's data.

    Usage
    -----
    strategy = ORBStrategy(config)
    trades = strategy.run_day(day_bars, start_line=None)   # start_line ignored
    """

    def __init__(self, config: ORBConfig | None = None) -> None:
        self.config = config or ORBConfig()

    def run_day(
        self,
        day_bars: pd.DataFrame,
        start_line: float | None = None,  # accepted but not used (backtester compat)
    ) -> list[ORBTrade]:
        cfg = self.config
        bars = day_bars.copy()

        if isinstance(bars.columns, pd.MultiIndex):
            bars.columns = bars.columns.get_level_values(0)

        if len(bars) < cfg.or_candles + 1:
            return []

        # ── Pre-compute indicators ─────────────────────────────────────
        bars["rsi"]     = calc_rsi(bars["Close"], period=14)
        bars["ema"]     = calc_ema(bars["Close"], period=cfg.ema_period)
        bars["avg_vol"] = average_volume(bars["Volume"], window=cfg.vol_avg_window)
        bars["vwap"]    = calc_vwap(bars)

        # ── Derive Opening Range from first N bars ─────────────────────
        or_bars  = bars.iloc[: cfg.or_candles]
        or_high  = float(or_bars["High"].max())
        or_low   = float(or_bars["Low"].min())
        range_h  = or_high - or_low

        if range_h <= 0:
            return []

        trade_end = pd.Timestamp(f"1970-01-01 {cfg.trade_end_time}").time()
        eod_exit  = pd.Timestamp(f"1970-01-01 {cfg.eod_exit_time}").time()

        open_trade: ORBTrade | None = None
        completed: list[ORBTrade] = []
        traded_today = False

        ema_values = bars["ema"].values

        for i, (ts, bar) in enumerate(bars.iterrows()):
            # Skip the range-forming candles — no trades during open range
            if i < cfg.or_candles:
                continue

            bar_time = ts.time()
            close    = float(bar["Close"])
            high_    = float(bar["High"])
            low_     = float(bar["Low"])
            volume   = float(bar["Volume"])
            avg_vol  = float(bar["avg_vol"]) if not np.isnan(bar["avg_vol"]) else 0.0
            bar_rsi  = float(bar["rsi"])     if not np.isnan(bar["rsi"])     else 50.0
            bar_vwap = float(bar["vwap"])    if not np.isnan(bar["vwap"])    else close
            bar_ema  = ema_values[i]
            prev_ema = ema_values[i - 1] if i > 0 else bar_ema

            # ── EOD force-exit ─────────────────────────────────────────
            if open_trade is not None and bar_time >= eod_exit:
                open_trade = self._close(open_trade, close, ts, "EOD", completed)
                continue

            # ── Manage open trade ──────────────────────────────────────
            if open_trade is not None:
                hit_sl, hit_tp = self._check_exit(open_trade, high_, low_, close)
                if hit_sl:
                    open_trade = self._close(open_trade, open_trade.sl_price, ts, "SL", completed)
                elif hit_tp:
                    open_trade = self._close(open_trade, open_trade.tp_price, ts, "TP", completed)
                continue

            # ── Look for entry ─────────────────────────────────────────
            if traded_today or bar_time > trade_end:
                continue

            ema_rising  = bar_ema > prev_ema
            ema_falling = bar_ema < prev_ema
            vol_ok      = avg_vol > 0 and volume >= cfg.vol_multiplier * avg_vol

            signal: Side | None = None

            if (
                close > or_high
                and close > bar_vwap
                and ema_rising
                and bar_rsi > cfg.rsi_long_min
                and vol_ok
            ):
                signal = "long"
            elif (
                close < or_low
                and close < bar_vwap
                and ema_falling
                and bar_rsi < cfg.rsi_short_max
                and vol_ok
            ):
                signal = "short"

            if signal is None:
                continue

            # SL at VWAP (opposite side of trade); TP = measured move
            if signal == "long":
                sl_price = bar_vwap
                tp_price = close + range_h
            else:
                sl_price = bar_vwap
                tp_price = close - range_h

            open_trade = ORBTrade(
                side=signal,
                entry_price=close,
                entry_time=ts,
                or_high=or_high,
                or_low=or_low,
                range_height=range_h,
                sl_price=sl_price,
                tp_price=tp_price,
                shares=cfg.shares_per_trade,
            )
            traded_today = True

        # Residual EOD close
        if open_trade is not None and not bars.empty:
            last_close = float(bars.iloc[-1]["Close"])
            last_ts    = bars.index[-1]
            open_trade = self._close(open_trade, last_close, last_ts, "EOD", completed)

        return completed

    # ------------------------------------------------------------------

    @staticmethod
    def _check_exit(
        trade: ORBTrade,
        high: float,
        low: float,
        close: float,
    ) -> tuple[bool, bool]:
        if trade.side == "long":
            hit_sl = low  <= trade.sl_price
            hit_tp = high >= trade.tp_price or close >= trade.tp_price
        else:
            hit_sl = high >= trade.sl_price
            hit_tp = low  <= trade.tp_price or close <= trade.tp_price
        return hit_sl, hit_tp

    @staticmethod
    def _close(
        trade: ORBTrade,
        exit_price: float,
        exit_time: pd.Timestamp,
        reason: str,
        completed: list[ORBTrade],
    ) -> None:
        trade.exit_price  = exit_price
        trade.exit_time   = exit_time
        trade.exit_reason = reason
        completed.append(trade)
        return None
