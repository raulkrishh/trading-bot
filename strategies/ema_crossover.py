"""
EMA Crossover Strategy — 1-Minute Intraday
============================================

Signal Logic
------------
Long entry:
  1. EMA8 crosses above EMA20 (was below or equal on prior bar, now above).
  2. EMA8 slope is positive (current EMA8 > previous EMA8) — confirms momentum.

Short entry (optional, enabled by allow_short):
  1. EMA8 crosses below EMA20.

Exit Logic
----------
- Long exit : EMA8 crosses back below EMA20, OR stop-loss hit, OR EOD.
- Short exit: EMA8 crosses back above EMA20 (with EMA8 rising), OR SL hit, OR EOD.
- EOD cutoff: any open position force-closed at eod_exit_time.
"""

from __future__ import annotations

from dataclasses import dataclass
import pandas as pd
import numpy as np

from utils.indicators import ema as calc_ema
from strategies.bounce_back import Trade, Side


@dataclass
class EmaCrossoverConfig:
    ema_fast: int = 8
    ema_slow: int = 20
    sl_pct: float = 1.0
    """Stop-loss distance from entry as a percentage; 0 disables fixed SL."""

    shares_per_trade: int = 100
    trade_start_time: str = "09:45"
    trade_end_time: str = "15:00"
    eod_exit_time: str = "15:30"
    allow_short: bool = True
    """If False, only long trades are taken; cross-downs simply close the long."""


class EmaCrossoverStrategy:
    """Runs EMA8/EMA20 crossover strategy bar-by-bar on a single day's data."""

    requires_start_line = False

    def __init__(self, config: EmaCrossoverConfig | None = None) -> None:
        self.config = config or EmaCrossoverConfig()

    def run_day(self, day_bars: pd.DataFrame, start_line: float = 0.0) -> list[Trade]:
        cfg = self.config
        bars = day_bars.copy()

        if isinstance(bars.columns, pd.MultiIndex):
            bars.columns = bars.columns.get_level_values(0)

        bars["ema_fast"] = calc_ema(bars["Close"], cfg.ema_fast)
        bars["ema_slow"] = calc_ema(bars["Close"], cfg.ema_slow)
        bars = bars.dropna(subset=["ema_fast", "ema_slow"])

        if len(bars) < 2:
            return []

        trade_start = pd.Timestamp(f"1970-01-01 {cfg.trade_start_time}").time()
        trade_end   = pd.Timestamp(f"1970-01-01 {cfg.trade_end_time}").time()
        eod_exit    = pd.Timestamp(f"1970-01-01 {cfg.eod_exit_time}").time()

        open_trade: Trade | None = None
        completed: list[Trade] = []

        prev_fast: float | None = None
        prev_slow: float | None = None

        for ts, bar in bars.iterrows():
            bar_time  = ts.time()
            close     = float(bar["Close"])
            curr_fast = float(bar["ema_fast"])
            curr_slow = float(bar["ema_slow"])

            # ── EOD force-close ────────────────────────────────────────
            if open_trade is not None and bar_time >= eod_exit:
                self._close(open_trade, close, ts, "EOD", completed)
                open_trade = None
                prev_fast, prev_slow = curr_fast, curr_slow
                continue

            # ── Manage open trade ──────────────────────────────────────
            if open_trade is not None and prev_fast is not None:
                # Stop-loss check
                if cfg.sl_pct > 0:
                    if open_trade.side == "long" and float(bar["Low"]) <= open_trade.sl_price:
                        self._close(open_trade, open_trade.sl_price, ts, "SL", completed)
                        open_trade = None
                        prev_fast, prev_slow = curr_fast, curr_slow
                        continue
                    if open_trade.side == "short" and float(bar["High"]) >= open_trade.sl_price:
                        self._close(open_trade, open_trade.sl_price, ts, "SL", completed)
                        open_trade = None
                        prev_fast, prev_slow = curr_fast, curr_slow
                        continue

                # Crossover exit
                cross_down = prev_fast >= prev_slow and curr_fast < curr_slow
                cross_up   = prev_fast <= prev_slow and curr_fast > curr_slow

                if open_trade.side == "long" and cross_down:
                    self._close(open_trade, close, ts, "cross_exit", completed)
                    open_trade = None
                    if cfg.allow_short and bar_time <= trade_end:
                        open_trade = self._open("short", close, ts, curr_slow, cfg)
                    prev_fast, prev_slow = curr_fast, curr_slow
                    continue

                if open_trade.side == "short" and cross_up and curr_fast > prev_fast:
                    self._close(open_trade, close, ts, "cross_exit", completed)
                    open_trade = None
                    if bar_time <= trade_end:
                        open_trade = self._open("long", close, ts, curr_slow, cfg)
                    prev_fast, prev_slow = curr_fast, curr_slow
                    continue

            # ── Look for new entry ─────────────────────────────────────
            if open_trade is None and prev_fast is not None:
                if trade_start <= bar_time <= trade_end:
                    cross_up   = prev_fast <= prev_slow and curr_fast > curr_slow
                    cross_down = prev_fast >= prev_slow and curr_fast < curr_slow
                    ema8_rising = curr_fast > prev_fast

                    if cross_up and ema8_rising:
                        open_trade = self._open("long", close, ts, curr_slow, cfg)
                    elif cross_down and cfg.allow_short:
                        open_trade = self._open("short", close, ts, curr_slow, cfg)

            prev_fast, prev_slow = curr_fast, curr_slow

        # Force-close any remaining position
        if open_trade is not None and not bars.empty:
            self._close(open_trade, float(bars.iloc[-1]["Close"]), bars.index[-1], "EOD", completed)

        return completed

    def _open(self, side: Side, price: float, ts: pd.Timestamp, ema_slow: float, cfg: EmaCrossoverConfig) -> Trade:
        sl_mult = cfg.sl_pct / 100
        if side == "long":
            sl_price = price * (1 - sl_mult) if cfg.sl_pct > 0 else 0.0
        else:
            sl_price = price * (1 + sl_mult) if cfg.sl_pct > 0 else float("inf")
        return Trade(
            side=side,
            entry_price=price,
            entry_time=ts,
            start_line=ema_slow,
            sl_price=sl_price,
            tp_price=0.0,
            shares=cfg.shares_per_trade,
        )

    @staticmethod
    def _close(trade: Trade, exit_price: float, exit_time: pd.Timestamp, reason: str, completed: list[Trade]) -> None:
        trade.exit_price  = exit_price
        trade.exit_time   = exit_time
        trade.exit_reason = reason
        completed.append(trade)
