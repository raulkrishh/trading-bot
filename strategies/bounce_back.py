"""
Bounce Back from Previous Day's Open – Intraday Strategy
=========================================================

Concept
-------
The *start line* is the **previous trading day's opening price**.  It acts as
a gravitational anchor: when intraday price deviates significantly from that
level and exhaustion signals appear, the strategy fades the move and targets
a return to the start line.

Signal Logic
------------
Long setup (fade a down-move):
  1. Current bar's close is at least `entry_pct`% BELOW the start line.
  2. RSI (14) < `rsi_oversold` (default 35) — momentum is exhausted.
  3. Bar volume > `vol_multiplier` × 20-bar average — institutional participation.
  4. No existing open position.

Short setup (fade an up-move):
  1. Current bar's close is at least `entry_pct`% ABOVE the start line.
  2. RSI (14) > `rsi_overbought` (default 65) — momentum is exhausted.
  3. Same volume filter.
  4. No existing open position.

Exit Logic
----------
- Take Profit : price crosses back to within `tp_buffer_pct`% of the start line.
- Stop Loss   : price moves `sl_pct`% further against the position from entry.
- EOD cutoff  : any open position is closed at the first bar on/after
                `eod_exit_time` (default 15:30 ET).

Time Filter
-----------
Trades are only opened between `trade_start_time` and `trade_end_time`
(default 09:45–15:00 ET) to avoid the chaotic open and pre-close liquidity drop.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal
import pandas as pd
import numpy as np

from utils.indicators import rsi as calc_rsi, average_volume


Side = Literal["long", "short"]


@dataclass
class Trade:
    side: Side
    entry_price: float
    entry_time: pd.Timestamp
    start_line: float
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
            "start_line": self.start_line,
            "sl_price": self.sl_price,
            "tp_price": self.tp_price,
            "pnl": self.pnl,
            "pnl_pct": self.pnl_pct,
        }


@dataclass
class BounceBackConfig:
    # --- Entry thresholds ---
    entry_pct: float = 0.75
    """Min % distance from start line required to trigger an entry."""

    rsi_oversold: float = 35.0
    """RSI threshold below which long entries are considered."""

    rsi_overbought: float = 65.0
    """RSI threshold above which short entries are considered."""

    vol_multiplier: float = 1.2
    """Current bar volume must exceed this multiple of the rolling avg."""

    vol_avg_window: int = 20
    """Rolling window (bars) used to compute average volume."""

    # --- Exit thresholds ---
    sl_pct: float = 0.50
    """Stop-loss distance from entry price as a percentage."""

    tp_buffer_pct: float = 0.10
    """
    Take-profit triggers when price is within this % of the start line.
    A value of 0.10 means TP fires when price is ≤ 0.10% away from start line.
    """

    # --- Time filters (HH:MM strings, Eastern time) ---
    trade_start_time: str = "09:45"
    """Earliest time a new position may be opened."""

    trade_end_time: str = "15:00"
    """Latest time a new position may be opened."""

    eod_exit_time: str = "15:30"
    """Any open position is force-closed at or after this time."""

    # --- VIX filter ---
    vix_min: float = 0.0
    """Only open new positions when prior day VIX close exceeds this level. 0 = disabled."""

    # --- EMA crossover exit ---
    ema_fast: int = 8
    """Fast EMA period for crossover exit signal."""

    ema_slow: int = 20
    """Slow EMA period for crossover exit signal."""

    # --- Position sizing ---
    shares_per_trade: int = 100
    """Fixed share size for every trade."""

    max_trades_per_day: int = 2
    """Maximum simultaneous open trades per day (one per direction)."""


class BounceBackStrategy:
    """
    Runs the Bounce Back strategy bar-by-bar on a single day's intraday data.

    Usage
    -----
    strategy = BounceBackStrategy(config)
    trades = strategy.run_day(day_bars, start_line)
    """

    def __init__(self, config: BounceBackConfig | None = None) -> None:
        self.config = config or BounceBackConfig()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run_day(
        self,
        day_bars: pd.DataFrame,
        start_line: float,
    ) -> list[Trade]:
        """
        Simulate one trading day bar-by-bar.

        Parameters
        ----------
        day_bars  : Intraday OHLCV DataFrame for a single session (ET tz-aware).
        start_line: Previous day's opening price — the gravitational anchor.

        Returns
        -------
        List of completed Trade objects for that session.
        """
        cfg = self.config
        bars = day_bars.copy()

        # Flatten MultiIndex columns if yfinance returned them
        if isinstance(bars.columns, pd.MultiIndex):
            bars.columns = bars.columns.get_level_values(0)

        bars["rsi"] = calc_rsi(bars["Close"], period=14)
        bars["avg_vol"] = average_volume(bars["Volume"], window=cfg.vol_avg_window)

        open_trade: Trade | None = None
        completed: list[Trade] = []
        daily_trade_count = 0

        trade_start = pd.Timestamp(f"1970-01-01 {cfg.trade_start_time}").time()
        trade_end   = pd.Timestamp(f"1970-01-01 {cfg.trade_end_time}").time()
        eod_exit    = pd.Timestamp(f"1970-01-01 {cfg.eod_exit_time}").time()

        for ts, bar in bars.iterrows():
            bar_time = ts.time()
            close    = float(bar["Close"])
            volume   = float(bar["Volume"])
            avg_vol  = float(bar["avg_vol"]) if not np.isnan(bar["avg_vol"]) else 0.0
            bar_rsi  = float(bar["rsi"]) if not np.isnan(bar["rsi"]) else 50.0

            # ── 1. Check EOD exit ──────────────────────────────────────
            if open_trade is not None and bar_time >= eod_exit:
                open_trade = self._close_trade(open_trade, close, ts, "EOD", completed)
                continue

            # ── 2. Manage open trade ───────────────────────────────────
            if open_trade is not None:
                hit_sl, hit_tp = self._check_exit(open_trade, bar, start_line)
                if hit_sl:
                    exit_px = open_trade.sl_price
                    open_trade = self._close_trade(open_trade, exit_px, ts, "SL", completed)
                elif hit_tp:
                    exit_px = open_trade.tp_price
                    open_trade = self._close_trade(open_trade, exit_px, ts, "TP", completed)
                continue  # one trade at a time per direction (simplified)

            # ── 3. Look for new entry ──────────────────────────────────
            if bar_time < trade_start or bar_time > trade_end:
                continue
            if daily_trade_count >= cfg.max_trades_per_day:
                continue

            signal = self._entry_signal(close, bar_rsi, volume, avg_vol, start_line)
            if signal is None:
                continue

            open_trade = self._open_trade(signal, close, ts, start_line)
            daily_trade_count += 1

        # Force-close any position still open at session end
        if open_trade is not None and not bars.empty:
            last_close = float(bars.iloc[-1]["Close"])
            last_ts    = bars.index[-1]
            open_trade = self._close_trade(open_trade, last_close, last_ts, "EOD", completed)

        return completed

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _entry_signal(
        self,
        close: float,
        bar_rsi: float,
        volume: float,
        avg_vol: float,
        start_line: float,
    ) -> Side | None:
        cfg = self.config
        distance_pct = (close - start_line) / start_line * 100
        vol_ok = avg_vol > 0 and volume >= cfg.vol_multiplier * avg_vol

        # Long: price is sufficiently below start line and RSI oversold
        if (
            distance_pct <= -cfg.entry_pct
            and bar_rsi <= cfg.rsi_oversold
            and vol_ok
        ):
            return "long"

        # Short: price is sufficiently above start line and RSI overbought
        if (
            distance_pct >= cfg.entry_pct
            and bar_rsi >= cfg.rsi_overbought
            and vol_ok
        ):
            return "short"

        return None

    def _open_trade(
        self,
        side: Side,
        price: float,
        ts: pd.Timestamp,
        start_line: float,
    ) -> Trade:
        cfg = self.config
        sl_mult = cfg.sl_pct / 100

        if side == "long":
            sl_price = price * (1 - sl_mult)
            # TP: price returns to within tp_buffer_pct% of start_line from below
            tp_price = start_line * (1 - cfg.tp_buffer_pct / 100)
        else:
            sl_price = price * (1 + sl_mult)
            tp_price = start_line * (1 + cfg.tp_buffer_pct / 100)

        return Trade(
            side=side,
            entry_price=price,
            entry_time=ts,
            start_line=start_line,
            sl_price=sl_price,
            tp_price=tp_price,
            shares=cfg.shares_per_trade,
        )

    def _check_exit(
        self,
        trade: Trade,
        bar: pd.Series,
        start_line: float,
    ) -> tuple[bool, bool]:
        """Return (hit_sl, hit_tp) based on bar's High/Low."""
        high  = float(bar["High"])
        low   = float(bar["Low"])
        close = float(bar["Close"])

        if trade.side == "long":
            hit_sl = low  <= trade.sl_price
            # TP when price climbs back toward start_line
            hit_tp = high >= trade.tp_price or close >= trade.tp_price
        else:
            hit_sl = high >= trade.sl_price
            # TP when price falls back toward start_line
            hit_tp = low  <= trade.tp_price or close <= trade.tp_price

        return hit_sl, hit_tp

    def _close_trade(
        self,
        trade: Trade,
        exit_price: float,
        exit_time: pd.Timestamp,
        reason: str,
        completed: list[Trade],
    ) -> None:
        trade.exit_price  = exit_price
        trade.exit_time   = exit_time
        trade.exit_reason = reason
        completed.append(trade)
        return None
