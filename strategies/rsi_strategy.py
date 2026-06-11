"""
RSI Mean-Reversion Strategy
============================

Concept
-------
Pure RSI-driven mean-reversion: enter long when the asset is oversold
(RSI < rsi_buy) and exit only when momentum flips to overbought (RSI > rsi_sell)
or the stop-loss is hit. Positions carry over overnight if RSI never recovers.

Signal Logic
------------
Entry  : RSI(14) crosses below `rsi_buy`  (default 35) → go long.
Exit   : RSI(14) crosses above `rsi_sell` (default 60) → close long.
Stop   : Price drops `sl_pct`% below entry price.

No time-based EOD force-close — positions hold until the signal fires.
No short selling — strategy is long-only.
"""

from __future__ import annotations

from dataclasses import dataclass
import pandas as pd
import numpy as np

from utils.indicators import rsi as calc_rsi


@dataclass
class RSITrade:
    entry_price: float
    entry_time: pd.Timestamp
    sl_price: float
    exit_price: float | None = None
    exit_time: pd.Timestamp | None = None
    exit_reason: str = ""
    shares: int = 1

    @property
    def pnl(self) -> float:
        if self.exit_price is None:
            return 0.0
        return (self.exit_price - self.entry_price) * self.shares

    @property
    def pnl_pct(self) -> float:
        if self.exit_price is None:
            return 0.0
        return (self.exit_price - self.entry_price) / self.entry_price * 100

    def to_dict(self) -> dict:
        return {
            "side": "long",
            "entry_time": self.entry_time,
            "entry_price": self.entry_price,
            "exit_time": self.exit_time,
            "exit_price": self.exit_price,
            "exit_reason": self.exit_reason,
            "sl_price": self.sl_price,
            "pnl": self.pnl,
            "pnl_pct": self.pnl_pct,
        }


@dataclass
class RSIConfig:
    rsi_period: int = 14
    """RSI lookback period."""

    rsi_buy: float = 35.0
    """Enter long when RSI falls below this level (oversold)."""

    rsi_sell: float = 60.0
    """Exit long when RSI rises above this level (overbought)."""

    sl_pct: float = 1.0
    """Stop-loss distance from entry as a percentage."""

    shares_per_trade: int = 100
    """Fixed share size per trade."""

    trade_start_time: str = "09:45"
    """Earliest bar at which a new position may be opened."""

    trade_end_time: str = "15:00"
    """Latest bar at which a new position may be opened."""

    eod_exit_time: str = "14:30"
    """Force-close any open position at or after this time."""


class RSIStrategy:
    """
    Long-only RSI mean-reversion strategy.

    Positions carry over between days until RSI > rsi_sell or SL is hit.

    Usage
    -----
    strategy = RSIStrategy(config)
    # Pass carry-over trade from previous session (or None for first day)
    completed, open_trade = strategy.run_day(day_bars, open_trade=None)
    """

    def __init__(self, config: RSIConfig | None = None) -> None:
        self.config = config or RSIConfig()

    def run_day(
        self,
        day_bars: pd.DataFrame,
        open_trade: RSITrade | None = None,
    ) -> tuple[list[RSITrade], RSITrade | None]:
        """
        Simulate one session bar-by-bar.

        Returns (completed_trades, still_open_trade).
        Pass still_open_trade into the next call to carry the position over.
        """
        cfg = self.config
        bars = day_bars.copy()

        if isinstance(bars.columns, pd.MultiIndex):
            bars.columns = bars.columns.get_level_values(0)

        bars["rsi"] = calc_rsi(bars["Close"], period=cfg.rsi_period)

        completed: list[RSITrade] = []

        trade_start = pd.Timestamp(f"1970-01-01 {cfg.trade_start_time}").time()
        trade_end   = pd.Timestamp(f"1970-01-01 {cfg.trade_end_time}").time()
        eod_exit    = pd.Timestamp(f"1970-01-01 {cfg.eod_exit_time}").time()

        for ts, bar in bars.iterrows():
            bar_time = ts.time()
            close    = float(bar["Close"])
            bar_rsi  = float(bar["rsi"]) if not np.isnan(bar["rsi"]) else 50.0

            # ── EOD force-close ────────────────────────────────────────
            if open_trade is not None and bar_time >= eod_exit:
                open_trade = self._close(open_trade, close, ts, "EOD", completed)
                continue

            # ── Manage open trade (including carried-over positions) ────
            if open_trade is not None:
                if float(bar["Low"]) <= open_trade.sl_price:
                    open_trade = self._close(open_trade, open_trade.sl_price, ts, "SL", completed)
                elif bar_rsi >= cfg.rsi_sell:
                    open_trade = self._close(open_trade, close, ts, "RSI_SELL", completed)
                continue

            # ── Look for new entry (only within trade window) ──────────
            if bar_time < trade_start or bar_time > trade_end:
                continue

            if bar_rsi <= cfg.rsi_buy:
                sl_price = close * (1 - cfg.sl_pct / 100)
                open_trade = RSITrade(
                    entry_price=close,
                    entry_time=ts,
                    sl_price=sl_price,
                    shares=cfg.shares_per_trade,
                )

        # Return open_trade so the backtester can carry it to the next day
        return completed, open_trade

    def _close(
        self,
        trade: RSITrade,
        price: float,
        ts: pd.Timestamp,
        reason: str,
        completed: list[RSITrade],
    ) -> None:
        trade.exit_price  = price
        trade.exit_time   = ts
        trade.exit_reason = reason
        completed.append(trade)
        return None
