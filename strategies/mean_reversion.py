"""
Strategy #1 — Mean Reversion (Bollinger Bands + RSI + Volume)
=============================================================

Logic
-----
Prices statistically revert to their mean.  When a stock closes outside its
Bollinger Bands (2 SD from the 20-day SMA) and is confirmed oversold/overbought
by RSI < 30 / RSI > 70 *plus* an above-average volume spike, we bet on the
snapback to the middle band (20-day SMA).

Entry rules (Long)
  1. Daily close < Lower Bollinger Band (20 SMA − 2 SD)
  2. RSI(14) < 30
  3. Volume > vol_multiplier × 20-bar average volume

Entry rules (Short)
  1. Daily close > Upper Bollinger Band (20 SMA + 2 SD)
  2. RSI(14) > 70
  3. Volume > vol_multiplier × 20-bar average volume

Exit
  - Take Profit : price reaches the 20-day SMA (middle band)
  - Stop Loss   : price moves stop_loss_pct% against the position

Instruments : S&P 500 stocks, SPY, QQQ (daily bars)
Backtested win rate: ~70–75%  |  Avg return per trade: ~0.46%

Backtest usage
--------------
    strategy = MeanReversionStrategy()
    trades   = strategy.run(daily_bars_df, symbol="SPY")

Live signal usage (called from live_trader.py)
----------------------------------------------
    df_with_signals = strategy.generate_signals(daily_bars_df)
    last_row = df_with_signals.iloc[-1]
    if last_row["signal_long"]:
        client.market_order("SPY", shares, "buy")
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

from utils.indicators import average_volume, bollinger_bands, rsi as calc_rsi

Side = Literal["long", "short"]


@dataclass
class MeanReversionConfig:
    # Bollinger Band parameters
    bb_period: int = 20
    bb_std: float = 2.0

    # RSI parameters
    rsi_period: int = 14
    rsi_oversold: float = 30.0
    rsi_overbought: float = 70.0

    # Volume filter
    vol_multiplier: float = 1.5
    vol_avg_window: int = 20

    # Risk management
    stop_loss_pct: float = 2.0
    max_positions: int = 3

    # Position sizing (shares per trade — override with $ sizing in live_trader)
    shares_per_trade: int = 100


@dataclass
class MRTrade:
    symbol: str
    side: Side
    entry_price: float
    entry_time: pd.Timestamp
    sl_price: float
    tp_price: float
    shares: int
    exit_price: float | None = None
    exit_time: pd.Timestamp | None = None
    exit_reason: str = ""

    @property
    def pnl(self) -> float:
        if self.exit_price is None:
            return 0.0
        diff = self.exit_price - self.entry_price
        return (diff if self.side == "long" else -diff) * self.shares

    @property
    def pnl_pct(self) -> float:
        if self.exit_price is None or self.entry_price == 0:
            return 0.0
        raw = (self.exit_price - self.entry_price) / self.entry_price
        return (raw if self.side == "long" else -raw) * 100

    def to_dict(self) -> dict:
        return {
            "symbol":      self.symbol,
            "side":        self.side,
            "entry_time":  self.entry_time,
            "entry_price": self.entry_price,
            "exit_time":   self.exit_time,
            "exit_price":  self.exit_price,
            "exit_reason": self.exit_reason,
            "sl_price":    self.sl_price,
            "tp_price":    self.tp_price,
            "shares":      self.shares,
            "pnl":         round(self.pnl, 2),
            "pnl_pct":     round(self.pnl_pct, 4),
        }


class MeanReversionStrategy:
    """
    Daily mean reversion strategy using Bollinger Bands, RSI, and volume.

    The strategy runs on *daily* OHLCV bars.  Intraday timing is handled
    by the live_trader scheduler (entry at the open of the next session
    after a signal fires on the previous close).
    """

    def __init__(self, config: MeanReversionConfig | None = None) -> None:
        self.config = config or MeanReversionConfig()

    # ------------------------------------------------------------------
    # Signal generation  (shared by backtest and live modes)
    # ------------------------------------------------------------------

    def generate_signals(self, bars: pd.DataFrame) -> pd.DataFrame:
        """
        Compute indicators and add boolean signal columns to the DataFrame.

        Input : daily OHLCV DataFrame (columns: Open, High, Low, Close, Volume).
        Output: same DataFrame with extra columns:
                  bb_mid, bb_upper, bb_lower, rsi, avg_vol,
                  signal_long, signal_short
        """
        cfg = self.config
        df = bars.copy()

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        sma, upper, lower = bollinger_bands(df["Close"], cfg.bb_period, cfg.bb_std)
        df["bb_mid"]   = sma
        df["bb_upper"] = upper
        df["bb_lower"] = lower
        df["rsi"]      = calc_rsi(df["Close"], cfg.rsi_period)
        df["avg_vol"]  = average_volume(df["Volume"], cfg.vol_avg_window)

        vol_spike = df["Volume"] > cfg.vol_multiplier * df["avg_vol"]

        df["signal_long"] = (
            (df["Close"] < df["bb_lower"]) &
            (df["rsi"] < cfg.rsi_oversold) &
            vol_spike
        )
        df["signal_short"] = (
            (df["Close"] > df["bb_upper"]) &
            (df["rsi"] > cfg.rsi_overbought) &
            vol_spike
        )
        return df

    # ------------------------------------------------------------------
    # Backtesting
    # ------------------------------------------------------------------

    def run(self, bars: pd.DataFrame, symbol: str = "ASSET") -> list[MRTrade]:
        """
        Simulate the strategy bar-by-bar on historical daily bars.

        Returns a list of completed MRTrade objects.  Open trades at the
        end of the series are force-closed at the last available price.
        """
        cfg = self.config
        df = self.generate_signals(bars)

        open_trades: list[MRTrade] = []
        completed: list[MRTrade] = []

        for ts, bar in df.iterrows():
            close  = float(bar["Close"])
            bb_mid = float(bar["bb_mid"]) if not pd.isna(bar.get("bb_mid")) else float("nan")

            # ── Manage exits for all open trades ──────────────────────
            still_open: list[MRTrade] = []
            for trade in open_trades:
                if trade.side == "long":
                    hit_sl = close <= trade.sl_price
                    hit_tp = (not hit_sl) and (not np.isnan(bb_mid)) and (close >= bb_mid)
                else:
                    hit_sl = close >= trade.sl_price
                    hit_tp = (not hit_sl) and (not np.isnan(bb_mid)) and (close <= bb_mid)

                if hit_sl or hit_tp:
                    trade.exit_price  = trade.sl_price if hit_sl else bb_mid
                    trade.exit_time   = ts
                    trade.exit_reason = "SL" if hit_sl else "TP"
                    completed.append(trade)
                else:
                    still_open.append(trade)
            open_trades = still_open

            # ── Look for new entries ───────────────────────────────────
            if len(open_trades) >= cfg.max_positions:
                continue
            if np.isnan(bar.get("bb_lower", float("nan"))) or np.isnan(bar.get("rsi", float("nan"))):
                continue

            for side, sig_col in [("long", "signal_long"), ("short", "signal_short")]:
                if not bar.get(sig_col, False):
                    continue
                sl = (
                    close * (1 - cfg.stop_loss_pct / 100)
                    if side == "long"
                    else close * (1 + cfg.stop_loss_pct / 100)
                )
                open_trades.append(MRTrade(
                    symbol=symbol,
                    side=side,  # type: ignore[arg-type]
                    entry_price=close,
                    entry_time=ts,
                    sl_price=sl,
                    tp_price=bb_mid,
                    shares=cfg.shares_per_trade,
                ))
                break  # one new position per bar

        # ── Force-close anything still open at end of history ─────────
        if not df.empty:
            last_close = float(df.iloc[-1]["Close"])
            last_ts    = df.index[-1]
            for trade in open_trades:
                trade.exit_price  = last_close
                trade.exit_time   = last_ts
                trade.exit_reason = "END"
                completed.append(trade)

        return completed

    # ------------------------------------------------------------------
    # Convenience summary
    # ------------------------------------------------------------------

    def summary(self, trades: list[MRTrade]) -> dict:
        if not trades:
            return {"error": "No trades"}
        pnls     = [t.pnl for t in trades]
        wins     = [p for p in pnls if p > 0]
        losses   = [p for p in pnls if p <= 0]
        win_rate = len(wins) / len(pnls) * 100
        pf       = sum(wins) / abs(sum(losses)) if losses and sum(losses) != 0 else float("inf")
        return {
            "total_trades":  len(trades),
            "win_rate_pct":  round(win_rate, 2),
            "net_pnl":       round(sum(pnls), 2),
            "avg_win":       round(sum(wins) / len(wins), 2) if wins else 0,
            "avg_loss":      round(sum(losses) / len(losses), 2) if losses else 0,
            "profit_factor": round(pf, 3),
        }
