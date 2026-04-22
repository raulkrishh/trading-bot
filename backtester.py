"""
Backtesting engine for intraday strategies.
Iterates over historical days, feeds each day to a strategy, and aggregates results.
"""

from __future__ import annotations

import json
from typing import Type
import pandas as pd
import numpy as np

from data.fetcher import fetch_daily, fetch_intraday, get_previous_day_open, split_by_day
from strategies.bounce_back import BounceBackStrategy, BounceBackConfig, Trade
from utils.indicators import rsi as calc_rsi, average_volume, ema as calc_ema


# ──────────────────────────────────────────────────────────────────────────────
# Core backtester
# ──────────────────────────────────────────────────────────────────────────────

class Backtester:
    """
    Walk-forward backtester that runs a BounceBackStrategy day-by-day.

    Parameters
    ----------
    ticker          : Stock symbol (e.g. 'SPY', 'AAPL').
    interval        : Intraday bar interval ('1m', '5m', '15m').
    lookback_days   : Calendar days of intraday history to fetch (≤ 59 for yfinance).
    config          : BounceBackConfig instance; uses defaults if None.
    initial_capital : Starting cash for equity curve calculation.
    """

    def __init__(
        self,
        ticker: str,
        interval: str = "5m",
        lookback_days: int = 59,
        config: BounceBackConfig | None = None,
        initial_capital: float = 100_000.0,
    ) -> None:
        self.ticker          = ticker.upper()
        self.interval        = interval
        self.lookback_days   = lookback_days
        self.config          = config or BounceBackConfig()
        self.initial_capital = initial_capital
        self.strategy        = BounceBackStrategy(self.config)

        self._daily_df    : pd.DataFrame | None = None
        self._intraday_df : pd.DataFrame | None = None
        self._trades      : list[Trade]          = []
        self._results_df  : pd.DataFrame | None = None

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------

    def load_data(self) -> None:
        print(f"[Backtester] Fetching daily data for {self.ticker}…")
        self._daily_df = fetch_daily(self.ticker, lookback_days=self.lookback_days + 10)

        print(f"[Backtester] Fetching {self.interval} intraday data for {self.ticker}…")
        self._intraday_df = fetch_intraday(
            self.ticker, interval=self.interval, lookback_days=self.lookback_days
        )
        print(f"[Backtester] Loaded {len(self._intraday_df)} intraday bars across "
              f"{self._intraday_df.index.normalize().nunique()} trading days.")

    # ------------------------------------------------------------------
    # Run
    # ------------------------------------------------------------------

    def run(self) -> pd.DataFrame:
        if self._daily_df is None or self._intraday_df is None:
            self.load_data()

        bars = self._intraday_df.copy()
        if isinstance(bars.columns, pd.MultiIndex):
            bars.columns = bars.columns.get_level_values(0)

        cfg = self.config
        bars["rsi"]      = calc_rsi(bars["Close"], period=14)
        bars["avg_vol"]  = average_volume(bars["Volume"], window=cfg.vol_avg_window)
        bars["ema_fast"] = calc_ema(bars["Close"], span=cfg.ema_fast)
        bars["ema_slow"] = calc_ema(bars["Close"], span=cfg.ema_slow)

        trade_start = pd.Timestamp(f"1970-01-01 {cfg.trade_start_time}").time()
        trade_end   = pd.Timestamp(f"1970-01-01 {cfg.trade_end_time}").time()

        all_trades: list[Trade] = []
        open_trade: Trade | None = None
        prev_ef: float | None = None
        prev_es: float | None = None

        for ts, bar in bars.iterrows():
            try:
                start_line = get_previous_day_open(self._daily_df, ts)
            except ValueError:
                continue

            close  = float(bar["Close"])
            high   = float(bar["High"])
            low    = float(bar["Low"])
            volume = float(bar["Volume"])
            avg_vol = float(bar["avg_vol"]) if not np.isnan(bar["avg_vol"]) else 0.0
            bar_rsi = float(bar["rsi"])     if not np.isnan(bar["rsi"])     else 50.0
            ef = float(bar["ema_fast"]) if not np.isnan(bar["ema_fast"]) else None
            es = float(bar["ema_slow"]) if not np.isnan(bar["ema_slow"]) else None

            if open_trade is not None:
                # SL check
                hit_sl = (
                    (open_trade.side == "long"  and low  <= open_trade.sl_price) or
                    (open_trade.side == "short" and high >= open_trade.sl_price)
                )
                if hit_sl:
                    open_trade = self.strategy._close_trade(
                        open_trade, open_trade.sl_price, ts, "SL", all_trades
                    )
                # EMA crossover exit
                elif ef is not None and es is not None and prev_ef is not None and prev_es is not None:
                    bearish_cross = prev_ef >= prev_es and ef < es
                    bullish_cross = prev_ef <= prev_es and ef > es
                    if open_trade.side == "long" and bearish_cross:
                        open_trade = self.strategy._close_trade(
                            open_trade, close, ts, "EMA_CROSS", all_trades
                        )
                    elif open_trade.side == "short" and bullish_cross:
                        open_trade = self.strategy._close_trade(
                            open_trade, close, ts, "EMA_CROSS", all_trades
                        )

            # Entry — only if no open position and within allowed hours
            if open_trade is None and trade_start <= ts.time() <= trade_end:
                signal = self.strategy._entry_signal(close, bar_rsi, volume, avg_vol, start_line)
                if signal is not None:
                    open_trade = self.strategy._open_trade(signal, close, ts, start_line)

            prev_ef = ef
            prev_es = es

        # Close any trade still open at end of data
        if open_trade is not None:
            self.strategy._close_trade(
                open_trade, float(bars["Close"].iloc[-1]), bars.index[-1], "END", all_trades
            )

        self._trades = all_trades
        self._results_df = self._build_results(all_trades)
        return self._results_df

    # ------------------------------------------------------------------
    # Results
    # ------------------------------------------------------------------

    def _build_results(self, trades: list[Trade]) -> pd.DataFrame:
        if not trades:
            return pd.DataFrame()
        rows = [t.to_dict() for t in trades]
        df = pd.DataFrame(rows)
        df["entry_time"] = pd.to_datetime(df["entry_time"])
        df["exit_time"]  = pd.to_datetime(df["exit_time"])
        df["date"]       = df["entry_time"].dt.normalize()
        return df

    def summary(self) -> dict:
        df = self._results_df
        if df is None or df.empty:
            return {"error": "No trades found. Run .run() first."}

        total_trades  = len(df)
        wins          = df[df["pnl"] > 0]
        losses        = df[df["pnl"] <= 0]
        win_rate      = len(wins) / total_trades * 100 if total_trades else 0
        avg_win       = wins["pnl"].mean()  if len(wins) else 0
        avg_loss      = losses["pnl"].mean() if len(losses) else 0
        profit_factor = (
            wins["pnl"].sum() / abs(losses["pnl"].sum())
            if losses["pnl"].sum() != 0 else float("inf")
        )
        net_pnl       = df["pnl"].sum()
        max_dd        = self._max_drawdown(df["pnl"])

        by_reason = df.groupby("exit_reason")["pnl"].agg(["count", "sum", "mean"])

        return {
            "ticker"          : self.ticker,
            "interval"        : self.interval,
            "total_trades"    : total_trades,
            "win_rate_pct"    : round(win_rate, 2),
            "avg_win_usd"     : round(avg_win, 2),
            "avg_loss_usd"    : round(avg_loss, 2),
            "profit_factor"   : round(profit_factor, 3),
            "net_pnl_usd"     : round(net_pnl, 2),
            "max_drawdown_usd": round(max_dd, 2),
            "by_exit_reason"  : by_reason.to_dict(),
        }

    @staticmethod
    def _max_drawdown(pnl_series: pd.Series) -> float:
        equity = pnl_series.cumsum()
        peak   = equity.cummax()
        dd     = equity - peak
        return float(dd.min())

    def equity_curve(self) -> pd.Series:
        """Return a daily equity curve (cumulative PnL added to initial capital)."""
        if self._results_df is None or self._results_df.empty:
            return pd.Series(dtype=float)
        daily = self._results_df.groupby("date")["pnl"].sum()
        return (daily.cumsum() + self.initial_capital).rename("equity")

    def print_summary(self) -> None:
        s = self.summary()
        if "error" in s:
            print(s["error"])
            return

        print("\n" + "=" * 55)
        print(f"  Bounce Back Strategy — Backtest Results [{s['ticker']}]")
        print("=" * 55)
        print(f"  Interval          : {s['interval']}")
        print(f"  Total Trades      : {s['total_trades']}")
        print(f"  Win Rate          : {s['win_rate_pct']}%")
        print(f"  Avg Win           : ${s['avg_win_usd']}")
        print(f"  Avg Loss          : ${s['avg_loss_usd']}")
        print(f"  Profit Factor     : {s['profit_factor']}")
        print(f"  Net PnL           : ${s['net_pnl_usd']}")
        print(f"  Max Drawdown      : ${s['max_drawdown_usd']}")
        print("-" * 55)
        print("  Exit Breakdown:")
        by_reason = s["by_exit_reason"]
        counts = by_reason.get("count", {})
        pnls   = by_reason.get("sum", {})
        for reason in counts:
            print(f"    {reason:6s} → {int(counts[reason]):3d} trades  |  ${pnls[reason]:,.2f}")
        print("=" * 55 + "\n")

    def save_trades(self, path: str) -> None:
        if self._results_df is not None:
            self._results_df.to_csv(path, index=False)
            print(f"[Backtester] Trades saved to {path}")
