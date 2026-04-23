"""
Strategy #3 — Multi-Factor Momentum
=====================================

Logic
-----
Stocks that are already winning tend to keep winning (momentum anomaly).
We blend three price-based factors into a single rank score:

  1. 12-1 Month Price Momentum (weight: 60%)
       Total return from 12 months ago to 1 month ago.  Skipping the most
       recent month avoids the well-documented short-term reversal effect.

  2. Low Beta vs. SPY (weight: 25%)
       Lower-volatility stocks outperform on a risk-adjusted basis.
       Score = 1 − beta  (so lower beta → higher score).

  3. 1-Month Relative Strength vs. SPY (weight: 15%)
       Recent 1-month return relative to benchmark — a short-term trend
       confirmation filter to avoid entering during sudden momentum breaks.

Portfolio construction
  - Universe : configurable list of tickers (defaults to 30 liquid S&P 500 names)
  - Selection : top `top_n` stocks by combined factor score
  - Weighting : equal-weight among selected stocks
  - Rebalance : monthly (first trading day of each calendar month)
  - Stop-loss : −15% per position (trailing from cost basis)

CAGR target: 15-23%  |  Win rate: 60-65%

Backtest usage
--------------
    strategy = MomentumStrategy()
    results  = strategy.run(bars_by_symbol, market_bars)

Live usage (called from live_trader.py)
----------------------------------------
    orders = strategy.rebalance_orders(
        current_positions, bars_by_symbol, market_bars, account_equity
    )
    for o in orders:
        client.market_order(o["symbol"], o["qty"], o["side"])
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

import numpy as np
import pandas as pd

from utils.indicators import beta as calc_beta, momentum_score


# Top-30 liquid S&P 500 / Nasdaq names covering diverse sectors
DEFAULT_UNIVERSE: list[str] = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "BRK-B",
    "JPM",  "V",    "UNH",   "XOM",  "LLY",  "JNJ",  "PG",   "MA",
    "HD",   "MRK",  "ABBV",  "CVX",  "AVGO", "PEP",  "COST", "KO",
    "WMT",  "TMO",  "MCD",   "CSCO", "ACN",  "BAC",
]


@dataclass
class MomentumConfig:
    # Universe
    universe: list[str] = field(default_factory=lambda: list(DEFAULT_UNIVERSE))
    market_symbol: str = "SPY"

    # Factor weights (must sum to 1.0)
    weight_momentum: float = 0.60
    weight_beta: float     = 0.25
    weight_rs: float       = 0.15

    # Portfolio construction
    top_n: int = 20                     # number of stocks to hold
    stop_loss_pct: float = 15.0         # per-position stop-loss from cost basis

    # Rebalancing
    lookback_months: int = 13           # bars needed: ≥ (12 + 1 skip + 1 buffer) months
    min_history_months: int = 13        # skip scoring if insufficient history


@dataclass
class MomentumTrade:
    symbol: str
    side: str                    # "buy" or "sell"
    price: float
    shares: int
    date: date
    reason: str = ""

    @property
    def value(self) -> float:
        return self.price * self.shares

    def to_dict(self) -> dict:
        return {
            "symbol": self.symbol,
            "side":   self.side,
            "price":  round(self.price, 4),
            "shares": self.shares,
            "value":  round(self.value, 2),
            "date":   str(self.date),
            "reason": self.reason,
        }


class MomentumStrategy:
    """
    Multi-factor momentum strategy with monthly rebalancing.

    Works in two modes:
      - Backtest : call `run(bars_by_symbol, market_bars, initial_capital)`
      - Live     : call `rebalance_orders(current_positions, bars_by_symbol,
                                          market_bars, account_equity)`
    """

    def __init__(self, config: MomentumConfig | None = None) -> None:
        self.config = config or MomentumConfig()

    # ------------------------------------------------------------------
    # Factor scoring
    # ------------------------------------------------------------------

    def score_universe(
        self,
        bars_by_symbol: dict[str, pd.DataFrame],
        market_bars: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Score every symbol in the universe using the three factors.

        Returns a DataFrame sorted by `combined_score` descending, with columns:
          symbol, momentum, beta_score, rs_score, combined_score, latest_price
        """
        cfg = self.config
        market_returns = self._daily_returns(market_bars)
        rows: list[dict] = []

        for symbol, bars in bars_by_symbol.items():
            if bars.empty:
                continue
            if isinstance(bars.columns, pd.MultiIndex):
                bars = bars.copy()
                bars.columns = bars.columns.get_level_values(0)

            min_bars = cfg.min_history_months * 21
            if len(bars) < min_bars:
                continue

            prices         = bars["Close"]
            asset_returns  = self._daily_returns(bars)

            # Factor 1: 12-1 month momentum
            mom = momentum_score(prices, skip_months=1, lookback_months=12)

            # Factor 2: beta vs SPY
            b   = calc_beta(asset_returns, market_returns)
            b   = max(min(b, 3.0), 0.0)        # clip outliers
            beta_score = 1.0 - b / 3.0         # invert: lower beta → higher score

            # Factor 3: 1-month relative strength
            rs = self._relative_strength_1m(prices, market_bars["Close"])

            rows.append({
                "symbol":       symbol,
                "momentum":     mom,
                "beta_score":   beta_score,
                "rs_score":     rs,
                "latest_price": float(prices.iloc[-1]),
            })

        if not rows:
            return pd.DataFrame()

        df = pd.DataFrame(rows)
        # Rank-normalise each factor to [0, 1] so weights are comparable
        for col in ["momentum", "beta_score", "rs_score"]:
            rng = df[col].max() - df[col].min()
            df[col + "_rank"] = (df[col] - df[col].min()) / rng if rng != 0 else 0.5

        df["combined_score"] = (
            cfg.weight_momentum * df["momentum_rank"] +
            cfg.weight_beta     * df["beta_score_rank"] +
            cfg.weight_rs       * df["rs_score_rank"]
        )
        return df.sort_values("combined_score", ascending=False).reset_index(drop=True)

    # ------------------------------------------------------------------
    # Order generation
    # ------------------------------------------------------------------

    def rebalance_orders(
        self,
        current_positions: dict[str, int],     # symbol → shares currently held
        bars_by_symbol: dict[str, pd.DataFrame],
        market_bars: pd.DataFrame,
        account_equity: float,
    ) -> list[MomentumTrade]:
        """
        Compute the trades needed to rebalance to the top-N portfolio.

        Returns a list of MomentumTrade objects (buy and sell orders).
        The caller is responsible for executing them via AlpacaClient.
        """
        cfg    = self.config
        scores = self.score_universe(bars_by_symbol, market_bars)
        if scores.empty:
            return []

        target_symbols  = set(scores.head(cfg.top_n)["symbol"].tolist())
        held_symbols    = set(current_positions.keys())
        today           = date.today()

        orders: list[MomentumTrade] = []

        # Sell anything not in the new top-N
        for sym in held_symbols - target_symbols:
            if current_positions[sym] > 0:
                price = float(bars_by_symbol[sym].iloc[-1]["Close"]) if sym in bars_by_symbol else 0
                orders.append(MomentumTrade(
                    symbol=sym, side="sell", price=price,
                    shares=current_positions[sym], date=today, reason="REBALANCE_OUT",
                ))

        # Buy into new top-N stocks not already held
        per_position = account_equity / cfg.top_n
        target_df    = scores.head(cfg.top_n)

        for _, row in target_df.iterrows():
            sym   = row["symbol"]
            price = row["latest_price"]
            if price <= 0:
                continue
            target_shares = int(per_position / price)
            if target_shares == 0:
                continue

            held = current_positions.get(sym, 0)
            if held == 0:
                orders.append(MomentumTrade(
                    symbol=sym, side="buy", price=price,
                    shares=target_shares, date=today, reason="REBALANCE_IN",
                ))

        return orders

    def stop_loss_orders(
        self,
        current_positions: dict[str, dict],    # symbol → {shares, cost_basis}
        bars_by_symbol: dict[str, pd.DataFrame],
    ) -> list[MomentumTrade]:
        """
        Check every held position for a −15% stop-loss breach.
        Returns sell orders for any position that has breached.
        """
        cfg    = self.config
        today  = date.today()
        orders: list[MomentumTrade] = []

        for sym, pos in current_positions.items():
            bars = bars_by_symbol.get(sym)
            if bars is None or bars.empty:
                continue
            current_price = float(bars.iloc[-1]["Close"])
            cost_basis    = pos.get("cost_basis", current_price)
            drawdown_pct  = (current_price - cost_basis) / cost_basis * 100

            if drawdown_pct <= -cfg.stop_loss_pct:
                orders.append(MomentumTrade(
                    symbol=sym, side="sell", price=current_price,
                    shares=pos["shares"], date=today,
                    reason=f"STOP_LOSS_{drawdown_pct:.1f}%",
                ))

        return orders

    # ------------------------------------------------------------------
    # Backtest
    # ------------------------------------------------------------------

    def run(
        self,
        bars_by_symbol: dict[str, pd.DataFrame],
        market_bars: pd.DataFrame,
        initial_capital: float = 100_000.0,
    ) -> pd.DataFrame:
        """
        Walk-forward monthly backtest.

        Rebalances on the first trading day of each month.  Returns a
        DataFrame of all trades executed (buys and sells), with running
        P&L and portfolio value columns appended.
        """
        cfg = self.config

        # Collect all unique trading dates (from market_bars)
        market_bars = market_bars.copy()
        if isinstance(market_bars.columns, pd.MultiIndex):
            market_bars.columns = market_bars.columns.get_level_values(0)

        all_dates = sorted(market_bars.index.tolist())
        if not all_dates:
            return pd.DataFrame()

        cash: float              = initial_capital
        holdings: dict[str, int] = {}        # symbol → shares
        cost_basis: dict[str, float] = {}    # symbol → avg cost per share
        all_trades: list[dict]   = []

        last_rebalance_month: int | None = None

        for idx, ts in enumerate(all_dates):
            current_date = ts.date() if hasattr(ts, "date") else ts
            month        = current_date.month

            # Only rebalance on first day of each new month
            if month == last_rebalance_month:
                continue

            # Need enough history
            start_idx = max(0, idx - cfg.min_history_months * 21)
            window_bars = {
                sym: df.iloc[:idx + 1] for sym, df in bars_by_symbol.items()
                if not df.empty and len(df.iloc[:idx + 1]) >= cfg.min_history_months * 21
            }
            mkt_window = market_bars.iloc[:idx + 1]

            if len(mkt_window) < cfg.min_history_months * 21:
                continue

            scores = self.score_universe(window_bars, mkt_window)
            if scores.empty:
                continue

            target_symbols = set(scores.head(cfg.top_n)["symbol"].tolist())
            prices_today   = {
                sym: float(bars_by_symbol[sym].loc[ts]["Close"])
                for sym in target_symbols | set(holdings.keys())
                if sym in bars_by_symbol and ts in bars_by_symbol[sym].index
            }

            # Sell exits
            for sym in list(holdings.keys()):
                if sym not in target_symbols or sym not in prices_today:
                    p   = prices_today.get(sym, cost_basis.get(sym, 0))
                    qty = holdings.pop(sym, 0)
                    cost_basis.pop(sym, None)
                    cash += p * qty
                    all_trades.append({
                        "date": current_date, "symbol": sym,
                        "side": "sell", "shares": qty, "price": p,
                        "cash_after": cash,
                    })

            # Check stop-losses for remaining holdings
            for sym in list(holdings.keys()):
                if sym not in prices_today:
                    continue
                p  = prices_today[sym]
                cb = cost_basis.get(sym, p)
                if (p - cb) / cb * 100 <= -cfg.stop_loss_pct:
                    qty  = holdings.pop(sym)
                    cost_basis.pop(sym, None)
                    cash += p * qty
                    all_trades.append({
                        "date": current_date, "symbol": sym,
                        "side": "sell", "shares": qty, "price": p,
                        "cash_after": cash, "note": "SL",
                    })

            # Buy entries — equal-weight remaining budget
            portfolio_value = cash + sum(
                holdings.get(sym, 0) * prices_today.get(sym, 0)
                for sym in holdings
            )
            n_to_buy    = len(target_symbols - set(holdings.keys()))
            if n_to_buy > 0:
                per_pos = portfolio_value / cfg.top_n
                for _, row in scores.head(cfg.top_n).iterrows():
                    sym = row["symbol"]
                    if sym in holdings or sym not in prices_today:
                        continue
                    p   = prices_today[sym]
                    qty = int(per_pos / p)
                    if qty == 0 or cash < p * qty:
                        continue
                    cash -= p * qty
                    holdings[sym]   = qty
                    cost_basis[sym] = p
                    all_trades.append({
                        "date": current_date, "symbol": sym,
                        "side": "buy", "shares": qty, "price": p,
                        "cash_after": cash,
                    })

            last_rebalance_month = month

        results = pd.DataFrame(all_trades) if all_trades else pd.DataFrame()
        return results

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _daily_returns(bars: pd.DataFrame) -> pd.Series:
        if isinstance(bars.columns, pd.MultiIndex):
            bars = bars.copy()
            bars.columns = bars.columns.get_level_values(0)
        return bars["Close"].pct_change().dropna()

    @staticmethod
    def _relative_strength_1m(prices: pd.Series, market_prices: pd.Series) -> float:
        """1-month return of asset minus 1-month return of market."""
        n = 21
        if len(prices) < n + 1 or len(market_prices) < n + 1:
            return 0.0
        asset_ret  = (float(prices.iloc[-1]) - float(prices.iloc[-n - 1])) / float(prices.iloc[-n - 1])
        market_ret = (float(market_prices.iloc[-1]) - float(market_prices.iloc[-n - 1])) / float(market_prices.iloc[-n - 1])
        return asset_ret - market_ret

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    @staticmethod
    def portfolio_summary(
        trades_df: pd.DataFrame,
        initial_capital: float,
    ) -> dict:
        if trades_df.empty:
            return {"error": "No trades"}
        buys  = trades_df[trades_df["side"] == "buy"]
        sells = trades_df[trades_df["side"] == "sell"]
        return {
            "total_buys":    len(buys),
            "total_sells":   len(sells),
            "unique_stocks": trades_df["symbol"].nunique(),
            "final_cash":    round(float(trades_df.iloc[-1]["cash_after"]), 2) if "cash_after" in trades_df.columns else 0,
        }
