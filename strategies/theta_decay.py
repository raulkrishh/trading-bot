"""
Strategy #2 — Options Premium Selling / Theta Decay (Wheel Strategy)
=====================================================================

Logic
-----
Time (theta) erodes option value every day.  By *selling* options we collect
premium upfront and let theta work for us.  This strategy runs the classic
"wheel":

  Phase 1 — Sell a cash-secured put:
    - Target ~30-delta put with 30-45 DTE on a high-quality underlying
    - Close (buy-to-close) at 50% profit OR when ≤ 21 DTE
    - If price falls below strike at expiry → take assignment (buy shares)

  Phase 2 — Sell a covered call (after assignment):
    - Sell ATM or slightly OTM call at same or higher strike as put
    - Close at 50% profit OR when ≤ 21 DTE
    - If called away → repeat from Phase 1

Instruments : AAPL, MSFT, SPY, QQQ and similar high-IV, high-liquidity names
Win rate    : ~72-75%  |  Target annual return: 15-40%

Backtest mode
-------------
Uses Black-Scholes pricing with historical volatility to simulate put premium
and track position P&L day-by-day.  Assignment is handled automatically.

Live mode (via Alpaca)
-----------------------
Uses `AlpacaClient.get_option_contracts()` to find real contracts near 30-delta
and `AlpacaClient.sell_to_open_option()` / `buy_to_close_option()` for execution.

Usage (backtest)
----------------
    strategy = ThetaDecayStrategy()
    result   = strategy.run(daily_bars, symbol="AAPL")
    print(strategy.summary(result))

Usage (live — called from live_trader.py)
-----------------------------------------
    signal = strategy.get_live_trade_params(client, "AAPL")
    if signal:
        client.sell_to_open_option(signal["option_symbol"], limit_price=signal["limit_price"])
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Literal

import numpy as np
import pandas as pd

from utils.indicators import (
    bsm_put_price,
    bsm_put_delta,
    find_put_strike_for_delta,
    historical_volatility,
)

Phase = Literal["put", "call"]


@dataclass
class WheelConfig:
    # Put-selling parameters
    target_delta: float = 0.30       # absolute delta (e.g. 0.30 → 30-delta put)
    target_dte: int     = 35         # preferred days-to-expiry at entry
    min_dte: int        = 25         # minimum DTE for new position
    max_dte: int        = 50         # maximum DTE for new position
    close_profit_pct: float = 0.50   # close early if P&L ≥ 50% of max profit
    close_dte: int      = 21         # also close if DTE falls to this level

    # Risk
    risk_free_rate: float = 0.05     # annualised risk-free rate for BSM
    vol_window: int = 20             # bars used for historical vol estimate

    # Position sizing
    contracts: int = 1               # option contracts (1 = 100 shares each)
    max_positions: int = 4           # max concurrent positions across symbols


@dataclass
class WheelTrade:
    symbol: str
    phase: Phase                     # "put" or "call"
    entry_date: date
    expiry_date: date
    strike: float
    entry_premium: float             # per-share premium collected (×100 per contract)
    contracts: int

    exit_date: date | None = None
    exit_premium: float | None = None
    assigned: bool = False           # True if put was assigned (stock purchased)
    exit_reason: str = ""

    @property
    def dte_at_entry(self) -> int:
        return (self.expiry_date - self.entry_date).days

    @property
    def pnl(self) -> float:
        """Net P&L in dollars.  Positive = profit for the premium seller."""
        if self.exit_premium is None:
            return 0.0
        return (self.entry_premium - self.exit_premium) * 100 * self.contracts

    @property
    def pnl_pct(self) -> float:
        """P&L as % of premium collected (100% = kept full premium)."""
        if self.exit_premium is None or self.entry_premium == 0:
            return 0.0
        return (self.entry_premium - self.exit_premium) / self.entry_premium * 100

    def to_dict(self) -> dict:
        return {
            "symbol":         self.symbol,
            "phase":          self.phase,
            "entry_date":     str(self.entry_date),
            "expiry_date":    str(self.expiry_date),
            "strike":         round(self.strike, 2),
            "entry_premium":  round(self.entry_premium, 4),
            "exit_premium":   round(self.exit_premium, 4) if self.exit_premium is not None else None,
            "contracts":      self.contracts,
            "exit_date":      str(self.exit_date) if self.exit_date else None,
            "exit_reason":    self.exit_reason,
            "assigned":       self.assigned,
            "pnl":            round(self.pnl, 2),
            "pnl_pct":        round(self.pnl_pct, 2),
        }


class ThetaDecayStrategy:
    """
    Wheel options strategy backtester and live-trading signal generator.

    Backtest uses Black-Scholes to price puts and track daily P&L.
    Live mode queries Alpaca's options API for real market contracts.
    """

    def __init__(self, config: WheelConfig | None = None) -> None:
        self.config = config or WheelConfig()

    # ------------------------------------------------------------------
    # Backtest
    # ------------------------------------------------------------------

    def run(self, bars: pd.DataFrame, symbol: str = "SPY") -> list[WheelTrade]:
        """
        Simulate the wheel strategy on historical daily OHLCV bars.

        The simulation:
          1. Enters a new put position whenever there is no open trade.
          2. Prices the put daily using BSM with rolling historical vol.
          3. Closes at 50% profit or 21 DTE.
          4. Records assignment if stock closes below strike at expiry.
          5. After assignment, sells a covered call (phase 2) at the put strike.

        Returns a list of completed WheelTrade objects.
        """
        cfg = self.config
        df = self._prepare(bars)

        completed: list[WheelTrade] = []
        open_trade: WheelTrade | None = None

        # Track shares held (for the covered-call phase)
        shares_held: int = 0

        for i, (ts, bar) in enumerate(df.iterrows()):
            current_date  = ts.date() if hasattr(ts, "date") else ts
            close         = float(bar["Close"])
            sigma         = float(bar["hist_vol"]) if not math.isnan(bar["hist_vol"]) else 0.25

            # ── 1. Manage open trade ───────────────────────────────────
            if open_trade is not None:
                days_elapsed = (current_date - open_trade.entry_date).days
                dte          = max(open_trade.dte_at_entry - days_elapsed, 0)
                T            = dte / 365.0

                if open_trade.phase == "put":
                    current_val = bsm_put_price(
                        close, open_trade.strike, T, cfg.risk_free_rate, sigma
                    )
                else:
                    # Covered call: use put-call parity approximation
                    # Call ≈ Put + S - K*e^{-rT}
                    put_equiv    = bsm_put_price(close, open_trade.strike, T, cfg.risk_free_rate, sigma)
                    current_val  = put_equiv + close - open_trade.strike * math.exp(-cfg.risk_free_rate * T)
                    current_val  = max(current_val, 0.0)

                profit_pct = (open_trade.entry_premium - current_val) / open_trade.entry_premium

                if dte == 0:
                    # Expiry
                    if open_trade.phase == "put" and close < open_trade.strike:
                        # Assigned: took delivery of shares
                        open_trade.assigned      = True
                        open_trade.exit_premium  = max(open_trade.strike - close, 0.0)
                        open_trade.exit_date      = current_date
                        open_trade.exit_reason    = "ASSIGNMENT"
                        shares_held               = 100 * open_trade.contracts
                    else:
                        # Expired worthless — full premium kept
                        open_trade.exit_premium = 0.0
                        open_trade.exit_date    = current_date
                        open_trade.exit_reason  = "EXPIRED"
                        shares_held = 0

                    completed.append(open_trade)
                    open_trade = None

                elif profit_pct >= cfg.close_profit_pct or dte <= cfg.close_dte:
                    open_trade.exit_premium = current_val
                    open_trade.exit_date    = current_date
                    open_trade.exit_reason  = (
                        f"PROFIT_50" if profit_pct >= cfg.close_profit_pct else "DTE_21"
                    )
                    completed.append(open_trade)
                    open_trade = None

                continue  # one trade at a time

            # ── 2. Open new trade ──────────────────────────────────────
            if math.isnan(sigma) or sigma <= 0:
                continue

            expiry_date = current_date + timedelta(days=cfg.target_dte)
            T           = cfg.target_dte / 365.0

            if shares_held > 0:
                # Phase 2: sell covered call at put strike (or ATM)
                strike        = round(close, 0)
                # Approximate call price via BSM put-call parity
                put_price     = bsm_put_price(close, strike, T, cfg.risk_free_rate, sigma)
                call_price    = put_price + close - strike * math.exp(-cfg.risk_free_rate * T)
                call_price    = max(call_price, 0.01)

                open_trade = WheelTrade(
                    symbol=symbol, phase="call",
                    entry_date=current_date, expiry_date=expiry_date,
                    strike=strike, entry_premium=call_price,
                    contracts=cfg.contracts,
                )
            else:
                # Phase 1: sell cash-secured put at ~30 delta
                strike      = find_put_strike_for_delta(
                    close, T, sigma, cfg.risk_free_rate, -cfg.target_delta
                )
                put_price   = bsm_put_price(close, strike, T, cfg.risk_free_rate, sigma)
                if put_price < 0.01:
                    continue

                open_trade = WheelTrade(
                    symbol=symbol, phase="put",
                    entry_date=current_date, expiry_date=expiry_date,
                    strike=strike, entry_premium=put_price,
                    contracts=cfg.contracts,
                )

        # Force-close any open position at end of history
        if open_trade is not None and not df.empty:
            last_date   = df.index[-1].date() if hasattr(df.index[-1], "date") else df.index[-1]
            last_close  = float(df.iloc[-1]["Close"])
            last_sigma  = float(df.iloc[-1]["hist_vol"]) if not math.isnan(df.iloc[-1]["hist_vol"]) else 0.25
            days_used   = (last_date - open_trade.entry_date).days
            dte_left    = max(open_trade.dte_at_entry - days_used, 0)
            final_val   = bsm_put_price(last_close, open_trade.strike, dte_left / 365, cfg.risk_free_rate, last_sigma)
            open_trade.exit_premium = final_val
            open_trade.exit_date    = last_date
            open_trade.exit_reason  = "END"
            completed.append(open_trade)

        return completed

    # ------------------------------------------------------------------
    # Live trade parameter generation
    # ------------------------------------------------------------------

    def get_live_trade_params(
        self,
        client: object,
        symbol: str,
        bars: pd.DataFrame | None = None,
    ) -> dict | None:
        """
        Query Alpaca for real option contracts and return parameters for
        the next sell-to-open order, or None if no valid contract is found.

        The caller (live_trader.py) places the actual order.

        Requires:
          client  : AlpacaClient instance
          symbol  : underlying symbol (e.g. "AAPL")
          bars    : recent daily bars used to estimate current IV range (optional)
        """
        cfg = self.config
        try:
            quote       = client.get_latest_quote(symbol)           # type: ignore[attr-defined]
            stock_price = quote["mid"]
        except Exception:
            return None

        # Find puts with ~30 delta via Alpaca's options API
        # We filter by strike range: roughly [ATM - 15%, ATM] for 30-delta puts
        strike_floor = round(stock_price * 0.85, 2)
        contracts    = client.get_option_contracts(   # type: ignore[attr-defined]
            underlying=symbol,
            contract_type="put",
            min_dte=cfg.min_dte,
            max_dte=cfg.max_dte,
            strike_gte=strike_floor,
            strike_lte=stock_price,
        )
        if not contracts:
            return None

        # Score contracts by how close their strike is to the theoretical 30-delta strike.
        # Use historical vol from bars if available, else default to 25%.
        sigma = 0.25
        if bars is not None and not bars.empty:
            df   = self._prepare(bars)
            last = df.iloc[-1]["hist_vol"]
            if not math.isnan(last):
                sigma = float(last)

        best_contract = None
        best_delta_diff = float("inf")

        for c in contracts:
            # Try to get greeks from the live snapshot
            snap = client.get_option_snapshot(c["symbol"])  # type: ignore[attr-defined]
            if snap and snap.get("delta") is not None:
                delta_diff = abs(abs(snap["delta"]) - cfg.target_delta)
            else:
                # Estimate delta via BSM
                dte       = (date.fromisoformat(c["expiry"]) - date.today()).days
                T         = max(dte, 1) / 365.0
                est_delta = bsm_put_delta(stock_price, c["strike"], T, cfg.risk_free_rate, sigma)
                delta_diff = abs(abs(est_delta) - cfg.target_delta)

            if delta_diff < best_delta_diff:
                best_delta_diff = delta_diff
                best_contract   = {**c, "est_delta_diff": delta_diff}

        if best_contract is None or best_delta_diff > 0.15:
            return None  # no contract within 15 delta-points of target

        # Suggest limit price at the mid of bid/ask (or BSM estimate)
        snap = client.get_option_snapshot(best_contract["symbol"])  # type: ignore[attr-defined]
        if snap and snap.get("bid") and snap.get("ask"):
            limit_price = round((snap["bid"] + snap["ask"]) / 2, 2)
        else:
            dte         = (date.fromisoformat(best_contract["expiry"]) - date.today()).days
            T           = max(dte, 1) / 365.0
            limit_price = round(bsm_put_price(stock_price, best_contract["strike"], T, cfg.risk_free_rate, sigma), 2)

        return {
            "symbol":          symbol,
            "option_symbol":   best_contract["symbol"],
            "strike":          best_contract["strike"],
            "expiry":          best_contract["expiry"],
            "limit_price":     limit_price,
            "contracts":       cfg.contracts,
            "phase":           "put",
            "target_delta":    cfg.target_delta,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _prepare(self, bars: pd.DataFrame) -> pd.DataFrame:
        df = bars.copy()
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        hv_series      = historical_volatility(df["Close"], self.config.vol_window)
        df["hist_vol"] = hv_series
        return df

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def summary(self, trades: list[WheelTrade]) -> dict:
        if not trades:
            return {"error": "No trades"}
        pnls       = [t.pnl for t in trades]
        pnl_pcts   = [t.pnl_pct for t in trades]
        wins       = [p for p in pnls if p > 0]
        losses     = [p for p in pnls if p <= 0]
        assignments = sum(1 for t in trades if t.assigned)
        return {
            "total_trades":    len(trades),
            "win_rate_pct":    round(len(wins) / len(trades) * 100, 2),
            "net_pnl":         round(sum(pnls), 2),
            "avg_pnl_pct":     round(sum(pnl_pcts) / len(pnl_pcts), 2),
            "avg_win":         round(sum(wins) / len(wins), 2) if wins else 0,
            "avg_loss":        round(sum(losses) / len(losses), 2) if losses else 0,
            "assignments":     assignments,
            "profit_factor":   round(
                sum(wins) / abs(sum(losses)), 3
            ) if losses and sum(losses) != 0 else float("inf"),
        }
