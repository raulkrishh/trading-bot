"""
Alpaca broker client — wraps alpaca-py for equity and options trading.

Supports paper and live modes. All credentials are read from environment
variables (ALPACA_API_KEY, ALPACA_SECRET_KEY) or passed directly.

Usage:
    from brokers.alpaca_client import AlpacaClient
    client = AlpacaClient(paper=True)           # reads env vars
    client = AlpacaClient("KEY", "SECRET")      # explicit credentials

Paper trading is the default and safe starting point. Switch to live only
when you have thoroughly validated the strategy in paper mode.
"""

from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone
from typing import Optional

import pandas as pd

from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest, StockLatestQuoteRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderSide, TimeInForce, ContractType
from alpaca.trading.requests import (
    GetOptionContractsRequest,
    LimitOrderRequest,
    MarketOrderRequest,
)


class AlpacaClient:
    """
    Thin, strategy-agnostic wrapper around alpaca-py.

    Provides:
    - Account info and positions
    - Historical OHLCV bars (daily and intraday)
    - Latest quotes (bid/ask)
    - Market and limit equity orders
    - Options contract lookup and sell-to-open / buy-to-close orders
    """

    def __init__(
        self,
        api_key: str | None = None,
        secret_key: str | None = None,
        paper: bool = True,
    ) -> None:
        key = api_key or os.getenv("ALPACA_API_KEY", "")
        secret = secret_key or os.getenv("ALPACA_SECRET_KEY", "")
        if not key or not secret:
            raise ValueError(
                "Alpaca credentials missing. Set ALPACA_API_KEY and "
                "ALPACA_SECRET_KEY environment variables, or pass them directly."
            )

        self.paper = paper
        self.trading = TradingClient(key, secret, paper=paper)
        self.data = StockHistoricalDataClient(key, secret)

        # Options data client (requires options subscription on the account)
        try:
            from alpaca.data.historical import OptionHistoricalDataClient
            self._option_data: object | None = OptionHistoricalDataClient(key, secret)
        except Exception:
            self._option_data = None

    # ------------------------------------------------------------------
    # Account
    # ------------------------------------------------------------------

    def get_account(self) -> dict:
        """Return key account metrics."""
        acct = self.trading.get_account()
        return {
            "buying_power":    float(acct.buying_power),
            "portfolio_value": float(acct.portfolio_value),
            "cash":            float(acct.cash),
            "equity":          float(acct.equity),
            "paper":           self.paper,
        }

    def get_positions(self) -> list[dict]:
        """Return all open positions as plain dicts."""
        return [
            {
                "symbol":          p.symbol,
                "qty":             float(p.qty),
                "side":            p.side.value,
                "avg_entry_price": float(p.avg_entry_price),
                "current_price":   float(p.current_price) if p.current_price else None,
                "unrealized_pl":   float(p.unrealized_pl) if p.unrealized_pl else 0.0,
                "asset_class":     p.asset_class.value if p.asset_class else "us_equity",
                "cost_basis":      float(p.cost_basis) if p.cost_basis else 0.0,
            }
            for p in self.trading.get_all_positions()
        ]

    def get_position(self, symbol: str) -> dict | None:
        """Return a single position or None if not held."""
        for pos in self.get_positions():
            if pos["symbol"] == symbol:
                return pos
        return None

    # ------------------------------------------------------------------
    # Market Data
    # ------------------------------------------------------------------

    def get_bars(
        self,
        symbols: list[str],
        timeframe: TimeFrame = TimeFrame.Day,
        start: datetime | None = None,
        end: datetime | None = None,
        limit: int | None = None,
    ) -> dict[str, pd.DataFrame]:
        """
        Fetch historical OHLCV bars for one or more symbols.

        Returns a dict mapping symbol → DataFrame (ET-localised index).
        Missing symbols map to an empty DataFrame.
        """
        if start is None:
            start = datetime.now(timezone.utc) - timedelta(days=400)

        req = StockBarsRequest(
            symbol_or_symbols=symbols,
            timeframe=timeframe,
            start=start,
            end=end,
            limit=limit,
            feed="iex",
        )
        bar_set = self.data.get_stock_bars(req)

        result: dict[str, pd.DataFrame] = {}
        for sym in symbols:
            try:
                df = bar_set[sym].df.copy()
                df.index = pd.to_datetime(df.index)
                if df.index.tz is None:
                    df.index = df.index.tz_localize("America/New_York")
                else:
                    df.index = df.index.tz_convert("America/New_York")
                # Normalise column names to Title Case so strategies use Close/Open/etc.
                df.columns = [c.capitalize() for c in df.columns]
                result[sym] = df
            except (KeyError, AttributeError):
                result[sym] = pd.DataFrame()

        return result

    def get_latest_quote(self, symbol: str) -> dict:
        """Return the latest bid/ask for a symbol."""
        req = StockLatestQuoteRequest(symbol_or_symbols=[symbol])
        quote = self.data.get_stock_latest_quote(req)[symbol]
        return {
            "ask": float(quote.ask_price),
            "bid": float(quote.bid_price),
            "mid": (float(quote.ask_price) + float(quote.bid_price)) / 2,
        }

    # ------------------------------------------------------------------
    # Equity Orders
    # ------------------------------------------------------------------

    def market_order(
        self,
        symbol: str,
        qty: int,
        side: str,
        tif: str = "day",
    ) -> dict:
        """Place a market order. side='buy' or 'sell'."""
        req = MarketOrderRequest(
            symbol=symbol,
            qty=qty,
            side=OrderSide.BUY if side == "buy" else OrderSide.SELL,
            time_in_force=TimeInForce.DAY if tif == "day" else TimeInForce.GTC,
        )
        order = self.trading.submit_order(req)
        return {
            "id":     str(order.id),
            "symbol": order.symbol,
            "status": order.status.value,
            "side":   side,
            "qty":    qty,
        }

    def limit_order(
        self,
        symbol: str,
        qty: int,
        side: str,
        limit_price: float,
        tif: str = "gtc",
    ) -> dict:
        """Place a limit order. side='buy' or 'sell'."""
        req = LimitOrderRequest(
            symbol=symbol,
            qty=qty,
            side=OrderSide.BUY if side == "buy" else OrderSide.SELL,
            time_in_force=TimeInForce.DAY if tif == "day" else TimeInForce.GTC,
            limit_price=round(limit_price, 2),
        )
        order = self.trading.submit_order(req)
        return {
            "id":          str(order.id),
            "symbol":      order.symbol,
            "status":      order.status.value,
            "side":        side,
            "qty":         qty,
            "limit_price": limit_price,
        }

    def close_position(self, symbol: str) -> dict:
        """Market-close an open equity position."""
        result = self.trading.close_position(symbol)
        return {"id": str(result.id), "symbol": result.symbol}

    def close_all_positions(self) -> None:
        """Market-close ALL open positions and cancel all open orders."""
        self.trading.close_all_positions(cancel_orders=True)

    # ------------------------------------------------------------------
    # Options
    # ------------------------------------------------------------------

    def get_option_contracts(
        self,
        underlying: str,
        contract_type: str = "put",
        min_dte: int = 25,
        max_dte: int = 50,
        strike_gte: float | None = None,
        strike_lte: float | None = None,
    ) -> list[dict]:
        """
        Return available option contracts for an underlying symbol.

        Filters by contract type (put/call), DTE range, and optionally
        by strike price range.  Returns plain dicts with symbol, strike,
        expiry, type, and underlying.
        """
        date_min = (datetime.now() + timedelta(days=min_dte)).strftime("%Y-%m-%d")
        date_max = (datetime.now() + timedelta(days=max_dte)).strftime("%Y-%m-%d")

        req = GetOptionContractsRequest(
            underlying_symbols=[underlying],
            type=ContractType.PUT if contract_type == "put" else ContractType.CALL,
            expiration_date_gte=date_min,
            expiration_date_lte=date_max,
            strike_price_gte=str(strike_gte) if strike_gte is not None else None,
            strike_price_lte=str(strike_lte) if strike_lte is not None else None,
        )
        contracts = self.trading.get_option_contracts(req)
        return [
            {
                "symbol":     c.symbol,
                "strike":     float(c.strike_price),
                "expiry":     str(c.expiration_date),
                "type":       c.type.value,
                "underlying": c.underlying_symbol,
            }
            for c in (contracts.option_contracts or [])
        ]

    def get_option_snapshot(self, option_symbol: str) -> dict | None:
        """
        Return a snapshot of an option contract (bid/ask, greeks, IV).
        Requires the options data subscription; returns None if unavailable.
        """
        if self._option_data is None:
            return None
        try:
            from alpaca.data.requests import OptionSnapshotRequest
            req = OptionSnapshotRequest(symbol_or_symbols=[option_symbol])
            snapshots = self._option_data.get_option_snapshot(req)  # type: ignore[attr-defined]
            snap = snapshots.get(option_symbol)
            if snap is None:
                return None
            greeks = snap.greeks
            return {
                "symbol":             option_symbol,
                "bid":                float(snap.latest_quote.bid_price) if snap.latest_quote else None,
                "ask":                float(snap.latest_quote.ask_price) if snap.latest_quote else None,
                "implied_volatility": float(snap.implied_volatility) if snap.implied_volatility else None,
                "delta":              float(greeks.delta) if greeks and greeks.delta else None,
                "theta":              float(greeks.theta) if greeks and greeks.theta else None,
                "gamma":              float(greeks.gamma) if greeks and greeks.gamma else None,
                "vega":               float(greeks.vega)  if greeks and greeks.vega  else None,
            }
        except Exception:
            return None

    def sell_to_open_option(
        self,
        option_symbol: str,
        qty: int = 1,
        limit_price: float | None = None,
    ) -> dict:
        """Sell-to-open: collect premium by shorting an option contract."""
        if limit_price is not None:
            req = LimitOrderRequest(
                symbol=option_symbol,
                qty=qty,
                side=OrderSide.SELL,
                time_in_force=TimeInForce.DAY,
                limit_price=round(limit_price, 2),
            )
        else:
            req = MarketOrderRequest(
                symbol=option_symbol,
                qty=qty,
                side=OrderSide.SELL,
                time_in_force=TimeInForce.DAY,
            )
        order = self.trading.submit_order(req)
        return {"id": str(order.id), "symbol": order.symbol, "status": order.status.value}

    def buy_to_close_option(
        self,
        option_symbol: str,
        qty: int = 1,
        limit_price: float | None = None,
    ) -> dict:
        """Buy-to-close: exit a short option position."""
        if limit_price is not None:
            req = LimitOrderRequest(
                symbol=option_symbol,
                qty=qty,
                side=OrderSide.BUY,
                time_in_force=TimeInForce.DAY,
                limit_price=round(limit_price, 2),
            )
        else:
            req = MarketOrderRequest(
                symbol=option_symbol,
                qty=qty,
                side=OrderSide.BUY,
                time_in_force=TimeInForce.DAY,
            )
        order = self.trading.submit_order(req)
        return {"id": str(order.id), "symbol": order.symbol, "status": order.status.value}
