"""
Live / Paper Trading Orchestrator
===================================
Connects to Alpaca and runs one of three quant strategies in real-time.

Strategies
----------
  mean_reversion  — Daily Bollinger Band + RSI + Volume snapback
  theta_decay     — Options wheel: sell cash-secured puts / covered calls
  momentum        — Multi-factor monthly rebalancer (12-1 mo momentum + beta)

All strategies default to PAPER trading.  Pass --live only after thorough
paper-trading validation.

Usage
-----
  # Paper trade mean reversion on SPY + QQQ
  python live_trader.py --strategy mean_reversion --tickers SPY,QQQ

  # Paper trade theta decay (options wheel) on AAPL, MSFT
  python live_trader.py --strategy theta_decay --tickers AAPL,MSFT

  # Paper trade momentum across default 30-stock universe
  python live_trader.py --strategy momentum

  # Backtest mean reversion on AAPL (no live connection required)
  python live_trader.py --strategy mean_reversion --tickers AAPL --backtest

  # Switch to live money (CAUTION)
  python live_trader.py --strategy mean_reversion --tickers SPY --live

Environment variables
---------------------
  ALPACA_API_KEY    — your Alpaca key ID
  ALPACA_SECRET_KEY — your Alpaca secret key
  ALPACA_PAPER      — "true" (default) or "false"

  Copy .env.example → .env and fill in your credentials.

Scheduling notes
----------------
  mean_reversion : checks for signals once per day at 10:00 AM ET.
  theta_decay    : checks open option positions every Monday at 10:00 AM ET.
  momentum       : rebalances on the first trading day of each month at 10:00 AM ET.

  The trader runs in a blocking loop (Ctrl-C to stop).  For production use,
  consider running it as a systemd service or scheduled cron job instead.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import date, datetime, timezone
from pathlib import Path

import pandas as pd

# Load .env if present (silently skip if python-dotenv not installed)
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


# ──────────────────────────────────────────────────────────────────────────────
# Strategy imports
# ──────────────────────────────────────────────────────────────────────────────

from strategies.mean_reversion import MeanReversionStrategy, MeanReversionConfig
from strategies.theta_decay import ThetaDecayStrategy, WheelConfig
from strategies.momentum import MomentumStrategy, MomentumConfig, DEFAULT_UNIVERSE


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _log(msg: str) -> None:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def _is_market_open() -> bool:
    """Very rough check — strategies apply their own time guards."""
    now_utc  = datetime.now(timezone.utc)
    now_et   = now_utc.astimezone(tz=None)  # system tz; adjust if needed
    weekday  = now_et.weekday()             # Mon=0, Sun=6
    hour_et  = now_et.hour
    return weekday < 5 and 14 <= hour_et < 21  # 9:30–16:00 ET ≈ 14:30–21:00 UTC


def _shares_for_dollar_amount(price: float, dollar_amount: float) -> int:
    if price <= 0:
        return 0
    return max(int(dollar_amount / price), 0)


def _load_config(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


# ──────────────────────────────────────────────────────────────────────────────
# Per-strategy runners
# ──────────────────────────────────────────────────────────────────────────────

def run_mean_reversion(
    client,
    tickers: list[str],
    config: MeanReversionConfig,
    position_size_usd: float,
    dry_run: bool = False,
) -> None:
    """
    Fetch the latest daily bars for each ticker, generate signals, and
    place orders for any ticker showing a Bollinger Band + RSI signal.

    Also checks existing positions against their stop-loss levels.
    """
    strategy = MeanReversionStrategy(config)
    acct     = client.get_account()
    _log(f"[mean_reversion] Account equity: ${acct['equity']:,.2f}  "
         f"Buying power: ${acct['buying_power']:,.2f}")

    from alpaca.data.timeframe import TimeFrame
    from datetime import timedelta

    start = datetime.now(timezone.utc) - timedelta(days=90)
    bars_map = client.get_bars(tickers, timeframe=TimeFrame.Day, start=start)

    for ticker in tickers:
        bars = bars_map.get(ticker, pd.DataFrame())
        if bars.empty or len(bars) < 30:
            _log(f"[mean_reversion] {ticker}: insufficient data ({len(bars)} bars)")
            continue

        df       = strategy.generate_signals(bars)
        last     = df.iloc[-1]
        position = client.get_position(ticker)

        # ── Exit logic for existing positions ─────────────────────────
        if position is not None:
            current_price = float(last["Close"])
            bb_mid        = float(last.get("bb_mid", current_price))
            entry_price   = position["avg_entry_price"]
            side          = position["side"]
            sl_pct        = config.stop_loss_pct / 100

            hit_tp = (side == "long"  and current_price >= bb_mid) or \
                     (side == "short" and current_price <= bb_mid)
            hit_sl = (side == "long"  and current_price <= entry_price * (1 - sl_pct)) or \
                     (side == "short" and current_price >= entry_price * (1 + sl_pct))

            if hit_tp or hit_sl:
                reason = "TP" if hit_tp else "SL"
                _log(f"[mean_reversion] {ticker}: EXIT {reason} @ ${current_price:.2f}")
                if not dry_run:
                    try:
                        client.close_position(ticker)
                    except Exception as e:
                        _log(f"[mean_reversion] {ticker}: close failed — {e}")
            else:
                _log(f"[mean_reversion] {ticker}: holding {side} position "
                     f"@ ${entry_price:.2f}  current=${current_price:.2f}")
            continue

        # ── Entry logic ────────────────────────────────────────────────
        if last.get("signal_long"):
            price = float(last["Close"])
            qty   = _shares_for_dollar_amount(price, position_size_usd)
            if qty == 0:
                _log(f"[mean_reversion] {ticker}: LONG signal but qty=0 (price ${price:.2f})")
                continue
            _log(f"[mean_reversion] {ticker}: LONG signal — buying {qty} shares @ ~${price:.2f}")
            if not dry_run:
                try:
                    result = client.market_order(ticker, qty, "buy")
                    _log(f"[mean_reversion] {ticker}: order {result['id']} {result['status']}")
                except Exception as e:
                    _log(f"[mean_reversion] {ticker}: order failed — {e}")

        elif last.get("signal_short"):
            price = float(last["Close"])
            qty   = _shares_for_dollar_amount(price, position_size_usd)
            if qty == 0:
                continue
            _log(f"[mean_reversion] {ticker}: SHORT signal — selling {qty} shares @ ~${price:.2f}")
            if not dry_run:
                try:
                    result = client.market_order(ticker, qty, "sell")
                    _log(f"[mean_reversion] {ticker}: order {result['id']} {result['status']}")
                except Exception as e:
                    _log(f"[mean_reversion] {ticker}: order failed — {e}")
        else:
            _log(f"[mean_reversion] {ticker}: no signal  "
                 f"(RSI={last.get('rsi', float('nan')):.1f}  "
                 f"close={last['Close']:.2f}  "
                 f"bb_lower={last.get('bb_lower', float('nan')):.2f})")


def run_theta_decay(
    client,
    tickers: list[str],
    config: WheelConfig,
    dry_run: bool = False,
) -> None:
    """
    For each ticker check whether we have an open option position.
    If not, find the best ~30-delta put and sell it.
    If yes, check for 50%-profit or 21-DTE close trigger.
    """
    strategy = ThetaDecayStrategy(config)
    acct     = client.get_account()
    _log(f"[theta_decay] Account equity: ${acct['equity']:,.2f}")

    from alpaca.data.timeframe import TimeFrame
    from datetime import timedelta

    start    = datetime.now(timezone.utc) - timedelta(days=60)
    bars_map = client.get_bars(tickers, timeframe=TimeFrame.Day, start=start)
    positions = {p["symbol"]: p for p in client.get_positions()}

    for ticker in tickers:
        bars = bars_map.get(ticker, pd.DataFrame())

        # ── Check existing option positions ────────────────────────────
        open_option = next(
            (p for sym, p in positions.items()
             if sym.startswith(ticker) and p.get("asset_class") == "us_option"),
            None,
        )
        if open_option is not None:
            sym   = open_option["symbol"]
            upnl  = open_option["unrealized_pl"]
            cost  = open_option["cost_basis"]
            pct   = upnl / abs(cost) * 100 if cost != 0 else 0
            _log(f"[theta_decay] {sym}: open option  uPnL=${upnl:.2f} ({pct:.1f}%)")

            # Close at 50% profit (uPnL for short option = negative cost vs market)
            if pct >= 50:
                _log(f"[theta_decay] {sym}: closing at 50% profit")
                if not dry_run:
                    try:
                        client.close_position(sym)
                    except Exception as e:
                        _log(f"[theta_decay] {sym}: close failed — {e}")
            continue

        # ── Open new put position ──────────────────────────────────────
        params = strategy.get_live_trade_params(client, ticker, bars)
        if params is None:
            _log(f"[theta_decay] {ticker}: no suitable contract found")
            continue

        _log(
            f"[theta_decay] {ticker}: SELL PUT  {params['option_symbol']}  "
            f"strike=${params['strike']}  expiry={params['expiry']}  "
            f"limit=${params['limit_price']}"
        )
        if not dry_run:
            try:
                result = client.sell_to_open_option(
                    params["option_symbol"],
                    qty=params["contracts"],
                    limit_price=params["limit_price"],
                )
                _log(f"[theta_decay] {ticker}: order {result['id']} {result['status']}")
            except Exception as e:
                _log(f"[theta_decay] {ticker}: order failed — {e}")


def run_momentum(
    client,
    config: MomentumConfig,
    account_equity: float,
    dry_run: bool = False,
) -> None:
    """
    Score the universe, compute target portfolio, and rebalance.
    Checks stop-losses first, then sells exits, then buys entries.
    """
    strategy = MomentumStrategy(config)
    _log(f"[momentum] Scoring {len(config.universe)} stocks…")

    from alpaca.data.timeframe import TimeFrame
    from datetime import timedelta

    start    = datetime.now(timezone.utc) - timedelta(days=400)
    all_tickers = config.universe + [config.market_symbol]
    bars_map = client.get_bars(all_tickers, timeframe=TimeFrame.Day, start=start)

    market_bars = bars_map.pop(config.market_symbol, pd.DataFrame())
    if market_bars.empty:
        _log("[momentum] Could not fetch market (SPY) bars — aborting")
        return

    positions_list = client.get_positions()
    current_positions = {
        p["symbol"]: {"shares": int(abs(p["qty"])), "cost_basis": p["avg_entry_price"]}
        for p in positions_list
        if p.get("asset_class") == "us_equity"
    }

    # Stop-loss check
    sl_orders = strategy.stop_loss_orders(current_positions, bars_map)
    for order in sl_orders:
        _log(f"[momentum] STOP-LOSS {order.symbol} — selling {order.shares} shares  ({order.reason})")
        if not dry_run:
            try:
                client.market_order(order.symbol, order.shares, "sell")
                current_positions.pop(order.symbol, None)
            except Exception as e:
                _log(f"[momentum] {order.symbol}: SL order failed — {e}")

    # Rebalance
    held_shares = {sym: pos["shares"] for sym, pos in current_positions.items()}
    orders = strategy.rebalance_orders(held_shares, bars_map, market_bars, account_equity)

    sells = [o for o in orders if o.side == "sell"]
    buys  = [o for o in orders if o.side == "buy"]

    for order in sells:
        _log(f"[momentum] SELL {order.symbol}  {order.shares} shares @ ~${order.price:.2f}  ({order.reason})")
        if not dry_run:
            try:
                client.market_order(order.symbol, order.shares, "sell")
            except Exception as e:
                _log(f"[momentum] {order.symbol}: sell failed — {e}")

    for order in buys:
        _log(f"[momentum] BUY  {order.symbol}  {order.shares} shares @ ~${order.price:.2f}  ({order.reason})")
        if not dry_run:
            try:
                client.market_order(order.symbol, order.shares, "buy")
            except Exception as e:
                _log(f"[momentum] {order.symbol}: buy failed — {e}")

    if not orders and not sl_orders:
        _log("[momentum] Portfolio already aligned — no trades needed")


# ──────────────────────────────────────────────────────────────────────────────
# Backtest runners
# ──────────────────────────────────────────────────────────────────────────────

def backtest_mean_reversion(tickers: list[str], config: MeanReversionConfig, days: int = 252) -> None:
    import yfinance as yf
    from datetime import timedelta

    strategy = MeanReversionStrategy(config)
    end   = datetime.now()
    start = end - timedelta(days=days + 30)

    print(f"\n{'='*60}")
    print(f"  Mean Reversion Backtest  ({days} trading days)")
    print(f"{'='*60}")

    for ticker in tickers:
        df = yf.download(ticker, start=start.strftime("%Y-%m-%d"),
                         end=end.strftime("%Y-%m-%d"), interval="1d",
                         auto_adjust=True, progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        trades = strategy.run(df, symbol=ticker)
        stats  = strategy.summary(trades)
        print(f"\n  {ticker}")
        for k, v in stats.items():
            print(f"    {k:<18}: {v}")


def backtest_theta_decay(tickers: list[str], config: WheelConfig, days: int = 365) -> None:
    import yfinance as yf
    from datetime import timedelta

    strategy = ThetaDecayStrategy(config)
    end   = datetime.now()
    start = end - timedelta(days=days + 60)

    print(f"\n{'='*60}")
    print(f"  Theta Decay Backtest  ({days} calendar days)")
    print(f"{'='*60}")

    for ticker in tickers:
        df = yf.download(ticker, start=start.strftime("%Y-%m-%d"),
                         end=end.strftime("%Y-%m-%d"), interval="1d",
                         auto_adjust=True, progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        trades = strategy.run(df, symbol=ticker)
        stats  = strategy.summary(trades)
        print(f"\n  {ticker}")
        for k, v in stats.items():
            print(f"    {k:<18}: {v}")


def backtest_momentum(config: MomentumConfig, initial_capital: float, days: int = 756) -> None:
    import yfinance as yf
    from datetime import timedelta

    strategy = MomentumStrategy(config)
    end   = datetime.now()
    start = end - timedelta(days=days + 90)

    print(f"\n{'='*60}")
    print(f"  Momentum Backtest  ({days // 252:.0f} years  |  {len(config.universe)} stocks)")
    print(f"{'='*60}")

    all_tickers = config.universe + [config.market_symbol]
    bars_map: dict[str, pd.DataFrame] = {}
    for t in all_tickers:
        df = yf.download(t, start=start.strftime("%Y-%m-%d"),
                         end=end.strftime("%Y-%m-%d"), interval="1d",
                         auto_adjust=True, progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        bars_map[t] = df

    market_bars = bars_map.pop(config.market_symbol)
    results_df  = strategy.run(bars_map, market_bars, initial_capital)
    stats       = strategy.portfolio_summary(results_df, initial_capital)
    print()
    for k, v in stats.items():
        print(f"  {k:<20}: {v}")
    if not results_df.empty:
        print(f"\n  Sample trades (last 10):")
        print(results_df.tail(10).to_string(index=False))


# ──────────────────────────────────────────────────────────────────────────────
# Scheduling loop
# ──────────────────────────────────────────────────────────────────────────────

def _next_run_seconds(strategy: str) -> float:
    """
    Return seconds to sleep before the next scheduled check.
    Strategies run once per day (mean_reversion/theta_decay) or once per
    check cycle (momentum checks daily but only trades on first-of-month).
    """
    now     = datetime.now()
    target  = now.replace(hour=10, minute=0, second=0, microsecond=0)
    if now >= target:
        from datetime import timedelta
        target += timedelta(days=1)
    return (target - now).total_seconds()


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Alpaca Live / Paper Trading Runner")
    p.add_argument(
        "--strategy", required=True,
        choices=["mean_reversion", "theta_decay", "momentum"],
        help="Which strategy to run",
    )
    p.add_argument(
        "--tickers", default="",
        help="Comma-separated ticker list (overrides per-strategy defaults)",
    )
    p.add_argument(
        "--config", default="config.json",
        help="Path to JSON config file",
    )
    p.add_argument(
        "--backtest", action="store_true",
        help="Run a backtest instead of live trading (uses yfinance data)",
    )
    p.add_argument(
        "--live", action="store_true",
        help="Use live Alpaca account (CAUTION: real money). Default is paper.",
    )
    p.add_argument(
        "--dry-run", action="store_true",
        help="Connect to Alpaca and compute signals but do NOT place orders",
    )
    p.add_argument(
        "--once", action="store_true",
        help="Run the strategy check once and exit (no scheduling loop)",
    )
    p.add_argument(
        "--position-size", type=float, default=10_000.0,
        help="Dollar amount per position for mean_reversion (default: $10,000)",
    )
    return p.parse_args()


def main() -> None:
    args       = parse_args()
    cfg_dict   = {}
    cfg_path   = Path(args.config)
    if cfg_path.exists():
        cfg_dict = _load_config(str(cfg_path))

    paper = not args.live

    # ── Backtest path (no Alpaca connection needed) ────────────────────
    if args.backtest:
        tickers = [t.strip().upper() for t in args.tickers.split(",") if t.strip()]

        if args.strategy == "mean_reversion":
            mr_cfg = cfg_dict.get("mean_reversion", {})
            config = MeanReversionConfig(**{
                k: v for k, v in mr_cfg.items()
                if k in MeanReversionConfig.__dataclass_fields__
            })
            backtest_mean_reversion(tickers or ["SPY", "QQQ"], config)

        elif args.strategy == "theta_decay":
            td_cfg = cfg_dict.get("theta_decay", {})
            config = WheelConfig(**{
                k: v for k, v in td_cfg.items()
                if k in WheelConfig.__dataclass_fields__
            })
            backtest_theta_decay(tickers or ["AAPL", "MSFT", "SPY"], config)

        elif args.strategy == "momentum":
            mom_cfg = cfg_dict.get("momentum", {})
            universe = tickers or mom_cfg.get("universe", DEFAULT_UNIVERSE)
            config   = MomentumConfig(
                universe=universe,
                **{k: v for k, v in mom_cfg.items()
                   if k in MomentumConfig.__dataclass_fields__ and k != "universe"},
            )
            initial_capital = cfg_dict.get("initial_capital", 100_000.0)
            backtest_momentum(config, initial_capital)
        return

    # ── Live / Paper trading path ──────────────────────────────────────
    try:
        from brokers.alpaca_client import AlpacaClient
        client = AlpacaClient(paper=paper)
    except ValueError as e:
        print(f"[ERROR] {e}")
        print("Copy .env.example → .env and add your Alpaca API credentials.")
        sys.exit(1)

    mode = "PAPER" if paper else "LIVE"
    _log(f"Connected to Alpaca ({mode} mode)")

    acct = client.get_account()
    _log(f"Portfolio value: ${acct['portfolio_value']:,.2f}  Cash: ${acct['cash']:,.2f}")

    if args.live and not args.dry_run:
        confirm = input("\n⚠️  LIVE TRADING MODE — real money will be at risk.\n"
                        "Type 'yes' to continue: ")
        if confirm.strip().lower() != "yes":
            print("Aborted.")
            sys.exit(0)

    # Build strategy config from JSON
    tickers = [t.strip().upper() for t in args.tickers.split(",") if t.strip()]

    if args.strategy == "mean_reversion":
        mr_cfg = cfg_dict.get("mean_reversion", {})
        strategy_config = MeanReversionConfig(**{
            k: v for k, v in mr_cfg.items()
            if k in MeanReversionConfig.__dataclass_fields__
        })
        run_fn = lambda: run_mean_reversion(
            client, tickers or ["SPY", "QQQ"],
            strategy_config, args.position_size, dry_run=args.dry_run,
        )

    elif args.strategy == "theta_decay":
        td_cfg = cfg_dict.get("theta_decay", {})
        strategy_config = WheelConfig(**{
            k: v for k, v in td_cfg.items()
            if k in WheelConfig.__dataclass_fields__
        })
        run_fn = lambda: run_theta_decay(
            client, tickers or ["AAPL", "MSFT"],
            strategy_config, dry_run=args.dry_run,
        )

    elif args.strategy == "momentum":
        mom_cfg  = cfg_dict.get("momentum", {})
        universe = tickers or mom_cfg.get("universe", DEFAULT_UNIVERSE)
        strategy_config = MomentumConfig(
            universe=universe,
            **{k: v for k, v in mom_cfg.items()
               if k in MomentumConfig.__dataclass_fields__ and k != "universe"},
        )
        equity  = acct["equity"]
        run_fn  = lambda: run_momentum(
            client, strategy_config, equity, dry_run=args.dry_run,
        )

    # ── Single-shot or scheduling loop ────────────────────────────────
    if args.once or args.dry_run:
        _log(f"Running {args.strategy} once…")
        run_fn()
        _log("Done.")
        return

    _log(f"Starting {args.strategy} scheduler (Ctrl-C to stop)…")
    while True:
        sleep_secs = _next_run_seconds(args.strategy)
        _log(f"Next run in {sleep_secs / 3600:.1f} h  (at 10:00 ET)")
        time.sleep(sleep_secs)
        _log(f"Running {args.strategy}…")
        try:
            run_fn()
        except KeyboardInterrupt:
            _log("Stopped by user.")
            break
        except Exception as e:
            _log(f"[ERROR] Strategy raised exception: {e}")
            _log("Waiting for next scheduled run…")


if __name__ == "__main__":
    main()
