"""
Entry point for the Bounce Back intraday trading strategy.

Usage
-----
Run a backtest with default config:
    python main.py

Run with a custom config file:
    python main.py --config my_config.json

Override individual parameters:
    python main.py --ticker AAPL --interval 15m --entry-pct 1.0 --sl-pct 0.75

Save trade log to CSV:
    python main.py --save-trades trades.csv
"""

import argparse
import json
import sys
from pathlib import Path

from backtester import Backtester
from strategies.bounce_back import BounceBackConfig


def load_config(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def build_config(cfg_dict: dict, args: argparse.Namespace) -> BounceBackConfig:
    s = cfg_dict.get("strategy", {})

    def get(key, default):
        return getattr(args, key.replace("-", "_"), None) or s.get(key, default)

    return BounceBackConfig(
        entry_pct          = float(get("entry_pct", 0.75)),
        rsi_oversold       = float(get("rsi_oversold", 35)),
        rsi_overbought     = float(get("rsi_overbought", 65)),
        vol_multiplier     = float(get("vol_multiplier", 1.2)),
        vol_avg_window     = int(get("vol_avg_window", 20)),
        sl_pct             = float(get("sl_pct", 0.50)),
        tp_buffer_pct      = float(get("tp_buffer_pct", 0.10)),
        trade_start_time   = str(get("trade_start_time", "09:45")),
        trade_end_time     = str(get("trade_end_time", "15:00")),
        eod_exit_time      = str(get("eod_exit_time", "15:30")),
        shares_per_trade   = int(get("shares_per_trade", 100)),
        max_trades_per_day = int(get("max_trades_per_day", 2)),
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Bounce Back from Previous Day Open — Intraday Strategy"
    )
    p.add_argument("--config",       default="config.json", help="Path to JSON config file")
    p.add_argument("--ticker",       help="Override ticker symbol")
    p.add_argument("--interval",     help="Override bar interval (1m/5m/15m)")
    p.add_argument("--lookback",     type=int, help="Override lookback days (≤59)")
    p.add_argument("--entry-pct",    type=float, help="Override entry_pct")
    p.add_argument("--sl-pct",       type=float, help="Override sl_pct")
    p.add_argument("--save-trades",  help="Save trade log to this CSV path")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        print(f"[ERROR] Config file not found: {config_path}")
        sys.exit(1)

    cfg_dict = load_config(str(config_path))

    ticker        = args.ticker   or cfg_dict.get("ticker", "SPY")
    interval      = args.interval or cfg_dict.get("interval", "5m")
    lookback_days = args.lookback or cfg_dict.get("lookback_days", 59)
    initial_cap   = cfg_dict.get("initial_capital", 100_000)

    strategy_cfg = build_config(cfg_dict, args)

    bt = Backtester(
        ticker          = ticker,
        interval        = interval,
        lookback_days   = lookback_days,
        config          = strategy_cfg,
        initial_capital = initial_cap,
    )

    bt.load_data()
    results = bt.run()
    bt.print_summary()

    if args.save_trades and not results.empty:
        bt.save_trades(args.save_trades)

    # Equity curve preview
    eq = bt.equity_curve()
    if not eq.empty:
        print(f"  Start equity : ${eq.iloc[0]:,.2f}")
        print(f"  End equity   : ${eq.iloc[-1]:,.2f}")
        print(f"  Total return : {(eq.iloc[-1] / initial_cap - 1) * 100:.2f}%\n")


if __name__ == "__main__":
    main()
