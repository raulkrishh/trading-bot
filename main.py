"""
Intraday Trading Strategy Backtester

Usage
-----
Bounce Back (default):
    python main.py
    python main.py --ticker AAPL --interval 5m --entry-pct 1.0 --sl-pct 0.75

EMA Crossover (1-minute):
    python main.py --strategy ema-crossover
    python main.py --strategy ema-crossover --ticker QQQ --interval 1m

Save trade log to CSV:
    python main.py --strategy ema-crossover --save-trades trades.csv
"""

import argparse
import json
import sys
from pathlib import Path

from backtester import Backtester
from strategies.bounce_back import BounceBackConfig, BounceBackStrategy
from strategies.ema_crossover import EmaCrossoverConfig, EmaCrossoverStrategy


def load_config(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def build_bounce_back_config(cfg_dict: dict, args: argparse.Namespace) -> BounceBackConfig:
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


def build_ema_config(cfg_dict: dict, args: argparse.Namespace) -> EmaCrossoverConfig:
    s = cfg_dict.get("ema_crossover", {})
    return EmaCrossoverConfig(
        ema_fast         = int(s.get("ema_fast", 8)),
        ema_slow         = int(s.get("ema_slow", 20)),
        sl_pct           = float(args.sl_pct or s.get("sl_pct", 1.0)),
        shares_per_trade = int(s.get("shares_per_trade", 100)),
        trade_start_time = str(s.get("trade_start_time", "09:45")),
        trade_end_time   = str(s.get("trade_end_time", "15:00")),
        eod_exit_time    = str(s.get("eod_exit_time", "15:30")),
        allow_short      = bool(s.get("allow_short", True)),
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Intraday Strategy Backtester")
    p.add_argument("--config",    default="config.json", help="Path to JSON config file")
    p.add_argument("--strategy",  choices=["bounce-back", "ema-crossover"],
                   default="bounce-back", help="Strategy to backtest")
    p.add_argument("--ticker",    help="Override ticker symbol")
    p.add_argument("--interval",  help="Override bar interval (1m/5m/15m)")
    p.add_argument("--lookback",  type=int, help="Override lookback days (≤7 for 1m, ≤59 for 5m+)")
    p.add_argument("--sl-pct",    type=float, help="Override stop-loss %")
    p.add_argument("--entry-pct", type=float, help="Override entry_pct (bounce-back only)")
    p.add_argument("--save-trades", help="Save trade log to this CSV path")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        print(f"[ERROR] Config file not found: {config_path}")
        sys.exit(1)

    cfg_dict      = load_config(str(config_path))
    initial_cap   = cfg_dict.get("initial_capital", 100_000)

    if args.strategy == "ema-crossover":
        ema_cfg  = build_ema_config(cfg_dict, args)
        strategy = EmaCrossoverStrategy(ema_cfg)

        ticker        = args.ticker   or cfg_dict.get("ticker", "SPY")
        interval      = args.interval or cfg_dict.get("ema_crossover", {}).get("interval", "1m")
        lookback_days = args.lookback or cfg_dict.get("ema_crossover", {}).get("lookback_days", 7)

        if interval == "1m" and lookback_days > 7:
            print(f"[WARNING] yfinance limits 1m data to 7 days. Capping lookback at 7.")
            lookback_days = 7

        strategy_name = f"EMA{ema_cfg.ema_fast}/EMA{ema_cfg.ema_slow} Crossover"
    else:
        bounce_cfg = build_bounce_back_config(cfg_dict, args)
        strategy   = BounceBackStrategy(bounce_cfg)

        ticker        = args.ticker   or cfg_dict.get("ticker", "SPY")
        interval      = args.interval or cfg_dict.get("interval", "5m")
        lookback_days = args.lookback or cfg_dict.get("lookback_days", 59)

        strategy_name = "Bounce Back"

    bt = Backtester(
        ticker          = ticker,
        interval        = interval,
        lookback_days   = lookback_days,
        strategy        = strategy,
        strategy_name   = strategy_name,
        initial_capital = initial_cap,
    )

    bt.load_data()
    results = bt.run()
    bt.print_summary()

    if args.save_trades and not results.empty:
        bt.save_trades(args.save_trades)

    eq = bt.equity_curve()
    if not eq.empty:
        print(f"  Start equity : ${eq.iloc[0]:,.2f}")
        print(f"  End equity   : ${eq.iloc[-1]:,.2f}")
        print(f"  Total return : {(eq.iloc[-1] / initial_cap - 1) * 100:.2f}%\n")


if __name__ == "__main__":
    main()
