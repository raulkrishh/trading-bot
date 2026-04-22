# trading-bot

Intraday trading bot implementing a **Bounce Back from Previous Day's Open** strategy.

## Strategy Overview

The *start line* is the **previous trading day's opening price** — a key reference level that acts as a gravitational anchor during the current session.

When price deviates significantly from the start line and exhaustion signals appear (oversold/overbought RSI + volume spike), the strategy fades the move and targets a return to the start line.

```
Start Line = Previous Day's Open

         ┌─────────────── Start Line ───────────────┐
         │                                           │
         │   Price drops -0.75% below → RSI < 35   │
         │   Volume confirms → LONG entry            │
         │   TP: price returns within 0.10% of line │
         │   SL: price drops another 0.50% from entry│
         └───────────────────────────────────────────┘
```

### Entry Conditions

| Direction | Price vs Start Line | RSI Condition | Volume |
|-----------|--------------------|-----------  --|--------|
| **Long**  | ≥ 0.75% below      | RSI < 35      | > 1.2× avg |
| **Short** | ≥ 0.75% above      | RSI > 65      | > 1.2× avg |

### Exit Conditions

| Exit Type | Trigger |
|-----------|---------|
| Take Profit | Price returns within 0.10% of the start line |
| Stop Loss   | Price moves 0.50% further against entry |
| EOD Exit    | Force-close all positions at 15:30 ET |

### Time Filter
- New positions opened only between **09:45–15:00 ET**
- All positions closed by **15:30 ET**

## Project Structure

```
trading-bot/
├── config.json              # Strategy & backtest configuration
├── main.py                  # CLI entry point
├── backtester.py            # Walk-forward backtesting engine
├── strategies/
│   └── bounce_back.py       # BounceBackStrategy + BounceBackConfig
├── data/
│   └── fetcher.py           # yfinance data fetching
└── utils/
    └── indicators.py        # RSI, VWAP, average volume
```

## Quick Start

```bash
pip install -r requirements.txt

# Backtest SPY with default config
python main.py

# Backtest AAPL on 15-minute bars
python main.py --ticker AAPL --interval 15m

# Tighter stop-loss, wider entry threshold
python main.py --entry-pct 1.0 --sl-pct 0.75

# Save trade log
python main.py --save-trades trades.csv
```

## Configuration (`config.json`)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `ticker` | `SPY` | Symbol to trade |
| `interval` | `5m` | Bar interval |
| `lookback_days` | `59` | Days of history (max 59 for yfinance) |
| `entry_pct` | `0.75` | Min % from start line to trigger entry |
| `rsi_oversold` | `35` | RSI threshold for long entries |
| `rsi_overbought` | `65` | RSI threshold for short entries |
| `vol_multiplier` | `1.2` | Volume must be N× rolling average |
| `sl_pct` | `0.50` | Stop-loss % from entry price |
| `tp_buffer_pct` | `0.10` | TP fires within this % of start line |
| `shares_per_trade` | `100` | Fixed position size in shares |
| `max_trades_per_day` | `2` | Max concurrent trades per session |

## Strategy Rationale

The previous day's open is the level where market participants began positioning. Price often "remembers" this level and reverts to it during the current session — especially when:

1. The deviation is sharp (momentum exhaustion)
2. RSI is at extremes (mean-reversion trigger)
3. Volume spikes confirm institutional participation (not just noise)

The strategy profits from this mean-reversion tendency while capping risk with intraday stop-losses and mandatory EOD exits (no overnight risk).
