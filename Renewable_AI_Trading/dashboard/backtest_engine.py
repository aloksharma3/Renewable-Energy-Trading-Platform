"""
Backtest Engine
===============
Standalone module that simulates the ML trading strategy against two baselines
using historical forecast data collected by the live pipeline.

STRATEGY LOGIC  (mirrors the live trading engine in trading_service/main.py)
    Signal      = ML_predicted_price − actual_market_price
    BUY         when signal  >  dynamic_buy_threshold
    SELL        when signal  < −dynamic_sell_threshold
    Threshold   = max(35th percentile of recent |signals| over warmup window, $1.50 floor)
    Trade size  = 50 MWh per round-trip

BASELINES
    Buy-and-Hold    : buy 50 MWh at open, hold to end of period — no active management
    Naive Mean-Rev  : buy when price drops >3% below 12-interval rolling mean,
                      sell when price rises >2% above rolling mean — no ML involved

OUTPUTS
    equity curves   : cumulative P&L over time for all three strategies
    trades_log      : per-trade record (buy price, sell price, signal, P&L, win/loss)
    stats           : summary metrics (Sharpe ratio, max drawdown, win rate, period)

USAGE
    from backtest_engine import run_backtest

    bt = run_backtest(forecast_history)   # list of forecast dicts from /forecast/history
    if bt:
        print(bt["stats"])
        print(bt["trades_log"])
"""

import numpy as np
import pandas as pd


# ─── Constants ───────────────────────────────────────────────────────────────

TRADE_SIZE_MWH  = 50    # MWh per round-trip trade
WARMUP_INTERVALS = 20   # intervals before dynamic threshold kicks in
THRESH_PERCENTILE = 35  # percentile of |signals| used as entry threshold
THRESH_FLOOR     = 1.5  # minimum signal threshold in $/MWh (avoids over-trading flat markets)
NR_BUY_PCT       = 0.97 # naive mean-rev: buy when price < mean * this
NR_SELL_PCT      = 1.02 # naive mean-rev: sell when price > mean * this
NR_WINDOW        = 12   # rolling window (intervals) for naive mean-reversion


# ─── Public API ──────────────────────────────────────────────────────────────

def run_backtest(forecast_history: list) -> dict | None:
    """
    Run the full backtest on a list of forecast records returned by the
    /forecast/history endpoint.

    Each record is expected to have the shape:
        {
            "timestamp":           str,
            "price": {
                "predicted":       float,
                "confidence_lower": float,   # optional
                "confidence_upper": float,   # optional
            },
            "actual_market_price": float | None,
        }

    Returns None if there is insufficient data (< 20 valid intervals).
    Otherwise returns a dict — see module docstring for full schema.
    """
    times, actual_prices, pred_prices = _extract_series(forecast_history)

    n = len(times)
    if n < WARMUP_INTERVALS:
        return None

    prices  = np.array(actual_prices, dtype=float)
    preds   = np.array(pred_prices,   dtype=float)
    signals = preds - prices                          # core ML signal

    ml_equity, ml_trades_log = _run_ml_strategy(times, prices, signals)
    bh_equity                = _run_buy_and_hold(prices)
    nr_equity                = _run_naive_mean_reversion(prices)

    stats = _compute_stats(
        times, ml_equity, bh_equity, nr_equity, ml_trades_log, n
    )

    return {
        "times":      times,
        "prices":     actual_prices,
        "signals":    signals.tolist(),
        "ml_equity":  ml_equity,
        "bh_equity":  bh_equity,
        "nr_equity":  nr_equity,
        "trades_log": ml_trades_log,
        "stats":      stats,
    }


# ─── Internal helpers ─────────────────────────────────────────────────────────

def _extract_series(forecast_history: list) -> tuple[list, list, list]:
    """
    Parse raw forecast dicts into parallel lists of timestamps, actual prices,
    and predicted prices.  Records missing either value are skipped.
    Input is reversed so records are oldest-first (chronological order).
    """
    times, actual_prices, pred_prices = [], [], []

    for f in reversed(forecast_history):
        pred   = f.get("price", {}).get("predicted")
        actual = f.get("actual_market_price") or pred  # fallback: use predicted if no actual yet
        if pred is not None and actual is not None:
            times.append(f.get("timestamp", ""))
            actual_prices.append(float(actual))
            pred_prices.append(float(pred))

    return times, actual_prices, pred_prices


def _run_ml_strategy(
    times: list,
    prices: np.ndarray,
    signals: np.ndarray,
) -> tuple[list, list]:
    """
    Simulate the ML signal strategy.

    Entry logic:
        - Long entry  : signal  >  +threshold  (model thinks price will rise)
        - Long exit   : signal  < −threshold   (model thinks price will fall)
        - Threshold is dynamic: 35th-pct of |signals| in a rolling warmup window,
          subject to a $1.50 floor to avoid noise trades.

    Returns (equity_curve, trades_log).
    """
    position   = 0      # 0 = flat, 1 = long
    cash       = 0.0
    buy_price  = None
    equity     = []
    trades_log = []

    for i in range(len(times)):
        # Rolling threshold — grows as more data accumulates
        window = np.abs(signals[max(0, i - WARMUP_INTERVALS): i + 1])
        thresh = float(np.percentile(window, THRESH_PERCENTILE)) \
                 if len(window) >= 5 else THRESH_FLOOR
        thresh = max(thresh, THRESH_FLOOR)

        if position == 0 and signals[i] > thresh:
            # Enter long
            position  = 1
            buy_price = prices[i]

        elif position == 1 and signals[i] < -thresh:
            # Exit long
            pnl    = (prices[i] - buy_price) * TRADE_SIZE_MWH
            cash  += pnl
            trades_log.append({
                "time":         times[i],
                "buy_price":    round(float(buy_price), 2),
                "sell_price":   round(float(prices[i]), 2),
                "signal_entry": round(float(signals[i - 1]) if i > 0 else float(signals[i]), 2),
                "signal_exit":  round(float(signals[i]), 2),
                "pnl":          round(float(pnl), 2),
                "win":          pnl > 0,
            })
            position  = 0
            buy_price = None

        equity.append(cash)

    return equity, trades_log


def _run_buy_and_hold(prices: np.ndarray) -> list:
    """
    Buy-and-Hold baseline: buy TRADE_SIZE_MWH at the first price,
    mark-to-market at every subsequent interval.
    """
    return [(float(p) - float(prices[0])) * TRADE_SIZE_MWH for p in prices]


def _run_naive_mean_reversion(prices: np.ndarray) -> list:
    """
    Naive mean-reversion baseline (no ML):
        - Buy  when price drops more than 3 % below the rolling mean
        - Sell when price rises more than 2 % above the rolling mean
    """
    position  = 0
    cash      = 0.0
    buy_price = None
    equity    = []

    for i in range(len(prices)):
        window = prices[max(0, i - NR_WINDOW): i + 1]
        mean_p = float(np.mean(window))

        if position == 0 and prices[i] < mean_p * NR_BUY_PCT:
            position  = 1
            buy_price = prices[i]
        elif position == 1 and prices[i] > mean_p * NR_SELL_PCT:
            cash     += (float(prices[i]) - float(buy_price)) * TRADE_SIZE_MWH
            position  = 0
            buy_price = None

        equity.append(cash)

    return equity


def _compute_stats(
    times: list,
    ml_equity: list,
    bh_equity: list,
    nr_equity: list,
    trades_log: list,
    n_intervals: int,
) -> dict:
    """
    Compute summary statistics for the ML strategy and baselines.

    Sharpe ratio is annualised assuming 96 intervals/day (15-min pipeline) × 252 trading days.
    Requires at least 96 intervals (1 day) to produce a meaningful Sharpe — below that
    the annualisation factor inflates it unrealistically (e.g. 36 from 20 intervals).
    Drawdown is computed as the maximum peak-to-trough decline in cumulative P&L.
    """
    n_trades = len(trades_log)
    n_wins   = sum(1 for t in trades_log if t["win"])
    win_rate = round(n_wins / n_trades * 100, 1) if n_trades > 0 else 0.0

    ml_arr   = np.array(ml_equity, dtype=float)
    step_ret = np.diff(ml_arr)

    # Annualised Sharpe (96 intervals/day × 252 days)
    # Only meaningful with >= 96 intervals (1 full day of data)
    # Cap at ±10 to avoid displaying nonsensical values during early accumulation
    if n_intervals >= 96 and float(np.std(step_ret)) > 1e-6:
        sharpe = (
            float(np.mean(step_ret)) / (float(np.std(step_ret)) + 1e-9)
        ) * np.sqrt(96 * 252)
        sharpe = round(max(-10.0, min(10.0, sharpe)), 2)
    else:
        sharpe = None  # not enough data for meaningful Sharpe

    # Max drawdown
    peak     = np.maximum.accumulate(ml_arr)
    drawdown = ml_arr - peak
    max_dd   = round(float(np.min(drawdown)), 2)

    total_pnl = round(float(ml_equity[-1]), 2) if ml_equity else 0.0
    bh_pnl    = round(float(bh_equity[-1]), 2) if bh_equity else 0.0
    nr_pnl    = round(float(nr_equity[-1]), 2) if nr_equity else 0.0

    # Human-readable period label
    try:
        t_start = pd.to_datetime(times[0]).strftime("%b %d, %Y")
        t_end   = pd.to_datetime(times[-1]).strftime("%b %d, %Y")
        period  = f"{t_start} → {t_end}"
    except Exception:
        period  = f"{n_intervals} intervals"

    return {
        "period":       period,
        "n_intervals":  n_intervals,
        "total_trades": n_trades,
        "win_rate":     win_rate,
        "sharpe":       sharpe,
        "max_drawdown": max_dd,
        "total_pnl":    total_pnl,
        "bh_pnl":       bh_pnl,
        "nr_pnl":       nr_pnl,
    }