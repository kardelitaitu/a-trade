"""
Backtest performance metrics for QuantumEdge.

Computes all metrics defined in docs/backtesting-standards.md
from an equity curve and optional trade list.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def compute_metrics(
    equity_curve: pd.Series,
    trades: Optional[list] = None,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 105120,  # 5-min candles in a 365-day year
) -> dict:
    """
    Compute all standard backtest metrics.

    Parameters
    ----------
    equity_curve : pd.Series
        Portfolio value over time (datetime index).
    trades : list[Trade], optional
        List of Trade objects from the engine. If provided, trade-level
        metrics are included.
    risk_free_rate : float
        Annual risk-free rate (e.g. 0.05 for 5%). Default 0.
    periods_per_year : int
        Number of periods in a year. 105120 for 5-min data.

    Returns
    -------
    dict
        Dictionary of all computed metrics.
    """
    metrics = {}

    # Basic equity stats
    initial = equity_curve.iloc[0]
    final = equity_curve.iloc[-1]
    total_return = (final / initial) - 1
    metrics["initial_capital"] = initial
    metrics["final_equity"] = final
    metrics["total_return_pct"] = total_return * 100

    # CAGR
    total_years = (equity_curve.index[-1] - equity_curve.index[0]).total_seconds() / (365.25 * 86400)
    metrics["cagr_pct"] = ((final / initial) ** (1 / max(total_years, 1e-6)) - 1) * 100
    metrics["total_years"] = total_years

    # ---------- Returns (log returns for statistics) ----------
    returns = equity_curve.pct_change().dropna()
    log_returns = np.log(equity_curve / equity_curve.shift(1)).dropna()

    metrics["avg_return_pct"] = returns.mean() * 100
    metrics["std_return_pct"] = returns.std() * 100

    # ---------- Drawdown ----------
    running_max = equity_curve.cummax()
    drawdown = (equity_curve - running_max) / running_max
    metrics["max_drawdown_pct"] = drawdown.min() * 100

    # Drawdown duration
    is_peak = equity_curve == running_max
    dd_start = None
    longest_dd = 0
    current_dd = 0
    for i, peak in enumerate(is_peak):
        if peak:
            current_dd = 0
        else:
            current_dd += 1
            if current_dd > longest_dd:
                longest_dd = current_dd
    metrics["longest_dd_periods"] = longest_dd
    metrics["longest_dd_days"] = longest_dd * 5 / (24 * 60)  # approx

    # ---------- Sharpe & Sortino ----------
    rf_per_period = risk_free_rate / periods_per_year
    excess_returns = returns - rf_per_period
    downside = returns[returns < 0]

    metrics["sharpe_ratio"] = float(
        excess_returns.mean() / excess_returns.std() * np.sqrt(periods_per_year)
        if excess_returns.std() > 0 else 0.0
    )
    metrics["sortino_ratio"] = float(
        excess_returns.mean() / downside.std() * np.sqrt(periods_per_year)
        if len(downside) > 0 and downside.std() > 0 else 0.0
    )

    # ---------- Calmar ----------
    max_dd_abs = abs(metrics["max_drawdown_pct"])
    metrics["calmar_ratio"] = float(
        metrics["cagr_pct"] / max_dd_abs if max_dd_abs > 0 else 0.0
    )

    # ---------- Trade-level metrics ----------
    if trades is not None and len(trades) > 0:
        metrics["total_trades"] = len(trades)

        winners = [t for t in trades if t.pnl > 0]
        losers = [t for t in trades if t.pnl <= 0]

        metrics["winning_trades"] = len(winners)
        metrics["losing_trades"] = len(losers)
        metrics["win_rate_pct"] = len(winners) / len(trades) * 100 if trades else 0.0

        avg_win = np.mean([t.pnl for t in winners]) if winners else 0.0
        avg_loss = np.mean([t.pnl for t in losers]) if losers else 0.0
        metrics["avg_win"] = avg_win
        metrics["avg_loss"] = avg_loss

        # Profit factor
        gross_profit = sum(t.pnl for t in winners)
        gross_loss = abs(sum(t.pnl for t in losers))
        metrics["gross_profit"] = gross_profit
        metrics["gross_loss"] = gross_loss
        metrics["profit_factor"] = float(
            gross_profit / gross_loss if gross_loss > 0 else float("inf")
        )

        # Avg RR (reward:risk)
        avg_win_pct = np.mean([t.pnl_pct for t in winners]) if winners else 0.0
        avg_loss_pct = np.mean([t.pnl_pct for t in losers]) if losers else 0.0
        metrics["avg_win_pct"] = avg_win_pct
        metrics["avg_loss_pct"] = avg_loss_pct
        metrics["avg_rr"] = float(
            abs(avg_win_pct / avg_loss_pct) if avg_loss_pct != 0 else float("inf")
        )

        # Expectancy (EV per trade)
        metrics["expectancy"] = float(
            (metrics["win_rate_pct"] / 100 * avg_win)
            + ((1 - metrics["win_rate_pct"] / 100) * avg_loss)
        )

        # Avg trade duration
        durations_sec = []
        for t in trades:
            sec = (t.exit_time - t.entry_time).total_seconds()
            durations_sec.append(sec)
        metrics["avg_trade_duration_min"] = np.mean(durations_sec) / 60 if durations_sec else 0.0
        metrics["median_trade_duration_min"] = np.median(durations_sec) / 60 if durations_sec else 0.0

        # Total fees paid
        metrics["total_fees"] = sum(t.fees for t in trades)

        # Best / worst trade
        metrics["best_trade_pnl"] = max(t.pnl for t in trades)
        metrics["worst_trade_pnl"] = min(t.pnl for t in trades)
    else:
        metrics["total_trades"] = 0
        metrics["profit_factor"] = 0.0
        metrics["win_rate_pct"] = 0.0
        metrics["avg_rr"] = 0.0
        metrics["expectancy"] = 0.0

    return metrics


def format_metrics_report(metrics: dict) -> str:
    """
    Format the metrics dict as a clean text report.

    Parameters
    ----------
    metrics : dict
        Output from compute_metrics().

    Returns
    -------
    str
        Formatted report.
    """
    lines = []
    lines.append("=" * 60)
    lines.append("QUANTUMEDGE — BACKTEST REPORT")
    lines.append("=" * 60)

    lines.append(f"\n{'Capital & Returns':-^60}")
    lines.append(f"  Initial capital:    ${metrics.get('initial_capital', 0):,.2f}")
    lines.append(f"  Final equity:       ${metrics.get('final_equity', 0):,.2f}")
    lines.append(f"  Total return:       {metrics.get('total_return_pct', 0):+.2f}%")
    lines.append(f"  CAGR:               {metrics.get('cagr_pct', 0):+.2f}%")
    lines.append(f"  Period:             {metrics.get('total_years', 0):.2f} years")

    lines.append(f"\n{'Risk Metrics':-^60}")
    lines.append(f"  Sharpe ratio:       {metrics.get('sharpe_ratio', 0):.2f}")
    lines.append(f"  Sortino ratio:      {metrics.get('sortino_ratio', 0):.2f}")
    lines.append(f"  Calmar ratio:       {metrics.get('calmar_ratio', 0):.2f}")
    lines.append(f"  Max drawdown:       {metrics.get('max_drawdown_pct', 0):.2f}%")
    dd_days = metrics.get('longest_dd_days', 0)
    lines.append(f"  Longest DD:         {metrics.get('longest_dd_periods', 0):,} periods ({dd_days:.1f} days)")

    lines.append(f"\n{'Trade Statistics':-^60}")
    lines.append(f"  Total trades:       {metrics.get('total_trades', 0):,}")
    lines.append(f"  Win rate:           {metrics.get('win_rate_pct', 0):.1f}%")
    lines.append(f"  Profit factor:      {metrics.get('profit_factor', 0):.2f}")
    lines.append(f"  Avg RR:             {metrics.get('avg_rr', 0):.2f}:1")
    lines.append(f"  Expectancy:         ${metrics.get('expectancy', 0):+.2f}")
    lines.append(f"  Avg win:            ${metrics.get('avg_win', 0):+.2f} ({metrics.get('avg_win_pct', 0):+.2f}%)")
    lines.append(f"  Avg loss:           ${metrics.get('avg_loss', 0):+.2f} ({metrics.get('avg_loss_pct', 0):+.2f}%)")
    lines.append(f"  Best trade:         ${metrics.get('best_trade_pnl', 0):+.2f}")
    lines.append(f"  Worst trade:        ${metrics.get('worst_trade_pnl', 0):+.2f}")
    lines.append(f"  Avg duration:       {metrics.get('avg_trade_duration_min', 0):.1f} min")
    lines.append(f"  Total fees:         ${metrics.get('total_fees', 0):,.2f}")

    lines.append("\n" + "=" * 60)
    return "\n".join(lines)
