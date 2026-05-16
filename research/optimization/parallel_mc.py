"""
Parallel Monte Carlo engine for QuantumEdge.

Distributes parameter combinations across all CPU cores using ProcessPoolExecutor.
Works with ANY strategy — no numba kernel required.
"""

from __future__ import annotations

import logging
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _worker_combo(
    strategy_name: str,
    params: dict,
    data_json: str,  # serialized to avoid pickling issues
    capital: float,
    fee_rate: float,
) -> dict:
    """Run a single combo in a worker process."""
    import json

    from research.strategies.factory import create_strategy
    from research.backtest.engine import VectorizedBacktest
    from research.backtest.metrics import compute_metrics

    # Deserialize data
    from io import StringIO
    data = pd.read_json(StringIO(data_json), orient="split")

    strat = create_strategy(strategy_name, params)
    signals = strat.generate_signals(data)
    bt = VectorizedBacktest(data, {"initial_capital": capital, "fee": fee_rate, "slippage": 0.0})
    result = bt.run(signals)
    metrics = compute_metrics(result.equity_curve, result.trades)

    return {
        "params": params,
        "sharpe": metrics["sharpe_ratio"],
        "profit_factor": metrics["profit_factor"],
        "max_dd": metrics["max_drawdown_pct"],
        "trades": metrics["total_trades"],
        "final_equity": metrics["final_equity"],
        "total_return": metrics["total_return_pct"],
    }


def parallel_monte_carlo(
    strategy_name: str,
    param_grid: list[dict[str, Any]],
    data: pd.DataFrame,
    capital: float = 10_000.0,
    fee_rate: float = 0.0001,
    n_jobs: int = 32,
    metric_sort: str = "sharpe",
) -> list[dict]:
    """
    Run Monte Carlo optimization in parallel across all CPU cores.

    Parameters
    ----------
    strategy_name : str
        Name passed to create_strategy().
    param_grid : list of dict
        Each dict is one parameter combination to test.
    data : pd.DataFrame
        OHLCV data.
    capital : float
    fee_rate : float
    n_jobs : int
        Number of parallel workers (default 32).
    metric_sort : str
        Metric to sort results by ('sharpe', 'profit_factor', etc).

    Returns
    -------
    list of dict, sorted by metric_sort descending.
    """
    from research.strategies.factory import list_strategies

    t0 = time.perf_counter()
    n_combos = len(param_grid)
    logger.info(f"Parallel MC: {strategy_name}, {n_combos} combos, {n_jobs} workers")

    # Serialize data once for all workers
    data_json = data.to_json(orient="split", date_format="iso")

    results = []
    n_jobs = min(n_jobs, n_combos)

    with ProcessPoolExecutor(max_workers=n_jobs) as executor:
        futures = {
            executor.submit(_worker_combo, strategy_name, p, data_json, capital, fee_rate): p
            for p in param_grid
        }

        for i, future in enumerate(as_completed(futures)):
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                params = futures[future]
                logger.warning(f"Combo failed: {params} — {e}")

            if (i + 1) % max(1, n_combos // 10) == 0 or (i + 1) == n_combos:
                elapsed = time.perf_counter() - t0
                rate = (i + 1) / elapsed
                logger.info(f"  [{i+1}/{n_combos}] {rate:.0f} combos/s, "
                           f"{elapsed:.1f}s elapsed")

    # Sort by metric
    reverse = metric_sort != "max_dd"
    results.sort(key=lambda r: r.get(metric_sort, -999), reverse=reverse)

    elapsed = time.perf_counter() - t0
    logger.info(f"Parallel MC complete: {n_combos} combos in {elapsed:.1f}s "
                f"({n_combos/elapsed:.0f} combos/s)")

    return results


def format_parallel_results(
    results: list[dict],
    top_n: int = 10,
) -> str:
    """Format parallel MC results as a human-readable summary."""
    lines = []
    lines.append("")
    lines.append("PARALLEL MONTE CARLO RESULTS")
    lines.append("-" * 80)
    lines.append(f"{'#':>3}  {'Parameters':<50}  {'Sharpe':>7}  {'PF':>5}  {'DD%':>6}  {'Trades':>6}  {'Return':>7}")
    lines.append("-" * 80)

    for i, r in enumerate(results[:top_n]):
        params_str = ", ".join(f"{k}={v}" for k, v in r["params"].items())
        if len(params_str) > 48:
            params_str = params_str[:45] + "..."
        lines.append(
            f"{i+1:>3}  {params_str:<50}  {r['sharpe']:>7.2f}  {r['profit_factor']:>5.2f}  "
            f"{r['max_dd']:>6.2f}%  {r['trades']:>6}  {r['total_return']:>+6.2f}%"
        )

    if results:
        best = results[0]
        lines.append("")
        lines.append("BEST PARAMETERS")
        lines.append("-" * 40)
        for k, v in best["params"].items():
            lines.append(f"  {k}: {v}")
        lines.append(f"  Sharpe:        {best['sharpe']:.4f}")
        lines.append(f"  Profit Factor: {best['profit_factor']:.4f}")
        lines.append(f"  Max DD:        {best['max_dd']:.2f}%")
        lines.append(f"  Total Return:  {best['total_return']:+.2f}%")
        lines.append(f"  Trades:        {best['trades']}")

    lines.append("")
    return "\n".join(lines)
