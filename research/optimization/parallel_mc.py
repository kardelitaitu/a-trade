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
    data_json: str,
    capital: float,
    fee_rate: float,
) -> dict:
    """Run a single combo in a worker process.

    Supports optional SL/TP via params dict:
      sl_pct (float, optional): stop loss as decimal (0.02 = 2%)
      tp_pct (float, optional): take profit as decimal (0.04 = 4%)
    """
    from io import StringIO
    import numpy as np

    from research.strategies.factory import create_strategy
    from research.backtest.engine import VectorizedBacktest
    from research.backtest.metrics import compute_metrics

    data = pd.read_json(StringIO(data_json), orient="split")

    strat = create_strategy(strategy_name, params)
    signals = strat.generate_signals(data)

    # Check if SL/TP is requested (using get to avoid mutating original param grid)
    sl_pct = params.get("sl_pct")
    tp_pct = params.get("tp_pct")

    if sl_pct is not None and tp_pct is not None and sl_pct > 0 and tp_pct > 0:
        # Use SL/TP-aware backtest via numba
        from research.backtest._batch import single_backtest_sltp
        close = data["close"].values.astype(np.float64)
        high = data["high"].values.astype(np.float64)
        low = data["low"].values.astype(np.float64)
        sig_arr = signals.values.astype(np.float64)
        eq, dd, sh, tr, wr, pf = single_backtest_sltp(
            close, sig_arr, capital, fee_rate,
            float(sl_pct), float(tp_pct), 0.0,
            high, low,
        )
        total_return = (eq / capital - 1) * 100
        metrics = {
            "sharpe_ratio": float(sh),
            "profit_factor": float(pf),
            "max_drawdown_pct": -abs(float(dd)),  # ensure negative
            "total_trades": int(tr),
            "final_equity": float(eq),
            "total_return_pct": float(total_return),
        }
        params["sl_pct"] = sl_pct * 100  # store as % for display
        params["tp_pct"] = tp_pct * 100
    else:
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
    fee_rate: float = 0.000011,
    n_jobs: int = 24,
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
    t0 = time.perf_counter()
    n_combos = len(param_grid)
    if n_combos == 0:
        return []
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
    strategy_name: str = "",
    capital: float = 10_000.0,
    fee_rate: float = 0.000011,
    command: str = "",
    asset: str = "BTCUSDT",
    data_range: str = "",
) -> str:
    """Format parallel MC results as a professional report."""
    import datetime

    lines = []
    now = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    fee_pct = fee_rate * 100
    sep = "-" * 80

    # Header
    lines.append("")
    lines.append("MONTE CARLO OPTIMIZATION")
    lines.append(sep)
    lines.append(f"  Generated:    {now}")
    lines.append(f"  Command:      {command}")
    lines.append(f"  Strategy:     {strategy_name}")
    lines.append(f"  Asset:        {asset}")
    lines.append(f"  Data range:   {data_range}")
    lines.append(f"  Initial capt: ${capital:>10,.2f}")
    lines.append(f"  Fee rate:     {fee_pct:.3f}%")
    hw = "32 threads \u00b7 96 GB RAM \u00b7 NVMe SSD"
    lines.append(f"  Hardware:     {hw}")
    lines.append(f"  Combinations: {len(results):>6,}")
    lines.append("")

    # Top N
    lines.append(f"TOP {top_n} BY SHARPE")
    lines.append(sep)
    header = f"  {'#':>3}  {'Sharpe':>7}  {'PF':>5}  {'DD%':>6}  {'Trades':>6}  {'Return':>7}  {'Parameters':<46}"
    lines.append(header)
    lines.append("  " + "-" * (len(header) - 2))

    for i, r in enumerate(results[:top_n]):
        p = r["params"]
        pstr = ", ".join(f"{k}={v}" for k, v in p.items())
        if len(pstr) > 44:
            pstr = pstr[:41] + "..."
        lines.append(
            f"  {i+1:>3}  {r['sharpe']:>7.2f}  {r['profit_factor']:>5.2f}  "
            f"{r['max_dd']:>6.2f}%  {r['trades']:>6}  {r['total_return']:>+6.2f}%  {pstr:<46}"
        )
    lines.append("")

    # Top N by DD (lowest drawdown)
    lines.append(f"TOP {top_n} BY DRAWDOWN")
    lines.append(sep)
    lines.append(header)
    lines.append("  " + "-" * (len(header) - 2))

    by_dd = sorted(results, key=lambda r: r["max_dd"])
    for i, r in enumerate(by_dd[:top_n]):
        p = r["params"]
        pstr = ", ".join(f"{k}={v}" for k, v in p.items())
        if len(pstr) > 44:
            pstr = pstr[:41] + "..."
        lines.append(
            f"  {i+1:>3}  {r['sharpe']:>7.2f}  {r['profit_factor']:>5.2f}  "
            f"{r['max_dd']:>6.2f}%  {r['trades']:>6}  {r['total_return']:>+6.2f}%  {pstr:<46}"
        )
    lines.append("")

    # Best
    if results:
        best = results[0]
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

    # Reproducibility
    lines.append("REPRODUCIBILITY")
    lines.append(sep)
    lines.append(f"  {command or 'parallel_monte_carlo(...)'}")
    lines.append("")

    return "\n".join(lines)


def save_parallel_results(
    results: list[dict],
    strategy_name: str,
    output_dir: str | Path = "results/reports",
    capital: float = 10_000.0,
    fee_rate: float = 0.000011,
    command: str = "",
    asset: str = "BTCUSDT",
    data_range: str = "",
    top_n: int = 20,
) -> Path:
    """Save full parallel MC results to a report file with sensitivity analysis."""
    import datetime
    from pathlib import Path

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    safe = strategy_name.lower().replace(" ", "_").replace("/", "_")
    now = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d_%H%M%S")
    path = output_dir / f"mc_{safe}_{now}.txt"

    content = format_parallel_results(
        results, top_n=top_n,
        strategy_name=strategy_name,
        capital=capital, fee_rate=fee_rate,
        command=command, asset=asset, data_range=data_range,
    )

    # Append sensitivity analysis
    from research.optimization.sensitivity import analyze_sensitivity
    try:
        analysis = analyze_sensitivity(results)
        content += "\n\n" + analysis["recommendations"]
    except Exception as e:
        content += f"\n\nSensitivity analysis unavailable: {e}"

    path.write_text(content)
    return path
