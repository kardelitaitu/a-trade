"""
Monte Carlo parameter optimizer for QuantumEdge.

Uses batched numba kernels to run thousands of parameter combinations
in parallel across all CPU cores.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from research.backtest._batch import (
    compute_sma_bank,
    _rsi_bank,
    batch_ma_crossover,
    batch_mean_reversion_rsi,
)

logger = logging.getLogger(__name__)


@dataclass
class MCMetrics:
    """Container for a single Monte Carlo run's results."""
    final_equity: float
    max_dd_pct: float
    sharpe: float
    n_trades: int
    win_rate: float
    profit_factor: float


@dataclass
class MCResult:
    """Results of a Monte Carlo optimization."""
    strategy: str
    n_combos: int
    duration: float
    combos_per_second: float
    params_list: list[dict]
    metrics: np.ndarray  # (n_combos, 6) — same order as MCMetrics
    best_idx: int
    sorted_indices: np.ndarray

    def best_metrics(self) -> MCMetrics:
        return MCMetrics(*self.metrics[self.best_idx])

    def summary(self) -> str:
        b = self.best_metrics()
        lines = [
            f"MC {self.strategy}: {self.n_combos} combos in {self.duration:.1f}s "
            f"({self.combos_per_second:.0f} combo/s)",
            f"Best: Sharpe={b.sharpe:.2f} PF={b.profit_factor:.2f} "
            f"DD={b.max_dd_pct:.1f}% Trades={b.n_trades}",
        ]
        return "\n".join(lines)


def monte_carlo_ma(
    close: np.ndarray,
    fast_range: tuple[int, int],
    slow_range: tuple[int, int],
    capital: float = 10_000.0,
    fee_rate: float = 0.00085,
    step: int = 1,
) -> MCResult:
    """
    Monte Carlo optimization for MA Crossover.

    Parameters
    ----------
    close : np.ndarray (n,)
        Close prices as float64.
    fast_range : (min, max)
        Range of fast MA periods (inclusive).
    slow_range : (min, max)
        Range of slow MA periods (inclusive).
    capital : float
    fee_rate : float
    step : int
        Step between periods (default 1 = exhaustive).

    Returns
    -------
    MCResult
    """
    t0 = time.perf_counter()

    # Build all period values
    fast_vals = list(range(fast_range[0], fast_range[1] + 1, step))
    slow_vals = list(range(slow_range[0], slow_range[1] + 1, step))
    all_periods = sorted(set(fast_vals + slow_vals))

    # Pre-compute SMA bank (once)
    periods_arr = np.array(all_periods, dtype=np.int32)
    sma_bank = compute_sma_bank(close, periods_arr)

    # Build period → index map
    period_to_idx = {p: i for i, p in enumerate(all_periods)}

    # Build combo arrays
    n_combos = len(fast_vals) * len(slow_vals)
    fast_idxs = np.zeros(n_combos, dtype=np.int32)
    slow_idxs = np.zeros(n_combos, dtype=np.int32)
    params_list = []

    idx = 0
    for f in fast_vals:
        for s in slow_vals:
            if f >= s:
                continue  # fast must be faster than slow
            fast_idxs[idx] = period_to_idx[f]
            slow_idxs[idx] = period_to_idx[s]
            params_list.append({"fast": f, "slow": s})
            idx += 1

    # Trim to actual count
    fast_idxs = fast_idxs[:idx]
    slow_idxs = slow_idxs[:idx]
    n_combos = idx

    if n_combos == 0:
        raise ValueError("No valid combos (all fast >= slow)")

    # Run batch kernel
    metrics = batch_ma_crossover(close, sma_bank, fast_idxs, slow_idxs, capital, fee_rate)

    duration = time.perf_counter() - t0
    sorted_idx = np.argsort(-metrics[:, 2])  # sort by Sharpe descending
    best_idx = int(sorted_idx[0])

    return MCResult(
        strategy="MA Crossover",
        n_combos=n_combos,
        duration=duration,
        combos_per_second=n_combos / duration,
        params_list=params_list,
        metrics=metrics,
        best_idx=best_idx,
        sorted_indices=sorted_idx,
    )


def monte_carlo_rsi(
    close: np.ndarray,
    period_range: tuple[int, int],
    oversold_values: list[int],
    overbought_values: list[int],
    capital: float = 10_000.0,
    fee_rate: float = 0.00085,
) -> MCResult:
    """
    Monte Carlo optimization for RSI Mean Reversion.
    """
    t0 = time.perf_counter()

    period_vals = list(range(period_range[0], period_range[1] + 1))

    # Pre-compute RSI bank (once)
    periods_arr = np.array(period_vals, dtype=np.int32)
    rsi_bank = _rsi_bank(close, periods_arr)
    period_to_idx = {p: i for i, p in enumerate(period_vals)}

    # Build combos
    n_combos = len(period_vals) * len(oversold_values) * len(overbought_values)
    period_idxs = np.zeros(n_combos, dtype=np.int32)
    oversold_arr = np.zeros(n_combos, dtype=np.float64)
    overbought_arr = np.zeros(n_combos, dtype=np.float64)
    params_list = []

    idx = 0
    for p in period_vals:
        for os_val in oversold_values:
            for ob_val in overbought_values:
                if os_val >= ob_val:
                    continue
                period_idxs[idx] = period_to_idx[p]
                oversold_arr[idx] = float(os_val)
                overbought_arr[idx] = float(ob_val)
                params_list.append({"period": p, "oversold": os_val, "overbought": ob_val})
                idx += 1

    period_idxs = period_idxs[:idx]
    oversold_arr = oversold_arr[:idx]
    overbought_arr = overbought_arr[:idx]
    n_combos = idx

    if n_combos == 0:
        raise ValueError("No valid combos (all oversold >= overbought)")

    metrics = batch_mean_reversion_rsi(
        close, rsi_bank, period_idxs, oversold_arr, overbought_arr, capital, fee_rate,
    )

    duration = time.perf_counter() - t0
    sorted_idx = np.argsort(-metrics[:, 2])
    best_idx = int(sorted_idx[0])

    return MCResult(
        strategy="RSI Mean Reversion",
        n_combos=n_combos,
        duration=duration,
        combos_per_second=n_combos / duration,
        params_list=params_list,
        metrics=metrics,
        best_idx=best_idx,
        sorted_indices=sorted_idx,
    )


def save_mc_report(
    result: MCResult,
    output_dir: Path = Path("results/reports"),
    top_n: int = 20,
) -> Path:
    """Generate a human-readable Monte Carlo report."""
    output_dir.mkdir(parents=True, exist_ok=True)
    safe_name = result.strategy.lower().replace(" ", "_")
    path = output_dir / f"mc_{safe_name}.txt"

    lines = []
    lines.append("=" * 70)
    lines.append(f"QUANTUMEDGE — MONTE CARLO: {result.strategy}")
    lines.append("=" * 70)
    lines.append(f"\nCombinations: {result.n_combos:,}")
    lines.append(f"Duration: {result.duration:.2f}s ({result.combos_per_second:.0f} combo/s)")
    lines.append(f"\n--- Top {top_n} by Sharpe ---")
    lines.append(f"{'Rank':>5}  {'Params':<35}  {'Equity':>10}  {'Sharpe':>8}  {'PF':>6}  {'DD%':>7}  {'Trades':>7}")
    lines.append("-" * 85)

    for rank in range(min(top_n, result.n_combos)):
        ci = int(result.sorted_indices[rank])
        p = result.params_list[ci]
        m = result.metrics[ci]
        params_str = ", ".join(f"{k}={v}" for k, v in p.items())
        lines.append(
            f"{rank+1:>5}  {params_str:<35}  ${m[0]:>8,.0f}  {m[2]:>8.2f}  "
            f"{m[5]:>6.2f}  {m[1]:>6.2f}%  {int(m[3]):>7,}"
        )

    best = result.best_metrics()
    lines.append(f"\n=== Best Parameters ===")
    for k, v in result.params_list[result.best_idx].items():
        lines.append(f"  {k}: {v}")
    lines.append(f"\n  Final equity: ${best.final_equity:,.2f}")
    lines.append(f"  Sharpe:       {best.sharpe:.4f}")
    lines.append(f"  Profit Factor:{best.profit_factor:.4f}")
    lines.append(f"  Max DD:       {best.max_dd_pct:.2f}%")
    lines.append(f"  Trades:       {best.n_trades}")
    lines.append(f"  Win Rate:     {best.win_rate:.1f}%")

    lines.append("\n" + "=" * 70)
    path.write_text("\n".join(lines))
    return path
