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
    batch_ma_crossover_sltp,
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
    metrics: np.ndarray  # (n_combos, 6) — final_eq, max_dd, sharpe, trades, wr, pf
    best_idx: int
    sorted_indices: np.ndarray
    initial_capital: float = 10_000.0
    command: str = ""
    asset: str = "BTCUSDT"
    data_range: str = ""
    fee_rate: float = 0.00085

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
        initial_capital=capital,
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
    """Generate a plain-text Monte Carlo optimization report."""
    import datetime

    output_dir.mkdir(parents=True, exist_ok=True)
    safe_name = result.strategy.lower().replace(" ", "_").replace("/", "_").replace("+", "plus")
    path = output_dir / f"mc_{safe_name}.txt"

    now = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    capital = result.initial_capital
    fee_pct = result.fee_rate * 100
    sub = "-" * 40

    lines = []
    lines.append("")
    lines.append("MONTE CARLO OPTIMIZATION")
    lines.append(f"  Generated:    {now}")
    lines.append(f"  Command:      {result.command}")
    lines.append(f"  Strategy:     {result.strategy}")
    lines.append(f"  Asset:        {result.asset}")
    lines.append(f"  Data range:   {result.data_range}")
    lines.append(f"  Initial capt: ${capital:>10,.2f}")
    lines.append(f"  Fee rate:     {fee_pct:.3f}%")
    hw = "32 threads \u00b7 96 GB RAM \u00b7 NVMe SSD"
    lines.append(f"  Hardware:     {hw}")
    lines.append(f"  Combinations: {result.n_combos:>6,} in {result.duration:.1f}s "
                 f"({result.combos_per_second:.0f} \u00b1 combo/s)")
    lines.append("")

    # ── Helper: fixed-width separator ──
    SEP = "-" * 80

    # Insert separators after header sections
    lines.insert(2, SEP)

    # ── Parameter Sweep Analysis ──
    lines.append("PARAMETER SWEEP ANALYSIS")
    lines.append(SEP)
    param_keys = list(result.params_list[0].keys()) if result.params_list else []
    for pk in param_keys:
        values = [p[pk] for p in result.params_list]
        if len(values) == 0:
            continue
        unique_vals = sorted(set(values))
        val_min = min(unique_vals)
        val_max = max(unique_vals)
        mean_val = sum(values) / len(values)
        std_val = (sum((v - mean_val) ** 2 for v in values) / len(values)) ** 0.5
        step = unique_vals[1] - unique_vals[0] if len(unique_vals) > 1 else 1

        lines.append(f"  {pk:<20}  {val_min} - {val_max}  (step {step})")
        lines.append(f"  {'':>20}  Mean: {mean_val:<8.1f}  Std Dev: {std_val:<8.1f}")

        sharpe_by_val = {}
        for i, p in enumerate(result.params_list):
            v = p[pk]
            if v not in sharpe_by_val:
                sharpe_by_val[v] = []
            sharpe_by_val[v].append(result.metrics[i, 2])

        best_val = None
        best_sharpe = -999
        for v, sharps in sharpe_by_val.items():
            avg_s = sum(sharps) / len(sharps)
            if avg_s > best_sharpe:
                best_sharpe = avg_s
                best_val = v

        std_sharpe = (
            sum((s - best_sharpe) ** 2 for s in sharpe_by_val.get(best_val, []))
            / max(len(sharpe_by_val.get(best_val, [])), 1)
        ) ** 0.5 if sharpe_by_val.get(best_val) else 0

        lines.append(f"  {'':>20}  Best value: {best_val:<12}  Sharpe avg: {best_sharpe:+6.2f}  (\u00b1{std_sharpe:.2f})")
        lines.append("")

    # ── Top N Table ──
    lines.append(f"TOP {top_n} BY SHARPE")
    header = f"  {'#':>3}  {'Parameters':<38}  {'Init $':>8}  {'Final $':>9}  {'Sharpe':>7}  {'PF':>5}  {'DD%':>6}  {'Trades':>7}"
    dash = "-" * (len(header) - 2)
    lines.append(f"  {dash}")
    lines.append(header)
    lines.append(f"  {dash}")

    for rank in range(min(top_n, result.n_combos)):
        ci = int(result.sorted_indices[rank])
        p = result.params_list[ci]
        m = result.metrics[ci]
        params_str = ", ".join(f"{k}={v}" for k, v in p.items())
        lines.append(
            f"  {rank+1:>3}  {params_str:<38}  ${capital:>7,.0f}  ${m[0]:>8,.0f}  "
            f"{m[2]:>7.2f}  {m[5]:>5.2f}  {m[1]:>6.2f}%  {int(m[3]):>7,}"
        )
    lines.append("")

    # ── Best Parameters ──
    best = result.best_metrics()
    lines.append("BEST PARAMETERS (Rank 1)")
    lines.append(SEP)
    for k, v in result.params_list[result.best_idx].items():
        lines.append(f"  {k:<20}  {v}")
    lines.append(f"  {sub}")
    lines.append(f"  Initial Equity        ${capital:>12,.2f}")
    lines.append(f"  Final Equity          ${best.final_equity:>12,.2f}")
    lines.append(f"  Total Return          {((best.final_equity / capital) - 1) * 100:>+11.2f}%")
    lines.append(f"  Sharpe Ratio          {best.sharpe:>15.4f}")
    lines.append(f"  Profit Factor         {best.profit_factor:>15.4f}")
    lines.append(f"  Max Drawdown          {best.max_dd_pct:>13.2f}%")
    lines.append(f"  Total Trades          {best.n_trades:>15,}")
    lines.append(f"  Win Rate              {best.win_rate:>14.1f}%")
    lines.append("")

    # ── Reproducibility ──
    lines.append("REPRODUCIBILITY")
    lines.append(SEP)
    cmd = result.command or "python -c \"from research.data.loader import load_parquet; ...\""
    lines.append(f"  {cmd}")
    lines.append("")

    path.write_text("\n".join(lines))
    return path


def monte_carlo_ma_sltp(
    close: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    fast_range: tuple[int, int],
    slow_range: tuple[int, int],
    sl_values: list[float],
    tp_values: list[float],
    capital: float = 10_000.0,
    fee_rate: float = 0.00085,
    step: int = 1,
) -> MCResult:
    """Monte Carlo for MA Crossover with SL/TP."""
    t0 = time.perf_counter()
    fast_vals = list(range(fast_range[0], fast_range[1] + 1, step))
    slow_vals = list(range(slow_range[0], slow_range[1] + 1, step))
    all_periods = sorted(set(fast_vals + slow_vals))
    periods_arr = np.array(all_periods, dtype=np.int32)
    sma_bank = compute_sma_bank(close, periods_arr)
    period_to_idx = {p: i for i, p in enumerate(all_periods)}
    n_combos = len(fast_vals) * len(slow_vals) * len(sl_values) * len(tp_values)
    fast_idxs = np.zeros(n_combos, dtype=np.int32)
    slow_idxs = np.zeros(n_combos, dtype=np.int32)
    sl_arr = np.zeros(n_combos, dtype=np.float64)
    tp_arr = np.zeros(n_combos, dtype=np.float64)
    params_list = []
    idx = 0
    for f in fast_vals:
        for s in slow_vals:
            if f >= s:
                continue
            for sl in sl_values:
                for tp in tp_values:
                    if sl >= tp:
                        continue
                    fast_idxs[idx] = period_to_idx[f]
                    slow_idxs[idx] = period_to_idx[s]
                    sl_arr[idx] = sl
                    tp_arr[idx] = tp
                    params_list.append({"fast": f, "slow": s, "sl_pct": sl * 100, "tp_pct": tp * 100})
                    idx += 1
    fast_idxs = fast_idxs[:idx]
    slow_idxs = slow_idxs[:idx]
    sl_arr = sl_arr[:idx]
    tp_arr = tp_arr[:idx]
    n_combos = idx
    if n_combos == 0:
        raise ValueError("No valid combos")
    metrics = batch_ma_crossover_sltp(
        close, high, low, sma_bank, fast_idxs, slow_idxs,
        sl_arr, tp_arr, capital, fee_rate,
    )
    duration = time.perf_counter() - t0
    sorted_idx = np.argsort(-metrics[:, 2])
    best_idx = int(sorted_idx[0])
    return MCResult(
        strategy="MA Crossover + SL/TP",
        n_combos=n_combos,
        duration=duration,
        combos_per_second=n_combos / duration,
        params_list=params_list,
        metrics=metrics,
        best_idx=best_idx,
        sorted_indices=sorted_idx,
        initial_capital=capital,
        fee_rate=fee_rate,
    )
