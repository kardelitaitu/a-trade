"""
Parameter optimization for QuantumEdge strategies.

Random search with walk-forward validation to find parameter sets
that generalize across market regimes.
"""

from __future__ import annotations

import itertools
import logging
import math
import random
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

from research.backtest.engine import VectorizedBacktest
from research.backtest.metrics import compute_metrics

logger = logging.getLogger(__name__)


def random_parameter_search(
    strategy_class: type,
    param_ranges: dict[str, list[Any]],
    data: pd.DataFrame,
    n_samples: int = 50,
    walk_forward_splits: Optional[list[tuple[str, str]]] = None,
    metric_target: str = "sharpe_ratio",
    seed: int = 42,
) -> list[dict]:
    """
    Random search over strategy parameters with walk-forward validation.

    Parameters
    ----------
    strategy_class : type
        A BaseStrategy subclass.
    param_ranges : dict
        {param_name: list_of_possible_values}. Random samples from these.
    data : pd.DataFrame
        Full OHLCV dataset.
    n_samples : int
        Number of random parameter combinations to try.
    walk_forward_splits : list of (start, end) tuples, optional
        Default: [(2017, 2022), (2023, 2023), (2024, 2025)] for train/val/test.
    metric_target : str
        Metric to optimize for (e.g. 'sharpe_ratio', 'profit_factor', 'calmar_ratio').
    seed : int

    Returns
    -------
    list of dicts: [{params, train_metrics, val_metrics, test_metrics, score}, ...]
        Sorted by val score descending.
    """
    if walk_forward_splits is None:
        years = sorted(data.index.year.unique())
        walk_forward_splits = [
            (f"{min(years)}-01-01", "2022-12-31"),   # train
            ("2023-01-01", "2023-12-31"),             # val
            ("2024-01-01", f"{max(years)}-12-31"),    # test
        ]

    rng = random.Random(seed)

    # Generate random parameter combinations
    param_combos = []
    for _ in range(n_samples):
        combo = {}
        for param_name, values in param_ranges.items():
            combo[param_name] = rng.choice(values)
        param_combos.append(combo)

    results = []

    for i, params in enumerate(param_combos):
        strat = strategy_class(params)

        split_results = {}
        for split_name, start_str, end_str in _split_labels(walk_forward_splits):
            split_data = data.loc[start_str:end_str]
            if len(split_data) < 1000:
                split_results[split_name] = {}
                continue

            signals = strat.generate_signals(split_data)
            bt = VectorizedBacktest(split_data)
            result = bt.run(signals)
            metrics = compute_metrics(result.equity_curve, result.trades)
            split_results[split_name] = metrics

        # Score = val metric (or train if val not available)
        val_metrics = split_results.get("val", split_results.get("train", {}))
        score = val_metrics.get(metric_target, -999)

        results.append({
            "params": params,
            "score": score,
            **{f"{k}_{sk}": sv for k, v in split_results.items()
               for sk, sv in v.items()},
        })

        if (i + 1) % 10 == 0 or i == 0:
            logger.info(f"  [{i+1}/{n_samples}] best so far: {metric_target}={max(r['score'] for r in results):.2f}")

    results.sort(key=lambda r: r["score"], reverse=True)
    return results


def _split_labels(splits: list[tuple[str, str]]) -> list[tuple[str, str, str]]:
    """Attach labels to split date ranges."""
    labels = ["train", "val", "test"]
    return [(labels[min(i, len(labels) - 1)], s, e) for i, (s, e) in enumerate(splits)]


def run_ma_crossover_search(data: pd.DataFrame, n: int = 50) -> list[dict]:
    """Convenience: run random search for MA Crossover."""
    param_ranges = {
        "fast_period": list(range(5, 50, 2)),
        "slow_period": list(range(20, 100, 5)),
        "ma_type": ["sma", "ema"],
    }
    return random_parameter_search(
        _import_strategy("ma_crossover", "MACrossover"),
        param_ranges, data, n_samples=n,
    )


def run_donchian_search(data: pd.DataFrame, n: int = 50) -> list[dict]:
    """Convenience: run random search for Donchian Breakout."""
    param_ranges = {
        "entry_period": list(range(10, 60, 5)),
        "exit_period": list(range(5, 30, 3)),
    }
    return random_parameter_search(
        _import_strategy("breakout", "DonchianBreakout"),
        param_ranges, data, n_samples=n,
    )


def run_mean_reversion_search(data: pd.DataFrame, n: int = 50) -> list[dict]:
    """Convenience: run random search for Mean Reversion."""
    param_ranges = {
        "mode": ["rsi", "bollinger"],
        "rsi_period": list(range(7, 28, 3)),
        "rsi_oversold": [20, 25, 30, 35],
        "rsi_overbought": [65, 70, 75, 80],
        "bb_period": list(range(10, 40, 5)),
        "bb_std": [1.5, 2.0, 2.5, 3.0],
    }
    return random_parameter_search(
        _import_strategy("mean_reversion", "MeanReversion"),
        param_ranges, data, n_samples=n,
    )


def _import_strategy(module_name: str, class_name: str):
    """Lazy import a strategy class by name."""
    import importlib
    mod = importlib.import_module(f"research.strategies.{module_name}")
    return getattr(mod, class_name)


def generate_optimization_report(
    results: list[dict],
    strategy_name: str,
    output_dir: Path,
    n_top: int = 10,
) -> Path:
    """Generate a human-readable optimization report."""
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"optimization_{strategy_name.lower().replace(' ', '_')}.txt"

    lines = []
    lines.append("=" * 70)
    lines.append(f"QUANTUMEDGE — PARAMETER OPTIMIZATION: {strategy_name}")
    lines.append("=" * 70)
    lines.append(f"\nTotal combinations tested: {len(results)}")
    lines.append(f"\n--- Top {n_top} Results ---")
    lines.append(f"{'Rank':>5}  {'Score':>7}  {'Params':<50}")
    lines.append("-" * 70)

    for i, r in enumerate(results[:n_top]):
        params_str = ", ".join(f"{k}={v}" for k, v in r["params"].items())
        lines.append(f"{i+1:>5}  {r['score']:>7.2f}  {params_str:<50}")

    # Best params
    best = results[0]
    lines.append(f"\n--- Best Parameters ---")
    for k, v in best["params"].items():
        lines.append(f"  {k}: {v}")

    # Per-split metrics for best
    lines.append(f"\n--- Walk-Forward Metrics (Best) ---")
    splits = set()
    for key in best:
        if "_" in key and key.split("_")[0] in ("train", "val", "test"):
            splits.add(key.split("_")[0])

    for split in sorted(splits):
        lines.append(f"\n  [{split}]")
        for metric in ["sharpe_ratio", "profit_factor", "max_drawdown_pct", "total_trades"]:
            key = f"{split}_{metric}"
            if key in best:
                lines.append(f"    {metric}: {best[key]:.4f}")

    lines.append("\n" + "=" * 70)
    path.write_text("\n".join(lines))
    return path
