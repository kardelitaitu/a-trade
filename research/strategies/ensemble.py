"""
Ensemble Signal Combiner for QuantumEdge.

Combines multiple strategy signals using rolling Sharpe-weighted averaging.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from research.backtest.engine import VectorizedBacktest
from research.backtest.metrics import compute_metrics, format_metrics_report


def compute_rolling_sharpe(
    signals: pd.Series,
    close: pd.Series,
    window: int = 1440,  # ~5 days on 5m data
    periods_per_year: int = 105120,
) -> pd.Series:
    """
    Compute rolling Sharpe ratio for a strategy's signals.

    Uses signal * close_return as daily returns, then computes
    rolling annualized Sharpe.

    Parameters
    ----------
    signals : pd.Series (-1, 0, +1 or continuous float)
        Trading signals aligned with close index.
    close : pd.Series
        Close price, same index as signals.
    window : int
        Rolling window in periods.
    periods_per_year : int
        For annualization (default 105120 for 5-min).

    Returns
    -------
    pd.Series
        Rolling annualized Sharpe ratio.
    """
    close_return = close.pct_change(fill_method=None).fillna(0)
    strategy_returns = signals.shift(1).fillna(0) * close_return

    rolling_mean = strategy_returns.rolling(window, min_periods=window // 4).mean()
    rolling_std = strategy_returns.rolling(window, min_periods=window // 4).std(ddof=0)

    # Annualize
    periods_per_window = min(window, len(strategy_returns))
    ann_factor = np.sqrt(periods_per_year / periods_per_window)
    sharpe = rolling_mean / rolling_std.replace(0, np.nan) * ann_factor

    return sharpe.fillna(0)


def ensemble_signals(
    signal_dict: dict[str, pd.Series],
    close: pd.Series,
    method: str = "sharpe_weight",
    window: int = 1440,
    equal_weight: Optional[list[float]] = None,
) -> pd.Series:
    """
    Combine multiple strategy signals into one ensemble signal.

    Parameters
    ----------
    signal_dict : dict[str, pd.Series]
        {name: signal_series}. All series must have the same index.
    close : pd.Series
        Close price with same index, used for rolling Sharpe computation.
    method : str
        'equal' — all strategies weighted equally.
        'sharpe_weight' — weight by rolling Sharpe (adaptive).
        'sharpe_rank' — weight by Sharpe rank (less extreme than raw weight).
    window : int
        Rolling window for Sharpe computation (only for sharpe methods).
    equal_weight : list[float], optional
        Custom weights for 'equal' method. Must sum to 1 and match signal_dict order.

    Returns
    -------
    pd.Series
        Combined ensemble signal (-1 to +1).
    """
    if not signal_dict:
        raise ValueError("At least one signal is required")

    # Validate all signals share the same index
    first_idx = list(signal_dict.values())[0].index
    for name, sig in signal_dict.items():
        if not sig.index.equals(first_idx):
            raise ValueError(f"Signal '{name}' has mismatched index")

    names = list(signal_dict.keys())
    signals = list(signal_dict.values())

    if method == "equal":
        if equal_weight is not None:
            weights = np.array(equal_weight)
        else:
            weights = np.ones(len(signals)) / len(signals)

        combined = sum(w * s for w, s in zip(weights, signals))

    elif method == "sharpe_weight":
        sharpe_values = []
        for name, sig in signal_dict.items():
            sh = compute_rolling_sharpe(sig, close, window=window)
            sharpe_values.append(sh)

        sharpe_df = pd.concat(sharpe_values, axis=1, keys=names)

        # Softmax weighting: positive Sharpe = positive weight
        # Negative Sharpe gets clipped to 0 (don't trade bad strategies)
        weights_df = sharpe_df.clip(lower=0)
        weight_sum = weights_df.sum(axis=1).replace(0, np.nan)

        combined = pd.Series(0.0, index=first_idx)
        for name in names:
            w = weights_df[name] / weight_sum
            combined += w.fillna(0) * signal_dict[name]

    elif method == "sharpe_rank":
        sharpe_values = []
        for name, sig in signal_dict.items():
            sh = compute_rolling_sharpe(sig, close, window=window)
            sharpe_values.append(sh)

        sharpe_df = pd.concat(sharpe_values, axis=1, keys=names)

        # Rank-based weighting (less extreme)
        ranks = sharpe_df.rank(axis=1)
        weights_df = ranks / ranks.sum(axis=1).replace(0, np.nan)

        combined = pd.Series(0.0, index=first_idx)
        for name in names:
            w = weights_df[name]
            combined += w.fillna(0) * signal_dict[name]

    else:
        raise ValueError(f"Unknown method: {method}")

    return combined.clip(-1, 1)


def backtest_ensemble(
    signals: pd.Series,
    data: pd.DataFrame,
    name: str = "ensemble",
) -> dict:
    """
    Run a backtest on ensemble signals and return results.

    Parameters
    ----------
    signals : pd.Series
        Ensemble signal series.
    data : pd.DataFrame
        OHLCV data.
    name : str
        Label for the output report.

    Returns
    -------
    dict with metrics and result.
    """
    bt = VectorizedBacktest(data)
    result = bt.run(signals)
    metrics = compute_metrics(result.equity_curve, result.trades)
    print(f"\n=== ENSEMBLE BACKTEST: {name} ===")
    print(format_metrics_report(metrics))
    return {"metrics": metrics, "result": result}
