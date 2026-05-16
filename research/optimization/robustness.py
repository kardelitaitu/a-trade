"""
Statistical robustness tests for QuantumEdge strategies.

Runs Monte Carlo simulation, regime breakdown, and parameter
stability tests on a strategy's backtest results.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from research.backtest.engine import VectorizedBacktest
from research.backtest.metrics import compute_metrics
from research.strategies.regime import detect_regime_sma


def monte_carlo_shuffle(
    trades: list,
    n_simulations: int = 1000,
    initial_capital: float = 10_000,
    seed: int = 42,
) -> dict:
    """
    Monte Carlo simulation by shuffling trade outcomes.

    Shuffles the order of trades, recomputes equity curves,
    and compares the actual Sharpe against the distribution
    of shuffled Sharps.

    Parameters
    ----------
    trades : list of Trade
        Trade objects from a backtest run.
    n_simulations : int
    initial_capital : float
    seed : int

    Returns
    -------
    dict with keys: actual_sharpe, sim_sharpes, p_value, percentile, is_significant
    """
    if len(trades) < 30:
        return {
            "actual_sharpe": 0.0,
            "sim_sharpes": [],
            "p_value": 1.0,
            "percentile": 50.0,
            "is_significant": False,
            "message": f"Too few trades ({len(trades)}), minimum 30 required.",
        }

    # Compute actual Sharpe from trade PnLs
    pnls = np.array([t.pnl for t in trades])
    actual_sharpe = pnls.mean() / pnls.std() * np.sqrt(len(trades)) if pnls.std() > 0 else 0.0

    rng = np.random.default_rng(seed)
    sim_sharpes = np.zeros(n_simulations)

    for i in range(n_simulations):
        shuffled = rng.permutation(pnls)
        sim_sharpes[i] = shuffled.mean() / shuffled.std() * np.sqrt(len(shuffled)) if shuffled.std() > 0 else 0.0

    # p-value: fraction of simulations with Sharpe >= actual
    p_value = float((sim_sharpes >= actual_sharpe).mean())
    percentile = float((sim_sharpes < actual_sharpe).mean() * 100)

    return {
        "actual_sharpe": float(actual_sharpe),
        "sim_sharpes": sim_sharpes.tolist(),
        "p_value": p_value,
        "percentile": percentile,
        "is_significant": p_value < 0.05,
        "n_simulations": n_simulations,
        "message": (
            f"Actual Sharpe {actual_sharpe:.2f} is at the {percentile:.0f}th percentile "
            f"of shuffled trades (p={p_value:.3f}). "
            f"{'Edge is statistically significant.' if p_value < 0.05 else 'Edge not statistically significant.'}"
        ),
    }


def regime_breakdown(
    data: pd.DataFrame,
    signals: pd.Series,
    regime_period: int = 200,
) -> pd.DataFrame:
    """
    Run backtest segregated by market regime (bull/bear/sideways).

    Parameters
    ----------
    data : pd.DataFrame
    signals : pd.Series (-1, 0, +1)
    regime_period : int
        SMA period for regime detection.

    Returns
    -------
    pd.DataFrame with rows = regimes, cols = metrics.
    """
    regime = detect_regime_sma(data["close"], period=regime_period)

    results = []
    for regime_name in ["bull", "bear", "sideways"]:
        mask = regime == regime_name
        if mask.sum() < 100:
            continue

        regime_data = data.loc[mask]
        regime_signals = signals.reindex(regime_data.index, method=None)

        # Only trade when regime matches and we have a signal
        bt = VectorizedBacktest(regime_data)
        result = bt.run(regime_signals)
        metrics = compute_metrics(result.equity_curve, result.trades)

        results.append({
            "regime": regime_name,
            "periods": mask.sum(),
            "trades": metrics["total_trades"],
            "sharpe": metrics["sharpe_ratio"],
            "profit_factor": metrics["profit_factor"],
            "max_dd": metrics["max_drawdown_pct"],
            "cagr": metrics["cagr_pct"],
            "win_rate": metrics["win_rate_pct"],
        })

    return pd.DataFrame(results)


def parameter_stability(
    strategy_class: type,
    base_params: dict,
    data: pd.DataFrame,
    variations: Optional[dict[str, list]] = None,
) -> pd.DataFrame:
    """
    Test how metrics change when each parameter is varied ±X%.

    Parameters
    ----------
    strategy_class : type
        BaseStrategy subclass.
    base_params : dict
        The "best" parameters found.
    data : pd.DataFrame
    variations : dict, optional
        {param: [values_to_test]}. Defaults to ±20% around base.

    Returns
    -------
    pd.DataFrame with rows = parameter variations, cols = metrics.
    """
    if variations is None:
        variations = {}
        for k, v in base_params.items():
            if isinstance(v, (int, float)) and v > 1:
                var_values = [max(1, int(v * 0.8)), int(v * 0.9), v, int(v * 1.1), int(v * 1.2)]
                var_values = sorted(set(var_values))
                if len(var_values) > 1:
                    variations[k] = var_values

    results = []
    baseline = _backtest_params(strategy_class, base_params, data)

    results.append({
        "param": "baseline",
        "value": "-",
        "sharpe": baseline["sharpe_ratio"],
        "profit_factor": baseline["profit_factor"],
        "max_dd": baseline["max_drawdown_pct"],
        "trades": baseline["total_trades"],
    })

    for param, values in variations.items():
        for val in values:
            params = base_params.copy()
            # Map param names for type mismatches
            if param == "rsi_oversold" and val >= base_params.get("rsi_overbought", 100):
                continue
            if param == "rsi_overbought" and val <= base_params.get("rsi_oversold", 0):
                continue
            params[param] = val

            try:
                metrics = _backtest_params(strategy_class, params, data)
                results.append({
                    "param": param,
                    "value": val,
                    "sharpe": metrics["sharpe_ratio"],
                    "profit_factor": metrics["profit_factor"],
                    "max_dd": metrics["max_drawdown_pct"],
                    "trades": metrics["total_trades"],
                })
            except Exception:
                continue

    return pd.DataFrame(results)


def _backtest_params(strategy_class, params, data) -> dict:
    """Run a single backtest with given params and return metrics."""
    strat = strategy_class(params)
    signals = strat.generate_signals(data)
    bt = VectorizedBacktest(data)
    result = bt.run(signals)
    return compute_metrics(result.equity_curve, result.trades)


def generate_robustness_report(
    monte_carlo_result: dict,
    regime_df: pd.DataFrame,
    stability_df: pd.DataFrame,
    strategy_name: str,
    output_dir: Path,
) -> Path:
    """Generate a comprehensive robustness report."""
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"robustness_{strategy_name.lower().replace(' ', '_')}.txt"

    lines = []
    lines.append("=" * 70)
    lines.append(f"QUANTUMEDGE — ROBUSTNESS REPORT: {strategy_name}")
    lines.append("=" * 70)

    # Monte Carlo
    lines.append(f"\n{'MONTE CARLO SIMULATION':-^70}")
    lines.append(f"  {monte_carlo_result.get('message', 'N/A')}")
    lines.append(f"  Simulations: {monte_carlo_result.get('n_simulations', 0)}")
    lines.append(f"  Actual Sharpe: {monte_carlo_result.get('actual_sharpe', 0):.4f}")
    lines.append(f"  p-value: {monte_carlo_result.get('p_value', 1):.4f}")
    lines.append(f"  Significant: {monte_carlo_result.get('is_significant', False)}")

    # Regime breakdown
    lines.append(f"\n{'REGIME BREAKDOWN':-^70}")
    if len(regime_df) > 0:
        lines.append(f"  {'Regime':<12} {'Periods':>8} {'Trades':>8} {'Sharpe':>8} {'PF':>8} {'DD%':>8}")
        lines.append("  " + "-" * 56)
        for _, row in regime_df.iterrows():
            lines.append(
                f"  {row['regime']:<12} {row['periods']:>8,} {row['trades']:>8} "
                f"{row['sharpe']:>8.2f} {row['profit_factor']:>8.2f} {row['max_dd']:>8.2f}"
            )
    else:
        lines.append("  No regime data available.")

    # Parameter stability
    lines.append(f"\n{'PARAMETER STABILITY':-^70}")
    if len(stability_df) > 0:
        for param in stability_df["param"].unique():
            subset = stability_df[stability_df["param"] == param]
            lines.append(f"\n  {param}:")
            for _, row in subset.iterrows():
                marker = " ← baseline" if row["param"] == "baseline" else ""
                lines.append(
                    f"    {str(row['value']):>10} → Sharpe={row['sharpe']:>6.2f} "
                    f"PF={row['profit_factor']:>5.2f} DD={row['max_dd']:>6.2f}%{marker}"
                )

    lines.append("\n" + "=" * 70)
    path.write_text("\n".join(lines))
    return path
