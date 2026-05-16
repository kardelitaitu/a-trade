"""
Parameter Sensitivity Analysis for QuantumEdge Monte Carlo results.

Analyzes the relationship between parameter values and performance metrics
(Sharpe, PF, DD, Return) to identify which parameters matter and what
values work best.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


def analyze_sensitivity(results: list[dict]) -> dict[str, Any]:
    """
    Analyze how each parameter affects Sharpe ratio.

    Parameters
    ----------
    results : list of dict
        Output from parallel_monte_carlo().

    Returns
    -------
    dict with keys:
        param_summaries : list of {param, correlation, best_value, importance, details}
        top_params      : list sorted by importance (highest first)
        recommendations : str — human-readable summary
    """
    if not results:
        return {"param_summaries": [], "top_params": [], "recommendations": "No results to analyze."}

    # Collect all parameter names from the first result
    param_names = list(results[0]["params"].keys())

    summaries = []
    for param in param_names:
        values = []
        sharpes = []
        # Group by parameter value
        groups: dict[Any, list[float]] = {}
        for r in results:
            v = r["params"][param]
            sh = r["sharpe"]
            if v not in groups:
                groups[v] = []
            groups[v].append(sh)
            values.append(v)
            sharpes.append(sh)

        # Compute mean Sharpe per value
        value_stats = []
        for v, sh_list in sorted(groups.items(), key=lambda x: str(x[0])):
            value_stats.append({
                "value": v,
                "mean_sharpe": np.mean(sh_list),
                "std_sharpe": np.std(sh_list),
                "count": len(sh_list),
                "sharpe_list": sh_list,
            })

        # Find best value
        best = max(value_stats, key=lambda x: x["mean_sharpe"])

        # Compute correlation and slope (only if numeric)
        corr = None
        slope = None
        if all(isinstance(v, (int, float)) for v in values):
            try:
                corr = float(np.corrcoef(values, sharpes)[0, 1])
                # Slope = correlation * (std_sharpe / std_param)
                if np.std(values) > 0:
                    slope = float(corr * np.std(sharpes) / np.std(values))
            except Exception:
                corr = 0.0
                slope = 0.0

        # Importance = max mean Sharpe - min mean Sharpe (spread)
        mean_sharpes = [s["mean_sharpe"] for s in value_stats]
        importance = max(mean_sharpes) - min(mean_sharpes) if mean_sharpes else 0.0

        summaries.append({
            "param": param,
            "correlation": corr,
            "slope": slope,
            "importance": importance,
            "best_value": best["value"],
            "best_mean_sharpe": best["mean_sharpe"],
            "best_std_sharpe": best["std_sharpe"],
            "best_count": best["count"],
            "value_stats": value_stats,
        })

    # Sort by importance descending
    summaries.sort(key=lambda s: abs(s["importance"]), reverse=True)
    top_params = [s["param"] for s in summaries]

    recommendations = _build_recommendations(summaries, results)

    return {
        "param_summaries": summaries,
        "top_params": top_params,
        "recommendations": recommendations,
    }


def _build_recommendations(summaries: list[dict], results: list[dict]) -> str:
    """Generate human-readable analysis text."""
    lines = []
    lines.append("Parameter Sensitivity Analysis")
    lines.append("=" * 60)
    lines.append(f"Total combos tested: {len(results)}")
    lines.append("")

    for s in summaries:
        p = s["param"]
        corr = s["correlation"]
        slope = s["slope"]
        imp = s["importance"]
        best_v = s["best_value"]
        best_sh = s["best_mean_sharpe"]

        corr_str = f"{corr:+.3f}" if corr is not None else "N/A (non-numeric)"
        slope_str = f"{slope:+.4f}" if slope is not None else "N/A"
        lines.append(f"{p}:")
        lines.append(f"  Per 1 unit:    Sharpe {slope_str}  (avg change per unit)")
        lines.append(f"  Best value:    {best_v}  (avg Sharpe {best_sh:+.3f})")
        lines.append(f"  Importance:    {imp:.3f}  (spread between best/worst value)")
        lines.append(f"  Correlation:   {corr_str}")

        # Show top 3 values
        top3 = sorted(s["value_stats"], key=lambda x: x["mean_sharpe"], reverse=True)[:3]
        values_str = ", ".join(f"{v['value']} (Sharpe {v['mean_sharpe']:+.2f})" for v in top3)
        lines.append(f"  Top values:    {values_str}")
        lines.append("")

    # Overall recommendation
    best = results[0]
    lines.append("BEST COMBO FOUND")
    lines.append("-" * 40)
    for k, v in best["params"].items():
        lines.append(f"  {k}: {v}")
    lines.append(f"  Sharpe:        {best['sharpe']:.4f}")
    lines.append(f"  Profit Factor: {best['profit_factor']:.4f}")
    lines.append(f"  Max DD:        {best['max_dd']:.2f}%")
    lines.append(f"  Total Return:  {best['total_return']:+.2f}%")
    lines.append(f"  Trades:        {best['trades']}")

    return "\n".join(lines)


def save_sensitivity_report(
    analysis: dict,
    strategy_name: str,
    output_dir: str | Path = "results/reports",
) -> Path:
    """Save sensitivity analysis to a report file."""
    import datetime
    from pathlib import Path

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    safe = strategy_name.lower().replace(" ", "_").replace("/", "_")
    now = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d_%H%M%S")
    path = output_dir / f"sensitivity_{safe}_{now}.txt"
    path.write_text(analysis["recommendations"])
    return path
