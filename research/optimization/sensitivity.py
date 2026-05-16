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


def _analyze_metric(
    results: list[dict],
    param_names: list[str],
    metric_key: str,
    higher_is_better: bool,
) -> list[dict]:
    """
    Analyze how each parameter affects a specific metric.

    Parameters
    ----------
    results : list of dict
    param_names : list of str
    metric_key : str ('sharpe', 'max_dd', 'profit_factor', 'total_return')
    higher_is_better : bool — True for Sharpe, False for drawdown

    Returns
    -------
    list of dict, one per parameter.
    """
    summaries = []
    for param in param_names:
        values = []
        metric_vals = []
        groups: dict[Any, list[float]] = {}

        for r in results:
            v = r["params"][param]
            mv = r[metric_key]
            if v not in groups:
                groups[v] = []
            groups[v].append(mv)
            values.append(v)
            metric_vals.append(mv)

        # Mean metric per value
        value_stats = []
        for v, mv_list in sorted(groups.items(), key=lambda x: str(x[0])):
            value_stats.append({
                "value": v,
                "mean": np.mean(mv_list),
                "std": np.std(mv_list),
                "count": len(mv_list),
                "values": mv_list,
            })

        # Best value (highest or lowest mean depending on metric)
        if higher_is_better:
            best = max(value_stats, key=lambda x: x["mean"])
        else:
            best = min(value_stats, key=lambda x: x["mean"])

        # Correlation and slope
        corr = None
        slope = None
        if all(isinstance(v, (int, float)) for v in values):
            try:
                corr = float(np.corrcoef(values, metric_vals)[0, 1])
                if np.std(values) > 0:
                    slope = float(corr * np.std(metric_vals) / np.std(values))
            except Exception:
                corr = 0.0
                slope = 0.0

        means = [s["mean"] for s in value_stats]
        importance = max(means) - min(means) if means else 0.0

        summaries.append({
            "param": param,
            "correlation": corr,
            "slope": slope,
            "importance": importance,
            "best_value": best["value"],
            "best_mean": best["mean"],
            "best_std": best["std"],
            "best_count": best["count"],
            "value_stats": value_stats,
        })

    summaries.sort(key=lambda s: abs(s["importance"]), reverse=True)
    return summaries


def analyze_sensitivity(results: list[dict]) -> dict[str, Any]:
    """
    Analyze how each parameter affects Sharpe AND Drawdown.

    Returns
    -------
    dict with keys:
        sharpe : list of param summaries vs Sharpe
        drawdown : list of param summaries vs Max DD
        recommendations : str
    """
    if not results:
        return {"sharpe": [], "drawdown": [], "recommendations": "No results."}

    param_names = list(results[0]["params"].keys())

    sharpe_analysis = _analyze_metric(results, param_names, "sharpe", higher_is_better=True)
    dd_analysis = _analyze_metric(results, param_names, "max_dd", higher_is_better=False)

    recommendations = _build_recommendations(results, sharpe_analysis, dd_analysis)

    return {
        "sharpe": sharpe_analysis,
        "drawdown": dd_analysis,
        "recommendations": recommendations,
    }


def _build_recommendations(
    results: list[dict],
    sharpe_analysis: list[dict],
    dd_analysis: list[dict],
) -> str:
    """Generate human-readable report with Sharpe and DD analysis side by side."""
    lines = []
    lines.append("Parameter Sensitivity Analysis")
    lines.append("=" * 60)
    lines.append(f"Total combos tested: {len(results)}")
    lines.append("")

    # Build a dict for quick DD lookup by param name
    dd_by_param = {s["param"]: s for s in dd_analysis}

    for s in sharpe_analysis:
        p = s["param"]
        d = dd_by_param.get(p, {})

        def fmt_corr(c):
            return f"{c:+.3f}" if c is not None else "N/A"

        def fmt_slope(sl):
            return f"{sl:+.4f}" if sl is not None else "N/A"

        lines.append(f"{p}:")
        lines.append(f"  vs Sharpe:")
        lines.append(f"    Per 1 unit:  Sharpe {fmt_slope(s['slope'])}")
        lines.append(f"    Best value:  {s['best_value']}  (avg Sharpe {s['best_mean']:+.3f})")
        lines.append(f"    Importance:  {s['importance']:.3f}  (spread)")
        lines.append(f"    Correlation: {fmt_corr(s['correlation'])}")
        top3_sh = sorted(s["value_stats"], key=lambda x: x["mean"], reverse=True)[:3]
        _vals = ", ".join(f'{v["value"]} (Sharpe {v["mean"]:+.2f})' for v in top3_sh)
        lines.append(f"    Top values:  {_vals}")
        lines.append("")
        lines.append(f"  vs Drawdown:")
        if d:
            lines.append(f"    Per 1 unit:  DD {fmt_slope(d['slope'])}")
            lines.append(f"    Best value:  {d['best_value']}  (avg DD {d['best_mean']:.2f}%)")
            lines.append(f"    Importance:  {d['importance']:.3f}  (spread)")
            lines.append(f"    Correlation: {fmt_corr(d['correlation'])}")
            top3_dd = sorted(d["value_stats"], key=lambda x: x["mean"])[:3]
            _vals_dd = ", ".join(f'{v["value"]} (DD {v["mean"]:.1f}%)' for v in top3_dd)
            lines.append(f"    Top values:  {_vals_dd}")
        else:
            lines.append("    (no data)")

        lines.append("")

    # Best combo
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
