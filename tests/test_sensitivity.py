"""Tests for research/optimization/sensitivity.py."""

import pytest

from research.optimization.sensitivity import analyze_sensitivity, save_sensitivity_report


@pytest.fixture
def sample_results():
    """Standard test results for sensitivity analysis."""
    return [
        {"params": {"fast": 10, "slow": 30, "sl": 0.01}, "sharpe": 0.8, "max_dd": -20,
         "profit_factor": 1.5, "total_return": 50, "trades": 100},
        {"params": {"fast": 10, "slow": 30, "sl": 0.02}, "sharpe": 0.6, "max_dd": -15,
         "profit_factor": 1.3, "total_return": 30, "trades": 80},
        {"params": {"fast": 20, "slow": 40, "sl": 0.01}, "sharpe": 0.3, "max_dd": -30,
         "profit_factor": 1.1, "total_return": 20, "trades": 60},
        {"params": {"fast": 20, "slow": 40, "sl": 0.02}, "sharpe": 0.1, "max_dd": -25,
         "profit_factor": 0.9, "total_return": 10, "trades": 40},
    ]


class TestSensitivityAnalysis:

    def test_analyze_returns_dict(self, sample_results):
        """Should return dict with expected keys."""
        analysis = analyze_sensitivity(sample_results)
        assert "sharpe" in analysis
        assert "drawdown" in analysis
        assert "calmar" in analysis
        assert "recommendations" in analysis

    def test_sharpe_analysis_has_params(self, sample_results):
        """Sharpe analysis should contain all parameter names."""
        analysis = analyze_sensitivity(sample_results)
        param_names = {s["param"] for s in analysis["sharpe"]}
        assert "fast" in param_names
        assert "slow" in param_names
        assert "sl" in param_names

    def test_drawdown_analysis_has_params(self, sample_results):
        """Drawdown analysis should contain all parameter names."""
        analysis = analyze_sensitivity(sample_results)
        param_names = {s["param"] for s in analysis["drawdown"]}
        assert "fast" in param_names

    def test_correlation_properties(self, sample_results):
        """Correlation should be between -1 and 1."""
        analysis = analyze_sensitivity(sample_results)
        for s in analysis["sharpe"]:
            if s["correlation"] is not None:
                assert -1.0 <= s["correlation"] <= 1.0

    def test_importance_positive(self, sample_results):
        """Importance should be non-negative."""
        analysis = analyze_sensitivity(sample_results)
        for s in analysis["sharpe"]:
            assert s["importance"] >= 0

    def test_best_value_in_value_stats(self, sample_results):
        """Best value should appear in value_stats."""
        analysis = analyze_sensitivity(sample_results)
        for s in analysis["sharpe"]:
            vals_in_stats = {v["value"] for v in s["value_stats"]}
            assert s["best_value"] in vals_in_stats

    def test_value_stats_count(self, sample_results):
        """value_stats should sum to total combos."""
        analysis = analyze_sensitivity(sample_results)
        for s in analysis["sharpe"]:
            total = sum(v["count"] for v in s["value_stats"])
            assert total == len(sample_results)

    def test_recommendations_contains_how_to_read(self, sample_results):
        """Report should include 'How to read' section."""
        analysis = analyze_sensitivity(sample_results)
        assert "How to read" in analysis["recommendations"]
        assert "vs Sharpe" in analysis["recommendations"]
        assert "vs Composite" in analysis["recommendations"]

    def test_recommendations_contains_best(self, sample_results):
        """Report should include BEST COMBO FOUND."""
        analysis = analyze_sensitivity(sample_results)
        assert "BEST COMBO FOUND" in analysis["recommendations"]

    def test_empty_results(self):
        """Empty results should not crash."""
        analysis = analyze_sensitivity([])
        assert "No results" in analysis["recommendations"]

    def test_slope_present(self, sample_results):
        """Each param summary should have slope field."""
        analysis = analyze_sensitivity(sample_results)
        for s in analysis["sharpe"]:
            assert "slope" in s

    def test_save_report(self, sample_results, tmp_path):
        """save_sensitivity_report should create a file."""
        analysis = analyze_sensitivity(sample_results)
        path = save_sensitivity_report(analysis, "test", output_dir=str(tmp_path))
        assert path.exists()
        content = open(path).read()
        assert "BEST COMBO FOUND" in content

    def test_calmar_analysis_present(self, sample_results):
        """Calmar analysis should be in return dict."""
        analysis = analyze_sensitivity(sample_results)
        assert len(analysis["calmar"]) > 0
        for s in analysis["calmar"]:
            assert "param" in s
            assert "best_value" in s

    def test_top_values_output(self, sample_results):
        """Report should show top 3 values per param."""
        analysis = analyze_sensitivity(sample_results)
        rec = analysis["recommendations"]
        assert "Top values" in rec
