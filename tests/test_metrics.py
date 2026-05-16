"""Tests for research/backtest/metrics.py — edge cases and format."""
import numpy as np
import pandas as pd
import pytest
from research.backtest.metrics import compute_metrics, format_metrics_report


@pytest.fixture
def dix():
    return pd.date_range("2024-01-01", periods=200, freq="5min")


class TestComputeMetrics:

    def test_single_value(self, dix):
        eq = pd.Series([10000], index=dix[:1])
        m = compute_metrics(eq)
        assert m["final_equity"] == 10000

    def test_two_values(self, dix):
        eq = pd.Series([10000, 11000], index=dix[:2])
        m = compute_metrics(eq)
        assert abs(m["total_return_pct"] - 10.0) < 0.5

    def test_exact_drawdown(self):
        idx = pd.date_range("2024-01-01", periods=5, freq="D")
        eq = pd.Series([10000, 12000, 8000, 9000, 7000], index=idx)
        m = compute_metrics(eq)
        assert m["max_drawdown_pct"] == pytest.approx(-41.67, abs=0.5)

    def test_no_drawdown_on_uptrend(self, dix):
        eq = pd.Series(10000 * (1 + np.arange(200) / 10000), index=dix)
        m = compute_metrics(eq)
        assert m["max_drawdown_pct"] >= -1.0

    def test_sharpe_positive(self, dix):
        idx = pd.date_range("2024-01-01", periods=100, freq="5min")
        eq = pd.Series(10000 * (1 + np.arange(100) / 10000), index=idx)
        m = compute_metrics(eq)
        assert m["sharpe_ratio"] > 0

    def test_sharpe_negative(self, dix):
        idx = pd.date_range("2024-01-01", periods=100, freq="5min")
        eq = pd.Series(10000 * (1 - np.arange(100) / 10000), index=idx)
        m = compute_metrics(eq)
        assert m["sharpe_ratio"] < 0

    def test_cagr_exists(self, dix):
        eq = pd.Series([10000, 11000], index=[dix[0], dix[-1]])
        m = compute_metrics(eq)
        assert "cagr_pct" in m

    def test_sortino_exists(self, dix):
        eq = pd.Series(10000 * (1 + np.arange(200) / 10000), index=dix)
        m = compute_metrics(eq)
        assert "sortino_ratio" in m

    def test_trade_metrics_without_trades(self, dix):
        eq = pd.Series([10000, 10500], index=dix[:2])
        m = compute_metrics(eq)
        assert m["total_trades"] == 0

    def test_win_rate_key(self, dix):
        m = compute_metrics(pd.Series([10000, 10500], index=dix[:2]))
        assert "win_rate_pct" in m

    def test_calmar_key(self, dix):
        m = compute_metrics(pd.Series([10000, 11000], index=dix[:2]))
        assert "calmar_ratio" in m

    def test_sharpe_is_float(self, dix):
        m = compute_metrics(pd.Series([10000, 10500], index=dix[:2]))
        assert isinstance(m["sharpe_ratio"], float)

    def test_profit_factor_key(self, dix):
        m = compute_metrics(pd.Series([10000, 10500], index=dix[:2]))
        assert "profit_factor" in m

    def test_avg_rr_key(self, dix):
        m = compute_metrics(pd.Series([10000, 10500], index=dix[:2]))
        assert "avg_rr" in m

    def test_expectancy_key(self, dix):
        m = compute_metrics(pd.Series([10000, 10500], index=dix[:2]))
        assert "expectancy" in m

    def test_return_pct_key(self, dix):
        m = compute_metrics(pd.Series([10000, 10500], index=dix[:2]))
        assert "total_return_pct" in m

    def test_final_equity_key(self, dix):
        m = compute_metrics(pd.Series([10000, 10500], index=dix[:2]))
        assert "final_equity" in m


class TestFormatMetricsReport:

    def test_report_string(self, dix):
        eq = pd.Series([10000, 10500], index=dix[:2])
        m = compute_metrics(eq)
        report = format_metrics_report(m)
        assert isinstance(report, str)
        assert len(report) > 50

    def test_report_has_sections(self, dix):
        eq = pd.Series([10000, 10500], index=dix[:2])
        m = compute_metrics(eq)
        report = format_metrics_report(m)
        assert "Initial" in report or "Sharpe" in report

    def test_report_with_trades(self, dix):
        eq = pd.Series([10000, 10500, 10200], index=dix[:3])
        m = compute_metrics(eq)
        report = format_metrics_report(m)
        assert len(report) > 50

    def test_report_for_loss(self, dix):
        eq = pd.Series([10000, 8000], index=dix[:2])
        m = compute_metrics(eq)
        report = format_metrics_report(m)
        assert isinstance(report, str)

    def test_report_length(self, dix):
        eq = pd.Series([10000, 20000, 15000, 25000], index=dix[:4])
        m = compute_metrics(eq)
        report = format_metrics_report(m)
        assert len(report) > 100
