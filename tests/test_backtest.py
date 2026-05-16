"""Tests for research/backtest/engine.py and metrics.py."""

import numpy as np
import pandas as pd
import pytest

from research.backtest.engine import VectorizedBacktest, Trade
from research.backtest.metrics import compute_metrics, format_metrics_report


@pytest.fixture
def sample_data():
    """Simple OHLCV dataset for testing (random walk)."""
    np.random.seed(42)
    periods = 1000
    idx = pd.date_range("2024-01-01", periods=periods, freq="5min")
    close = 50000 + np.cumsum(np.random.randn(periods) * 10)
    df = pd.DataFrame({
        "open": close - 5,
        "high": close + 10,
        "low": close - 10,
        "close": close,
        "volume": np.random.uniform(10, 100, periods),
    }, index=idx)
    return df


@pytest.fixture
def bt(sample_data):
    """Default backtest engine instance."""
    return VectorizedBacktest(sample_data)


class TestVectorizedBacktest:

    def test_init_validates_columns(self, sample_data):
        """Should raise on missing columns."""
        bad = sample_data.drop(columns=["volume"])
        with pytest.raises(ValueError, match="Missing columns"):
            VectorizedBacktest(bad)

    def test_run_flat_equity(self, bt):
        """No signals = flat equity = initial capital."""
        signals = pd.Series(0, index=bt.data.index)
        result = bt.run(signals)
        assert result.equity_curve.iloc[-1] == pytest.approx(10_000.0, abs=0.01)
        assert len(result.trades) == 0

    def test_run_constant_long(self, bt):
        """Constant long signal = equity moves with price."""
        signals = pd.Series(1, index=bt.data.index)
        result = bt.run(signals)
        # Equity should follow close price (roughly)
        final_close_ret = bt.data["close"].iloc[-1] / bt.data["close"].iloc[0]
        expected_equity = 10_000 * final_close_ret
        assert result.equity_curve.iloc[-1] == pytest.approx(expected_equity, rel=0.1)
        # No closed trades (always in position)
        assert len(result.trades) == 0

    def test_run_single_trade(self, bt):
        """Long entry, then exit should produce exactly 1 trade."""
        signals = pd.Series(0, index=bt.data.index)
        signals.iloc[10:50] = 1  # Signal active at indices 10-49
        result = bt.run(signals)
        # Position shifts by 1: entry at index 11, exit at index 51
        assert len(result.trades) == 1
        t = result.trades[0]
        assert t.side == 1
        assert t.entry_time == bt.data.index[11]
        assert t.exit_time == bt.data.index[51]

    def test_run_alternating(self, bt):
        """Alternating long/short signals should produce trades with correct sides."""
        n = len(bt.data)
        arr = np.zeros(n)
        arr[::100] = 1          # Long signal at 0, 100, 200, ...
        arr[50::100] = -1       # Short signal at 50, 150, 250, ...
        signals = pd.Series(arr, index=bt.data.index)
        result = bt.run(signals)
        # Should have multiple closed trades
        assert len(result.trades) >= 10
        # First trade should be long, second short, third long, etc.
        assert result.trades[0].side == 1
        assert result.trades[1].side == -1
        assert result.trades[2].side == 1

    def test_custom_config(self, sample_data):
        """Custom config should be respected."""
        bt = VectorizedBacktest(sample_data, {"initial_capital": 100_000})
        signals = pd.Series(1, index=sample_data.index)
        result = bt.run(signals)
        assert result.config["initial_capital"] == 100_000

    def test_signal_shift_no_lookahead(self, bt):
        """Position should lag signal by 1 period (no look-ahead bias)."""
        signals = pd.Series(0, index=bt.data.index)
        signals.iloc[5] = 1  # Signal at index 5
        result = bt.run(signals)
        # Position should activate at index 6 (shifted)
        first_pos_idx = result.positions[result.positions != 0].index[0]
        assert first_pos_idx == signals.index[6]


class TestMetrics:

    def test_flat_equity(self):
        """Flat equity should produce zeros."""
        equity = pd.Series(10_000, index=pd.date_range("2024-01-01", periods=100, freq="5min"))
        m = compute_metrics(equity)
        assert m["total_return_pct"] == pytest.approx(0, abs=0.01)
        assert m["max_drawdown_pct"] == pytest.approx(0, abs=0.01)
        assert m["total_trades"] == 0

    def test_linear_growth(self):
        """Linear growth should produce positive metrics."""
        idx = pd.date_range("2024-01-01", periods=1000, freq="5min")
        equity = pd.Series(10_000 * (1 + np.linspace(0, 0.5, 1000)), index=idx)
        m = compute_metrics(equity)
        assert m["total_return_pct"] > 0
        assert m["sharpe_ratio"] > 0
        assert m["cagr_pct"] > 0

    def test_sharpe_on_flat_equity(self):
        """Flat equity with 0 risk-free rate should have Sharpe ~0."""
        idx = pd.date_range("2024-01-01", periods=100, freq="1D")  # daily
        equity = pd.Series(10_000, index=idx)
        m = compute_metrics(equity, risk_free_rate=0, periods_per_year=365)
        assert abs(m["sharpe_ratio"]) < 0.1

    def test_metrics_with_trades(self):
        """Metrics with trade data should include trade-level stats."""
        idx = pd.date_range("2024-01-01", periods=100, freq="5min")
        equity = pd.Series(10_000 * (1 + np.linspace(0, 0.2, 100)), index=idx)

        trades = [
            Trade(idx[10], idx[20], 1, 50000, 51000, 1.0, 100, 1.0, 2.0, 0.15, "10min"),
            Trade(idx[30], idx[40], 1, 51000, 50500, 1.0, -50, -0.5, -1.0, 0.15, "10min"),
            Trade(idx[50], idx[60], 1, 50500, 52000, 1.0, 150, 1.5, 3.0, 0.15, "10min"),
        ]
        m = compute_metrics(equity, trades)
        assert m["total_trades"] == 3
        assert m["win_rate_pct"] == pytest.approx(66.67, abs=0.1)
        assert m["profit_factor"] >= 1.0

    def test_format_report(self):
        """Report string should contain key metric labels."""
        idx = pd.date_range("2024-01-01", periods=100, freq="5min")
        equity = pd.Series(10_000 * (1 + np.linspace(0, 0.1, 100)), index=idx)
        m = compute_metrics(equity)
        report = format_metrics_report(m)
        assert "Sharpe ratio" in report
        assert "Max drawdown" in report
        assert "Total trades" in report
