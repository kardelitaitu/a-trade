"""Tests for research/backtest/engine.py and metrics.py."""

import numpy as np
import pandas as pd
import pytest

from research.backtest.engine import VectorizedBacktest, Trade
from research.backtest.metrics import compute_metrics, format_metrics_report
from research.backtest._numba_ops import extract_trades_numba


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

    # ------------------------------------------------------------------
    # Continuous / Fractional position sizing
    # ------------------------------------------------------------------

    def test_continuous_scale_up_single_trade(self, bt):
        """Continuous signal scaling up then to zero should produce 1 trade."""
        signals = pd.Series(0, index=bt.data.index, dtype=float)
        signals.iloc[10:30] = np.linspace(0.3, 1.0, 20)  # ramps up
        # Drop back to 0
        signals.iloc[30:40] = 0
        result = bt.run(signals)
        # One continuous long position: all positive values are same direction
        assert len(result.trades) == 1
        assert result.trades[0].side == 1

    def test_continuous_scale_down_to_zero(self, bt):
        """Continuous signal decreasing from 1.0 to 0 should produce exactly 1 trade."""
        signals = pd.Series(0, index=bt.data.index, dtype=float)
        signals.iloc[10:40] = 1.0
        signals.iloc[40:50] = np.linspace(1.0, 0.0, 10)  # ramps down
        signals.iloc[50:60] = 0.0
        result = bt.run(signals)
        # Still one position: all entries stay positive until hitting 0
        assert len(result.trades) == 1

    def test_continuous_alternating_fractional(self, bt):
        """Alternating fractional long/short should produce trades with correct sides."""
        n = len(bt.data)
        arr = np.zeros(n)
        arr[10:40] = 0.6    # fractional long
        arr[50:80] = -0.4   # fractional short
        arr[90:120] = 0.8   # fractional long
        signals = pd.Series(arr, index=bt.data.index)
        result = bt.run(signals)
        assert len(result.trades) >= 2
        assert result.trades[0].side == 1
        assert result.trades[1].side == -1
        if len(result.trades) > 2:
            assert result.trades[2].side == 1

    def test_continuous_subtle_signal_entry(self, bt):
        """Tiny positive signal from flat should still open a trade."""
        signals = pd.Series(0, index=bt.data.index, dtype=float)
        signals.iloc[10:15] = 0.001   # barely positive
        signals.iloc[15:20] = 0.0
        result = bt.run(signals)
        # Any positive value, no matter how small, is an open position
        assert len(result.trades) == 1
        assert result.trades[0].side == 1

    def test_continuous_noise_within_same_sign(self, bt):
        """Fluctuating within positive territory should not create extra trades."""
        signals = pd.Series(0, index=bt.data.index, dtype=float)
        # Fluctuating positive values — no direction change
        signals.iloc[10:50] = [0.3, 0.5, 0.7, 0.4, 0.6][::-1] * 8
        signals.iloc[50:60] = 0.0
        result = bt.run(signals)
        # All positive → same direction → single trade
        assert len(result.trades) == 1


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


class TestExtractTradesNumba:
    """Direct unit tests for extract_trades_numba with continuous/fractional positions."""

    def test_fractional_long_open_close(self):
        """0.0 → 0.5 → 0.0: entry at index 1, exit at index 2, side=1."""
        pos = np.array([0.0, 0.5, 0.0], dtype=np.float64)
        entries, exits, sides = extract_trades_numba(pos)
        assert len(entries) == 1
        assert entries[0] == 1
        assert exits[0] == 2
        assert sides[0] == 1

    def test_fractional_short_open_close(self):
        """0.0 → -0.3 → 0.0: entry at index 1, exit at index 2, side=-1."""
        pos = np.array([0.0, -0.3, 0.0], dtype=np.float64)
        entries, exits, sides = extract_trades_numba(pos)
        assert len(entries) == 1
        assert entries[0] == 1
        assert exits[0] == 2
        assert sides[0] == -1

    def test_continuous_scaling_no_extra_trades(self):
        """0.0 → 0.3 → 0.6 → 0.9 → 0.0: 1 trade (all same sign)."""
        pos = np.array([0.0, 0.3, 0.6, 0.9, 0.0], dtype=np.float64)
        entries, exits, sides = extract_trades_numba(pos)
        assert len(entries) == 1
        assert sides[0] == 1

    def test_fractional_flip_long_to_short(self):
        """0.0 → 0.5 → 0.0 → -0.3 → 0.0: 2 trades (long then short)."""
        pos = np.array([0.0, 0.5, 0.0, -0.3, 0.0], dtype=np.float64)
        entries, exits, sides = extract_trades_numba(pos)
        assert len(entries) == 2
        assert sides[0] == 1
        assert sides[1] == -1
        # First trade: entry at 1, exit at 2
        assert entries[0] == 1
        assert exits[0] == 2
        # Second trade: entry at 3, exit at 4
        assert entries[1] == 3
        assert exits[1] == 4

    def test_fractional_flip_short_to_long(self):
        """0.0 → -0.3 → 0.0 → 0.5 → 0.0: 2 trades (short then long)."""
        pos = np.array([0.0, -0.3, 0.0, 0.5, 0.0], dtype=np.float64)
        entries, exits, sides = extract_trades_numba(pos)
        assert len(entries) == 2
        assert sides[0] == -1
        assert sides[1] == 1

    def test_subtle_positive_is_still_a_trade(self):
        """0.0 → 0.001 → 0.0: 1 trade (any positive is a direction)."""
        pos = np.array([0.0, 0.001, 0.0], dtype=np.float64)
        entries, exits, sides = extract_trades_numba(pos)
        assert len(entries) == 1
        assert sides[0] == 1

    def test_subtle_negative_is_still_short(self):
        """0.0 → -0.001 → 0.0: 1 short trade."""
        pos = np.array([0.0, -0.001, 0.0], dtype=np.float64)
        entries, exits, sides = extract_trades_numba(pos)
        assert len(entries) == 1
        assert sides[0] == -1

    def test_fractional_to_full_and_back(self):
        """0.0 → 0.3 → 1.0 → 0.0: 1 trade (same direction throughout)."""
        pos = np.array([0.0, 0.3, 1.0, 0.0], dtype=np.float64)
        entries, exits, sides = extract_trades_numba(pos)
        assert len(entries) == 1
        assert sides[0] == 1

    def test_no_noise_on_same_sign_fluctuations(self):
        """0.0 → 0.5 → 0.8 → 0.3 → 0.0: 1 trade, not 3."""
        pos = np.array([0.0, 0.5, 0.8, 0.3, 0.0], dtype=np.float64)
        entries, exits, sides = extract_trades_numba(pos)
        assert len(entries) == 1
        assert sides[0] == 1

    def test_direct_flip_no_zero(self):
        """0.0 → 0.5 → -0.4 → 0.0: flip (no intermediate zero), 2 trades."""
        pos = np.array([0.0, 0.5, -0.4, 0.0], dtype=np.float64)
        entries, exits, sides = extract_trades_numba(pos)
        assert len(entries) == 2
        assert sides[0] == 1
        assert sides[1] == -1
        # First trade closes at flip index (2), second opens at same index
        assert exits[0] == 2
        assert entries[1] == 2

    def test_flat_no_trades(self):
        """All zeros: no trades."""
        pos = np.zeros(10, dtype=np.float64)
        entries, exits, sides = extract_trades_numba(pos)
        assert len(entries) == 0
        assert len(exits) == 0
        assert len(sides) == 0

    def test_continuous_never_exits(self):
        """0.0 → 0.5 → 0.8: still in position at end, no closed trade."""
        pos = np.array([0.0, 0.5, 0.8], dtype=np.float64)
        entries, exits, sides = extract_trades_numba(pos)
        # No closed trade because position never returns to 0
        assert len(entries) == 0
        assert len(exits) == 0
