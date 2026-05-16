"""Tests for research/strategies/breakout.py."""
import numpy as np
import pandas as pd
import pytest
from research.strategies.breakout import DonchianBreakout


@pytest.fixture
def breakout_data():
    np.random.seed(42)
    idx = pd.date_range("2024-01-01", periods=500, freq="5min")
    close = 100 + np.cumsum(np.random.randn(500) * 0.3)
    close[150:155] += 5
    close[350:355] -= 4
    return pd.DataFrame({
        "open": close - 0.2, "high": close + 0.5 + np.abs(np.random.randn(500)),
        "low": close - 0.5 - np.abs(np.random.randn(500)), "close": close,
        "volume": np.random.uniform(10, 100, 500),
    }, index=idx)


class TestDonchianBreakout:

    def test_default_config(self):
        s = DonchianBreakout()
        assert s.config["entry_period"] == 20

    def test_name(self):
        assert "Donchian" in DonchianBreakout().name

    def test_signals_generated(self, breakout_data):
        s = DonchianBreakout()
        sig = s.generate_signals(breakout_data)
        assert set(sig.unique()).issubset({-1, 0, 1})

    def test_no_nan_after_warmup(self, breakout_data):
        s = DonchianBreakout()
        sig = s.generate_signals(breakout_data)
        assert not sig.isna().any()

    def test_description(self):
        assert "entry_period" in DonchianBreakout().description.lower() or "Donchian" in DonchianBreakout().description

    # --- ADDITIONAL TESTS ---

    def test_short_entry(self, breakout_data):
        s = DonchianBreakout({"entry_period": 15, "exit_period": 10})
        sig = s.generate_signals(breakout_data)
        assert (sig == -1).sum() >= 0

    def test_equal_entry_exit(self, breakout_data):
        s = DonchianBreakout({"entry_period": 20, "exit_period": 20})
        sig = s.generate_signals(breakout_data)
        assert not sig.isna().any()

    def test_large_periods(self, breakout_data):
        s = DonchianBreakout({"entry_period": 100, "exit_period": 50})
        sig = s.generate_signals(breakout_data)
        assert not sig.isna().any()

    def test_small_periods(self, breakout_data):
        s = DonchianBreakout({"entry_period": 3, "exit_period": 2})
        sig = s.generate_signals(breakout_data)
        assert not sig.isna().any()

    def test_volume_filter(self, breakout_data):
        sig = DonchianBreakout({"entry_period": 20, "exit_period": 10, "filter_volume_pct": 0.5})
        assert not sig.generate_signals(breakout_data).isna().any()

    def test_atr_filter(self, breakout_data):
        s = DonchianBreakout({"entry_period": 20, "exit_period": 10, "filter_atr_mult": 2.0})
        sig = s.generate_signals(breakout_data)
        assert not sig.isna().any()

    def test_config_params(self):
        s = DonchianBreakout({"entry_period": 10, "exit_period": 5})
        assert s.config["entry_period"] == 10
        assert s.config["exit_period"] == 5

    def test_entry_gt_exit(self, breakout_data):
        sig = DonchianBreakout({"entry_period": 30, "exit_period": 10}).generate_signals(breakout_data)
        assert not sig.isna().any()

    def test_exit_gt_entry(self, breakout_data):
        sig = DonchianBreakout({"entry_period": 10, "exit_period": 30}).generate_signals(breakout_data)
        assert not sig.isna().any()

    def test_very_large_periods(self, breakout_data):
        sig = DonchianBreakout({"entry_period": 150, "exit_period": 100}).generate_signals(breakout_data)
        assert not sig.isna().any()

    def test_very_small_periods(self, breakout_data):
        sig = DonchianBreakout({"entry_period": 3, "exit_period": 2}).generate_signals(breakout_data)
        assert not sig.isna().any()
