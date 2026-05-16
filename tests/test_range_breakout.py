"""Tests for research/strategies/range_breakout.py."""
import numpy as np
import pandas as pd
import pytest
from research.strategies.range_breakout import RangeBreakout


@pytest.fixture
def rng_data():
    np.random.seed(42)
    idx = pd.date_range("2024-01-01", periods=1000, freq="5min")
    close = 100 + np.cumsum(np.random.randn(1000) * 0.2)
    close[200:250] += np.cumsum(np.abs(np.random.randn(50)) * 0.5)
    close[500:550] -= np.cumsum(np.abs(np.random.randn(50)) * 0.5)
    return pd.DataFrame({
        "open": close - 0.2, "high": close + np.abs(np.random.randn(1000)) * 0.5,
        "low": close - np.abs(np.random.randn(1000)) * 0.5, "close": close,
        "volume": np.random.uniform(10, 100, 1000),
    }, index=idx)


class TestRangeBreakout:
    def test_default_config(self):
        s = RangeBreakout()
        assert s.config["range_period"] == 48
        assert s.config["exit_period"] == 20

    def test_name(self):
        assert "Range" in RangeBreakout().name

    def test_signals_generated(self, rng_data):
        s = RangeBreakout()
        sig = s.generate_signals(rng_data)
        assert set(sig.unique()).issubset({-1, 0, 1})

    def test_no_nan(self, rng_data):
        s = RangeBreakout()
        assert not s.generate_signals(rng_data).isna().any()

    def test_volume_filter(self, rng_data):
        unfiltered = RangeBreakout().generate_signals(rng_data)
        filtered = RangeBreakout({"filter_volume": True, "filter_vol_mult": 2.0}).generate_signals(rng_data)
        non_zero_unf = (unfiltered != 0).sum()
        non_zero_fil = (filtered != 0).sum()
        assert non_zero_fil <= non_zero_unf

    def test_trading_activity(self, rng_data):
        """Should produce several trades (1-2/day = ~12-24 on 1000 5m bars)."""
        s = RangeBreakout({"range_period": 12, "exit_period": 6})
        sig = s.generate_signals(rng_data)
        non_zero = (sig != 0).sum()
        assert non_zero > 10

    def test_shorter_range_more_trades(self, rng_data):
        short = RangeBreakout({"range_period": 12}).generate_signals(rng_data)
        long = RangeBreakout({"range_period": 96}).generate_signals(rng_data)
        assert (short != 0).sum() >= (long != 0).sum()

    def test_description(self):
        s = RangeBreakout({"range_period": 48})
        assert "48" in s.description

    def test_various_periods(self, rng_data):
        for rp in [12, 24, 48, 96]:
            s = RangeBreakout({"range_period": rp})
            assert not s.generate_signals(rng_data).isna().any()

    def test_volume_filter_extreme(self, rng_data):
        s = RangeBreakout({"filter_volume": True, "filter_vol_mult": 5.0})
        sig = s.generate_signals(rng_data)
        assert set(sig.unique()).issubset({-1, 0, 1})
