"""Tests for research/strategies/breakout.py."""

import numpy as np
import pandas as pd
import pytest

from research.strategies.breakout import DonchianBreakout


@pytest.fixture
def breakout_data():
    """Data with clear breakout patterns — range followed by spikes."""
    np.random.seed(42)
    idx = pd.date_range("2024-01-01", periods=500, freq="5min")

    close = np.zeros(500)
    # Phase 1: range (0-100)
    close[:100] = 100 + np.random.randn(100) * 0.5
    # Phase 2: spike up (100-110)
    close[100] = 103
    close[101:110] = np.linspace(103, 108, 9) + np.random.randn(9) * 0.2
    close[110:140] = 108 + np.random.randn(30) * 0.3
    # Phase 3: range (140-250)
    close[140:250] = 105 + np.random.randn(110) * 0.5
    # Phase 4: drop down (250-260)
    close[250] = 102
    close[251:260] = np.linspace(102, 95, 9) + np.random.randn(9) * 0.2
    close[260:290] = 95 + np.random.randn(30) * 0.3
    # Phase 5: range
    close[290:] = 100 + np.random.randn(210) * 0.5

    df = pd.DataFrame({
        "open": close - 0.3,
        "high": close + 0.5,
        "low": close - 0.5,
        "close": close,
        "volume": np.random.uniform(10, 100, 500),
    }, index=idx)
    return df


class TestDonchianBreakout:

    def test_default_config(self):
        s = DonchianBreakout()
        assert s.config["entry_period"] == 20
        assert s.config["exit_period"] == 10

    def test_name(self):
        assert "Donchian" in DonchianBreakout().name

    def test_long_on_spike_up(self, breakout_data):
        """A price spike above recent range should trigger long."""
        s = DonchianBreakout()
        signals = s.generate_signals(breakout_data)
        # After the spike up (around index 100-110), should have longs
        post_spike = signals.iloc[100:150]
        assert (post_spike == 1).sum() > 0, "No long signals after upward spike"

    def test_short_on_drop_down(self, breakout_data):
        """A price drop below recent range should trigger short."""
        s = DonchianBreakout()
        signals = s.generate_signals(breakout_data)
        # After the drop (around index 250-260), should have shorts
        post_drop = signals.iloc[250:300]
        assert (post_drop == -1).sum() > 0, "No short signals after downward drop"

    def test_no_nan_signals(self, breakout_data):
        """Signals should not contain NaN."""
        s = DonchianBreakout()
        signals = s.generate_signals(breakout_data)
        assert not signals.isna().any()

    def test_signals_in_range(self, breakout_data):
        """All signal values should be -1, 0, or 1."""
        s = DonchianBreakout()
        signals = s.generate_signals(breakout_data)
        assert set(signals.unique()).issubset({-1, 0, 1})

    def test_volume_filter(self, breakout_data):
        """Volume filter should increase zero signals."""
        s_filtered = DonchianBreakout({"filter_volume_pct": 0.5})
        s_raw = DonchianBreakout()
        signals_raw = s_raw.generate_signals(breakout_data)
        signals_filtered = s_filtered.generate_signals(breakout_data)
        assert (signals_filtered == 0).sum() >= (signals_raw == 0).sum()
