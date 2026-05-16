"""Tests for research/strategies/ma_crossover.py."""

import numpy as np
import pandas as pd
import pytest

from research.strategies.ma_crossover import MACrossover


@pytest.fixture
def trend_data():
    """Uptrend data (price goes up)."""
    np.random.seed(42)
    idx = pd.date_range("2024-01-01", periods=500, freq="5min")
    close = 100 + np.cumsum(np.random.randn(500) * 0.3 + 0.05)  # slight uptrend
    df = pd.DataFrame({
        "open": close - 0.2,
        "high": close + 0.5,
        "low": close - 0.5,
        "close": close,
        "volume": np.random.uniform(10, 100, 500),
    }, index=idx)
    return df


class TestMACrossover:

    def test_default_config(self):
        s = MACrossover()
        assert s.config["fast_period"] == 12
        assert s.config["slow_period"] == 26
        assert s.config["ma_type"] == "ema"

    def test_name(self):
        assert "MA Crossover" in MACrossover().name

    def test_long_signals_in_uptrend(self, trend_data):
        """In an uptrend, should produce more long signals than short."""
        s = MACrossover()
        signals = s.generate_signals(trend_data)
        longs = (signals == 1).sum()
        shorts = (signals == -1).sum()
        # After warm-up, long should dominate in uptrend
        warmup = signals.iloc[50:]
        assert (warmup == 1).sum() > (warmup == -1).sum()

    def test_no_nan_signals(self, trend_data):
        """Signals should not contain NaN."""
        s = MACrossover()
        signals = s.generate_signals(trend_data)
        assert not signals.isna().any()

    def test_volume_filter(self, trend_data):
        """With volume filter, some signals should be zeroed."""
        s = MACrossover({"filter_volume_pct": 0.5})
        signals_raw = MACrossover().generate_signals(trend_data)
        signals_filtered = s.generate_signals(trend_data)
        # Filtered should have more zeros
        assert (signals_filtered == 0).sum() >= (signals_raw == 0).sum()

    def test_ma_type_sma(self, trend_data):
        """SMA mode should produce signals without error."""
        s = MACrossover({"ma_type": "sma"})
        signals = s.generate_signals(trend_data)
        assert not signals.isna().any()
        assert set(signals.unique()).issubset({-1, 0, 1})
