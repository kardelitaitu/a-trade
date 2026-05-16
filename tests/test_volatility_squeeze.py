"""Tests for research/strategies/volatility_squeeze.py."""

import numpy as np
import pandas as pd
import pytest

from research.strategies.volatility_squeeze import VolatilitySqueeze


@pytest.fixture
def squeeze_data():
    """Data with clear squeeze patterns: low vol → expansion."""
    np.random.seed(42)
    idx = pd.date_range("2024-01-01", periods=500, freq="5min")

    close = np.zeros(500)
    # Low vol range (100-150)
    close[:150] = 100 + np.random.randn(150) * 0.3
    # Volatility expansion upward (150-250)
    close[150:250] = 102 + np.cumsum(np.abs(np.random.randn(100)) * 0.5 + 0.05)
    # Another squeeze (250-350)
    close[250:350] = 105 + np.random.randn(100) * 0.3
    # Vol expansion downward (350-450)
    close[350:450] = 104 - np.cumsum(np.abs(np.random.randn(100)) * 0.5 + 0.05)
    # Back to range
    close[450:] = 98 + np.random.randn(50) * 0.3

    df = pd.DataFrame({
        "open": close - 0.2,
        "high": close + np.abs(np.random.randn(500)) * 1.0 + 0.3,
        "low": close - np.abs(np.random.randn(500)) * 1.0 - 0.3,
        "close": close,
        "volume": np.random.uniform(10, 100, 500),
    }, index=idx)
    return df


class TestVolatilitySqueeze:

    def test_default_config(self):
        s = VolatilitySqueeze()
        assert s.config["bb_period"] == 20
        assert s.config["keltner_atr_mult"] == 1.5

    def test_name(self):
        assert "Squeeze" in VolatilitySqueeze().name

    def test_signals_generated(self, squeeze_data):
        s = VolatilitySqueeze()
        signals = s.generate_signals(squeeze_data)
        assert not signals.isna().any()
        assert set(signals.unique()).issubset({-1, 0, 1})

    def test_no_nan(self, squeeze_data):
        s = VolatilitySqueeze()
        signals = s.generate_signals(squeeze_data)
        assert not signals.isna().any()
