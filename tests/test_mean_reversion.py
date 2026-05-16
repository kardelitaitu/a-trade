"""Tests for research/strategies/mean_reversion.py."""

import numpy as np
import pandas as pd
import pytest

from research.strategies.mean_reversion import MeanReversion


@pytest.fixture
def meanrev_data():
    """Data with clear mean reversion patterns — spikes that snap back."""
    np.random.seed(42)
    idx = pd.date_range("2024-01-01", periods=500, freq="5min")

    close = np.zeros(500)
    # Base random walk
    close[:] = 100 + np.cumsum(np.random.randn(500) * 0.2)
    # Inject sharp spikes
    close[100] = close[99] + 3.0   # spike up
    close[101:110] = close[100] - np.linspace(3, 0.5, 9)  # snap back
    close[300] = close[299] - 3.0  # spike down
    close[301:310] = close[300] + np.linspace(3, 0.5, 9)  # snap back

    df = pd.DataFrame({
        "open": close - 0.2,
        "high": close + 0.5,
        "low": close - 0.5,
        "close": close,
        "volume": np.random.uniform(10, 100, 500),
    }, index=idx)
    return df


class TestMeanReversionRSI:

    def test_default_config(self):
        s = MeanReversion()
        assert s.config["mode"] == "rsi"

    def test_name(self):
        assert "RSI" in MeanReversion().name

    def test_rsi_signals_generated(self, meanrev_data):
        """RSI mode should produce signals."""
        s = MeanReversion()
        signals = s.generate_signals(meanrev_data)
        assert not signals.isna().any()
        assert set(signals.unique()).issubset({-1, 0, 1})

    def test_rsi_short_on_overbought(self):
        """When price spikes up dramatically, RSI should go above 70 → short."""
        idx = pd.date_range("2024-01-01", periods=100, freq="5min")
        close = pd.Series(np.linspace(100, 105, 80), index=idx[:80])  # slow uptrend
        spike = pd.Series([108, 107, 106, 105, 104, 103, 102, 101, 100, 99,
                           98, 97, 96, 95, 94, 93, 92, 91, 90, 89],
                          index=idx[80:])
        close = pd.concat([close, spike])
        data = pd.DataFrame({
            "open": close - 0.2, "high": close + 0.5,
            "low": close - 0.5, "close": close, "volume": 50,
        })
        s = MeanReversion()
        signals = s.generate_signals(data)
        # After the spike, should have some short signals
        post_spike = signals.iloc[85:]
        assert (post_spike == -1).sum() > 0, "No short signals after overbought spike"


class TestMeanReversionBollinger:

    def test_bollinger_config(self):
        s = MeanReversion({"mode": "bollinger"})
        assert s.config["mode"] == "bollinger"

    def test_bollinger_signals_generated(self, meanrev_data):
        """Bollinger mode should produce signals."""
        s = MeanReversion({"mode": "bollinger"})
        signals = s.generate_signals(meanrev_data)
        assert not signals.isna().any()
        assert set(signals.unique()).issubset({-1, 0, 1})
