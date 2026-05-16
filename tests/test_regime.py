"""Tests for research/strategies/regime.py."""

import numpy as np
import pandas as pd
import pytest

from research.strategies.regime import (
    detect_regime_sma,
    detect_regime_adx,
    filter_by_regime,
)


@pytest.fixture
def trend_data():
    """Data with clear bull/bear phases."""
    np.random.seed(42)
    idx = pd.date_range("2024-01-01", periods=1000, freq="5min")

    # 500 bars up, 500 bars down
    close = np.zeros(1000)
    close[:500] = 100 + np.linspace(0, 30, 500) + np.random.randn(500) * 1.0  # bull
    close[500:] = close[499] - np.linspace(0, 20, 500) + np.random.randn(500) * 1.0  # bear

    df = pd.DataFrame({
        "open": close - 0.5,
        "high": close + 1.0,
        "low": close - 1.0,
        "close": close,
        "volume": np.random.uniform(10, 100, 1000),
    }, index=idx)
    return df


class TestDetectRegimeSMA:

    def test_bull_detected(self, trend_data):
        """First half should be bull."""
        regime = detect_regime_sma(trend_data["close"], period=30, bull_threshold=1.02)
        bull_count = (regime.iloc[100:500] == "bull").sum()
        assert bull_count > 0, "No bull regime detected"

    def test_bear_detected(self, trend_data):
        """Second half should be bear."""
        regime = detect_regime_sma(trend_data["close"], period=30, bear_threshold=0.98)
        bear_count = (regime.iloc[600:] == "bear").sum()
        assert bear_count > 0, "No bear regime detected"

    def test_returns_sideways_for_flat(self):
        """Flat price should be sideways."""
        idx = pd.date_range("2024-01-01", periods=500, freq="5min")
        close = pd.Series(100 + np.random.randn(500) * 0.5, index=idx)
        regime = detect_regime_sma(close, period=50)
        # Most should be sideways (no strong trend)
        sideways_pct = (regime == "sideways").mean()
        assert sideways_pct > 0.5


class TestDetectRegimeADX:

    def test_adx_bull_detected(self, trend_data):
        """ADX should detect bull trend."""
        regime = detect_regime_adx(
            trend_data["high"], trend_data["low"], trend_data["close"],
            period=10, trending_threshold=18,
        )
        bull_count = (regime.iloc[250:500] == "bull").sum()
        # ADX needs warmup — check middle of bull phase
        assert bull_count > 0

    def test_adx_output_type(self, trend_data):
        """Output should be Series of strings."""
        regime = detect_regime_adx(
            trend_data["high"], trend_data["low"], trend_data["close"],
        )
        assert isinstance(regime, pd.Series)
        assert set(regime.dropna().unique()).issubset({"bull", "bear", "sideways"})


class TestFilterByRegime:

    @pytest.fixture
    def signals(self):
        idx = pd.date_range("2024-01-01", periods=10, freq="5min")
        return pd.Series([1, 1, 0, -1, -1, 0, 1, -1, 1, 0], index=idx)

    def test_default_filter(self, signals):
        """Default filter should zero longs in bear, shorts in bull."""
        idx = signals.index
        regime = pd.Series(["bull"] * 5 + ["bear"] * 5, index=idx)
        filtered = filter_by_regime(signals, regime)
        # First 5 (bull): longs stay, shorts zeroed
        assert filtered.iloc[0] == 1    # long in bull
        assert filtered.iloc[3] == 0    # short in bull → zeroed
        # Last 5 (bear): shorts stay, longs zeroed
        assert filtered.iloc[7] == -1   # short in bear
        assert filtered.iloc[6] == 0    # long in bear → zeroed

    def test_custom_allowed(self, signals):
        """Custom allowed regimes should work."""
        idx = signals.index
        regime = pd.Series(["bull"] * 5 + ["bear"] * 5, index=idx)
        filtered = filter_by_regime(signals, regime, allowed_regimes=["bull"])
        assert filtered.iloc[0] == 1    # long in bull → allowed
        assert filtered.iloc[7] == 0    # short in bear → not in ["bull"]
