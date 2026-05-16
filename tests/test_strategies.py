"""Tests for research/strategies/base.py."""

import pandas as pd
import pytest

from research.strategies import BaseStrategy


class DummyStrategy(BaseStrategy):
    """Minimal concrete strategy for testing the base class."""

    DEFAULT_CONFIG = {"period": 20, "threshold": 0.5}

    @property
    def name(self) -> str:
        return "Dummy"

    @property
    def description(self) -> str:
        return "A dummy strategy for testing."

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        return pd.Series(0, index=data.index)


class TestBaseStrategy:

    def test_cannot_instantiate_abstract(self):
        """BaseStrategy should not be instantiable directly."""
        with pytest.raises(TypeError):
            BaseStrategy()  # type: ignore

    def test_concrete_instantiation(self):
        """Concrete strategy should instantiate with defaults."""
        s = DummyStrategy()
        assert s.name == "Dummy"
        assert s.config["period"] == 20
        assert s.config["threshold"] == 0.5

    def test_custom_config_overrides_defaults(self):
        """Custom config should merge with defaults."""
        s = DummyStrategy({"period": 50})
        assert s.config["period"] == 50
        assert s.config["threshold"] == 0.5  # unchanged

    def test_set_params(self):
        """set_params should update config."""
        s = DummyStrategy()
        s.set_params(period=100, threshold=0.9)
        assert s.config["period"] == 100
        assert s.config["threshold"] == 0.9

    def test_set_params_unknown_raises(self):
        """set_params with unknown key should raise KeyError."""
        s = DummyStrategy()
        with pytest.raises(KeyError):
            s.set_params(nonexistent=42)

    def test_generate_signals_output(self):
        """Signals should be a Series with same index as input."""
        idx = pd.date_range("2024-01-01", periods=10, freq="5min")
        data = pd.DataFrame({
            "open": [100]*10, "high": [101]*10,
            "low": [99]*10, "close": [100]*10, "volume": [50]*10,
        }, index=idx)
        s = DummyStrategy()
        signals = s.generate_signals(data)
        assert isinstance(signals, pd.Series)
        assert len(signals) == len(data)
        assert list(signals.index) == list(data.index)

    def test_repr(self):
        """__repr__ should show strategy name and config."""
        s = DummyStrategy({"period": 30})
        r = repr(s)
        assert "Dummy" in r
        assert "period=30" in r
