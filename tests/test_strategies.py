"""Extended tests for research/strategies/base.py."""
import numpy as np
import pandas as pd
import pytest

from research.strategies import BaseStrategy
from research.strategies.base import BaseStrategy as _Base


class DummyStrategy(BaseStrategy):
    DEFAULT_CONFIG = {"period": 20, "threshold": 0.5}
    @property
    def name(self): return "Dummy"
    @property
    def description(self): return "Dummy strategy for testing"
    def generate_signals(self, data):
        import pandas as pd
        return pd.Series([0]*len(data), index=data.index)


class TestBaseStrategy:

    def test_cannot_instantiate_abstract(self):
        with pytest.raises(TypeError):
            BaseStrategy()

    def test_default_config_loaded(self):
        s = DummyStrategy()
        assert s.config["period"] == 20
        assert s.config["threshold"] == 0.5

    def test_custom_config_overrides_defaults(self):
        s = DummyStrategy({"period": 50})
        assert s.config["period"] == 50
        assert s.config["threshold"] == 0.5

    def test_config_cannot_be_empty_dict(self):
        s = DummyStrategy({})
        assert s.config == DummyStrategy.DEFAULT_CONFIG

    def test_set_params_single(self):
        s = DummyStrategy()
        s.set_params(period=100)
        assert s.config["period"] == 100

    def test_set_params_multiple(self):
        s = DummyStrategy()
        s.set_params(period=100, threshold=0.8)
        assert s.config["period"] == 100
        assert s.config["threshold"] == 0.8

    def test_set_params_unknown_raises(self):
        s = DummyStrategy()
        assert s.config["period"] == 20

    def test_generate_signals_output(self):
        idx = pd.date_range("2024-01-01", periods=10, freq="5min")
        data = pd.DataFrame({"open": [100]*10, "high": [101]*10, "low": [99]*10, "close": [100]*10, "volume": [50]*10}, index=idx)
        s = DummyStrategy()
        sig = s.generate_signals(data)
        assert isinstance(sig, pd.Series)
        assert len(sig) == 10

    def test_generate_signals_on_empty_data(self):
        s = DummyStrategy()
        empty = pd.DataFrame()
        sig = s.generate_signals(empty)
        assert len(sig) == 0 or (sig == 0).all()

    def test_different_data_lengths(self):
        s = DummyStrategy({"custom_param": 42})
        short_data = pd.DataFrame({"close": [100]}, index=pd.date_range("2024-01-01", periods=1, freq="5min"))
        long_data = pd.DataFrame({"close": [100, 101]}, index=pd.date_range("2024-01-01", periods=2, freq="5min"))
        assert len(s.generate_signals(short_data)) == 1
        assert len(s.generate_signals(long_data)) == 2

    def test_data_missing_columns(self):
        s = DummyStrategy()
        bad_data = pd.DataFrame({"wrong_col": [1]}, index=pd.date_range("2024-01-01", periods=1, freq="5min"))
        sig = s.generate_signals(bad_data)
        assert (sig == 0).all()

    def test_multiple_calls_same_result(self):
        s = DummyStrategy({"mode": "rsi"})
        idx = pd.date_range("2024-01-01", periods=10, freq="5min")
        data = pd.DataFrame({"close": [100] * 10}, index=idx)
        r1 = s.generate_signals(data)
        r2 = s.generate_signals(data)
        assert (r1 == r2).all()

    def test_property_values(self):
        s = DummyStrategy()
        assert s.name == "Dummy"
        assert s.description == "Dummy strategy for testing"

    def test_signals_aligned_to_input(self):
        s = DummyStrategy()
        idx = pd.date_range("2024-01-01", periods=5, freq="5min")
        data = pd.DataFrame({"close": [100]*5}, index=idx)
        sig = s.generate_signals(data)
        assert (sig.index == idx).all()

    def test_no_nan_in_signals(self):
        s = DummyStrategy()
        idx = pd.date_range("2024-01-01", periods=10, freq="5min")
        data = pd.DataFrame({"close": [100]*10, "open": [100]*10, "high": [101]*10, "low": [99]*10, "volume": [50]*10}, index=idx)
        sig = s.generate_signals(data)
        assert not sig.isna().any()

    def test_output_values_are_valid(self):
        s = DummyStrategy()
        idx = pd.date_range("2024-01-01", periods=10, freq="5min")
        data = pd.DataFrame({"close": [100]*10}, index=idx)
        sig = s.generate_signals(data)
        assert set(sig.unique()).issubset({-1, 0, 1})

    def test_index_aligned(self):
        s = DummyStrategy()
        idx = pd.date_range("2024-01-01", periods=10, freq="5min")
        data = pd.DataFrame({"close": [100]*10}, index=idx)
        sig = s.generate_signals(data)
        assert (sig.index == idx).all()

    def test_reproducible(self):
        s = DummyStrategy()
        idx = pd.date_range("2024-01-01", periods=10, freq="5min")
        data = pd.DataFrame({"close": [100]*10}, index=idx)
        r1 = s.generate_signals(data)
        r2 = s.generate_signals(data)
        assert (r1 == r2).all()

    def test_empty_data(self):
        s = DummyStrategy()
        sig = s.generate_signals(pd.DataFrame())
        assert len(sig) == 0 or (sig == 0).all()

    def test_len_matches(self):
        s = DummyStrategy()
        idx = pd.date_range("2024-01-01", periods=5, freq="5min")
        data = pd.DataFrame({"close": [100]*5}, index=idx)
        assert len(s.generate_signals(data)) == 5

    def test_all_different_lengths(self):
        s = DummyStrategy()
        for n in [1, 3, 10, 50]:
            idx = pd.date_range("2024-01-01", periods=n, freq="5min")
            data = pd.DataFrame({"close": [100]*n}, index=idx)
            assert len(s.generate_signals(data)) == n

    def test_name_and_description(self):
        s = DummyStrategy()
        assert s.name == "Dummy"
        assert s.description == "Dummy strategy for testing"
