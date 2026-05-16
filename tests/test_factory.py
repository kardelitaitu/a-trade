"""Tests for research/strategies/factory.py."""

import numpy as np
import pandas as pd
import pytest

from research.strategies.factory import create_strategy, list_strategies


class TestCreateStrategy:

    def test_ma_crossover_default(self):
        strat = create_strategy("ma_crossover")
        assert strat.name == "MA Crossover"
        assert strat.config["fast_period"] == 12
        assert strat.config["slow_period"] == 26

    def test_ma_crossover_with_indicator(self):
        strat = create_strategy("ma_crossover", {
            "fast_indicator": "ema",
            "fast_period": 10,
            "slow_indicator": "sma",
            "slow_period": 40,
        })
        assert strat.config["fast_period"] == 10
        assert strat.config["slow_period"] == 40

    def test_ma_crossover_invalid_indicator(self):
        with pytest.raises(ValueError, match="Unknown indicator"):
            create_strategy("ma_crossover", {"fast_indicator": "nonexistent"})

    def test_donchian_default(self):
        strat = create_strategy("donchian_breakout")
        assert "Donchian" in strat.name

    def test_mean_reversion(self):
        strat = create_strategy("mean_reversion", {"mode": "rsi"})
        assert strat.config["mode"] == "rsi"

    def test_volatility_breakout(self):
        strat = create_strategy("volatility_breakout")
        assert "Volatility" in strat.name

    def test_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown strategy"):
            create_strategy("nonexistent")

    def test_generates_signals(self):
        """Created strategy should produce valid signals with real data."""
        np.random.seed(42)
        idx = pd.date_range("2024-01-01", periods=200, freq="5min")
        close = 100 + np.cumsum(np.random.randn(200) * 0.3)
        data = pd.DataFrame({
            "open": close - 0.2, "high": close + 0.5,
            "low": close - 0.5, "close": close,
            "volume": np.random.uniform(10, 100, 200),
        }, index=idx)

        strat = create_strategy("ma_crossover")
        signals = strat.generate_signals(data)
        assert set(signals.dropna().unique()).issubset({-1, 0, 1})


class TestListStrategies:

    def test_list_output(self):
        result = list_strategies()
        assert "ma_crossover" in result
        assert "donchian_breakout" in result
        assert "mean_reversion" in result
        assert "volatility_breakout" in result
        assert "fast_period" in result

    def test_list_has_vol_squeeze(self):
        result = list_strategies()
        assert "volatility_squeeze" in result

    def test_list_has_bb_std(self):
        result = list_strategies()
        assert "bb_std" in result

    def test_list_output_length(self):
        result = list_strategies()
        assert len(result) > 200

    def test_all_strategies_generatable(self):
        for name in ["ma_crossover", "donchian_breakout", "mean_reversion", "volatility_breakout", "volatility_squeeze"]:
            from research.strategies.factory import create_strategy
            s = create_strategy(name)
            assert s is not None

    def test_create_with_indicator(self):
        from research.strategies.factory import create_strategy
        s = create_strategy("ma_crossover", {"fast_indicator":"ema","slow_indicator":"sma"})
        assert s.config["fast_indicator"] == "ema"

    def test_create_all_indicator_names(self):
        from research.strategies.factory import create_strategy
        for n in ["ma_crossover","donchian_breakout","mean_reversion","volatility_breakout","volatility_squeeze"]:
            s = create_strategy(n, {"fast_period": 5})
            assert s.config is not None
