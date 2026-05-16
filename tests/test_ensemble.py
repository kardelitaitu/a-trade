"""Tests for research/strategies/ensemble.py."""
import numpy as np
import pandas as pd
import pytest
from research.strategies.ensemble import ensemble_signals, compute_rolling_sharpe


@pytest.fixture
def dix():
    return pd.date_range("2024-01-01", periods=500, freq="5min")


class TestEnsemble:

    def test_ensemble_equal_two_signals(self, dix):
        close = pd.Series(100 + np.arange(10), index=dix[:10])
        a = pd.Series([1, 1, 1, 0, 0, -1, -1, -1, 0, 0], index=dix[:10])
        b = pd.Series([0, 0, 1, 1, 0, -1, -1, 0, 0, 1], index=dix[:10])
        r = ensemble_signals({"a": a, "b": b}, close, method="equal")
        assert abs(r.iloc[0] - 0.5) < 0.01

    def test_ensemble_sharpe_weight(self, dix):
        close = pd.Series(100 + np.cumsum(np.random.randn(100) * 0.3), index=dix[:100])
        a = pd.Series(np.random.choice([-1, 0, 1], 100), index=dix[:100])
        b = pd.Series(np.random.choice([-1, 0, 1], 100), index=dix[:100])
        r = ensemble_signals({"a": a, "b": b}, close, method="sharpe_weight")
        assert not r.isna().all()

    def test_ensemble_sharpe_rank(self, dix):
        close = pd.Series(100 + np.cumsum(np.random.randn(100) * 0.3), index=dix[:100])
        a = pd.Series(np.random.choice([-1, 0, 1], 100), index=dix[:100])
        b = pd.Series(np.random.choice([-1, 0, 1], 100), index=dix[:100])
        c = pd.Series(np.random.choice([-1, 0, 1], 100), index=dix[:100])
        r = ensemble_signals({"a": a, "b": b, "c": c}, close, method="sharpe_rank")
        assert not r.isna().all()

    def test_rolling_sharpe_output(self, dix):
        close = pd.Series(100 + np.cumsum(np.random.randn(200) * 0.3), index=dix[:200])
        sig = pd.Series(np.random.choice([-1, 0, 1], 200), index=dix[:200])
        rs = compute_rolling_sharpe(sig, close, window=50)
        assert len(rs) == 200
        assert rs.notna().sum() > 0

    def test_ensemble_single(self, dix):
        close = pd.Series(100 + np.arange(5), index=dix[:5])
        sig = pd.Series([1, 1, 0, -1, -1], index=dix[:5])
        r = ensemble_signals({"a": sig}, close, method="equal")
        assert (r.values == sig.values).all()

    def test_ensemble_equal_weight_specified(self, dix):
        close = pd.Series(100 + np.arange(10), index=dix[:10])
        a = pd.Series([1]*10, index=dix[:10]); b = pd.Series([0]*10, index=dix[:10])
        r = ensemble_signals({"a": a, "b": b}, close, method="equal", equal_weight=[0.8, 0.2])
        assert abs(r.mean() - 0.8) < 0.01

    def test_three_signals_equal(self, dix):
        close = pd.Series(100 + np.arange(10), index=dix[:10])
        a = pd.Series([1]*10, index=dix[:10])
        b = pd.Series([1]*10, index=dix[:10])
        c = pd.Series([1]*10, index=dix[:10])
        r = ensemble_signals({"a": a, "b": b, "c": c}, close, method="equal")
        assert abs(r.mean() - 1.0) < 0.01

    def test_nan_signals_handling(self, dix):
        close = pd.Series(100 + np.arange(10), index=dix[:10])
        a = pd.Series([1, np.nan, 1, np.nan, 1, 1, 1, 1, 1, 1], index=dix[:10])
        r = ensemble_signals({"a": a}, close, method="equal")
        assert not r.isna().all()

    def test_backtest_ensemble_exists(self, dix):
        from research.strategies.ensemble import backtest_ensemble
        assert callable(backtest_ensemble)
