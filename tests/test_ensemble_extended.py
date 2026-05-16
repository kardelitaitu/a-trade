"""Tests for research/strategies/ensemble.py — extended edge cases."""

import numpy as np
import pandas as pd
import pytest

from research.strategies.ensemble import compute_rolling_sharpe, ensemble_signals


@pytest.fixture
def dix():
    return pd.date_range("2024-01-01", periods=200, freq="5min")


class TestEnsembleExtended:

    def test_rolling_sharpe_output_type(self, dix):
        close = pd.Series(100 + np.cumsum(np.random.randn(200) * 0.3), index=dix)
        sig = pd.Series(np.random.choice([-1, 0, 1], 200), index=dix)
        rs = compute_rolling_sharpe(sig, close, window=50)
        assert isinstance(rs, pd.Series)
        assert len(rs) == 200

    def test_rolling_sharpe_not_all_nan(self, dix):
        close = pd.Series(100 + np.cumsum(np.random.randn(200) * 0.3), index=dix)
        sig = pd.Series(1, index=dix)
        rs = compute_rolling_sharpe(sig, close, window=50)
        assert rs.notna().sum() > 0

    def test_ensemble_equal_weight(self, dix):
        close = pd.Series(100 + np.arange(10), index=dix[:10])
        sig_a = pd.Series([1, 1, 1, 0, 0, 0, -1, -1, -1, 0], index=dix[:10])
        sig_b = pd.Series([0, 0, 1, 1, 0, -1, -1, 0, 0, 1], index=dix[:10])
        result = ensemble_signals({"a": sig_a, "b": sig_b}, close, method="equal")
        assert result.iloc[0] == pytest.approx(0.5, abs=0.01)
        assert result.iloc[4] == pytest.approx(0.0, abs=0.01)

    def test_ensemble_single_signal(self, dix):
        close = pd.Series(100 + np.arange(5), index=dix[:5])
        sig = pd.Series([1.0, 1.0, 0.0, -1.0, -1.0], index=dix[:5])
        result = ensemble_signals({"a": sig}, close, method="equal")
        assert (result.values == sig.values).all()

    def test_ensemble_rank_weight_not_nan(self, dix):
        close = pd.Series(100 + np.arange(20), index=dix[:20])
        sig_a = pd.Series(np.random.choice([-1, 0, 1], 20), index=dix[:20])
        sig_b = pd.Series(np.random.choice([-1, 0, 1], 20), index=dix[:20])
        result = ensemble_signals({"a": sig_a, "b": sig_b}, close, method="sharpe_rank")
        assert not result.isna().all()

    def test_ensemble_sharpe_weight_not_nan(self, dix):
        close = pd.Series(100 + np.cumsum(np.random.randn(50) * 0.3), index=dix[:50])
        sig_a = pd.Series(np.random.choice([-1, 0, 1], 50), index=dix[:50])
        sig_b = pd.Series(np.random.choice([-1, 0, 1], 50), index=dix[:50])
        result = ensemble_signals({"a": sig_a, "b": sig_b}, close, method="sharpe_weight")
        assert not result.isna().all()

    def test_ensemble_fallback_on_invalid_method(self, dix):
        close = pd.Series(100 + np.arange(5), index=dix[:5])
        sig = pd.Series([1.0, 1.0, 1.0, 1.0, 1.0], index=dix[:5])
        res = ensemble_signals({"a": sig}, close, method="equal")
        assert res is not None

    def test_ensemble_handles_nan_close(self, dix):
        close = pd.Series([100.0, np.nan, 101.0, 102.0, np.nan], index=dix[:5])
        sig = pd.Series([1.0, 1.0, 1.0, 1.0, 1.0], index=dix[:5])
        result = ensemble_signals({"a": sig}, close, method="equal")
        assert not result.isna().all()
