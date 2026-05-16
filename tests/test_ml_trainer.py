"""Tests for research/ml/trainer.py."""

import numpy as np
import pandas as pd
import pytest

from research.ml.trainer import (
    signals_from_probabilities,
    predict_probabilities,
)


class TestSignalsFromProbabilities:

    def test_high_prob_long(self):
        """Probability > 0.55 should give long signal."""
        idx = pd.date_range("2024-01-01", periods=5, freq="5min")
        probs = pd.Series([0.6, 0.8, 0.55, 0.3, 0.45], index=idx)
        signals = signals_from_probabilities(probs)
        assert signals.iloc[0] == 1
        assert signals.iloc[3] == -1
        assert signals.iloc[4] == 0  # neutral zone

    def test_all_signal_values(self):
        """Signals should only be -1, 0, or 1."""
        np.random.seed(42)
        idx = pd.date_range("2024-01-01", periods=100, freq="5min")
        probs = pd.Series(np.random.uniform(0, 1, 100), index=idx)
        signals = signals_from_probabilities(probs, threshold=0.5)
        assert set(signals.unique()).issubset({-1, 0, 1})

    def test_continuous_range(self):
        """Continuous signals should be floats in [-1, 1]."""
        idx = pd.date_range("2024-01-01", periods=5, freq="5min")
        probs = pd.Series([0.0, 0.25, 0.5, 0.75, 1.0], index=idx)
        signals = signals_from_probabilities(probs, continuous=True)
        assert signals.iloc[0] == pytest.approx(-1.0)
        assert signals.iloc[2] == pytest.approx(0.0)
        assert signals.iloc[4] == pytest.approx(1.0)
        assert signals.iloc[1] > -1 and signals.iloc[1] < 0
        assert signals.iloc[3] > 0 and signals.iloc[3] < 1
