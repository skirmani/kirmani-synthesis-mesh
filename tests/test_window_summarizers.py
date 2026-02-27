"""Tests for Window Signal Summarizers (Pattern 5)."""

import importlib.util
import math
import os
import sys
from datetime import datetime, timezone

import pytest

# Direct import of the window_summarizers module to avoid pydantic chain
_MODULE_PATH = os.path.join(
    os.path.dirname(__file__), "..", "kirmani_synthesis", "window_summarizers.py"
)
_spec = importlib.util.spec_from_file_location(
    "kirmani_synthesis.window_summarizers", _MODULE_PATH
)
_mod = importlib.util.module_from_spec(_spec)
sys.modules["kirmani_synthesis.window_summarizers"] = _mod
_spec.loader.exec_module(_mod)

RollingZScore = _mod.RollingZScore
DecayWeightedMean = _mod.DecayWeightedMean
NthMoment = _mod.NthMoment
CrossCorrelationWindow = _mod.CrossCorrelationWindow
SignalSummarizerPipeline = _mod.SignalSummarizerPipeline
create_summarizer_pipeline = _mod.create_summarizer_pipeline


class TestRollingZScore:
    def test_basic_zscore(self):
        zscore = RollingZScore(window=20)
        results = []
        for i in range(30):
            result = zscore.update(float(i))
            results.append(result)
        assert results[-1].value > 0
        assert results[-1].window_size == 20

    def test_anomaly_detection(self):
        zscore = RollingZScore(window=50)
        for i in range(100):
            zscore.update(10.0 + (i % 5) * 0.1)
        result = zscore.update(100.0)
        assert abs(result.value) > 2.0

    def test_mean_and_std(self):
        zscore = RollingZScore(window=100)
        for i in range(100):
            zscore.update(10.0)
        assert abs(zscore.mean - 10.0) < 0.01
        assert zscore.std < 0.01


class TestDecayWeightedMean:
    def test_recency_bias(self):
        dwm = DecayWeightedMean(halflife=5)
        for _ in range(20):
            dwm.update(1.0)
        for _ in range(5):
            dwm.update(10.0)
        assert dwm.value > 5.0

    def test_exponential_decay(self):
        dwm = DecayWeightedMean(halflife=10)
        result_1 = dwm.update(100.0)
        result_2 = dwm.update(0.0)
        assert result_2.value < result_1.value


class TestNthMoment:
    def test_variance(self):
        moments = NthMoment(max_moment=2)
        for v in [2, 4, 4, 4, 5, 5, 7, 9]:
            result = moments.update(float(v))
        assert result.metadata["count"] == 8
        assert abs(result.metadata["mean"] - 5.0) < 0.1

    def test_skewness(self):
        moments = NthMoment(max_moment=3)
        # Generate clearly right-skewed data: many small values, few large
        for v in [1, 1, 1, 1, 1, 2, 2, 2, 3, 3, 50, 100]:
            result = moments.update(float(v))
        # Skewness should be non-zero for asymmetric data
        assert result.metadata["skewness"] != 0

    def test_windowed_mode(self):
        moments = NthMoment(max_moment=4, window=10)
        for i in range(50):
            result = moments.update(float(i))
        assert result.window_size == 10


class TestCrossCorrelation:
    def test_perfect_correlation(self):
        corr = CrossCorrelationWindow(window=30)
        for i in range(30):
            result = corr.update(float(i), float(i))
        assert result.value > 0.99

    def test_negative_correlation(self):
        corr = CrossCorrelationWindow(window=30)
        for i in range(30):
            result = corr.update(float(i), float(-i))
        assert result.value < -0.99


class TestSummarizerPipeline:
    def test_pipeline_processing(self):
        pipeline = create_summarizer_pipeline(
            zscore_windows=[10, 20], decay_halflife=10, moment_order=4,
        )
        for i in range(30):
            results = pipeline.process(float(i))
        assert "zscore_10" in results
        assert "zscore_20" in results
        assert "decay_weighted" in results
        assert "moments" in results

    def test_pipeline_names(self):
        pipeline = create_summarizer_pipeline()
        assert "zscore_20" in pipeline.names
        assert "decay_weighted" in pipeline.names
