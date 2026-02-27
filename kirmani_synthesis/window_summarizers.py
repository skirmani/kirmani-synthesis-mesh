"""Window-based Signal Summarizers — Flint Pattern.

Implements rolling window statistical summarizers for signal
aggregation across multiple timeframes:
- Rolling z-score with configurable lookback
- Decay-weighted mean with exponential weighting
- Nth-moment calculators for skew/kurtosis
- Cross-correlation windows for lead-lag detection
- Regime-aware windowing with breakpoint detection

Inspired by Two Sigma's Flint window summarizer library.

Kirmani Partners LP — 2026
"""

from __future__ import annotations

import collections
import logging
import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Deque, Dict, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)


class WindowType(Enum):
    """Types of rolling windows."""

    FIXED = "fixed"
    EXPANDING = "expanding"
    EXPONENTIAL = "exponential"
    REGIME_AWARE = "regime_aware"


@dataclass
class WindowState:
    """Internal state for a rolling window."""

    values: Deque[float]
    timestamps: Deque[datetime]
    sum_x: float = 0.0
    sum_x2: float = 0.0
    sum_x3: float = 0.0
    sum_x4: float = 0.0
    count: int = 0


@dataclass
class SummarizerResult:
    """Result of a window summarization."""

    value: float
    timestamp: datetime
    window_size: int
    window_type: WindowType
    metadata: Dict[str, Any] = field(default_factory=dict)


class RollingZScore:
    """Rolling z-score calculator.

    Tracks mean and standard deviation over a rolling window and
    computes z-scores for incoming values in O(1) amortized time.

    Usage:
        zscore = RollingZScore(window=252)
        for price in prices:
            z = zscore.update(price)
            if abs(z) > 2.0:
                print("Anomaly detected")
    """

    def __init__(self, window: int = 252):
        self._window = window
        self._state = WindowState(
            values=collections.deque(maxlen=window),
            timestamps=collections.deque(maxlen=window),
        )

    def update(
        self, value: float, timestamp: Optional[datetime] = None
    ) -> SummarizerResult:
        """Add a value and compute the z-score.

        Args:
            value: New observation.
            timestamp: Optional timestamp.

        Returns:
            SummarizerResult with z-score.
        """
        ts = timestamp or datetime.now(timezone.utc)
        state = self._state

        # Remove oldest if at capacity
        if len(state.values) == self._window:
            old = state.values[0]
            state.sum_x -= old
            state.sum_x2 -= old * old
        else:
            state.count += 1

        state.values.append(value)
        state.timestamps.append(ts)
        state.sum_x += value
        state.sum_x2 += value * value

        n = len(state.values)
        if n < 2:
            return SummarizerResult(
                value=0.0, timestamp=ts, window_size=n, window_type=WindowType.FIXED
            )

        mean = state.sum_x / n
        variance = (state.sum_x2 / n) - (mean * mean)
        std = math.sqrt(max(variance, 1e-12))
        z = (value - mean) / std

        return SummarizerResult(
            value=z,
            timestamp=ts,
            window_size=n,
            window_type=WindowType.FIXED,
            metadata={"mean": mean, "std": std, "variance": variance},
        )

    @property
    def mean(self) -> float:
        n = len(self._state.values)
        return self._state.sum_x / n if n > 0 else 0.0

    @property
    def std(self) -> float:
        n = len(self._state.values)
        if n < 2:
            return 0.0
        mean = self._state.sum_x / n
        var = (self._state.sum_x2 / n) - (mean * mean)
        return math.sqrt(max(var, 0))


class DecayWeightedMean:
    """Exponentially decay-weighted mean calculator.

    Gives more weight to recent observations using an exponential
    decay factor. Useful for signal confidence aggregation where
    recency matters.

    Usage:
        dwm = DecayWeightedMean(halflife=20)
        for confidence in confidences:
            weighted = dwm.update(confidence)
    """

    def __init__(self, halflife: float = 20.0, time_based: bool = False):
        """Initialize decay-weighted mean.

        Args:
            halflife: Number of steps (or seconds if time_based) for half decay.
            time_based: If True, decay based on time delta instead of steps.
        """
        self._halflife = halflife
        self._decay = math.log(2) / halflife
        self._time_based = time_based
        self._weighted_sum = 0.0
        self._weight_total = 0.0
        self._last_timestamp: Optional[datetime] = None
        self._history: Deque[Tuple[float, float]] = collections.deque(maxlen=10000)

    def update(
        self, value: float, timestamp: Optional[datetime] = None
    ) -> SummarizerResult:
        """Add a value with exponential decay weighting.

        Args:
            value: New observation.
            timestamp: Optional timestamp for time-based decay.

        Returns:
            SummarizerResult with decay-weighted mean.
        """
        ts = timestamp or datetime.now(timezone.utc)

        if self._time_based and self._last_timestamp:
            dt = (ts - self._last_timestamp).total_seconds()
            decay_factor = math.exp(-self._decay * dt)
        else:
            decay_factor = math.exp(-self._decay)

        # Decay existing weights
        self._weighted_sum *= decay_factor
        self._weight_total *= decay_factor

        # Add new observation with weight 1.0
        self._weighted_sum += value
        self._weight_total += 1.0

        self._last_timestamp = ts
        weighted_mean = (
            self._weighted_sum / self._weight_total
            if self._weight_total > 0
            else 0.0
        )

        self._history.append((value, weighted_mean))

        return SummarizerResult(
            value=weighted_mean,
            timestamp=ts,
            window_size=len(self._history),
            window_type=WindowType.EXPONENTIAL,
            metadata={
                "halflife": self._halflife,
                "decay_factor": decay_factor,
                "effective_weight": self._weight_total,
            },
        )

    @property
    def value(self) -> float:
        return (
            self._weighted_sum / self._weight_total
            if self._weight_total > 0
            else 0.0
        )


class NthMoment:
    """Nth statistical moment calculator.

    Computes running moments (variance, skewness, kurtosis)
    using Welford's online algorithm for numerical stability.

    Usage:
        moments = NthMoment(max_moment=4)
        for ret in returns:
            result = moments.update(ret)
            print(f"Skew: {result.metadata['skewness']}")
    """

    def __init__(self, max_moment: int = 4, window: Optional[int] = None):
        """Initialize moment calculator.

        Args:
            max_moment: Maximum moment to compute (2=var, 3=skew, 4=kurt).
            window: Optional rolling window size.
        """
        self._max_moment = min(max_moment, 4)
        self._window = window
        self._values: Deque[float] = collections.deque(
            maxlen=window if window else None
        )
        self._n = 0
        self._mean = 0.0
        self._m2 = 0.0
        self._m3 = 0.0
        self._m4 = 0.0

    def update(
        self, value: float, timestamp: Optional[datetime] = None
    ) -> SummarizerResult:
        """Add a value and compute moments.

        Uses Welford's algorithm for numerical stability.

        Args:
            value: New observation.
            timestamp: Optional timestamp.

        Returns:
            SummarizerResult with moments in metadata.
        """
        ts = timestamp or datetime.now(timezone.utc)

        if self._window and len(self._values) == self._window:
            # Recompute from scratch for windowed version
            self._values.append(value)
            self._recompute()
        else:
            self._values.append(value)
            self._n += 1
            n = self._n
            delta = value - self._mean
            delta_n = delta / n
            self._mean += delta_n

            if self._max_moment >= 2:
                self._m2 += delta * (value - self._mean)

            if self._max_moment >= 3 and n >= 3:
                self._m3 += (
                    delta_n * delta_n * delta * (n - 1) * (n - 2) / n
                    - 3 * delta_n * self._m2
                )

            if self._max_moment >= 4 and n >= 4:
                self._m4 += (
                    delta_n * delta_n * delta * delta * (n - 1) * (n * n - 3 * n + 3) / (n * n)
                    + 6 * delta_n * delta_n * self._m2
                    - 4 * delta_n * self._m3
                )

        n = len(self._values)
        meta: Dict[str, Any] = {"mean": self._mean, "count": n}

        variance = self._m2 / n if n > 1 else 0.0
        std = math.sqrt(max(variance, 0))
        meta["variance"] = variance
        meta["std"] = std

        if self._max_moment >= 3 and n > 2 and std > 0:
            meta["skewness"] = (self._m3 / n) / (std ** 3)
        else:
            meta["skewness"] = 0.0

        if self._max_moment >= 4 and n > 3 and std > 0:
            meta["kurtosis"] = (self._m4 / n) / (std ** 4) - 3.0
        else:
            meta["kurtosis"] = 0.0

        return SummarizerResult(
            value=variance,
            timestamp=ts,
            window_size=n,
            window_type=WindowType.FIXED if self._window else WindowType.EXPANDING,
            metadata=meta,
        )

    def _recompute(self) -> None:
        """Recompute moments from scratch (for windowed mode)."""
        values = list(self._values)
        n = len(values)
        self._n = n
        if n == 0:
            self._mean = self._m2 = self._m3 = self._m4 = 0.0
            return

        self._mean = sum(values) / n
        self._m2 = sum((x - self._mean) ** 2 for x in values)
        if self._max_moment >= 3:
            self._m3 = sum((x - self._mean) ** 3 for x in values)
        if self._max_moment >= 4:
            self._m4 = sum((x - self._mean) ** 4 for x in values)


class CrossCorrelationWindow:
    """Rolling cross-correlation calculator for lead-lag detection.

    Computes Pearson correlation between two signal streams over
    a rolling window, useful for detecting which system leads.

    Usage:
        corr = CrossCorrelationWindow(window=60)
        for momentum, regime in zip(momentum_stream, regime_stream):
            result = corr.update(momentum, regime)
            if result.value > 0.7:
                print("Strong positive correlation")
    """

    def __init__(self, window: int = 60):
        self._window = window
        self._x: Deque[float] = collections.deque(maxlen=window)
        self._y: Deque[float] = collections.deque(maxlen=window)

    def update(
        self, x: float, y: float, timestamp: Optional[datetime] = None
    ) -> SummarizerResult:
        """Add paired observations and compute correlation.

        Args:
            x: Value from first stream.
            y: Value from second stream.
            timestamp: Optional timestamp.

        Returns:
            SummarizerResult with correlation coefficient.
        """
        ts = timestamp or datetime.now(timezone.utc)
        self._x.append(x)
        self._y.append(y)

        n = len(self._x)
        if n < 3:
            return SummarizerResult(
                value=0.0, timestamp=ts, window_size=n, window_type=WindowType.FIXED
            )

        x_list = list(self._x)
        y_list = list(self._y)

        mean_x = sum(x_list) / n
        mean_y = sum(y_list) / n

        cov = sum((xi - mean_x) * (yi - mean_y) for xi, yi in zip(x_list, y_list)) / n
        var_x = sum((xi - mean_x) ** 2 for xi in x_list) / n
        var_y = sum((yi - mean_y) ** 2 for yi in y_list) / n

        denom = math.sqrt(max(var_x, 1e-12)) * math.sqrt(max(var_y, 1e-12))
        corr = cov / denom if denom > 0 else 0.0

        return SummarizerResult(
            value=corr,
            timestamp=ts,
            window_size=n,
            window_type=WindowType.FIXED,
            metadata={
                "covariance": cov,
                "var_x": var_x,
                "var_y": var_y,
                "mean_x": mean_x,
                "mean_y": mean_y,
            },
        )


class SignalSummarizerPipeline:
    """Pipeline of multiple summarizers for comprehensive signal analysis.

    Combines multiple window-based summarizers into a single
    processing pipeline for real-time signal characterization.

    Usage:
        pipeline = SignalSummarizerPipeline()
        pipeline.add("zscore_20", RollingZScore(20))
        pipeline.add("zscore_60", RollingZScore(60))
        pipeline.add("dwm", DecayWeightedMean(halflife=10))
        pipeline.add("moments", NthMoment(max_moment=4))

        for confidence in signal_stream:
            results = pipeline.process(confidence)
    """

    def __init__(self) -> None:
        self._summarizers: Dict[str, Any] = {}

    def add(self, name: str, summarizer: Any) -> "SignalSummarizerPipeline":
        """Add a summarizer to the pipeline.

        Args:
            name: Unique name for this summarizer.
            summarizer: Summarizer instance with an `update` method.

        Returns:
            Self for chaining.
        """
        self._summarizers[name] = summarizer
        return self

    def process(
        self, value: float, timestamp: Optional[datetime] = None
    ) -> Dict[str, SummarizerResult]:
        """Process a value through all summarizers.

        Args:
            value: Input value.
            timestamp: Optional timestamp.

        Returns:
            Dict of name -> SummarizerResult.
        """
        results = {}
        for name, summarizer in self._summarizers.items():
            if hasattr(summarizer, "update"):
                results[name] = summarizer.update(value, timestamp)
        return results

    @property
    def names(self) -> List[str]:
        return list(self._summarizers.keys())


def create_summarizer_pipeline(
    zscore_windows: Optional[List[int]] = None,
    decay_halflife: float = 20.0,
    moment_order: int = 4,
) -> SignalSummarizerPipeline:
    """Factory to create a standard summarizer pipeline.

    Args:
        zscore_windows: Z-score window sizes (default [20, 60, 252]).
        decay_halflife: Halflife for decay-weighted mean.
        moment_order: Maximum moment order.

    Returns:
        Configured SignalSummarizerPipeline.
    """
    windows = zscore_windows or [20, 60, 252]
    pipeline = SignalSummarizerPipeline()

    for w in windows:
        pipeline.add(f"zscore_{w}", RollingZScore(window=w))

    pipeline.add("decay_weighted", DecayWeightedMean(halflife=decay_halflife))
    pipeline.add("moments", NthMoment(max_moment=moment_order))

    return pipeline
