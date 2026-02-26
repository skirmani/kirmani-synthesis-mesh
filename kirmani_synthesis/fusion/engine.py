"""
Bayesian Fusion Engine

Aggregates signals from all systems using Bayesian belief updating,
confidence weighting, and cross-system validation.
"""

import logging
from collections import defaultdict
from datetime import datetime, timedelta
from typing import Any, Optional

import numpy as np
from pydantic import BaseModel, Field

from ..models import MarketState, MarketRegime

logger = logging.getLogger(__name__)


# System accuracy weights (historical performance)
SYSTEM_WEIGHTS = {
    # Tier 1 - A+ grade
    "nexus-core": 0.96,
    "athena-sovereign": 0.96,
    "kirmani-unified": 0.95,

    # Tier 2 - A grade
    "strat-engine": 0.92,
    "athena-frm": 0.92,
    "sentinel-8": 0.91,
    "sentinel-squeeze": 0.90,
    "sornette-lppls": 0.89,
    "hurst-cycles": 0.89,

    # Tier 3 - A- grade
    "elite-scanner": 0.88,
    "murphy-intermarket": 0.88,
    "panorama-nis": 0.88,
    "cgma-system": 0.88,
    "wyckoff-elite": 0.88,

    # Tier 4 - B+ grade
    "bubble-radar": 0.85,
    "stealth-scanner": 0.86,
    "power-hour": 0.85,
    "bubble-detector": 0.84,
    "oneil-short": 0.84,
    "capitol-alpha": 0.84,

    # Default
    "default": 0.75,
}


class SignalAggregation(BaseModel):
    """Aggregated signal from multiple sources"""

    direction: str = "NEUTRAL"  # BULLISH, BEARISH, NEUTRAL
    confidence: float = 0.5
    magnitude: float = 0.5
    system_count: int = 0
    systems: list[str] = Field(default_factory=list)
    agreement_score: float = 0.5


class SynthesizedView(BaseModel):
    """
    Complete synthesized market view

    This is the output of the fusion engine after processing
    all signals across all layers.
    """

    timestamp: datetime = Field(default_factory=datetime.utcnow)

    # Layer aggregations
    macro_view: SignalAggregation = Field(default_factory=SignalAggregation)
    risk_view: SignalAggregation = Field(default_factory=SignalAggregation)
    alpha_view: SignalAggregation = Field(default_factory=SignalAggregation)

    # Unified view
    unified_direction: str = "NEUTRAL"
    unified_confidence: float = 0.5
    unified_magnitude: float = 0.5

    # Market state
    market_state: MarketState = Field(default_factory=MarketState)

    # Individual components
    crash_hazard: float = 0.0
    bubble_phase: int = 1
    squeeze_risk: float = 0.0
    intermarket_regime: str = "NEUTRAL"
    vol_regime: str = "NORMAL"

    # Ticker-specific signals
    ticker_signals: dict[str, SignalAggregation] = Field(default_factory=dict)

    # System health
    systems_reporting: list[str] = Field(default_factory=list)
    total_signals: int = 0


class FusionEngine:
    """
    Bayesian Fusion Engine

    Aggregates signals from all 48+ trading systems using:
    1. Bayesian belief updating with prior from historical accuracy
    2. Confidence-weighted averaging
    3. Cross-layer validation
    4. Contradiction detection
    5. Regime-aware weighting

    Usage:
        engine = FusionEngine()

        # Process signals
        for signal in signals:
            engine.ingest(signal)

        # Get synthesized view
        view = engine.synthesize()
    """

    def __init__(self):
        # Store signals by layer
        self._macro_signals: list[dict] = []
        self._risk_signals: list[dict] = []
        self._alpha_signals: list[dict] = []

        # Store by ticker
        self._ticker_signals: dict[str, list[dict]] = defaultdict(list)

        # Store by system
        self._system_signals: dict[str, list[dict]] = defaultdict(list)

        # Track last synthesis
        self._last_synthesis: Optional[SynthesizedView] = None

    def clear(self) -> None:
        """Clear all ingested signals"""
        self._macro_signals.clear()
        self._risk_signals.clear()
        self._alpha_signals.clear()
        self._ticker_signals.clear()
        self._system_signals.clear()

    def ingest(self, signal: dict) -> None:
        """
        Ingest a signal into the fusion engine

        Args:
            signal: Signal dictionary with system_id, signal_type, direction, confidence, etc.
        """
        system_id = signal.get("system_id", "unknown")
        signal_type = signal.get("signal_type", "")
        layer = self._determine_layer(signal_type)

        # Add system weight
        signal["_weight"] = SYSTEM_WEIGHTS.get(system_id, SYSTEM_WEIGHTS["default"])

        # Store by layer
        if layer == "MACRO":
            self._macro_signals.append(signal)
        elif layer == "RISK":
            self._risk_signals.append(signal)
        else:
            self._alpha_signals.append(signal)

        # Store by ticker
        ticker = signal.get("ticker", "SPY")
        self._ticker_signals[ticker].append(signal)

        # Store by system
        self._system_signals[system_id].append(signal)

    def ingest_batch(self, signals: list[dict]) -> None:
        """Ingest multiple signals"""
        for signal in signals:
            self.ingest(signal)

    def _determine_layer(self, signal_type: str) -> str:
        """Determine which layer a signal type belongs to"""
        macro_types = {
            "CRASH_HAZARD", "BUBBLE_PHASE", "INTERMARKET_REGIME",
            "GEOPOLITICAL_RISK", "HISTORICAL_MATCH", "BUBBLE_SCORE"
        }
        risk_types = {
            "SQUEEZE_ALERT", "PORTFOLIO_HEAT", "VOL_REGIME",
            "GREEK_EXPOSURE", "OPTION_SKEW", "SECTOR_RISK"
        }

        if signal_type in macro_types:
            return "MACRO"
        elif signal_type in risk_types:
            return "RISK"
        else:
            return "ALPHA"

    def synthesize(self) -> SynthesizedView:
        """
        Synthesize all ingested signals into unified view

        Returns:
            SynthesizedView with aggregated intelligence
        """
        view = SynthesizedView()

        # Aggregate each layer
        view.macro_view = self._aggregate_layer(self._macro_signals)
        view.risk_view = self._aggregate_layer(self._risk_signals)
        view.alpha_view = self._aggregate_layer(self._alpha_signals)

        # Extract specific metrics
        view.crash_hazard = self._extract_crash_hazard()
        view.bubble_phase = self._extract_bubble_phase()
        view.squeeze_risk = self._extract_squeeze_risk()
        view.intermarket_regime = self._extract_intermarket_regime()
        view.vol_regime = self._extract_vol_regime()

        # Compute unified view using Bayesian fusion
        view.unified_direction, view.unified_confidence, view.unified_magnitude = \
            self._bayesian_fusion(view.macro_view, view.risk_view, view.alpha_view)

        # Aggregate ticker-specific signals
        for ticker, signals in self._ticker_signals.items():
            view.ticker_signals[ticker] = self._aggregate_layer(signals)

        # Build market state
        view.market_state = self._build_market_state(view)

        # Stats
        view.systems_reporting = list(self._system_signals.keys())
        view.total_signals = len(self._macro_signals) + len(self._risk_signals) + len(self._alpha_signals)

        self._last_synthesis = view
        return view

    def _aggregate_layer(self, signals: list[dict]) -> SignalAggregation:
        """
        Aggregate signals within a layer using weighted confidence

        Uses Bayesian-style weighted averaging where weights come from
        historical system accuracy.
        """
        if not signals:
            return SignalAggregation()

        # Separate by direction
        bullish_weight = 0.0
        bearish_weight = 0.0
        neutral_weight = 0.0
        total_weight = 0.0

        magnitudes = []
        systems = set()

        for sig in signals:
            direction = sig.get("direction", "NEUTRAL")
            confidence = sig.get("confidence", 0.5)
            magnitude = sig.get("magnitude", 0.5)
            weight = sig.get("_weight", 0.75) * confidence

            systems.add(sig.get("system_id", "unknown"))
            magnitudes.append(magnitude * weight)
            total_weight += weight

            if direction == "BULLISH":
                bullish_weight += weight
            elif direction == "BEARISH":
                bearish_weight += weight
            else:
                neutral_weight += weight

        # Determine consensus direction
        if bullish_weight > bearish_weight and bullish_weight > neutral_weight:
            direction = "BULLISH"
            consensus_weight = bullish_weight
        elif bearish_weight > bullish_weight and bearish_weight > neutral_weight:
            direction = "BEARISH"
            consensus_weight = bearish_weight
        else:
            direction = "NEUTRAL"
            consensus_weight = neutral_weight

        # Calculate agreement score (how much systems agree)
        if total_weight > 0:
            agreement = consensus_weight / total_weight
            confidence = min(1.0, agreement * 1.2)  # Boost when agreement is high
        else:
            agreement = 0.5
            confidence = 0.5

        # Average magnitude
        avg_magnitude = sum(magnitudes) / len(magnitudes) if magnitudes else 0.5

        return SignalAggregation(
            direction=direction,
            confidence=confidence,
            magnitude=avg_magnitude,
            system_count=len(systems),
            systems=list(systems),
            agreement_score=agreement,
        )

    def _bayesian_fusion(
        self,
        macro: SignalAggregation,
        risk: SignalAggregation,
        alpha: SignalAggregation,
    ) -> tuple[str, float, float]:
        """
        Bayesian fusion across layers

        Combines layer-level aggregations with different weights
        based on regime and confidence levels.
        """
        # Layer weights (can be adjusted based on regime)
        macro_weight = 0.35  # Macro is important for direction
        risk_weight = 0.35   # Risk is important for sizing
        alpha_weight = 0.30  # Alpha for entry timing

        # Direction votes (weighted)
        direction_scores = {"BULLISH": 0.0, "BEARISH": 0.0, "NEUTRAL": 0.0}

        for layer, weight in [
            (macro, macro_weight),
            (risk, risk_weight),
            (alpha, alpha_weight),
        ]:
            score = layer.confidence * layer.agreement_score * weight
            direction_scores[layer.direction] += score

        # Determine unified direction
        max_dir = max(direction_scores, key=direction_scores.get)
        max_score = direction_scores[max_dir]
        total_score = sum(direction_scores.values())

        # Confidence from score ratio
        unified_confidence = max_score / total_score if total_score > 0 else 0.5

        # Magnitude as weighted average
        unified_magnitude = (
            macro.magnitude * macro_weight +
            risk.magnitude * risk_weight +
            alpha.magnitude * alpha_weight
        )

        return max_dir, unified_confidence, unified_magnitude

    def _extract_crash_hazard(self) -> float:
        """Extract crash hazard from macro signals"""
        for sig in self._macro_signals:
            if sig.get("signal_type") == "CRASH_HAZARD":
                return sig.get("magnitude", 0.0)
            # Check components
            components = sig.get("components", {})
            if "hazard_rate" in components:
                return components["hazard_rate"]
        return 0.0

    def _extract_bubble_phase(self) -> int:
        """Extract Minsky bubble phase"""
        for sig in self._macro_signals:
            if sig.get("signal_type") == "BUBBLE_PHASE":
                components = sig.get("components", {})
                return components.get("stage", 1)
        return 1

    def _extract_squeeze_risk(self) -> float:
        """Extract squeeze risk from risk signals"""
        max_squeeze = 0.0
        for sig in self._risk_signals:
            if sig.get("signal_type") == "SQUEEZE_ALERT":
                score = sig.get("components", {}).get("squeeze_score", 0) / 100
                max_squeeze = max(max_squeeze, score)
        return max_squeeze

    def _extract_intermarket_regime(self) -> str:
        """Extract intermarket regime"""
        for sig in self._macro_signals:
            if sig.get("signal_type") == "INTERMARKET_REGIME":
                components = sig.get("components", {})
                return components.get("regime", "NEUTRAL")
        return "NEUTRAL"

    def _extract_vol_regime(self) -> str:
        """Extract volatility regime"""
        for sig in self._risk_signals:
            if sig.get("signal_type") == "VOL_REGIME":
                components = sig.get("components", {})
                return components.get("regime", "NORMAL")
        return "NORMAL"

    def _build_market_state(self, view: SynthesizedView) -> MarketState:
        """Build complete market state from synthesized view"""

        # Determine regime
        if view.crash_hazard > 0.7 or view.bubble_phase >= 4:
            regime = MarketRegime.CRISIS
        elif view.intermarket_regime == "RISK_OFF" or view.macro_view.direction == "BEARISH":
            regime = MarketRegime.RISK_OFF
        elif view.intermarket_regime == "RISK_ON" or view.macro_view.direction == "BULLISH":
            regime = MarketRegime.RISK_ON
        elif view.macro_view.agreement_score < 0.6:
            regime = MarketRegime.TRANSITION
        else:
            regime = MarketRegime.NEUTRAL

        # Calculate overall risk score (0-100)
        risk_score = (
            view.crash_hazard * 30 +
            (view.bubble_phase / 5) * 25 +
            view.squeeze_risk * 20 +
            (1 - view.macro_view.agreement_score) * 25
        )

        return MarketState(
            regime=regime,
            regime_confidence=view.unified_confidence,
            crash_hazard=view.crash_hazard,
            bubble_phase=view.bubble_phase,
            bubble_confidence=view.macro_view.confidence,
            overall_risk_score=min(100, risk_score),
            squeeze_risk=view.squeeze_risk,
            vol_regime=view.vol_regime,
            intermarket_signal=view.intermarket_regime,
            signal_count=view.total_signals,
            system_agreement=view.macro_view.agreement_score,
            components={
                "macro_direction": view.macro_view.direction,
                "risk_direction": view.risk_view.direction,
                "alpha_direction": view.alpha_view.direction,
            }
        )

    def get_last_synthesis(self) -> Optional[SynthesizedView]:
        """Get the last synthesized view"""
        return self._last_synthesis
