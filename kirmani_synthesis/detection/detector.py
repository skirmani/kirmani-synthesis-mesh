"""
Contradiction and Confirmation Detectors

Identifies when systems agree (confirmation) or disagree (contradiction),
enabling better decision-making through cross-validation.
"""

import logging
from collections import defaultdict
from datetime import datetime
from typing import Optional

from ..models import ContradictionAlert, ConfirmationAlert

logger = logging.getLogger(__name__)


# Systems that should agree if market direction is clear
AGREEMENT_PAIRS = [
    # If both signal same ticker, they should align
    ("sornette-lppls", "minsky-analyzer"),      # Crash/bubble systems
    ("murphy-intermarket", "sornette-lppls"),   # Macro alignment
    ("elite-scanner", "wyckoff-elite"),         # Entry systems
    ("sentinel-8", "minsky-analyzer"),          # Risk systems
]


class ContradictionDetector:
    """
    Detects conflicting signals between systems

    A contradiction occurs when two systems emit signals for the same
    ticker with opposite directions and both have high confidence.

    Example:
        - Wyckoff says ACCUMULATION (bullish) for AAPL
        - Minsky says EUPHORIA stage 4 (bearish) for AAPL
        → Contradiction detected!

    Usage:
        detector = ContradictionDetector()
        contradictions = detector.detect(signals)
    """

    def __init__(self, min_confidence: float = 0.6):
        self.min_confidence = min_confidence

    def detect(self, signals: list[dict]) -> list[ContradictionAlert]:
        """
        Detect contradictions in a set of signals

        Args:
            signals: List of signal dictionaries

        Returns:
            List of ContradictionAlert objects
        """
        contradictions = []

        # Group signals by ticker
        by_ticker: dict[str, list[dict]] = defaultdict(list)
        for sig in signals:
            ticker = sig.get("ticker", "SPY")
            by_ticker[ticker].append(sig)

        # Check each ticker for contradictions
        for ticker, ticker_signals in by_ticker.items():
            # Get directional signals with sufficient confidence
            bullish = [
                s for s in ticker_signals
                if s.get("direction") == "BULLISH" and s.get("confidence", 0) >= self.min_confidence
            ]
            bearish = [
                s for s in ticker_signals
                if s.get("direction") == "BEARISH" and s.get("confidence", 0) >= self.min_confidence
            ]

            # If both bullish and bearish exist, we have a contradiction
            if bullish and bearish:
                # Find the strongest from each side
                strongest_bull = max(bullish, key=lambda s: s.get("confidence", 0))
                strongest_bear = max(bearish, key=lambda s: s.get("confidence", 0))

                severity = (
                    strongest_bull.get("confidence", 0) +
                    strongest_bear.get("confidence", 0)
                ) / 2

                # Determine recommended action
                if severity > 0.8:
                    recommended = "WAIT"  # High severity = unclear
                elif strongest_bull.get("confidence", 0) > strongest_bear.get("confidence", 0) + 0.15:
                    recommended = "FAVOR_BULLISH"
                elif strongest_bear.get("confidence", 0) > strongest_bull.get("confidence", 0) + 0.15:
                    recommended = "FAVOR_BEARISH"
                else:
                    recommended = "REDUCE"  # Reduce exposure when unclear

                alert = ContradictionAlert(
                    system_a=strongest_bull.get("system_id", "unknown"),
                    system_b=strongest_bear.get("system_id", "unknown"),
                    signal_a_type=strongest_bull.get("signal_type", ""),
                    signal_a_direction="BULLISH",
                    signal_a_confidence=strongest_bull.get("confidence", 0),
                    signal_b_type=strongest_bear.get("signal_type", ""),
                    signal_b_direction="BEARISH",
                    signal_b_confidence=strongest_bear.get("confidence", 0),
                    ticker=ticker,
                    severity=severity,
                    recommended_action=recommended,
                    resolution_rationale=self._generate_rationale(
                        strongest_bull, strongest_bear, recommended
                    ),
                )

                contradictions.append(alert)
                logger.warning(
                    f"Contradiction detected: {ticker} - "
                    f"{alert.system_a} (BULLISH) vs {alert.system_b} (BEARISH)"
                )

        return contradictions

    def _generate_rationale(
        self,
        bull_signal: dict,
        bear_signal: dict,
        recommended: str,
    ) -> str:
        """Generate explanation for the contradiction"""
        bull_sys = bull_signal.get("system_id", "unknown")
        bear_sys = bear_signal.get("system_id", "unknown")
        bull_conf = bull_signal.get("confidence", 0)
        bear_conf = bear_signal.get("confidence", 0)

        rationale = f"{bull_sys} signals BULLISH ({bull_conf:.0%}) while {bear_sys} signals BEARISH ({bear_conf:.0%}). "

        if recommended == "WAIT":
            rationale += "High confidence from both sides - wait for clarity before acting."
        elif recommended == "FAVOR_BULLISH":
            rationale += f"Favor bullish signal due to higher confidence from {bull_sys}."
        elif recommended == "FAVOR_BEARISH":
            rationale += f"Favor bearish signal due to higher confidence from {bear_sys}."
        else:
            rationale += "Consider reducing position size until conflict resolves."

        return rationale


class ConfirmationDetector:
    """
    Detects when multiple systems confirm the same signal

    A confirmation occurs when multiple systems emit signals for the same
    ticker with the same direction. Stronger confirmations involve:
    - More systems agreeing
    - Multiple layers (MACRO + RISK + ALPHA) agreeing
    - Higher average confidence

    Usage:
        detector = ConfirmationDetector()
        confirmations = detector.detect(signals)
    """

    def __init__(
        self,
        min_systems: int = 2,
        min_confidence: float = 0.5,
    ):
        self.min_systems = min_systems
        self.min_confidence = min_confidence

    def detect(self, signals: list[dict]) -> list[ConfirmationAlert]:
        """
        Detect confirmations in a set of signals

        Args:
            signals: List of signal dictionaries

        Returns:
            List of ConfirmationAlert objects
        """
        confirmations = []

        # Group by ticker and direction
        groups: dict[tuple[str, str], list[dict]] = defaultdict(list)

        for sig in signals:
            ticker = sig.get("ticker", "SPY")
            direction = sig.get("direction", "NEUTRAL")
            confidence = sig.get("confidence", 0)

            if direction != "NEUTRAL" and confidence >= self.min_confidence:
                groups[(ticker, direction)].append(sig)

        # Check each group for confirmations
        for (ticker, direction), group_signals in groups.items():
            if len(group_signals) < self.min_systems:
                continue

            # Get unique systems
            systems = list(set(s.get("system_id", "unknown") for s in group_signals))

            if len(systems) < self.min_systems:
                continue

            # Count unique layers
            layers = set()
            for sig in group_signals:
                layer = self._get_layer(sig.get("signal_type", ""))
                layers.add(layer)

            # Calculate combined confidence
            confidences = [s.get("confidence", 0) for s in group_signals]
            combined = sum(confidences) / len(confidences)

            # Boost for multi-layer confirmation
            if len(layers) >= 2:
                combined = min(1.0, combined * 1.1)
            if len(layers) >= 3:
                combined = min(1.0, combined * 1.1)

            # Determine strength
            if len(systems) >= 4 and len(layers) >= 2:
                strength = "VERY_STRONG"
            elif len(systems) >= 3 or len(layers) >= 2:
                strength = "STRONG"
            elif combined >= 0.7:
                strength = "MODERATE"
            else:
                strength = "WEAK"

            alert = ConfirmationAlert(
                systems=systems,
                system_count=len(systems),
                layer_count=len(layers),
                ticker=ticker,
                direction=direction,
                combined_confidence=combined,
                confirmation_strength=strength,
            )

            confirmations.append(alert)
            logger.info(
                f"Confirmation detected: {ticker} {direction} - "
                f"{len(systems)} systems, {len(layers)} layers, strength={strength}"
            )

        return confirmations

    def _get_layer(self, signal_type: str) -> str:
        """Determine layer from signal type"""
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

    def get_strongest(
        self,
        confirmations: list[ConfirmationAlert],
    ) -> Optional[ConfirmationAlert]:
        """Get the strongest confirmation"""
        if not confirmations:
            return None

        return max(
            confirmations,
            key=lambda c: (
                c.combined_confidence * c.system_count * (c.layer_count ** 0.5)
            )
        )
