"""
S++ Tier Consensus Cascade
==========================

Tracks when multiple systems agree on a signal and measures
the conviction boost from cross-system confirmation.

Key Features:
- Identifies consensus signals (3+ systems agreeing)
- Tracks consensus performance vs single-system signals
- Implements conviction cascade (more systems = higher weight)
- Provides consensus-adjusted confidence scores

This is what separates amateur from institutional quant.
"""

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Set, Tuple
from enum import Enum

logger = logging.getLogger(__name__)


class ConsensusLevel(str, Enum):
    """Levels of cross-system agreement"""
    NONE = "NONE"           # 1 system
    WEAK = "WEAK"           # 2 systems
    MODERATE = "MODERATE"   # 3 systems
    STRONG = "STRONG"       # 4-5 systems
    UNANIMOUS = "UNANIMOUS" # 6+ systems


@dataclass
class ConsensusResult:
    """Result of consensus detection"""
    ticker: str
    direction: str
    systems: List[str]
    layers: Set[str]
    level: ConsensusLevel

    # Aggregated metrics
    avg_confidence: float
    max_confidence: float
    min_confidence: float
    confidence_spread: float

    # Cascade-adjusted confidence
    cascade_confidence: float
    cascade_multiplier: float

    # Cross-layer validation
    layers_agreeing: int
    cross_layer_bonus: float

    # Signal details
    signals: List[Dict[str, Any]]
    timestamp: datetime = field(default_factory=datetime.utcnow)


class ConsensusCascade:
    """
    S++ Consensus Detection and Cascade Logic

    Implements the conviction cascade:
    - 2 systems: 1.1x confidence multiplier
    - 3 systems: 1.25x confidence multiplier
    - 4 systems: 1.4x confidence multiplier
    - 5+ systems: 1.5x confidence multiplier
    - Cross-layer bonus: +0.1 per additional layer

    Maximum cascade confidence: 0.98
    """

    # Cascade multipliers by system count
    CASCADE_MULTIPLIERS = {
        1: 1.0,
        2: 1.1,
        3: 1.25,
        4: 1.4,
        5: 1.5,
    }

    # Cross-layer bonus per additional layer
    CROSS_LAYER_BONUS = 0.1

    # Maximum final confidence
    MAX_CONFIDENCE = 0.98

    # Minimum systems for consensus
    MIN_CONSENSUS = 3

    # Layer mappings
    SYSTEM_LAYERS = {
        # MACRO layer
        "sornette": "MACRO",
        "minsky": "MACRO",
        "murphy": "MACRO",
        "cgma": "MACRO",
        "gkhy": "MACRO",
        "nations": "MACRO",

        # RISK layer
        "sentinel": "RISK",
        "frm": "RISK",
        "greeks": "RISK",
        "vixlab": "RISK",
        "squeeze": "RISK",

        # ALPHA layer
        "scanner": "ALPHA",
        "elite_scanner": "ALPHA",
        "wyckoff": "ALPHA",
        "hurst": "ALPHA",
        "strat": "ALPHA",
        "stealth": "ALPHA",
        "power_hour": "ALPHA",
        "capitol": "ALPHA",
        "bitcoin": "ALPHA",
    }

    def __init__(self):
        self.recent_signals: Dict[str, List[Dict]] = defaultdict(list)
        self.consensus_history: List[ConsensusResult] = []

    def _get_layer(self, system_id: str) -> str:
        """Determine layer for a system"""
        system_lower = system_id.lower()

        for keyword, layer in self.SYSTEM_LAYERS.items():
            if keyword in system_lower:
                return layer

        return "ALPHA"  # Default

    def _get_cascade_multiplier(self, system_count: int) -> float:
        """Get cascade multiplier for system count"""
        if system_count >= 5:
            return self.CASCADE_MULTIPLIERS[5]
        return self.CASCADE_MULTIPLIERS.get(system_count, 1.0)

    def ingest_signal(self, signal: Dict[str, Any]) -> Optional[ConsensusResult]:
        """
        Ingest a new signal and check for consensus.

        Returns ConsensusResult if consensus is reached.
        """
        ticker = signal.get("ticker", "UNKNOWN")
        direction = signal.get("direction", "NEUTRAL")
        system_id = signal.get("system_id", "unknown")
        timestamp = signal.get("timestamp", datetime.utcnow().isoformat())

        # Skip neutral signals for consensus
        if direction == "NEUTRAL":
            return None

        # Create signal key (ticker + direction)
        key = f"{ticker}:{direction}"

        # Add to recent signals
        signal_entry = {
            **signal,
            "layer": self._get_layer(system_id),
            "ingested_at": datetime.utcnow(),
        }
        self.recent_signals[key].append(signal_entry)

        # Clean old signals (keep last 5 minutes)
        self._clean_old_signals(key)

        # Check for consensus
        signals = self.recent_signals[key]
        if len(signals) >= self.MIN_CONSENSUS:
            return self._create_consensus(ticker, direction, signals)

        return None

    def _clean_old_signals(self, key: str, max_age_minutes: int = 5):
        """Remove signals older than max_age"""
        cutoff = datetime.utcnow() - timedelta(minutes=max_age_minutes)

        self.recent_signals[key] = [
            s for s in self.recent_signals[key]
            if s.get("ingested_at", datetime.utcnow()) > cutoff
        ]

    def _create_consensus(
        self,
        ticker: str,
        direction: str,
        signals: List[Dict],
    ) -> ConsensusResult:
        """Create a consensus result from agreeing signals"""
        # Get unique systems
        systems = list(set(s.get("system_id", "unknown") for s in signals))

        # Get layers
        layers = set(s.get("layer", "ALPHA") for s in signals)

        # Calculate level
        if len(systems) >= 6:
            level = ConsensusLevel.UNANIMOUS
        elif len(systems) >= 4:
            level = ConsensusLevel.STRONG
        elif len(systems) >= 3:
            level = ConsensusLevel.MODERATE
        elif len(systems) >= 2:
            level = ConsensusLevel.WEAK
        else:
            level = ConsensusLevel.NONE

        # Calculate confidence metrics
        confidences = [s.get("confidence", 0.5) for s in signals]
        avg_conf = sum(confidences) / len(confidences)
        max_conf = max(confidences)
        min_conf = min(confidences)

        # Apply cascade multiplier
        multiplier = self._get_cascade_multiplier(len(systems))

        # Apply cross-layer bonus
        cross_layer_bonus = (len(layers) - 1) * self.CROSS_LAYER_BONUS

        # Calculate cascade confidence
        cascade_conf = min(
            self.MAX_CONFIDENCE,
            avg_conf * multiplier + cross_layer_bonus
        )

        result = ConsensusResult(
            ticker=ticker,
            direction=direction,
            systems=systems,
            layers=layers,
            level=level,
            avg_confidence=round(avg_conf, 3),
            max_confidence=round(max_conf, 3),
            min_confidence=round(min_conf, 3),
            confidence_spread=round(max_conf - min_conf, 3),
            cascade_confidence=round(cascade_conf, 3),
            cascade_multiplier=multiplier,
            layers_agreeing=len(layers),
            cross_layer_bonus=round(cross_layer_bonus, 2),
            signals=signals,
        )

        self.consensus_history.append(result)
        logger.info(
            f"Consensus detected: {ticker} {direction} | "
            f"{len(systems)} systems | {level.value} | "
            f"cascade_conf={cascade_conf:.2f}"
        )

        return result

    def find_consensus(
        self,
        signals: List[Dict[str, Any]],
        time_window_minutes: int = 60,
    ) -> List[ConsensusResult]:
        """
        Find all consensus signals from a batch of signals.

        Groups by ticker+direction and identifies where 3+ systems agree.
        """
        # Group by ticker + direction
        grouped: Dict[str, List[Dict]] = defaultdict(list)

        for signal in signals:
            ticker = signal.get("ticker", "UNKNOWN")
            direction = signal.get("direction", "NEUTRAL")

            if direction == "NEUTRAL":
                continue

            key = f"{ticker}:{direction}"
            signal_entry = {
                **signal,
                "layer": self._get_layer(signal.get("system_id", "")),
            }
            grouped[key].append(signal_entry)

        # Find consensus groups
        results = []

        for key, group_signals in grouped.items():
            # Count unique systems
            systems = set(s.get("system_id", "unknown") for s in group_signals)

            if len(systems) >= self.MIN_CONSENSUS:
                ticker, direction = key.split(":")
                result = self._create_consensus(ticker, direction, group_signals)
                results.append(result)

        return results

    def get_consensus_signals(
        self,
        min_level: ConsensusLevel = ConsensusLevel.MODERATE,
    ) -> List[ConsensusResult]:
        """Get recent consensus signals above a minimum level"""
        level_order = [
            ConsensusLevel.NONE,
            ConsensusLevel.WEAK,
            ConsensusLevel.MODERATE,
            ConsensusLevel.STRONG,
            ConsensusLevel.UNANIMOUS,
        ]

        min_idx = level_order.index(min_level)

        return [
            c for c in self.consensus_history
            if level_order.index(c.level) >= min_idx
        ]

    def get_consensus_summary(self) -> Dict[str, Any]:
        """Get summary of consensus detection"""
        if not self.consensus_history:
            return {
                "total_consensus": 0,
                "by_level": {},
                "avg_cascade_multiplier": 1.0,
                "avg_confidence_boost": 0.0,
            }

        by_level = defaultdict(int)
        for c in self.consensus_history:
            by_level[c.level.value] += 1

        avg_multiplier = sum(c.cascade_multiplier for c in self.consensus_history) / len(self.consensus_history)
        avg_boost = sum(
            c.cascade_confidence - c.avg_confidence
            for c in self.consensus_history
        ) / len(self.consensus_history)

        cross_layer = len([
            c for c in self.consensus_history if c.layers_agreeing >= 2
        ])

        return {
            "total_consensus": len(self.consensus_history),
            "by_level": dict(by_level),
            "avg_cascade_multiplier": round(avg_multiplier, 2),
            "avg_confidence_boost": round(avg_boost, 3),
            "cross_layer_consensus": cross_layer,
        }


class CrossLayerValidator:
    """
    Validates signals across layers (MACRO, RISK, ALPHA).

    S++ tier requires cross-layer validation to reduce false signals.
    """

    # Layer alignment rules
    ALIGNMENT_RULES = {
        # MACRO bearish + RISK high + ALPHA bearish = Strong short
        ("BEARISH", "HIGH", "BEARISH"): ("SHORT", 1.5),
        # MACRO bullish + RISK low + ALPHA bullish = Strong long
        ("BULLISH", "LOW", "BULLISH"): ("LONG", 1.5),
        # MACRO neutral + RISK low + ALPHA bullish = Moderate long
        ("NEUTRAL", "LOW", "BULLISH"): ("LONG", 1.2),
        # MACRO bearish + RISK high + ALPHA bullish = CONTRADICTION
        ("BEARISH", "HIGH", "BULLISH"): ("CONTRADICTION", 0.5),
        # MACRO bullish + RISK low + ALPHA bearish = CONTRADICTION
        ("BULLISH", "LOW", "BEARISH"): ("CONTRADICTION", 0.5),
    }

    def __init__(self):
        pass

    def classify_signal(self, signal: Dict[str, Any]) -> str:
        """Classify signal as BULLISH, BEARISH, or NEUTRAL"""
        direction = signal.get("direction", "NEUTRAL")
        if direction in ["BULLISH", "LONG"]:
            return "BULLISH"
        elif direction in ["BEARISH", "SHORT"]:
            return "BEARISH"
        return "NEUTRAL"

    def classify_risk(self, risk_score: float) -> str:
        """Classify risk level"""
        if risk_score >= 60:
            return "HIGH"
        elif risk_score <= 30:
            return "LOW"
        return "MODERATE"

    def validate_cross_layer(
        self,
        macro_signal: Optional[Dict] = None,
        risk_score: float = 50,
        alpha_signal: Optional[Dict] = None,
    ) -> Tuple[str, float, str]:
        """
        Validate signal across layers.

        Returns:
            Tuple of (recommendation, multiplier, rationale)
        """
        # Classify each layer
        macro_class = self.classify_signal(macro_signal) if macro_signal else "NEUTRAL"
        risk_class = self.classify_risk(risk_score)
        alpha_class = self.classify_signal(alpha_signal) if alpha_signal else "NEUTRAL"

        key = (macro_class, risk_class, alpha_class)

        # Check alignment rules
        if key in self.ALIGNMENT_RULES:
            recommendation, multiplier = self.ALIGNMENT_RULES[key]
            return (
                recommendation,
                multiplier,
                f"Macro={macro_class}, Risk={risk_class}, Alpha={alpha_class}"
            )

        # Default: neutral stance
        # Count agreements
        signals = [macro_class, alpha_class]
        bullish = signals.count("BULLISH")
        bearish = signals.count("BEARISH")

        if bullish > bearish:
            return ("CAUTIOUS_LONG", 0.8, "Partial bullish alignment")
        elif bearish > bullish:
            return ("CAUTIOUS_SHORT", 0.8, "Partial bearish alignment")
        else:
            return ("NEUTRAL", 1.0, "No clear alignment")


def run_consensus_test():
    """Test consensus detection with sample signals"""
    cascade = ConsensusCascade()

    # Sample signals - 4 systems agree on NVDA BULLISH
    signals = [
        {"system_id": "sornette", "ticker": "NVDA", "direction": "BULLISH", "confidence": 0.7},
        {"system_id": "murphy", "ticker": "NVDA", "direction": "BULLISH", "confidence": 0.75},
        {"system_id": "elite_scanner", "ticker": "NVDA", "direction": "BULLISH", "confidence": 0.8},
        {"system_id": "wyckoff", "ticker": "NVDA", "direction": "BULLISH", "confidence": 0.72},
        # Different ticker
        {"system_id": "sornette", "ticker": "AAPL", "direction": "BEARISH", "confidence": 0.65},
        {"system_id": "minsky", "ticker": "AAPL", "direction": "BEARISH", "confidence": 0.7},
    ]

    print("=" * 70)
    print("CONSENSUS CASCADE TEST")
    print("=" * 70)

    # Find consensus
    results = cascade.find_consensus(signals)

    for r in results:
        print(f"\n{r.ticker} {r.direction}:")
        print(f"  Systems: {r.systems}")
        print(f"  Layers: {r.layers}")
        print(f"  Level: {r.level.value}")
        print(f"  Avg Confidence: {r.avg_confidence:.2f}")
        print(f"  Cascade Confidence: {r.cascade_confidence:.2f}")
        print(f"  Multiplier: {r.cascade_multiplier}x")
        print(f"  Cross-Layer Bonus: {r.cross_layer_bonus}")

    print("\n" + "-" * 70)
    print("SUMMARY")
    print("-" * 70)
    summary = cascade.get_consensus_summary()
    for k, v in summary.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    run_consensus_test()
