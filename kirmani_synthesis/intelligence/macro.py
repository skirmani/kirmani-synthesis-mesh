"""
Macro Intelligence Synthesis

Combines signals from:
- Sornette-LPPLS: Crash hazard probability, critical time estimates
- Minsky-Analyzer: 5-stage bubble phase, bubble score
- Murphy-Intermarket: Asset class relationships, regime classification
- CGMA-System: Global macro factors, geopolitical risk

Output: Unified crash hazard score with regime classification
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional
from collections import defaultdict

logger = logging.getLogger(__name__)


class MacroRegime(str, Enum):
    """Macro market regime classification"""
    EXPANSION = "EXPANSION"           # Growth phase, low risk
    LATE_CYCLE = "LATE_CYCLE"         # Late expansion, rising risk
    BUBBLE = "BUBBLE"                 # Bubble formation detected
    CRISIS_WARNING = "CRISIS_WARNING" # High crash probability
    CRISIS = "CRISIS"                 # Active crisis/crash
    RECOVERY = "RECOVERY"             # Post-crisis recovery


class CrashTimeframe(str, Enum):
    """Estimated timeframe for potential crash"""
    IMMINENT = "IMMINENT"       # Within days
    SHORT_TERM = "SHORT_TERM"   # Within weeks
    MEDIUM_TERM = "MEDIUM_TERM" # Within months
    LONG_TERM = "LONG_TERM"     # 6+ months out
    NONE = "NONE"               # No crash signal


@dataclass
class MacroIntelligenceReport:
    """Unified macro intelligence output"""
    # Core metrics
    unified_crash_hazard: float          # 0-1 probability
    regime: MacroRegime
    regime_confidence: float             # 0-1

    # Bubble analysis
    bubble_phase: int                    # Minsky 1-5
    bubble_score: float                  # 0-100
    bubble_systems_agreeing: int         # Count of systems seeing bubble

    # Crash timing
    crash_timeframe: CrashTimeframe
    critical_time_estimate: Optional[str] # Date string if available

    # Intermarket
    intermarket_regime: str              # RISK_ON, RISK_OFF, NEUTRAL
    asset_class_divergences: list[str]   # Notable divergences

    # Geopolitical
    geopolitical_risk_score: float       # 0-100
    geopolitical_hotspots: list[str]

    # System contributions
    sornette_hazard: float
    minsky_phase: int
    murphy_regime: str
    cgma_risk: float

    # Meta
    systems_reporting: list[str]
    confidence_weights: dict[str, float]

    # Recommendations
    recommended_equity_exposure: float   # 0-1.5 multiplier
    hedge_recommendation: str

    # Generated timestamp (with default)
    generated_at: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> dict:
        return {
            "unified_crash_hazard": self.unified_crash_hazard,
            "regime": self.regime.value,
            "regime_confidence": self.regime_confidence,
            "bubble_phase": self.bubble_phase,
            "bubble_score": self.bubble_score,
            "bubble_systems_agreeing": self.bubble_systems_agreeing,
            "crash_timeframe": self.crash_timeframe.value,
            "critical_time_estimate": self.critical_time_estimate,
            "intermarket_regime": self.intermarket_regime,
            "asset_class_divergences": self.asset_class_divergences,
            "geopolitical_risk_score": self.geopolitical_risk_score,
            "geopolitical_hotspots": self.geopolitical_hotspots,
            "sornette_hazard": self.sornette_hazard,
            "minsky_phase": self.minsky_phase,
            "murphy_regime": self.murphy_regime,
            "cgma_risk": self.cgma_risk,
            "systems_reporting": self.systems_reporting,
            "confidence_weights": self.confidence_weights,
            "generated_at": self.generated_at.isoformat(),
            "recommended_equity_exposure": self.recommended_equity_exposure,
            "hedge_recommendation": self.hedge_recommendation,
        }


class MacroIntelligence:
    """
    Macro Intelligence Synthesis Engine

    Combines Sornette, Minsky, Murphy, and CGMA signals into unified
    macro intelligence with crash hazard scoring and regime classification.

    Usage:
        macro = MacroIntelligence()
        report = macro.synthesize(signals)
    """

    # System IDs we look for
    SORNETTE_SYSTEMS = {"sornette-lppls", "sornette", "lppls"}
    MINSKY_SYSTEMS = {"minsky-analyzer", "minsky", "minsky-bubble"}
    MURPHY_SYSTEMS = {"murphy-intermarket", "murphy", "intermarket"}
    CGMA_SYSTEMS = {"cgma-system", "cgma", "global-macro"}
    BUBBLE_SYSTEMS = {"bubble-ensemble", "bubble-radar", "bubble-detector", "nations-framework"}

    # Confidence weights (calibrated from backtests)
    DEFAULT_WEIGHTS = {
        "sornette": 0.30,   # Strong crash timing signal
        "minsky": 0.25,     # Strong bubble phase signal
        "murphy": 0.25,     # Intermarket confirmation
        "cgma": 0.20,       # Macro backdrop
    }

    def __init__(self, weights: Optional[dict] = None):
        self.weights = weights or self.DEFAULT_WEIGHTS.copy()

    def synthesize(self, signals: list[dict]) -> MacroIntelligenceReport:
        """
        Synthesize macro intelligence from system signals

        Args:
            signals: List of signal dictionaries from macro systems

        Returns:
            MacroIntelligenceReport with unified intelligence
        """
        # Extract signals by system type
        sornette_signals = self._extract_signals(signals, self.SORNETTE_SYSTEMS)
        minsky_signals = self._extract_signals(signals, self.MINSKY_SYSTEMS)
        murphy_signals = self._extract_signals(signals, self.MURPHY_SYSTEMS)
        cgma_signals = self._extract_signals(signals, self.CGMA_SYSTEMS)
        bubble_signals = self._extract_signals(signals, self.BUBBLE_SYSTEMS)

        # Combine bubble signals with Minsky
        all_bubble_signals = minsky_signals + bubble_signals

        # Extract key metrics from each system
        sornette_hazard = self._get_sornette_hazard(sornette_signals)
        minsky_phase, bubble_score = self._get_minsky_metrics(all_bubble_signals)
        murphy_regime, divergences = self._get_murphy_metrics(murphy_signals)
        cgma_risk, hotspots = self._get_cgma_metrics(cgma_signals)

        # Synthesize unified crash hazard
        unified_hazard = self._calculate_unified_hazard(
            sornette_hazard, minsky_phase, murphy_regime, cgma_risk
        )

        # Determine regime
        regime, regime_confidence = self._classify_regime(
            unified_hazard, minsky_phase, murphy_regime
        )

        # Determine crash timeframe
        crash_timeframe, critical_time = self._estimate_crash_timing(
            sornette_signals, unified_hazard
        )

        # Count bubble agreements
        bubble_agreeing = self._count_bubble_agreements(all_bubble_signals)

        # Calculate recommendations
        equity_exposure = self._recommend_equity_exposure(unified_hazard, regime)
        hedge_rec = self._recommend_hedge(unified_hazard, regime, minsky_phase)

        # Build systems reporting list
        systems_reporting = []
        if sornette_signals:
            systems_reporting.append("sornette-lppls")
        if minsky_signals:
            systems_reporting.append("minsky-analyzer")
        if murphy_signals:
            systems_reporting.append("murphy-intermarket")
        if cgma_signals:
            systems_reporting.append("cgma-system")
        if bubble_signals:
            systems_reporting.extend([s.get("system_id") for s in bubble_signals[:2]])

        return MacroIntelligenceReport(
            unified_crash_hazard=unified_hazard,
            regime=regime,
            regime_confidence=regime_confidence,
            bubble_phase=minsky_phase,
            bubble_score=bubble_score,
            bubble_systems_agreeing=bubble_agreeing,
            crash_timeframe=crash_timeframe,
            critical_time_estimate=critical_time,
            intermarket_regime=murphy_regime,
            asset_class_divergences=divergences,
            geopolitical_risk_score=cgma_risk,
            geopolitical_hotspots=hotspots,
            sornette_hazard=sornette_hazard,
            minsky_phase=minsky_phase,
            murphy_regime=murphy_regime,
            cgma_risk=cgma_risk,
            systems_reporting=systems_reporting,
            confidence_weights=self.weights,
            recommended_equity_exposure=equity_exposure,
            hedge_recommendation=hedge_rec,
        )

    def _extract_signals(self, signals: list[dict], system_ids: set) -> list[dict]:
        """Extract signals from specific systems"""
        return [
            s for s in signals
            if s.get("system_id", "").lower() in system_ids
            or any(sid in s.get("system_id", "").lower() for sid in system_ids)
        ]

    def _get_sornette_hazard(self, signals: list[dict]) -> float:
        """Extract crash hazard from Sornette signals"""
        if not signals:
            return 0.0

        # Look for hazard_rate in metadata
        hazards = []
        for sig in signals:
            meta = sig.get("metadata", {})
            if "hazard_rate" in meta:
                hazards.append(meta["hazard_rate"])
            elif "crash_probability" in meta:
                hazards.append(meta["crash_probability"])
            elif sig.get("signal_type") == "CRASH_HAZARD":
                hazards.append(sig.get("confidence", 0.5))

        return max(hazards) if hazards else 0.0

    def _get_minsky_metrics(self, signals: list[dict]) -> tuple[int, float]:
        """Extract bubble phase and score from Minsky signals"""
        if not signals:
            return 1, 0.0

        phases = []
        scores = []

        for sig in signals:
            meta = sig.get("metadata", {})
            if "phase" in meta:
                phases.append(meta["phase"])
            if "bubble_score" in meta:
                scores.append(meta["bubble_score"])
            elif "bubble_phase" in meta:
                phases.append(meta["bubble_phase"])

        phase = max(phases) if phases else 1
        score = max(scores) if scores else phase * 20  # Estimate from phase

        return phase, score

    def _get_murphy_metrics(self, signals: list[dict]) -> tuple[str, list[str]]:
        """Extract intermarket regime and divergences from Murphy signals"""
        if not signals:
            return "NEUTRAL", []

        regimes = []
        divergences = []

        for sig in signals:
            meta = sig.get("metadata", {})
            if "regime" in meta:
                regimes.append(meta["regime"])
            if "divergences" in meta:
                divergences.extend(meta["divergences"])
            if meta.get("bond_equity_divergence"):
                divergences.append("Bond-Equity Divergence")
            if meta.get("dollar_commodity_divergence"):
                divergences.append("Dollar-Commodity Divergence")

        # Determine regime by majority
        if regimes:
            regime_counts = defaultdict(int)
            for r in regimes:
                regime_counts[r.upper()] += 1
            regime = max(regime_counts, key=regime_counts.get)
        else:
            regime = "NEUTRAL"

        return regime, list(set(divergences))

    def _get_cgma_metrics(self, signals: list[dict]) -> tuple[float, list[str]]:
        """Extract geopolitical risk from CGMA signals"""
        if not signals:
            return 0.0, []

        risks = []
        hotspots = []

        for sig in signals:
            meta = sig.get("metadata", {})
            if "geopolitical_risk" in meta:
                risks.append(meta["geopolitical_risk"])
            if "risk_score" in meta:
                risks.append(meta["risk_score"])
            if "hotspots" in meta:
                hotspots.extend(meta["hotspots"])
            if "conflict_zones" in meta:
                hotspots.extend(meta["conflict_zones"])

        risk = max(risks) if risks else 0.0
        return risk, list(set(hotspots))

    def _calculate_unified_hazard(
        self,
        sornette: float,
        minsky_phase: int,
        murphy_regime: str,
        cgma_risk: float,
    ) -> float:
        """
        Calculate unified crash hazard using Bayesian-weighted fusion

        Formula:
        hazard = w_s * sornette + w_m * minsky_norm + w_mu * murphy_adj + w_c * cgma_norm

        With adjustments for cross-confirmation
        """
        # Normalize Minsky phase to 0-1
        minsky_norm = (minsky_phase - 1) / 4  # Phase 1-5 -> 0-1

        # Murphy regime adjustment
        murphy_adj = {
            "RISK_OFF": 0.7,
            "CRISIS": 1.0,
            "NEUTRAL": 0.3,
            "RISK_ON": 0.1,
        }.get(murphy_regime.upper(), 0.3)

        # Normalize CGMA risk
        cgma_norm = min(1.0, cgma_risk / 100)

        # Weighted combination
        hazard = (
            self.weights["sornette"] * sornette +
            self.weights["minsky"] * minsky_norm +
            self.weights["murphy"] * murphy_adj +
            self.weights["cgma"] * cgma_norm
        )

        # Cross-confirmation boost
        # If multiple systems agree on high risk, boost the signal
        high_risk_count = sum([
            sornette >= 0.6,
            minsky_phase >= 4,
            murphy_regime.upper() in ("RISK_OFF", "CRISIS"),
            cgma_risk >= 60,
        ])

        if high_risk_count >= 3:
            hazard = min(1.0, hazard * 1.3)  # 30% boost for triple confirmation
        elif high_risk_count >= 2:
            hazard = min(1.0, hazard * 1.15)  # 15% boost for double confirmation

        return round(hazard, 3)

    def _classify_regime(
        self,
        hazard: float,
        minsky_phase: int,
        murphy_regime: str,
    ) -> tuple[MacroRegime, float]:
        """Classify macro regime based on signals"""

        # Crisis detection
        if hazard >= 0.8 or murphy_regime.upper() == "CRISIS":
            return MacroRegime.CRISIS, 0.9

        # Crisis warning
        if hazard >= 0.6 or (minsky_phase >= 4 and hazard >= 0.4):
            return MacroRegime.CRISIS_WARNING, 0.8

        # Bubble
        if minsky_phase >= 3 and hazard >= 0.3:
            return MacroRegime.BUBBLE, 0.75

        # Late cycle
        if minsky_phase >= 2 or murphy_regime.upper() == "RISK_OFF":
            return MacroRegime.LATE_CYCLE, 0.7

        # Recovery
        if murphy_regime.upper() == "RISK_ON" and minsky_phase == 1:
            return MacroRegime.RECOVERY, 0.65

        # Default: Expansion
        return MacroRegime.EXPANSION, 0.6

    def _estimate_crash_timing(
        self,
        sornette_signals: list[dict],
        hazard: float,
    ) -> tuple[CrashTimeframe, Optional[str]]:
        """Estimate crash timing from Sornette critical time"""

        if hazard < 0.3:
            return CrashTimeframe.NONE, None

        # Look for tc estimate in Sornette signals
        tc_estimate = None
        for sig in sornette_signals:
            meta = sig.get("metadata", {})
            if "tc_estimate" in meta:
                tc_estimate = meta["tc_estimate"]
                break
            if "critical_time" in meta:
                tc_estimate = meta["critical_time"]
                break

        # Determine timeframe from hazard level
        if hazard >= 0.8:
            timeframe = CrashTimeframe.IMMINENT
        elif hazard >= 0.6:
            timeframe = CrashTimeframe.SHORT_TERM
        elif hazard >= 0.4:
            timeframe = CrashTimeframe.MEDIUM_TERM
        else:
            timeframe = CrashTimeframe.LONG_TERM

        return timeframe, tc_estimate

    def _count_bubble_agreements(self, signals: list[dict]) -> int:
        """Count how many systems see bubble conditions"""
        count = 0
        seen_systems = set()

        for sig in signals:
            system_id = sig.get("system_id", "")
            if system_id in seen_systems:
                continue

            meta = sig.get("metadata", {})
            phase = meta.get("phase", meta.get("bubble_phase", 0))
            score = meta.get("bubble_score", 0)

            if phase >= 3 or score >= 60:
                count += 1
                seen_systems.add(system_id)

        return count

    def _recommend_equity_exposure(self, hazard: float, regime: MacroRegime) -> float:
        """Recommend equity exposure multiplier"""
        base_exposure = {
            MacroRegime.EXPANSION: 1.2,
            MacroRegime.RECOVERY: 1.1,
            MacroRegime.LATE_CYCLE: 0.9,
            MacroRegime.BUBBLE: 0.7,
            MacroRegime.CRISIS_WARNING: 0.5,
            MacroRegime.CRISIS: 0.3,
        }.get(regime, 1.0)

        # Adjust for hazard
        hazard_adj = 1.0 - (hazard * 0.5)  # High hazard reduces exposure

        return round(base_exposure * hazard_adj, 2)

    def _recommend_hedge(
        self,
        hazard: float,
        regime: MacroRegime,
        minsky_phase: int,
    ) -> str:
        """Generate hedge recommendation"""

        if regime == MacroRegime.CRISIS:
            return "MAXIMUM HEDGE: Long VIX calls, put spreads on SPY, reduce to 30% equity"

        if regime == MacroRegime.CRISIS_WARNING:
            return "DEFENSIVE: Buy VIX calls, collar existing positions, raise cash to 40%"

        if regime == MacroRegime.BUBBLE and minsky_phase >= 4:
            return "CAUTION: Buy protective puts, reduce concentrated positions, hedge 50%"

        if hazard >= 0.5:
            return "MODERATE: Consider put protection on largest positions"

        if regime == MacroRegime.LATE_CYCLE:
            return "MILD: Maintain modest put protection, rebalance to defensive sectors"

        return "NONE: No immediate hedge needed, maintain normal risk parameters"
