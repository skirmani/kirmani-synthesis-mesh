"""
Cross-System Intelligence Engine

The master orchestrator for Phase 3 intelligence synthesis.
Combines Macro, Risk, and Alpha intelligence into unified
actionable intelligence with cross-layer validation.

This is the brain of the Kirmani Synthesis Mesh.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional

from .macro import MacroIntelligence, MacroIntelligenceReport, MacroRegime
from .risk import RiskIntelligence, RiskIntelligenceReport, RiskLevel, VolRegime
from .alpha import AlphaIntelligence, AlphaIntelligenceReport, AlphaSignal, SignalStrength

logger = logging.getLogger(__name__)


class MarketPosture(str, Enum):
    """Overall market posture recommendation"""
    FULL_DEFENSE = "FULL_DEFENSE"       # Crisis mode, max protection
    DEFENSIVE = "DEFENSIVE"              # Elevated risk, reduce exposure
    CAUTIOUS = "CAUTIOUS"               # Some concerns, hedge core
    NEUTRAL = "NEUTRAL"                 # Normal operations
    OPPORTUNISTIC = "OPPORTUNISTIC"     # Good conditions, seek alpha
    AGGRESSIVE = "AGGRESSIVE"           # Strong conditions, max exposure


class ActionPriority(str, Enum):
    """Priority level for recommended actions"""
    CRITICAL = "CRITICAL"   # Execute immediately
    HIGH = "HIGH"           # Execute today
    MEDIUM = "MEDIUM"       # Execute this week
    LOW = "LOW"             # Execute when convenient
    MONITOR = "MONITOR"     # Watch and wait


@dataclass
class IntelligenceAction:
    """Specific action recommendation from intelligence synthesis"""
    action_type: str              # HEDGE, ENTER, EXIT, REDUCE, INCREASE, ROTATE
    priority: ActionPriority
    ticker: Optional[str]
    direction: Optional[str]      # LONG, SHORT, NEUTRAL

    description: str
    rationale: str
    source_layers: list[str]      # MACRO, RISK, ALPHA

    position_change_pct: Optional[float]
    target_price: Optional[float]
    stop_loss: Optional[float]

    confidence: float
    cross_validation_score: float  # How many layers agree

    def to_dict(self) -> dict:
        return {
            "action_type": self.action_type,
            "priority": self.priority.value,
            "ticker": self.ticker,
            "direction": self.direction,
            "description": self.description,
            "rationale": self.rationale,
            "source_layers": self.source_layers,
            "position_change_pct": self.position_change_pct,
            "target_price": self.target_price,
            "stop_loss": self.stop_loss,
            "confidence": self.confidence,
            "cross_validation_score": self.cross_validation_score,
        }


@dataclass
class UnifiedIntelligenceReport:
    """
    The master intelligence report combining all layers.

    This is the primary output of Phase 3 Cross-System Intelligence.
    """
    # Overall assessment
    market_posture: MarketPosture
    posture_confidence: float
    posture_rationale: str

    # Component reports
    macro_intelligence: MacroIntelligenceReport
    risk_intelligence: RiskIntelligenceReport
    alpha_intelligence: AlphaIntelligenceReport

    # Unified metrics
    unified_risk_score: float          # 0-100, weighted from all layers
    unified_opportunity_score: float   # 0-100, alpha potential
    risk_reward_balance: float         # -1 to +1, negative = risk dominant

    # Cross-layer validations
    macro_risk_alignment: float        # 0-1, how well macro and risk agree
    risk_alpha_alignment: float        # 0-1, how well risk and alpha agree
    full_stack_alignment: float        # 0-1, all three layers agree

    # Recommended actions (priority ordered)
    actions: list[IntelligenceAction]
    critical_actions: list[IntelligenceAction]  # CRITICAL priority only

    # Position guidance
    recommended_equity_exposure: float  # 0-1.5 multiplier
    recommended_cash_level: float       # 0-1
    max_single_position_pct: float      # Max % in any single name

    # Hedging
    hedge_ratio: float                  # 0-1, portion of portfolio to hedge
    recommended_hedges: list[str]

    # Meta
    systems_contributing: list[str]
    total_signals_processed: int
    intelligence_confidence: float      # Overall confidence in this report
    generated_at: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> dict:
        return {
            "market_posture": self.market_posture.value,
            "posture_confidence": self.posture_confidence,
            "posture_rationale": self.posture_rationale,
            "macro_intelligence": self.macro_intelligence.to_dict(),
            "risk_intelligence": self.risk_intelligence.to_dict(),
            "alpha_intelligence": self.alpha_intelligence.to_dict(),
            "unified_risk_score": self.unified_risk_score,
            "unified_opportunity_score": self.unified_opportunity_score,
            "risk_reward_balance": self.risk_reward_balance,
            "macro_risk_alignment": self.macro_risk_alignment,
            "risk_alpha_alignment": self.risk_alpha_alignment,
            "full_stack_alignment": self.full_stack_alignment,
            "actions": [a.to_dict() for a in self.actions],
            "critical_actions": [a.to_dict() for a in self.critical_actions],
            "recommended_equity_exposure": self.recommended_equity_exposure,
            "recommended_cash_level": self.recommended_cash_level,
            "max_single_position_pct": self.max_single_position_pct,
            "hedge_ratio": self.hedge_ratio,
            "recommended_hedges": self.recommended_hedges,
            "systems_contributing": self.systems_contributing,
            "total_signals_processed": self.total_signals_processed,
            "intelligence_confidence": self.intelligence_confidence,
            "generated_at": self.generated_at.isoformat(),
        }

    def print_summary(self) -> None:
        """Print human-readable summary"""
        print("\n" + "=" * 70)
        print("KIRMANI CROSS-SYSTEM INTELLIGENCE REPORT")
        print("=" * 70)
        print(f"\nGenerated: {self.generated_at.strftime('%Y-%m-%d %H:%M:%S UTC')}")
        print(f"Signals Processed: {self.total_signals_processed}")
        print(f"Systems Contributing: {len(self.systems_contributing)}")

        print(f"\n{'=' * 70}")
        print("MARKET POSTURE")
        print(f"{'=' * 70}")
        print(f"Posture: {self.market_posture.value} (Confidence: {self.posture_confidence:.0%})")
        print(f"Rationale: {self.posture_rationale}")

        print(f"\n{'=' * 70}")
        print("UNIFIED METRICS")
        print(f"{'=' * 70}")
        print(f"Risk Score: {self.unified_risk_score:.0f}/100")
        print(f"Opportunity Score: {self.unified_opportunity_score:.0f}/100")
        print(f"Risk/Reward Balance: {self.risk_reward_balance:+.2f}")
        print(f"Full Stack Alignment: {self.full_stack_alignment:.0%}")

        print(f"\n{'=' * 70}")
        print("POSITION GUIDANCE")
        print(f"{'=' * 70}")
        print(f"Equity Exposure: {self.recommended_equity_exposure:.0%}")
        print(f"Cash Level: {self.recommended_cash_level:.0%}")
        print(f"Max Single Position: {self.max_single_position_pct:.0%}")
        print(f"Hedge Ratio: {self.hedge_ratio:.0%}")

        if self.critical_actions:
            print(f"\n{'=' * 70}")
            print(f"CRITICAL ACTIONS ({len(self.critical_actions)})")
            print(f"{'=' * 70}")
            for action in self.critical_actions:
                print(f"  [{action.priority.value}] {action.action_type}: {action.description}")

        if self.actions[:5]:
            print(f"\n{'=' * 70}")
            print("TOP RECOMMENDED ACTIONS")
            print(f"{'=' * 70}")
            for action in self.actions[:5]:
                ticker_str = f" {action.ticker}" if action.ticker else ""
                print(f"  [{action.priority.value}] {action.action_type}{ticker_str}: {action.description}")

        print("\n")


class CrossSystemIntelligenceEngine:
    """
    Cross-System Intelligence Engine

    The master orchestrator that combines Macro, Risk, and Alpha intelligence
    into unified actionable recommendations with cross-layer validation.

    This is Phase 3 of the Kirmani Synthesis Mesh integration.

    Usage:
        engine = CrossSystemIntelligenceEngine()
        report = engine.synthesize(signals)
        report.print_summary()
    """

    def __init__(self):
        self.macro = MacroIntelligence()
        self.risk = RiskIntelligence()
        self.alpha = AlphaIntelligence()

    def synthesize(self, signals: list[dict]) -> UnifiedIntelligenceReport:
        """
        Run complete cross-system intelligence synthesis

        Args:
            signals: List of signal dictionaries from all systems

        Returns:
            UnifiedIntelligenceReport with full intelligence synthesis
        """
        logger.info(f"Running cross-system intelligence on {len(signals)} signals...")

        # Run each intelligence layer
        macro_report = self.macro.synthesize(signals)
        logger.info(f"Macro: {macro_report.regime.value}, hazard={macro_report.unified_crash_hazard:.0%}")

        risk_report = self.risk.synthesize(signals)
        logger.info(f"Risk: {risk_report.risk_level.value}, score={risk_report.risk_score:.0f}")

        alpha_report = self.alpha.synthesize(signals)
        logger.info(f"Alpha: {alpha_report.total_opportunities} opportunities, {alpha_report.high_conviction_count} high-conviction")

        # Calculate unified metrics
        unified_risk = self._calculate_unified_risk(macro_report, risk_report)
        unified_opp = self._calculate_opportunity_score(alpha_report, risk_report)
        rr_balance = self._calculate_risk_reward_balance(unified_risk, unified_opp)

        # Cross-layer alignment
        macro_risk_align = self._calculate_macro_risk_alignment(macro_report, risk_report)
        risk_alpha_align = self._calculate_risk_alpha_alignment(risk_report, alpha_report)
        full_align = (macro_risk_align + risk_alpha_align) / 2

        # Determine market posture
        posture, posture_conf, posture_rationale = self._determine_posture(
            macro_report, risk_report, alpha_report, unified_risk, unified_opp
        )

        # Generate actions
        actions = self._generate_actions(
            macro_report, risk_report, alpha_report, posture
        )

        # Sort by priority
        actions.sort(key=lambda a: self._priority_value(a.priority), reverse=True)
        critical = [a for a in actions if a.priority == ActionPriority.CRITICAL]

        # Position guidance
        equity_exp = self._recommend_equity_exposure(posture, unified_risk)
        cash_level = self._recommend_cash_level(posture, unified_risk)
        max_pos = self._recommend_max_position(posture, risk_report)

        # Hedging
        hedge_ratio = self._calculate_hedge_ratio(posture, macro_report, risk_report)
        hedges = self._compile_hedge_recommendations(macro_report, risk_report)

        # Systems contributing
        systems = list(set(
            macro_report.systems_reporting +
            risk_report.systems_reporting +
            alpha_report.systems_reporting
        ))

        # Intelligence confidence
        intel_conf = self._calculate_intelligence_confidence(
            len(systems), full_align, len(signals)
        )

        return UnifiedIntelligenceReport(
            market_posture=posture,
            posture_confidence=posture_conf,
            posture_rationale=posture_rationale,
            macro_intelligence=macro_report,
            risk_intelligence=risk_report,
            alpha_intelligence=alpha_report,
            unified_risk_score=unified_risk,
            unified_opportunity_score=unified_opp,
            risk_reward_balance=rr_balance,
            macro_risk_alignment=macro_risk_align,
            risk_alpha_alignment=risk_alpha_align,
            full_stack_alignment=full_align,
            actions=actions,
            critical_actions=critical,
            recommended_equity_exposure=equity_exp,
            recommended_cash_level=cash_level,
            max_single_position_pct=max_pos,
            hedge_ratio=hedge_ratio,
            recommended_hedges=hedges,
            systems_contributing=systems,
            total_signals_processed=len(signals),
            intelligence_confidence=intel_conf,
        )

    def _calculate_unified_risk(
        self,
        macro: MacroIntelligenceReport,
        risk: RiskIntelligenceReport,
    ) -> float:
        """Calculate unified risk score from macro and risk layers"""
        # Macro contributes 40%, Risk contributes 60%
        macro_risk = macro.unified_crash_hazard * 100
        risk_score = risk.risk_score

        unified = (macro_risk * 0.4) + (risk_score * 0.6)

        # Amplify if both layers show high risk
        if macro_risk >= 60 and risk_score >= 60:
            unified = min(100, unified * 1.15)

        return round(unified, 1)

    def _calculate_opportunity_score(
        self,
        alpha: AlphaIntelligenceReport,
        risk: RiskIntelligenceReport,
    ) -> float:
        """Calculate opportunity score from alpha signals"""
        # Base from alpha signals
        if alpha.total_opportunities == 0:
            return 0

        # High conviction signals weighted heavily
        hc_ratio = alpha.high_conviction_count / max(1, alpha.total_opportunities)
        base_score = (hc_ratio * 60) + (min(alpha.total_opportunities, 20) * 2)

        # Adjust for risk (high risk dampens opportunity)
        risk_adj = 1.0 - (risk.risk_score / 200)  # 50% max reduction

        return round(base_score * risk_adj, 1)

    def _calculate_risk_reward_balance(
        self,
        risk: float,
        opportunity: float,
    ) -> float:
        """Calculate risk/reward balance (-1 to +1)"""
        # Normalize both to 0-1
        risk_norm = risk / 100
        opp_norm = opportunity / 100

        # Balance: positive = opportunity dominant, negative = risk dominant
        balance = opp_norm - risk_norm

        return round(max(-1, min(1, balance)), 2)

    def _calculate_macro_risk_alignment(
        self,
        macro: MacroIntelligenceReport,
        risk: RiskIntelligenceReport,
    ) -> float:
        """Calculate alignment between macro and risk layers"""
        # Both should agree on risk level
        macro_risk_level = macro.unified_crash_hazard

        # Map risk level to 0-1
        risk_level_map = {
            RiskLevel.MINIMAL: 0.1,
            RiskLevel.LOW: 0.25,
            RiskLevel.MODERATE: 0.5,
            RiskLevel.HIGH: 0.75,
            RiskLevel.CRITICAL: 0.95,
        }
        risk_risk_level = risk_level_map.get(risk.risk_level, 0.5)

        # Agreement = 1 - difference
        alignment = 1.0 - abs(macro_risk_level - risk_risk_level)

        return round(alignment, 2)

    def _calculate_risk_alpha_alignment(
        self,
        risk: RiskIntelligenceReport,
        alpha: AlphaIntelligenceReport,
    ) -> float:
        """Calculate alignment between risk and alpha layers"""
        # In low risk environments, alpha should be finding opportunities
        # In high risk environments, alpha should be cautious

        risk_norm = risk.risk_score / 100
        opp_norm = alpha.high_conviction_count / max(1, alpha.total_opportunities)

        # They're aligned if:
        # - Low risk + high opportunity
        # - High risk + low opportunity
        expected_opp = 1.0 - risk_norm
        alignment = 1.0 - abs(expected_opp - opp_norm)

        return round(alignment, 2)

    def _determine_posture(
        self,
        macro: MacroIntelligenceReport,
        risk: RiskIntelligenceReport,
        alpha: AlphaIntelligenceReport,
        unified_risk: float,
        unified_opp: float,
    ) -> tuple[MarketPosture, float, str]:
        """Determine overall market posture"""

        # Crisis detection
        if macro.regime == MacroRegime.CRISIS or risk.risk_level == RiskLevel.CRITICAL:
            return (
                MarketPosture.FULL_DEFENSE,
                0.95,
                f"Crisis conditions: {macro.regime.value}, risk={unified_risk:.0f}"
            )

        # Crisis warning
        if macro.regime == MacroRegime.CRISIS_WARNING or unified_risk >= 75:
            return (
                MarketPosture.DEFENSIVE,
                0.85,
                f"Elevated risk: hazard={macro.unified_crash_hazard:.0%}, vol={risk.vol_regime.value}"
            )

        # High risk but not crisis
        if unified_risk >= 60 or macro.regime == MacroRegime.BUBBLE:
            return (
                MarketPosture.CAUTIOUS,
                0.75,
                f"Caution warranted: risk={unified_risk:.0f}, bubble phase={macro.bubble_phase}"
            )

        # Good opportunities with moderate risk
        if unified_opp >= 50 and unified_risk < 50:
            if alpha.high_conviction_count >= 5:
                return (
                    MarketPosture.AGGRESSIVE,
                    0.80,
                    f"Strong alpha: {alpha.high_conviction_count} high-conviction signals, risk={unified_risk:.0f}"
                )
            return (
                MarketPosture.OPPORTUNISTIC,
                0.75,
                f"Favorable conditions: opportunity={unified_opp:.0f}, risk={unified_risk:.0f}"
            )

        # Default: Neutral
        return (
            MarketPosture.NEUTRAL,
            0.70,
            f"Balanced conditions: risk={unified_risk:.0f}, opportunity={unified_opp:.0f}"
        )

    def _generate_actions(
        self,
        macro: MacroIntelligenceReport,
        risk: RiskIntelligenceReport,
        alpha: AlphaIntelligenceReport,
        posture: MarketPosture,
    ) -> list[IntelligenceAction]:
        """Generate prioritized action list"""
        actions = []

        # Macro-driven actions
        if macro.regime in (MacroRegime.CRISIS, MacroRegime.CRISIS_WARNING):
            actions.append(IntelligenceAction(
                action_type="HEDGE",
                priority=ActionPriority.CRITICAL,
                ticker="SPY",
                direction="SHORT",
                description=f"Implement crisis hedge: {macro.hedge_recommendation}",
                rationale=f"Crash hazard {macro.unified_crash_hazard:.0%}, regime={macro.regime.value}",
                source_layers=["MACRO"],
                position_change_pct=-25,
                target_price=None,
                stop_loss=None,
                confidence=macro.regime_confidence,
                cross_validation_score=0.8 if risk.risk_level in (RiskLevel.HIGH, RiskLevel.CRITICAL) else 0.5,
            ))

        # Risk-driven actions (squeezes)
        for alert in risk.squeeze_alerts[:3]:
            if alert.alert_level == "IMMINENT":
                priority = ActionPriority.CRITICAL
            elif alert.alert_level == "ELEVATED":
                priority = ActionPriority.HIGH
            else:
                priority = ActionPriority.MEDIUM

            actions.append(IntelligenceAction(
                action_type="EXIT" if alert.squeeze_score >= 80 else "REDUCE",
                priority=priority,
                ticker=alert.ticker,
                direction="COVER_SHORT",
                description=f"Squeeze risk on {alert.ticker}: score={alert.squeeze_score:.0f}",
                rationale=f"SI={alert.short_interest:.1f}%, DTC={alert.days_to_cover:.1f}, CTB={alert.ctb_rate:.0f}%",
                source_layers=["RISK"],
                position_change_pct=-100 if alert.squeeze_score >= 80 else -50,
                target_price=alert.trigger_price,
                stop_loss=None,
                confidence=0.8,
                cross_validation_score=0.6,
            ))

        # Alpha-driven actions (high conviction entries)
        for sig in alpha.alpha_signals[:5]:
            if sig.strength in (SignalStrength.VERY_STRONG, SignalStrength.EXCEPTIONAL):
                # Check if risk supports this entry
                risk_ok = risk.risk_score < 60

                if risk_ok:
                    priority = ActionPriority.HIGH if sig.strength == SignalStrength.EXCEPTIONAL else ActionPriority.MEDIUM

                    actions.append(IntelligenceAction(
                        action_type="ENTER",
                        priority=priority,
                        ticker=sig.ticker,
                        direction=sig.direction,
                        description=f"{sig.direction} {sig.ticker}: {sig.pattern}",
                        rationale=f"{sig.confirmation_count} systems confirm, confidence={sig.confidence:.0%}",
                        source_layers=["ALPHA"],
                        position_change_pct=5 if sig.direction == "LONG" else -5,
                        target_price=sig.target_1,
                        stop_loss=sig.stop_loss,
                        confidence=sig.confidence,
                        cross_validation_score=sig.confirmation_count / 4,
                    ))

        # Greeks-driven actions
        if risk.greeks_imbalance != "BALANCED":
            actions.append(IntelligenceAction(
                action_type="REBALANCE",
                priority=ActionPriority.MEDIUM,
                ticker=None,
                direction="NEUTRAL",
                description=f"Rebalance Greeks: {risk.greeks_imbalance}",
                rationale=risk.delta_recommendation,
                source_layers=["RISK"],
                position_change_pct=None,
                target_price=None,
                stop_loss=None,
                confidence=0.7,
                cross_validation_score=0.5,
            ))

        # Posture-driven actions
        if posture == MarketPosture.FULL_DEFENSE:
            actions.append(IntelligenceAction(
                action_type="REDUCE",
                priority=ActionPriority.CRITICAL,
                ticker=None,
                direction="NEUTRAL",
                description="Reduce overall equity exposure to 30%",
                rationale="Full defense posture activated",
                source_layers=["MACRO", "RISK"],
                position_change_pct=-50,
                target_price=None,
                stop_loss=None,
                confidence=0.9,
                cross_validation_score=1.0,
            ))

        return actions

    def _priority_value(self, priority: ActionPriority) -> int:
        """Convert priority to numeric for sorting"""
        return {
            ActionPriority.CRITICAL: 5,
            ActionPriority.HIGH: 4,
            ActionPriority.MEDIUM: 3,
            ActionPriority.LOW: 2,
            ActionPriority.MONITOR: 1,
        }.get(priority, 0)

    def _recommend_equity_exposure(
        self,
        posture: MarketPosture,
        risk: float,
    ) -> float:
        """Recommend equity exposure level"""
        base = {
            MarketPosture.FULL_DEFENSE: 0.30,
            MarketPosture.DEFENSIVE: 0.50,
            MarketPosture.CAUTIOUS: 0.70,
            MarketPosture.NEUTRAL: 1.00,
            MarketPosture.OPPORTUNISTIC: 1.15,
            MarketPosture.AGGRESSIVE: 1.30,
        }.get(posture, 1.0)

        # Adjust for risk
        risk_adj = 1.0 - (max(0, risk - 50) / 100)

        return round(base * risk_adj, 2)

    def _recommend_cash_level(
        self,
        posture: MarketPosture,
        risk: float,
    ) -> float:
        """Recommend cash level"""
        base = {
            MarketPosture.FULL_DEFENSE: 0.50,
            MarketPosture.DEFENSIVE: 0.35,
            MarketPosture.CAUTIOUS: 0.20,
            MarketPosture.NEUTRAL: 0.10,
            MarketPosture.OPPORTUNISTIC: 0.05,
            MarketPosture.AGGRESSIVE: 0.05,
        }.get(posture, 0.10)

        return round(base, 2)

    def _recommend_max_position(
        self,
        posture: MarketPosture,
        risk: RiskIntelligenceReport,
    ) -> float:
        """Recommend maximum single position size"""
        base = {
            MarketPosture.FULL_DEFENSE: 0.03,
            MarketPosture.DEFENSIVE: 0.04,
            MarketPosture.CAUTIOUS: 0.05,
            MarketPosture.NEUTRAL: 0.06,
            MarketPosture.OPPORTUNISTIC: 0.07,
            MarketPosture.AGGRESSIVE: 0.08,
        }.get(posture, 0.05)

        # Reduce if vol is extreme
        if risk.vol_regime == VolRegime.EXTREME:
            base *= 0.6
        elif risk.vol_regime == VolRegime.HIGH_VOL:
            base *= 0.8

        return round(base, 3)

    def _calculate_hedge_ratio(
        self,
        posture: MarketPosture,
        macro: MacroIntelligenceReport,
        risk: RiskIntelligenceReport,
    ) -> float:
        """Calculate portion of portfolio to hedge"""
        base = {
            MarketPosture.FULL_DEFENSE: 0.70,
            MarketPosture.DEFENSIVE: 0.50,
            MarketPosture.CAUTIOUS: 0.30,
            MarketPosture.NEUTRAL: 0.15,
            MarketPosture.OPPORTUNISTIC: 0.10,
            MarketPosture.AGGRESSIVE: 0.05,
        }.get(posture, 0.15)

        # Increase for high crash hazard
        if macro.unified_crash_hazard >= 0.7:
            base = min(1.0, base * 1.3)

        return round(base, 2)

    def _compile_hedge_recommendations(
        self,
        macro: MacroIntelligenceReport,
        risk: RiskIntelligenceReport,
    ) -> list[str]:
        """Compile hedge recommendations from all layers"""
        hedges = []

        # From macro
        if macro.hedge_recommendation and "NONE" not in macro.hedge_recommendation:
            hedges.append(f"MACRO: {macro.hedge_recommendation}")

        # From risk
        hedges.extend(risk.specific_hedges)

        return hedges[:5]  # Top 5

    def _calculate_intelligence_confidence(
        self,
        system_count: int,
        alignment: float,
        signal_count: int,
    ) -> float:
        """Calculate overall confidence in intelligence report"""
        # More systems = more confidence
        system_factor = min(1.0, system_count / 8)

        # Better alignment = more confidence
        align_factor = alignment

        # More signals = more confidence (up to a point)
        signal_factor = min(1.0, signal_count / 20)

        confidence = (system_factor * 0.4) + (align_factor * 0.4) + (signal_factor * 0.2)

        return round(confidence, 2)
