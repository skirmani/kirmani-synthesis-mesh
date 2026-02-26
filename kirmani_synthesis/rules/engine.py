"""
Quick Wins Rule Engine

Implements the 4 Quick Win rules plus additional cross-system rules
that fire when specific conditions are met across multiple systems.

Quick Wins:
1. Squeeze + Bubble: SENTINEL squeeze AND Minsky Euphoria → 75% reduction
2. Murphy + Sornette: Risk-off AND hazard > 0.6 → 50% equity reduction
3. Scanner + Wyckoff: Momentum setup AND accumulation → Full position
4. Hurst + Greeks: Cycle bottom AND extreme P/C ratio → Buy calls
"""

import logging
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Callable, Optional

from pydantic import BaseModel, Field

from ..models import (
    ActionType,
    UrgencyLevel,
    PositionRecommendation,
    QuickWinTrigger,
)
from ..fusion import SynthesizedView

logger = logging.getLogger(__name__)


class RuleResult(BaseModel):
    """Result of evaluating a rule"""

    triggered: bool = False
    rule_id: str = ""
    rule_name: str = ""
    confidence: float = 0.0

    # Action details
    action: Optional[ActionType] = None
    target_ticker: str = "SPY"
    position_change_pct: float = 0.0
    urgency: UrgencyLevel = UrgencyLevel.NORMAL

    # Supporting info
    conditions_met: dict[str, Any] = Field(default_factory=dict)
    source_signals: list[str] = Field(default_factory=list)
    rationale: str = ""


class Rule(ABC):
    """
    Base class for Quick Win rules

    Each rule checks specific conditions across multiple systems
    and returns an action when conditions are met.
    """

    rule_id: str = "base"
    rule_name: str = "Base Rule"
    description: str = ""

    @abstractmethod
    def evaluate(self, view: SynthesizedView, signals: list[dict]) -> RuleResult:
        """
        Evaluate the rule against current market state

        Args:
            view: Synthesized market view
            signals: Raw signals for detailed inspection

        Returns:
            RuleResult indicating if rule triggered and recommended action
        """
        pass


class SqueezeAndBubbleRule(Rule):
    """
    Quick Win #1: Squeeze + Bubble Cross-Check

    WHEN: SENTINEL detects squeeze risk (score > 65) AND Minsky shows Euphoria (stage 3+)
    ACTION: Immediate 75% position reduction on affected shorts
    RATIONALE: Squeeze in bubble = maximum danger for short positions
    """

    rule_id = "squeeze_bubble"
    rule_name = "Squeeze + Bubble Cross-Check"
    description = "Reduce short exposure when squeeze risk meets bubble euphoria"

    def evaluate(self, view: SynthesizedView, signals: list[dict]) -> RuleResult:
        result = RuleResult(rule_id=self.rule_id, rule_name=self.rule_name)

        # Check conditions
        squeeze_risk = view.squeeze_risk
        bubble_phase = view.bubble_phase

        squeeze_triggered = squeeze_risk >= 0.65
        bubble_triggered = bubble_phase >= 3

        if squeeze_triggered and bubble_triggered:
            result.triggered = True
            result.action = ActionType.REDUCE_SHORT
            result.position_change_pct = -75.0  # Reduce by 75%
            result.urgency = UrgencyLevel.IMMEDIATE
            result.confidence = min(squeeze_risk, bubble_phase / 5) * 1.2

            result.conditions_met = {
                "squeeze_risk": squeeze_risk,
                "bubble_phase": bubble_phase,
                "squeeze_threshold": 0.65,
                "bubble_threshold": 3,
            }

            result.source_signals = ["sentinel-8", "minsky-analyzer"]
            result.rationale = (
                f"CRITICAL: Squeeze risk ({squeeze_risk:.0%}) detected during "
                f"Minsky Phase {bubble_phase} (Euphoria). Maximum danger for shorts. "
                f"Recommend immediate 75% reduction in short exposure."
            )

            # Find affected tickers from squeeze signals
            for sig in signals:
                if sig.get("signal_type") == "SQUEEZE_ALERT":
                    score = sig.get("components", {}).get("squeeze_score", 0)
                    if score >= 65:
                        result.target_ticker = sig.get("ticker", "SPY")
                        break

        return result


class MurphySornetteRule(Rule):
    """
    Quick Win #2: Murphy + Sornette Confirmation

    WHEN: Murphy shows risk-off regime AND Sornette hazard > 0.6
    ACTION: Reduce equity exposure to 50%
    RATIONALE: Intermarket + LPPLS double confirmation of elevated crash risk
    """

    rule_id = "murphy_sornette"
    rule_name = "Murphy + Sornette Confirmation"
    description = "Reduce equity when intermarket risk-off aligns with crash hazard"

    def evaluate(self, view: SynthesizedView, signals: list[dict]) -> RuleResult:
        result = RuleResult(rule_id=self.rule_id, rule_name=self.rule_name)

        # Check conditions
        intermarket = view.intermarket_regime
        crash_hazard = view.crash_hazard

        murphy_triggered = intermarket in ("RISK_OFF", "DEFENSIVE")
        sornette_triggered = crash_hazard >= 0.6

        if murphy_triggered and sornette_triggered:
            result.triggered = True
            result.action = ActionType.REDUCE_LONG
            result.target_ticker = "SPY"  # Market-wide
            result.position_change_pct = -50.0  # Reduce to 50%
            result.urgency = UrgencyLevel.URGENT
            result.confidence = (crash_hazard + 0.8) / 2  # High confidence

            result.conditions_met = {
                "intermarket_regime": intermarket,
                "crash_hazard": crash_hazard,
                "murphy_threshold": "RISK_OFF",
                "sornette_threshold": 0.6,
            }

            result.source_signals = ["murphy-intermarket", "sornette-lppls"]
            result.rationale = (
                f"DEFENSIVE: Murphy intermarket shows {intermarket} regime while "
                f"Sornette crash hazard is {crash_hazard:.0%}. Double confirmation "
                f"of elevated risk. Recommend reducing equity exposure to 50%."
            )

        return result


class ScannerWyckoffRule(Rule):
    """
    Quick Win #3: Elite Scanner + Wyckoff Filter

    WHEN: Elite Scanner finds momentum setup (score > 80) AND Wyckoff confirms accumulation
    ACTION: Take full position size
    RATIONALE: Technical momentum + volume confirmation = high conviction entry
    """

    rule_id = "scanner_wyckoff"
    rule_name = "Scanner + Wyckoff Confirmation"
    description = "Full position when momentum setup aligns with accumulation"

    def evaluate(self, view: SynthesizedView, signals: list[dict]) -> RuleResult:
        result = RuleResult(rule_id=self.rule_id, rule_name=self.rule_name)

        # Find matching signals
        momentum_setups = {}
        accumulation_tickers = set()

        for sig in signals:
            ticker = sig.get("ticker", "")

            if sig.get("signal_type") == "MOMENTUM_SETUP":
                score = sig.get("components", {}).get("composite_score", 0)
                if score >= 80:
                    momentum_setups[ticker] = {
                        "score": score,
                        "setup_type": sig.get("components", {}).get("setup_type"),
                        "confidence": sig.get("confidence", 0.5),
                    }

            if sig.get("signal_type") == "ACCUMULATION":
                if sig.get("confidence", 0) >= 0.7:
                    accumulation_tickers.add(ticker)

        # Check for overlap
        confirmed_tickers = set(momentum_setups.keys()) & accumulation_tickers

        if confirmed_tickers:
            # Take the highest scoring ticker
            best_ticker = max(confirmed_tickers, key=lambda t: momentum_setups[t]["score"])
            setup = momentum_setups[best_ticker]

            result.triggered = True
            result.action = ActionType.ENTER_LONG
            result.target_ticker = best_ticker
            result.position_change_pct = 100.0  # Full position
            result.urgency = UrgencyLevel.NORMAL
            result.confidence = setup["confidence"]

            result.conditions_met = {
                "ticker": best_ticker,
                "momentum_score": setup["score"],
                "setup_type": setup["setup_type"],
                "wyckoff_confirmed": True,
            }

            result.source_signals = ["elite-momentum-scanner", "wyckoff-elite"]
            result.rationale = (
                f"BULLISH ENTRY: {best_ticker} has Elite Scanner score of {setup['score']} "
                f"({setup['setup_type']}) with Wyckoff accumulation confirmation. "
                f"High conviction long entry recommended."
            )

        return result


class HurstGreeksRule(Rule):
    """
    Quick Win #4: Hurst Cycle + Options Greeks

    WHEN: Hurst signals cycle bottom AND options show extreme put/call ratio
    ACTION: Buy calls, consider selling puts
    RATIONALE: Cycle timing + sentiment alignment for options play
    """

    rule_id = "hurst_greeks"
    rule_name = "Hurst Cycle + Greeks Alignment"
    description = "Options play when cycle bottom aligns with extreme sentiment"

    def evaluate(self, view: SynthesizedView, signals: list[dict]) -> RuleResult:
        result = RuleResult(rule_id=self.rule_id, rule_name=self.rule_name)

        # Check for Hurst cycle signals
        cycle_bottom_tickers = {}
        extreme_put_call = {}

        for sig in signals:
            ticker = sig.get("ticker", "")

            if sig.get("signal_type") == "CYCLE_TURN":
                components = sig.get("components", {})
                cycle_phase = components.get("phase", "")
                if cycle_phase in ("BOTTOM", "TROUGH", "CYCLE_LOW"):
                    cycle_bottom_tickers[ticker] = {
                        "phase": cycle_phase,
                        "confidence": sig.get("confidence", 0.5),
                    }

            if sig.get("signal_type") == "GREEK_EXPOSURE":
                components = sig.get("components", {})
                put_call = components.get("put_call_ratio", 1.0)
                # Extreme put/call (high = bearish sentiment = contrarian bullish)
                if put_call >= 1.5:  # Very bearish sentiment
                    extreme_put_call[ticker] = {
                        "put_call_ratio": put_call,
                        "confidence": sig.get("confidence", 0.5),
                    }

        # Check for overlap
        confirmed_tickers = set(cycle_bottom_tickers.keys()) & set(extreme_put_call.keys())

        if confirmed_tickers:
            best_ticker = list(confirmed_tickers)[0]
            cycle = cycle_bottom_tickers[best_ticker]
            greeks = extreme_put_call[best_ticker]

            result.triggered = True
            result.action = ActionType.ENTER_LONG  # Buy calls
            result.target_ticker = best_ticker
            result.position_change_pct = 50.0  # Moderate position for options
            result.urgency = UrgencyLevel.NORMAL
            result.confidence = (cycle["confidence"] + greeks["confidence"]) / 2

            result.conditions_met = {
                "ticker": best_ticker,
                "cycle_phase": cycle["phase"],
                "put_call_ratio": greeks["put_call_ratio"],
            }

            result.source_signals = ["hurst-cycles", "greeks-trader"]
            result.rationale = (
                f"OPTIONS OPPORTUNITY: {best_ticker} at Hurst cycle bottom "
                f"with extreme put/call ratio ({greeks['put_call_ratio']:.1f}). "
                f"Contrarian bullish setup. Consider buying calls."
            )

        return result


class TripleConfirmationRule(Rule):
    """
    Bonus Rule: Triple Layer Confirmation

    WHEN: All three layers (MACRO, RISK, ALPHA) agree on direction
    ACTION: Increase position size by 25%
    RATIONALE: Rare multi-layer agreement = highest conviction
    """

    rule_id = "triple_confirmation"
    rule_name = "Triple Layer Confirmation"
    description = "Increase size when all layers agree"

    def evaluate(self, view: SynthesizedView, signals: list[dict]) -> RuleResult:
        result = RuleResult(rule_id=self.rule_id, rule_name=self.rule_name)

        # Check if all layers agree
        macro_dir = view.macro_view.direction
        risk_dir = view.risk_view.direction
        alpha_dir = view.alpha_view.direction

        # All must agree and not be NEUTRAL
        if (macro_dir == risk_dir == alpha_dir) and macro_dir != "NEUTRAL":
            # Check confidence levels
            min_conf = min(
                view.macro_view.confidence,
                view.risk_view.confidence,
                view.alpha_view.confidence
            )

            if min_conf >= 0.6:
                result.triggered = True

                if macro_dir == "BULLISH":
                    result.action = ActionType.INCREASE_LONG
                else:
                    result.action = ActionType.INCREASE_SHORT

                result.target_ticker = "SPY"
                result.position_change_pct = 25.0  # Increase by 25%
                result.urgency = UrgencyLevel.NORMAL
                result.confidence = min_conf

                result.conditions_met = {
                    "macro_direction": macro_dir,
                    "risk_direction": risk_dir,
                    "alpha_direction": alpha_dir,
                    "macro_confidence": view.macro_view.confidence,
                    "risk_confidence": view.risk_view.confidence,
                    "alpha_confidence": view.alpha_view.confidence,
                }

                result.source_signals = list(
                    set(view.macro_view.systems + view.risk_view.systems + view.alpha_view.systems)
                )
                result.rationale = (
                    f"TRIPLE CONFIRMATION: All layers agree on {macro_dir} direction "
                    f"with minimum confidence {min_conf:.0%}. Rare alignment across "
                    f"MACRO, RISK, and ALPHA systems. Consider increasing position size."
                )

        return result


class CrisisExitRule(Rule):
    """
    Safety Rule: Crisis Mode Exit

    WHEN: Crash hazard > 0.8 OR bubble phase = 5 (Panic) OR vol regime = CRISIS
    ACTION: Reduce all positions by 50%, hedge remainder
    RATIONALE: Capital preservation in crisis conditions
    """

    rule_id = "crisis_exit"
    rule_name = "Crisis Mode Exit"
    description = "Defensive action in crisis conditions"

    def evaluate(self, view: SynthesizedView, signals: list[dict]) -> RuleResult:
        result = RuleResult(rule_id=self.rule_id, rule_name=self.rule_name)

        crash_hazard = view.crash_hazard
        bubble_phase = view.bubble_phase
        vol_regime = view.vol_regime

        crisis_triggered = (
            crash_hazard >= 0.8 or
            bubble_phase >= 5 or
            vol_regime == "CRISIS"
        )

        if crisis_triggered:
            result.triggered = True
            result.action = ActionType.HEDGE
            result.target_ticker = "SPY"
            result.position_change_pct = -50.0
            result.urgency = UrgencyLevel.IMMEDIATE
            result.confidence = max(crash_hazard, bubble_phase / 5)

            result.conditions_met = {
                "crash_hazard": crash_hazard,
                "bubble_phase": bubble_phase,
                "vol_regime": vol_regime,
            }

            result.source_signals = ["sornette-lppls", "minsky-analyzer", "vix-lab"]
            result.rationale = (
                f"CRISIS MODE: Emergency defensive action required. "
                f"Crash hazard: {crash_hazard:.0%}, Minsky Phase: {bubble_phase}, "
                f"Vol Regime: {vol_regime}. Reduce exposure 50% and hedge remainder."
            )

        return result


class QuickWinsEngine:
    """
    Quick Wins Rule Engine

    Evaluates all Quick Win rules against the synthesized market view
    and returns triggered actions.

    Usage:
        engine = QuickWinsEngine()

        # Evaluate rules
        results = engine.evaluate(view, signals)

        # Get triggered actions
        for result in results:
            if result.triggered:
                print(f"Rule {result.rule_name} triggered: {result.action}")
    """

    def __init__(self):
        # Register all rules
        self.rules: list[Rule] = [
            SqueezeAndBubbleRule(),
            MurphySornetteRule(),
            ScannerWyckoffRule(),
            HurstGreeksRule(),
            TripleConfirmationRule(),
            CrisisExitRule(),
        ]

    def add_rule(self, rule: Rule) -> None:
        """Add a custom rule"""
        self.rules.append(rule)

    def evaluate(
        self,
        view: SynthesizedView,
        signals: list[dict],
    ) -> list[RuleResult]:
        """
        Evaluate all rules against current state

        Args:
            view: Synthesized market view
            signals: Raw signals for detailed inspection

        Returns:
            List of RuleResults for all rules (triggered and not)
        """
        results = []

        for rule in self.rules:
            try:
                result = rule.evaluate(view, signals)
                results.append(result)

                if result.triggered:
                    logger.info(
                        f"Rule triggered: {rule.rule_name} -> "
                        f"{result.action} {result.target_ticker} "
                        f"({result.position_change_pct:+.0f}%)"
                    )

            except Exception as e:
                logger.error(f"Rule {rule.rule_id} failed: {e}")
                results.append(RuleResult(
                    rule_id=rule.rule_id,
                    rule_name=rule.rule_name,
                    triggered=False,
                    rationale=f"Error: {e}",
                ))

        return results

    def get_triggered(
        self,
        view: SynthesizedView,
        signals: list[dict],
    ) -> list[RuleResult]:
        """Get only triggered rules"""
        results = self.evaluate(view, signals)
        return [r for r in results if r.triggered]

    def to_recommendations(
        self,
        results: list[RuleResult],
    ) -> list[PositionRecommendation]:
        """Convert triggered rules to position recommendations"""
        recommendations = []

        for result in results:
            if not result.triggered or not result.action:
                continue

            rec = PositionRecommendation(
                ticker=result.target_ticker,
                action=result.action,
                urgency=result.urgency,
                change_pct=result.position_change_pct,
                execution_algo="VWAP" if result.urgency != UrgencyLevel.IMMEDIATE else "MARKET",
                execution_window="10:30" if result.urgency != UrgencyLevel.IMMEDIATE else "NOW",
                primary_trigger=result.rule_name,
                supporting_signals=result.source_signals,
                confidence=result.confidence,
            )

            recommendations.append(rec)

        return recommendations

    def to_triggers(self, results: list[RuleResult]) -> list[QuickWinTrigger]:
        """Convert triggered rules to QuickWinTrigger objects"""
        triggers = []

        for result in results:
            if not result.triggered or not result.action:
                continue

            trigger = QuickWinTrigger(
                rule_name=result.rule_name,
                rule_id=result.rule_id,
                conditions_met=result.conditions_met,
                action=result.action,
                target_ticker=result.target_ticker,
                position_change_pct=result.position_change_pct,
                confidence=result.confidence,
                source_signals=result.source_signals,
            )

            triggers.append(trigger)

        return triggers
