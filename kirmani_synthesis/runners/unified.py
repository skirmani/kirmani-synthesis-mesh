"""
Unified Synthesis Runner

The main orchestrator that ties together:
- Signal ingestion from all systems
- Bayesian fusion
- Quick Wins evaluation
- Contradiction/Confirmation detection
- Report generation
- Dashboard output

This is the primary entry point for running the Synthesis Mesh.
"""

import asyncio
import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Optional

from ..models import SynthesisReport, MarketRegime
from ..fusion import FusionEngine, SynthesizedView
from ..rules import QuickWinsEngine
from ..detection import ContradictionDetector, ConfirmationDetector
from ..dashboard import DashboardGenerator

logger = logging.getLogger(__name__)


class UnifiedSynthesisRunner:
    """
    Unified Synthesis Runner

    Orchestrates the complete synthesis pipeline:
    1. Collect signals from all systems
    2. Run Bayesian fusion to create unified market view
    3. Evaluate Quick Wins rules
    4. Detect contradictions and confirmations
    5. Generate synthesis report
    6. Output dashboard

    Usage:
        runner = UnifiedSynthesisRunner()

        # Add signals (from any source)
        runner.add_signal(signal_dict)
        runner.add_signals(signal_list)

        # Run synthesis
        report = runner.run()

        # Save outputs
        runner.save_report("/path/to/report.json")
        runner.save_dashboard("/path/to/dashboard.html")
    """

    def __init__(
        self,
        output_dir: Optional[str] = None,
    ):
        self.output_dir = output_dir or os.path.expanduser("~/.kirmani_reports")
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)

        # Initialize engines
        self.fusion_engine = FusionEngine()
        self.rules_engine = QuickWinsEngine()
        self.contradiction_detector = ContradictionDetector()
        self.confirmation_detector = ConfirmationDetector()
        self.dashboard_generator = DashboardGenerator()

        # Store signals
        self._signals: list[dict] = []

        # Last report
        self._last_report: Optional[SynthesisReport] = None

    def clear(self) -> None:
        """Clear all signals and reset state"""
        self._signals.clear()
        self.fusion_engine.clear()

    def add_signal(self, signal: dict) -> None:
        """Add a single signal"""
        self._signals.append(signal)
        self.fusion_engine.ingest(signal)

    def add_signals(self, signals: list[dict]) -> None:
        """Add multiple signals"""
        for signal in signals:
            self.add_signal(signal)

    def run(self) -> SynthesisReport:
        """
        Run the complete synthesis pipeline

        Returns:
            SynthesisReport with all synthesized intelligence
        """
        logger.info(f"Running synthesis on {len(self._signals)} signals...")

        # Step 1: Fusion
        view = self.fusion_engine.synthesize()
        logger.info(f"Fusion complete: {view.unified_direction} ({view.unified_confidence:.0%})")

        # Step 2: Quick Wins
        rule_results = self.rules_engine.evaluate(view, self._signals)
        triggered_rules = [r for r in rule_results if r.triggered]
        quick_wins = self.rules_engine.to_triggers(triggered_rules)
        recommendations = self.rules_engine.to_recommendations(triggered_rules)
        logger.info(f"Quick Wins: {len(triggered_rules)} rules triggered")

        # Step 3: Contradictions
        contradictions = self.contradiction_detector.detect(self._signals)
        logger.info(f"Contradictions: {len(contradictions)} detected")

        # Step 4: Confirmations
        confirmations = self.confirmation_detector.detect(self._signals)
        logger.info(f"Confirmations: {len(confirmations)} detected")

        # Step 5: Determine overall stance
        overall_stance = self._determine_stance(view, triggered_rules)

        # Step 6: Calculate recommended exposure
        recommended_exposure = self._calculate_exposure(view, overall_stance)

        # Step 7: Build report
        report = SynthesisReport(
            market_state=view.market_state,
            recommendations=recommendations,
            quick_wins_triggered=quick_wins,
            contradictions=contradictions,
            confirmations=confirmations,
            total_signals_processed=len(self._signals),
            systems_reporting=len(view.systems_reporting),
            critical_alerts=len([r for r in triggered_rules if r.urgency.value == "IMMEDIATE"]),
            overall_stance=overall_stance,
            recommended_exposure=recommended_exposure,
        )

        self._last_report = report
        logger.info(f"Synthesis complete: {overall_stance} stance, {recommended_exposure:.0%} exposure")

        return report

    def _determine_stance(self, view: SynthesizedView, triggered_rules: list) -> str:
        """Determine overall investment stance"""
        state = view.market_state

        # Check for crisis indicators
        if state.regime == MarketRegime.CRISIS:
            return "DEFENSIVE"

        # Check for defensive rules triggered
        defensive_rules = ["crisis_exit", "murphy_sornette", "squeeze_bubble"]
        if any(r.rule_id in defensive_rules for r in triggered_rules):
            return "DEFENSIVE"

        # Check market state
        if state.crash_hazard >= 0.6 or state.bubble_phase >= 4:
            return "DEFENSIVE"

        # Check for bullish signals
        if view.unified_direction == "BULLISH" and view.unified_confidence >= 0.7:
            if any(r.rule_id in ("scanner_wyckoff", "triple_confirmation") for r in triggered_rules):
                return "AGGRESSIVE"

        # Check for risk-on regime
        if state.regime == MarketRegime.RISK_ON and state.overall_risk_score < 40:
            return "AGGRESSIVE"

        return "NEUTRAL"

    def _calculate_exposure(self, view: SynthesizedView, stance: str) -> float:
        """Calculate recommended position exposure multiplier"""
        state = view.market_state

        # Base exposure by stance
        base = {"DEFENSIVE": 0.5, "NEUTRAL": 1.0, "AGGRESSIVE": 1.25}
        exposure = base.get(stance, 1.0)

        # Adjust for risk score
        if state.overall_risk_score >= 70:
            exposure *= 0.7
        elif state.overall_risk_score >= 50:
            exposure *= 0.85

        # Adjust for crash hazard
        if state.crash_hazard >= 0.7:
            exposure *= 0.5
        elif state.crash_hazard >= 0.5:
            exposure *= 0.75

        # Adjust for system agreement
        if state.system_agreement >= 0.8:
            exposure *= 1.1
        elif state.system_agreement < 0.5:
            exposure *= 0.9

        return min(1.5, max(0.25, exposure))

    def save_report(self, filepath: Optional[str] = None) -> str:
        """Save report as JSON"""
        if not self._last_report:
            raise ValueError("No report generated. Call run() first.")

        if filepath is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = os.path.join(self.output_dir, f"synthesis_report_{timestamp}.json")

        with open(filepath, 'w') as f:
            json.dump(self._last_report.to_dict(), f, indent=2, default=str)

        logger.info(f"Report saved: {filepath}")
        return filepath

    def save_dashboard(self, filepath: Optional[str] = None) -> str:
        """Save dashboard as HTML"""
        if not self._last_report:
            raise ValueError("No report generated. Call run() first.")

        if filepath is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = os.path.join(self.output_dir, f"synthesis_dashboard_{timestamp}.html")

        self.dashboard_generator.save(self._last_report, filepath)
        logger.info(f"Dashboard saved: {filepath}")
        return filepath

    def get_last_report(self) -> Optional[SynthesisReport]:
        """Get the last generated report"""
        return self._last_report

    def print_summary(self) -> None:
        """Print a summary of the last report to console"""
        if not self._last_report:
            print("No report generated. Call run() first.")
            return

        report = self._last_report
        state = report.market_state

        print("\n" + "=" * 60)
        print("KIRMANI SYNTHESIS MESH - SUMMARY")
        print("=" * 60)
        print(f"\nGenerated: {report.generated_at.strftime('%Y-%m-%d %H:%M:%S UTC')}")
        print(f"Signals Processed: {report.total_signals_processed}")
        print(f"Systems Reporting: {report.systems_reporting}")

        print(f"\n{'='*60}")
        print("MARKET STATE")
        print(f"{'='*60}")
        print(f"Regime: {state.regime.value}")
        print(f"Crash Hazard: {state.crash_hazard:.0%}")
        print(f"Bubble Phase: {state.bubble_phase}/5")
        print(f"Risk Score: {state.overall_risk_score:.0f}/100")
        print(f"System Agreement: {state.system_agreement:.0%}")

        print(f"\n{'='*60}")
        print("OVERALL ASSESSMENT")
        print(f"{'='*60}")
        print(f"Stance: {report.overall_stance}")
        print(f"Recommended Exposure: {report.recommended_exposure:.0%}")
        print(f"Critical Alerts: {report.critical_alerts}")

        if report.quick_wins_triggered:
            print(f"\n{'='*60}")
            print("QUICK WINS TRIGGERED")
            print(f"{'='*60}")
            for qw in report.quick_wins_triggered:
                print(f"  - {qw.rule_name}: {qw.action.value} {qw.target_ticker} ({qw.position_change_pct:+.0f}%)")

        if report.recommendations:
            print(f"\n{'='*60}")
            print("RECOMMENDATIONS")
            print(f"{'='*60}")
            for rec in report.recommendations:
                print(f"  - {rec.action.value}: {rec.ticker} ({rec.change_pct:+.0f}%) [{rec.urgency.value}]")

        if report.contradictions:
            print(f"\n{'='*60}")
            print(f"CONTRADICTIONS ({len(report.contradictions)})")
            print(f"{'='*60}")
            for c in report.contradictions[:3]:
                print(f"  - {c.ticker}: {c.system_a} vs {c.system_b}")

        if report.confirmations:
            print(f"\n{'='*60}")
            print(f"CONFIRMATIONS ({len(report.confirmations)})")
            print(f"{'='*60}")
            for c in report.confirmations[:3]:
                print(f"  - {c.ticker}: {c.direction} ({c.system_count} systems, {c.confirmation_strength})")

        print("\n")
