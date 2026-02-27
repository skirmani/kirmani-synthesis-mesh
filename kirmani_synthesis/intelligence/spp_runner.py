"""
S++ Tier Elite Intelligence Runner
===================================

The ultimate integration of all S++ components:
- Real data from IBKR + Polygon + FRED
- 25+ signal producers across 3 layers
- Consensus cascade with cross-layer validation
- S++ validation with drift detection
- Elite dashboard generation

This is institutional-grade quant intelligence.

Usage:
    python -m kirmani_synthesis.intelligence.spp_runner

Or programmatically:
    from kirmani_synthesis.intelligence import run_spp_intelligence
    report, dashboard_path = run_spp_intelligence()
"""

import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)


def run_spp_intelligence(
    portfolio_dir: str = None,
    output_dir: str = None,
    enable_consensus: bool = True,
    enable_drift_detection: bool = True,
    open_dashboard: bool = True,
) -> Tuple[Any, str]:
    """
    Run complete S++ tier intelligence pipeline.

    Args:
        portfolio_dir: Directory containing portfolio CSVs
        output_dir: Directory for output files
        enable_consensus: Enable consensus cascade detection
        enable_drift_detection: Run drift detection on systems
        open_dashboard: Open dashboard in browser

    Returns:
        Tuple of (UnifiedIntelligenceReport, dashboard_path)
    """
    from .real_data import RealDataConnector, RealSignalGenerator
    from .signal_producers import SignalProducerRegistry
    from .engine import CrossSystemIntelligenceEngine
    from .dashboard import IntelligenceDashboardGenerator
    from .spp_validation import SPPValidationTracker
    from .consensus import ConsensusCascade, CrossLayerValidator

    print("=" * 80)
    print("KIRMANI S++ TIER ELITE INTELLIGENCE SYSTEM")
    print("=" * 80)
    print(f"Timestamp: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}")
    print(f"Mode: S++ (Consensus + Validation + Drift Detection)")

    # Initialize output directory
    if output_dir is None:
        output_dir = Path.home() / ".kirmani_reports"
    else:
        output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    # =========================================================================
    # PHASE 1: Load Real Data
    # =========================================================================
    print("\n" + "=" * 80)
    print("[PHASE 1] LOADING REAL DATA FROM IBKR + MARKET FEEDS")
    print("=" * 80)

    connector = RealDataConnector(portfolio_dir=portfolio_dir)
    bundle = connector.get_full_data_bundle()

    print(f"\n  PORTFOLIO")
    print(f"  ─────────────────────────────────────────────")
    print(f"  Positions:       {len(bundle.portfolio)}")
    if bundle.portfolio:
        total_value = sum(abs(p.market_value) for p in bundle.portfolio)
        total_pnl = sum(p.unrealized_pnl for p in bundle.portfolio)
        print(f"  Total Value:     ${total_value:,.2f}")
        print(f"  Unrealized P&L:  ${total_pnl:+,.2f}")

    print(f"\n  MARKET DATA")
    print(f"  ─────────────────────────────────────────────")
    print(f"  Tickers:         {len(bundle.market_data)}")
    print(f"  VIX Level:       {bundle.vix_level:.2f}")
    print(f"  SPY:             ${bundle.spy_price:,.2f}")
    print(f"  QQQ:             ${bundle.qqq_price:,.2f}")

    print(f"\n  MACRO")
    print(f"  ─────────────────────────────────────────────")
    print(f"  Fed Funds:       {bundle.fed_funds_rate:.2f}%")
    print(f"  10Y Treasury:    {bundle.treasury_10y:.2f}%")

    # =========================================================================
    # PHASE 2: Generate Signals from All Systems
    # =========================================================================
    print("\n" + "=" * 80)
    print("[PHASE 2] GENERATING SIGNALS FROM 25+ SYSTEMS")
    print("=" * 80)

    signal_generator = RealSignalGenerator(connector)
    signals = signal_generator.generate_signals_from_portfolio(bundle)

    # Group by layer
    layers = {"MACRO": [], "RISK": [], "ALPHA": []}
    for sig in signals:
        system_id = sig.get("system_id", "").lower()

        if any(k in system_id for k in ["sornette", "minsky", "murphy", "cgma", "gkhy"]):
            layers["MACRO"].append(sig)
        elif any(k in system_id for k in ["sentinel", "frm", "greeks", "vix", "squeeze"]):
            layers["RISK"].append(sig)
        else:
            layers["ALPHA"].append(sig)

    print(f"\n  SIGNAL DISTRIBUTION")
    print(f"  ─────────────────────────────────────────────")
    print(f"  Total Signals:   {len(signals)}")
    print(f"  MACRO Layer:     {len(layers['MACRO'])} signals")
    print(f"  RISK Layer:      {len(layers['RISK'])} signals")
    print(f"  ALPHA Layer:     {len(layers['ALPHA'])} signals")

    # Count unique systems
    systems = set(s.get("system_id", "unknown") for s in signals)
    print(f"  Systems Active:  {len(systems)}")

    # =========================================================================
    # PHASE 3: Consensus Cascade Detection
    # =========================================================================
    consensus_results = []

    if enable_consensus:
        print("\n" + "=" * 80)
        print("[PHASE 3] RUNNING CONSENSUS CASCADE DETECTION")
        print("=" * 80)

        cascade = ConsensusCascade()
        consensus_results = cascade.find_consensus(signals)

        summary = cascade.get_consensus_summary()

        print(f"\n  CONSENSUS SUMMARY")
        print(f"  ─────────────────────────────────────────────")
        print(f"  Total Consensus:       {summary.get('total_consensus', 0)}")
        print(f"  By Level:              {summary.get('by_level', {})}")
        print(f"  Avg Cascade Mult:      {summary.get('avg_cascade_multiplier', 1.0):.2f}x")
        print(f"  Avg Confidence Boost:  {summary.get('avg_confidence_boost', 0):+.3f}")
        print(f"  Cross-Layer Consensus: {summary.get('cross_layer_consensus', 0)}")

        if consensus_results:
            print(f"\n  TOP CONSENSUS SIGNALS")
            print(f"  ─────────────────────────────────────────────")
            for c in sorted(consensus_results, key=lambda x: x.cascade_confidence, reverse=True)[:5]:
                print(f"  {c.ticker:8s} {c.direction:8s} | "
                      f"{len(c.systems)} systems | "
                      f"cascade_conf={c.cascade_confidence:.2f} | "
                      f"{c.level.value}")

    # =========================================================================
    # PHASE 4: Cross-System Intelligence Synthesis
    # =========================================================================
    print("\n" + "=" * 80)
    print("[PHASE 4] RUNNING CROSS-SYSTEM INTELLIGENCE ENGINE")
    print("=" * 80)

    engine = CrossSystemIntelligenceEngine()
    report = engine.synthesize(signals)

    print(f"\n  MARKET POSTURE: {report.market_posture.value}")
    print(f"  ─────────────────────────────────────────────")
    print(f"  Confidence:      {report.posture_confidence:.0%}")
    print(f"  Rationale:       {report.posture_rationale}")

    print(f"\n  UNIFIED METRICS")
    print(f"  ─────────────────────────────────────────────")
    print(f"  Risk Score:      {report.unified_risk_score:.0f}/100")
    print(f"  Opportunity:     {report.unified_opportunity_score:.0f}/100")
    print(f"  R/R Balance:     {report.risk_reward_balance:+.2f}")
    print(f"  Stack Alignment: {report.full_stack_alignment:.0%}")

    print(f"\n  POSITION GUIDANCE")
    print(f"  ─────────────────────────────────────────────")
    print(f"  Equity Exposure: {report.recommended_equity_exposure:.0%}")
    print(f"  Cash Level:      {report.recommended_cash_level:.0%}")
    print(f"  Max Position:    {report.max_single_position_pct:.0%}")
    print(f"  Hedge Ratio:     {report.hedge_ratio:.0%}")

    if report.critical_actions:
        print(f"\n  CRITICAL ACTIONS ({len(report.critical_actions)})")
        print(f"  ─────────────────────────────────────────────")
        for action in report.critical_actions[:3]:
            ticker_str = f" {action.ticker}" if action.ticker else ""
            print(f"  [{action.priority.value}] {action.action_type}{ticker_str}")
            print(f"      {action.description}")

    # =========================================================================
    # PHASE 5: S++ Validation Tracking
    # =========================================================================
    print("\n" + "=" * 80)
    print("[PHASE 5] S++ VALIDATION TRACKING")
    print("=" * 80)

    spp_tracker = SPPValidationTracker()

    # Get prices for validation
    prices = {t: bundle.market_data[t].price
              for t in bundle.market_data}

    # Record signals for validation
    actionable = [s for s in signals
                  if s.get("direction") in ["BULLISH", "BEARISH"]
                  and s.get("confidence", 0) >= 0.5]

    recorded = 0
    for sig in actionable:
        ticker = sig.get("ticker", "")
        if ticker in prices:
            spp_tracker.record_signal(sig, prices[ticker])
            recorded += 1

    print(f"\n  SIGNAL RECORDING")
    print(f"  ─────────────────────────────────────────────")
    print(f"  Actionable Signals:  {len(actionable)}")
    print(f"  Recorded for Track:  {recorded}")

    # Update existing outcomes
    updates = spp_tracker.update_outcomes(prices)
    print(f"\n  OUTCOME UPDATES")
    print(f"  ─────────────────────────────────────────────")
    for horizon, count in updates.items():
        if count > 0:
            print(f"  {horizon}d horizon:       {count} updated")

    # Record consensus signals
    for c in consensus_results:
        if c.ticker in prices:
            spp_tracker.record_consensus(
                c.ticker,
                c.direction,
                c.systems,
                c.cascade_confidence,
                prices[c.ticker]
            )

    # =========================================================================
    # PHASE 6: Drift Detection
    # =========================================================================
    if enable_drift_detection:
        print("\n" + "=" * 80)
        print("[PHASE 6] MODEL DRIFT DETECTION")
        print("=" * 80)

        drift_alerts = spp_tracker.detect_drift()

        if drift_alerts:
            print(f"\n  DRIFT ALERTS ({len(drift_alerts)})")
            print(f"  ─────────────────────────────────────────────")
            for alert in drift_alerts:
                print(f"  [{alert.severity.value}] {alert.system_id}")
                print(f"      Historical WR: {alert.historical_win_rate:.1%}")
                print(f"      Recent WR:     {alert.recent_win_rate:.1%}")
                print(f"      Degradation:   {alert.degradation_pct:+.1f}%")
        else:
            print(f"\n  No drift alerts - all systems performing within expectations")

    # =========================================================================
    # PHASE 7: Generate S++ Report
    # =========================================================================
    print("\n" + "=" * 80)
    print("[PHASE 7] GENERATING S++ VALIDATION REPORT")
    print("=" * 80)

    spp_report = spp_tracker.generate_spp_report()

    print(f"\n  VALIDATION SUMMARY")
    print(f"  ─────────────────────────────────────────────")
    print(f"  Total Tracked:       {spp_report.total_signals_tracked}")
    print(f"  Validated:           {spp_report.total_validated}")
    print(f"  Validation Rate:     {spp_report.validation_rate:.0%}")

    print(f"\n  WIN RATES BY HORIZON")
    print(f"  ─────────────────────────────────────────────")
    print(f"  1-day:  {spp_report.win_rate_1d:.1%}  ({spp_report.avg_return_1d:+.2%})")
    print(f"  5-day:  {spp_report.win_rate_5d:.1%}  ({spp_report.avg_return_5d:+.2%})")
    print(f"  20-day: {spp_report.win_rate_20d:.1%}  ({spp_report.avg_return_20d:+.2%})")
    print(f"  60-day: {spp_report.win_rate_60d:.1%}  ({spp_report.avg_return_60d:+.2%})")

    print(f"\n  CONSENSUS PERFORMANCE")
    print(f"  ─────────────────────────────────────────────")
    print(f"  Consensus WR:        {spp_report.consensus_win_rate:.1%}")
    print(f"  Consensus Premium:   {spp_report.consensus_premium:+.1%}")
    print(f"  Calibration:         {spp_report.overall_calibration:.1%}")

    # =========================================================================
    # PHASE 8: Generate Dashboard
    # =========================================================================
    print("\n" + "=" * 80)
    print("[PHASE 8] GENERATING S++ DASHBOARD")
    print("=" * 80)

    dashboard_gen = IntelligenceDashboardGenerator()
    html = dashboard_gen.generate(report)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dashboard_path = output_dir / f"spp_intelligence_{timestamp}.html"
    dashboard_path.write_text(html)

    print(f"\n  Dashboard saved: {dashboard_path}")

    # Open in browser
    if open_dashboard:
        import subprocess
        try:
            subprocess.run(["open", str(dashboard_path)], check=True)
            print("  Dashboard opened in browser")
        except Exception:
            pass

    # =========================================================================
    # SUMMARY
    # =========================================================================
    print("\n" + "=" * 80)
    print("S++ TIER INTELLIGENCE COMPLETE")
    print("=" * 80)

    print(f"""
  POSTURE:        {report.market_posture.value}
  CONFIDENCE:     {report.posture_confidence:.0%}
  RISK SCORE:     {report.unified_risk_score:.0f}/100
  OPPORTUNITY:    {report.unified_opportunity_score:.0f}/100

  SYSTEMS:        {len(systems)} active
  SIGNALS:        {len(signals)} processed
  CONSENSUS:      {len(consensus_results)} detected
  ACTIONS:        {len(report.actions)} recommended

  VALIDATION:
    Tracked:      {spp_report.total_signals_tracked} signals
    Win Rate:     {spp_report.win_rate_20d:.1%} (20d)
    Calibration:  {spp_report.overall_calibration:.1%}
    Drift Alerts: {spp_report.systems_with_drift}

  DASHBOARD:      {dashboard_path}
    """)

    return report, str(dashboard_path)


def print_spp_status():
    """Print S++ system status"""
    from .real_data import RealDataConnector
    from .spp_validation import SPPValidationTracker

    print("=" * 80)
    print("KIRMANI S++ SYSTEM STATUS")
    print("=" * 80)

    # Check data sources
    connector = RealDataConnector()

    print("\n[DATA SOURCES]")
    if connector._yf:
        print("  yfinance:        AVAILABLE")
    else:
        print("  yfinance:        NOT INSTALLED")

    portfolio_files = list(connector.portfolio_dir.glob("portfolio.*.csv"))
    if portfolio_files:
        latest = max(portfolio_files, key=lambda p: p.stat().st_mtime)
        print(f"  IBKR Portfolio:  AVAILABLE ({latest.name})")
    else:
        print("  IBKR Portfolio:  NO FILES FOUND")

    # Check API keys
    import os
    print("\n[API KEYS]")
    for key in ["POLYGON_API_KEY", "FRED_API_KEY"]:
        val = os.environ.get(key, "")
        print(f"  {key}: {'SET' if val else 'NOT SET'}")

    # Check validation database
    tracker = SPPValidationTracker()
    report = tracker.generate_spp_report()

    print("\n[S++ VALIDATION DATABASE]")
    print(f"  Total Signals:   {report.total_signals_tracked}")
    print(f"  Validated:       {report.total_validated}")
    print(f"  20d Win Rate:    {report.win_rate_20d:.1%}")
    print(f"  Calibration:     {report.overall_calibration:.1%}")
    print(f"  Top Systems:     {', '.join(report.top_systems[:5]) or 'N/A'}")
    print(f"  Degraded:        {', '.join(report.degraded_systems[:3]) or 'None'}")

    print()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Kirmani S++ Tier Intelligence")
    parser.add_argument("--status", action="store_true", help="Print system status")
    parser.add_argument("--no-consensus", action="store_true", help="Disable consensus")
    parser.add_argument("--no-drift", action="store_true", help="Disable drift detection")
    parser.add_argument("--no-open", action="store_true", help="Don't open dashboard")
    parser.add_argument("--portfolio-dir", type=str, help="Portfolio CSV directory")
    parser.add_argument("--output-dir", type=str, help="Output directory")

    args = parser.parse_args()

    if args.status:
        print_spp_status()
    else:
        run_spp_intelligence(
            portfolio_dir=args.portfolio_dir,
            output_dir=args.output_dir,
            enable_consensus=not args.no_consensus,
            enable_drift_detection=not args.no_drift,
            open_dashboard=not args.no_open,
        )
