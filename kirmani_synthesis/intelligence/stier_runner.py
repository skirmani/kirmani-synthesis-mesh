"""
S-Tier Intelligence Runner
==========================

The ultimate integration of all S-tier components:
- Real data from IBKR + Polygon + FRED
- Standardized signal producers for 9 key systems
- Cross-system intelligence synthesis
- Live validation tracking
- Elite dashboard generation

This is the production-grade runner for S-tier operations.

Usage:
    python -m kirmani_synthesis.intelligence.stier_runner

Or programmatically:
    from kirmani_synthesis.intelligence import run_stier_intelligence
    report, dashboard_path = run_stier_intelligence()
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


def run_stier_intelligence(
    portfolio_dir: str = None,
    output_dir: str = None,
    track_validation: bool = True,
    open_dashboard: bool = True,
) -> Tuple[Any, str]:
    """
    Run complete S-tier intelligence pipeline.

    Args:
        portfolio_dir: Directory containing portfolio CSVs
        output_dir: Directory for output files
        track_validation: Record signals for validation tracking
        open_dashboard: Open dashboard in browser

    Returns:
        Tuple of (UnifiedIntelligenceReport, dashboard_path)
    """
    from .real_data import RealDataConnector, RealSignalGenerator
    from .signal_producers import SignalProducerRegistry
    from .engine import CrossSystemIntelligenceEngine
    from .dashboard import IntelligenceDashboardGenerator
    from .validation import LiveValidationTracker

    print("=" * 70)
    print("KIRMANI S-TIER INTELLIGENCE SYSTEM")
    print("=" * 70)
    print(f"Timestamp: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}")

    # Initialize output directory
    if output_dir is None:
        output_dir = Path.home() / ".kirmani_reports"
    else:
        output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    # =========================================================================
    # PHASE 1: Load Real Data
    # =========================================================================
    print("\n" + "-" * 70)
    print("[PHASE 1] LOADING REAL DATA")
    print("-" * 70)

    connector = RealDataConnector(portfolio_dir=portfolio_dir)
    bundle = connector.get_full_data_bundle()

    print(f"  Portfolio Positions: {len(bundle.portfolio)}")
    print(f"  Market Data Tickers: {len(bundle.market_data)}")
    print(f"  VIX Level: {bundle.vix_level:.2f}")
    print(f"  SPY Price: ${bundle.spy_price:.2f}")
    print(f"  QQQ Price: ${bundle.qqq_price:.2f}")
    print(f"  Fed Funds Rate: {bundle.fed_funds_rate:.2f}%")
    print(f"  10Y Treasury: {bundle.treasury_10y:.2f}%")

    # Print top positions
    if bundle.portfolio:
        print("\n  Top 5 Positions:")
        sorted_positions = sorted(
            bundle.portfolio,
            key=lambda p: abs(p.market_value),
            reverse=True
        )[:5]
        for pos in sorted_positions:
            pnl_str = f"{pos.pnl_pct:+.1f}%" if pos.pnl_pct else "N/A"
            print(f"    {pos.ticker:8s} | {pos.side:5s} | ${pos.market_value:>10,.2f} | P&L: {pnl_str}")

    # =========================================================================
    # PHASE 2: Generate Signals from Real Data
    # =========================================================================
    print("\n" + "-" * 70)
    print("[PHASE 2] GENERATING SIGNALS FROM REAL DATA")
    print("-" * 70)

    signal_generator = RealSignalGenerator(connector)
    signals = signal_generator.generate_signals_from_portfolio(bundle)

    # Group signals by system
    signals_by_system: Dict[str, List] = {}
    for sig in signals:
        sys_id = sig.get("system_id", "unknown")
        if sys_id not in signals_by_system:
            signals_by_system[sys_id] = []
        signals_by_system[sys_id].append(sig)

    print(f"  Total Signals: {len(signals)}")
    print("\n  Signals by System:")
    for sys_id, sys_signals in sorted(signals_by_system.items()):
        print(f"    {sys_id:20s}: {len(sys_signals)} signals")

    # =========================================================================
    # PHASE 3: Run Cross-System Intelligence
    # =========================================================================
    print("\n" + "-" * 70)
    print("[PHASE 3] RUNNING CROSS-SYSTEM INTELLIGENCE")
    print("-" * 70)

    engine = CrossSystemIntelligenceEngine()
    report = engine.synthesize(signals)

    print(f"\n  MARKET POSTURE: {report.market_posture.value}")
    print(f"  Confidence: {report.posture_confidence:.0%}")
    print(f"  Rationale: {report.posture_rationale}")

    print(f"\n  Unified Metrics:")
    print(f"    Risk Score: {report.unified_risk_score:.0f}/100")
    print(f"    Opportunity Score: {report.unified_opportunity_score:.0f}/100")
    print(f"    Risk/Reward Balance: {report.risk_reward_balance:+.2f}")
    print(f"    Full Stack Alignment: {report.full_stack_alignment:.0%}")

    print(f"\n  Position Guidance:")
    print(f"    Equity Exposure: {report.recommended_equity_exposure:.0%}")
    print(f"    Cash Level: {report.recommended_cash_level:.0%}")
    print(f"    Max Single Position: {report.max_single_position_pct:.0%}")
    print(f"    Hedge Ratio: {report.hedge_ratio:.0%}")

    if report.critical_actions:
        print(f"\n  CRITICAL ACTIONS ({len(report.critical_actions)}):")
        for action in report.critical_actions:
            ticker_str = f" {action.ticker}" if action.ticker else ""
            print(f"    [{action.priority.value}] {action.action_type}{ticker_str}")
            print(f"       {action.description}")

    # =========================================================================
    # PHASE 4: Track for Validation
    # =========================================================================
    if track_validation:
        print("\n" + "-" * 70)
        print("[PHASE 4] RECORDING SIGNALS FOR VALIDATION")
        print("-" * 70)

        tracker = LiveValidationTracker()

        # Get prices for all signaled tickers
        signaled_tickers = set(s.get("ticker") for s in signals if s.get("ticker"))
        prices = {t: bundle.market_data[t].price
                  for t in signaled_tickers
                  if t in bundle.market_data}

        # Record actionable signals
        actionable_signals = [s for s in signals
                             if s.get("direction") in ["BULLISH", "BEARISH"]
                             and s.get("confidence", 0) >= 0.6]

        signal_ids = tracker.record_signals_batch(actionable_signals, prices)
        print(f"  Recorded {len(signal_ids)} signals for validation tracking")

        # Update any pending outcomes
        updated = tracker.update_outcomes(prices)
        print(f"  Updated {updated} existing signal outcomes")

        # Print validation summary
        val_report = tracker.generate_validation_report()
        print(f"\n  Validation Summary:")
        print(f"    Total Tracked: {val_report.total_signals}")
        print(f"    Validated: {val_report.validated_signals}")
        print(f"    Overall Win Rate: {val_report.overall_win_rate:.1%}")
        print(f"    Model Drift: {val_report.model_drift_score:.3f}")
        print(f"    Confidence Calibration: {val_report.confidence_calibration:.3f}")

    # =========================================================================
    # PHASE 5: Generate Dashboard
    # =========================================================================
    print("\n" + "-" * 70)
    print("[PHASE 5] GENERATING DASHBOARD")
    print("-" * 70)

    dashboard_gen = IntelligenceDashboardGenerator()
    html = dashboard_gen.generate(report)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dashboard_path = output_dir / f"stier_intelligence_{timestamp}.html"
    dashboard_path.write_text(html)

    print(f"  Dashboard saved: {dashboard_path}")

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
    print("\n" + "=" * 70)
    print("S-TIER INTELLIGENCE COMPLETE")
    print("=" * 70)
    print(f"\nPosture: {report.market_posture.value}")
    print(f"Systems: {len(report.systems_contributing)} contributing")
    print(f"Signals: {len(signals)} processed")
    print(f"Actions: {len(report.actions)} recommended, {len(report.critical_actions)} critical")
    print(f"\nDashboard: {dashboard_path}")
    print()

    return report, str(dashboard_path)


def run_validation_only():
    """Run validation update without generating new signals"""
    from .validation import run_validation_update
    run_validation_update()


def print_system_status():
    """Print system connectivity status"""
    from .real_data import RealDataConnector

    print("=" * 70)
    print("KIRMANI S-TIER SYSTEM STATUS")
    print("=" * 70)

    connector = RealDataConnector()

    # Check yfinance
    print("\n[Data Sources]")
    if connector._yf:
        print("  yfinance: AVAILABLE")
    else:
        print("  yfinance: NOT INSTALLED")

    # Check for portfolio files
    portfolio_files = list(connector.portfolio_dir.glob("portfolio.*.csv"))
    if portfolio_files:
        latest = max(portfolio_files, key=lambda p: p.stat().st_mtime)
        print(f"  IBKR Portfolio: AVAILABLE ({latest.name})")
    else:
        print("  IBKR Portfolio: NO FILES FOUND")

    # Check API keys
    import os
    print("\n[API Keys]")
    polygon_key = os.environ.get("POLYGON_API_KEY", "")
    fred_key = os.environ.get("FRED_API_KEY", "")
    print(f"  POLYGON_API_KEY: {'SET' if polygon_key else 'NOT SET'}")
    print(f"  FRED_API_KEY: {'SET' if fred_key else 'NOT SET'}")

    # Check validation database
    val_db = Path.home() / ".kirmani_reports" / "validation_tracker.db"
    if val_db.exists():
        import sqlite3
        conn = sqlite3.connect(str(val_db))
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM signals")
        count = cursor.fetchone()[0]
        conn.close()
        print(f"\n[Validation Database]")
        print(f"  Location: {val_db}")
        print(f"  Tracked Signals: {count}")
    else:
        print(f"\n[Validation Database]")
        print(f"  Not initialized yet")

    print()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Kirmani S-Tier Intelligence")
    parser.add_argument("--status", action="store_true", help="Print system status")
    parser.add_argument("--validate", action="store_true", help="Run validation update only")
    parser.add_argument("--no-track", action="store_true", help="Skip validation tracking")
    parser.add_argument("--no-open", action="store_true", help="Don't open dashboard")
    parser.add_argument("--portfolio-dir", type=str, help="Portfolio CSV directory")
    parser.add_argument("--output-dir", type=str, help="Output directory")

    args = parser.parse_args()

    if args.status:
        print_system_status()
    elif args.validate:
        run_validation_only()
    else:
        run_stier_intelligence(
            portfolio_dir=args.portfolio_dir,
            output_dir=args.output_dir,
            track_validation=not args.no_track,
            open_dashboard=not args.no_open,
        )
