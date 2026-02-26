"""
Kirmani Synthesis Mesh CLI

Command-line interface for running the Synthesis Mesh,
generating reports, and viewing dashboards.
"""

import argparse
import json
import logging
import os
import sys
import webbrowser
from datetime import datetime
from pathlib import Path
from typing import Optional

from .models import SynthesisReport
from .runners import UnifiedSynthesisRunner

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def load_signals_from_file(filepath: str) -> list[dict]:
    """Load signals from JSON file"""
    with open(filepath) as f:
        data = json.load(f)

    # Handle both single signal and list of signals
    if isinstance(data, list):
        return data
    elif isinstance(data, dict):
        if "signals" in data:
            return data["signals"]
        else:
            return [data]
    return []


def load_signals_from_dir(dirpath: str) -> list[dict]:
    """Load all signals from a directory of JSON files"""
    signals = []
    path = Path(dirpath)

    for json_file in path.glob("*.json"):
        try:
            with open(json_file) as f:
                data = json.load(f)
            if isinstance(data, list):
                signals.extend(data)
            elif isinstance(data, dict):
                if "signals" in data:
                    signals.extend(data["signals"])
                else:
                    signals.append(data)
        except Exception as e:
            logger.warning(f"Failed to load {json_file}: {e}")

    return signals


def cmd_run(args):
    """Run synthesis on input signals"""
    runner = UnifiedSynthesisRunner(output_dir=args.output_dir)

    # Load signals
    signals = []

    if args.file:
        signals.extend(load_signals_from_file(args.file))

    if args.dir:
        signals.extend(load_signals_from_dir(args.dir))

    if args.stdin:
        stdin_data = sys.stdin.read()
        try:
            data = json.loads(stdin_data)
            if isinstance(data, list):
                signals.extend(data)
            else:
                signals.append(data)
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON from stdin: {e}")
            return 1

    if not signals:
        logger.error("No signals provided. Use --file, --dir, or --stdin")
        return 1

    logger.info(f"Loaded {len(signals)} signals")

    # Add signals to runner
    runner.add_signals(signals)

    # Run synthesis
    report = runner.run()

    # Output
    if args.json:
        print(json.dumps(report.to_dict(), indent=2, default=str))
    else:
        runner.print_summary()

    # Save outputs
    if args.save_report:
        report_path = runner.save_report(args.save_report)
        logger.info(f"Report saved: {report_path}")

    if args.save_dashboard or args.open_dashboard:
        dashboard_path = args.save_dashboard or os.path.join(
            args.output_dir or os.path.expanduser("~/.kirmani_reports"),
            f"synthesis_dashboard_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
        )
        runner.save_dashboard(dashboard_path)
        logger.info(f"Dashboard saved: {dashboard_path}")

        if args.open_dashboard:
            webbrowser.open(f"file://{os.path.abspath(dashboard_path)}")

    return 0


def cmd_demo(args):
    """Run synthesis with demo signals"""
    demo_signals = [
        # Sornette LPPLS - Crash hazard warning
        {
            "system_id": "sornette-lppls",
            "signal_type": "CRASH_HAZARD",
            "ticker": "SPY",
            "direction": "BEARISH",
            "confidence": 0.72,
            "metadata": {
                "hazard_rate": 0.72,
                "tc_estimate": "2026-03-15",
                "lppl_fit_r2": 0.89,
            }
        },
        # Minsky bubble analyzer
        {
            "system_id": "minsky-analyzer",
            "signal_type": "BUBBLE_PHASE",
            "ticker": "SPY",
            "direction": "BEARISH",
            "confidence": 0.68,
            "metadata": {
                "phase": 3,
                "phase_name": "Euphoria",
                "bubble_score": 78,
            }
        },
        # Murphy intermarket
        {
            "system_id": "murphy-intermarket",
            "signal_type": "INTERMARKET_REGIME",
            "ticker": "SPY",
            "direction": "BEARISH",
            "confidence": 0.65,
            "metadata": {
                "regime": "RISK_OFF",
                "dollar_signal": "STRONG",
                "bond_equity_divergence": True,
            }
        },
        # SENTINEL-8 squeeze
        {
            "system_id": "sentinel-8",
            "signal_type": "SQUEEZE_ALERT",
            "ticker": "APPN",
            "direction": "BULLISH",
            "confidence": 0.75,
            "metadata": {
                "squeeze_score": 85,
                "short_interest": 32.5,
                "days_to_cover": 8.2,
                "ctb": 45.0,
            }
        },
        # Elite scanner entry
        {
            "system_id": "elite-scanner",
            "signal_type": "ACCUMULATION",
            "ticker": "NVDA",
            "direction": "BULLISH",
            "confidence": 0.82,
            "metadata": {
                "pattern": "CUP_AND_HANDLE",
                "volume_surge": 2.3,
                "relative_strength": 95,
            }
        },
        # Wyckoff confirmation
        {
            "system_id": "wyckoff-elite",
            "signal_type": "ACCUMULATION",
            "ticker": "NVDA",
            "direction": "BULLISH",
            "confidence": 0.78,
            "metadata": {
                "phase": "SPRING",
                "volume_analysis": "DEMAND_DOMINATES",
            }
        },
        # Greeks warning
        {
            "system_id": "greeks-trader",
            "signal_type": "GREEK_EXPOSURE",
            "ticker": "SPY",
            "direction": "NEUTRAL",
            "confidence": 0.60,
            "metadata": {
                "portfolio_delta": 0.65,
                "portfolio_gamma": 0.12,
                "vega_exposure": -15000,
                "theta_bleed": -850,
            }
        },
        # Hurst cycle
        {
            "system_id": "hurst-cycles",
            "signal_type": "CYCLE_TURN",
            "ticker": "QQQ",
            "direction": "BULLISH",
            "confidence": 0.70,
            "metadata": {
                "cycle_phase": "TROUGH",
                "dominant_cycle": 40,
                "next_peak_bars": 20,
            }
        },
    ]

    logger.info("Running synthesis with demo signals...")

    runner = UnifiedSynthesisRunner(output_dir=args.output_dir)
    runner.add_signals(demo_signals)
    report = runner.run()

    if args.json:
        print(json.dumps(report.to_dict(), indent=2, default=str))
    else:
        runner.print_summary()

    # Save dashboard
    dashboard_path = os.path.join(
        args.output_dir or os.path.expanduser("~/.kirmani_reports"),
        f"synthesis_demo_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
    )
    runner.save_dashboard(dashboard_path)
    logger.info(f"Dashboard saved: {dashboard_path}")

    if args.open:
        webbrowser.open(f"file://{os.path.abspath(dashboard_path)}")

    return 0


def cmd_status(args):
    """Show system status and configuration"""
    print("\n" + "=" * 50)
    print("KIRMANI SYNTHESIS MESH - STATUS")
    print("=" * 50)

    output_dir = args.output_dir or os.path.expanduser("~/.kirmani_reports")
    print(f"\nOutput Directory: {output_dir}")

    # Check for recent reports
    reports_path = Path(output_dir)
    if reports_path.exists():
        reports = list(reports_path.glob("synthesis_report_*.json"))
        dashboards = list(reports_path.glob("synthesis_dashboard_*.html"))

        print(f"Reports Found: {len(reports)}")
        print(f"Dashboards Found: {len(dashboards)}")

        if reports:
            latest = max(reports, key=lambda p: p.stat().st_mtime)
            print(f"\nLatest Report: {latest.name}")
            print(f"  Modified: {datetime.fromtimestamp(latest.stat().st_mtime)}")
    else:
        print("Output directory does not exist yet")

    print("\nSupported Systems:")
    systems = [
        "sornette-lppls", "minsky-analyzer", "murphy-intermarket",
        "sentinel-8", "elite-scanner", "wyckoff-elite",
        "hurst-cycles", "greeks-trader", "natenberg-options",
        "bfo-quant", "dow-theory", "cgma-system",
    ]
    for sys in systems:
        print(f"  - {sys}")

    print("\nQuick Win Rules:")
    rules = [
        "squeeze_bubble - Squeeze + Bubble = 75% reduction",
        "murphy_sornette - Murphy + Sornette = 50% equity reduction",
        "scanner_wyckoff - Scanner + Wyckoff = Full position entry",
        "hurst_greeks - Hurst trough + Greeks = Buy calls",
        "triple_confirmation - 3+ systems agree = Increase 25%",
        "crisis_exit - Crisis mode = 50% reduction + hedge",
    ]
    for rule in rules:
        print(f"  - {rule}")

    print()
    return 0


def cmd_open(args):
    """Open the most recent dashboard"""
    output_dir = args.output_dir or os.path.expanduser("~/.kirmani_reports")
    reports_path = Path(output_dir)

    if not reports_path.exists():
        logger.error("No reports directory found")
        return 1

    dashboards = list(reports_path.glob("synthesis_dashboard_*.html"))
    dashboards.extend(reports_path.glob("synthesis_demo_*.html"))

    if not dashboards:
        logger.error("No dashboards found")
        return 1

    latest = max(dashboards, key=lambda p: p.stat().st_mtime)
    logger.info(f"Opening: {latest}")
    webbrowser.open(f"file://{latest.absolute()}")

    return 0


def main():
    parser = argparse.ArgumentParser(
        description="Kirmani Synthesis Mesh - Cross-System Intelligence Fusion",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run synthesis on signals from a file
  synthesis run --file signals.json --open-dashboard

  # Run synthesis on all signals in a directory
  synthesis run --dir ./signal_outputs/ --save-report report.json

  # Pipe signals from another command
  cat signals.json | synthesis run --stdin --json

  # Run demo with sample signals
  synthesis demo --open

  # Check system status
  synthesis status

  # Open latest dashboard
  synthesis open
        """
    )

    parser.add_argument(
        "--output-dir", "-o",
        help="Directory for output files (default: ~/.kirmani_reports)",
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Run command
    run_parser = subparsers.add_parser("run", help="Run synthesis on signals")
    run_parser.add_argument("--file", "-f", help="JSON file with signals")
    run_parser.add_argument("--dir", "-d", help="Directory of signal JSON files")
    run_parser.add_argument("--stdin", action="store_true", help="Read signals from stdin")
    run_parser.add_argument("--json", "-j", action="store_true", help="Output as JSON")
    run_parser.add_argument("--save-report", help="Save report to this path")
    run_parser.add_argument("--save-dashboard", help="Save dashboard to this path")
    run_parser.add_argument("--open-dashboard", action="store_true", help="Open dashboard in browser")
    run_parser.set_defaults(func=cmd_run)

    # Demo command
    demo_parser = subparsers.add_parser("demo", help="Run with demo signals")
    demo_parser.add_argument("--json", "-j", action="store_true", help="Output as JSON")
    demo_parser.add_argument("--open", action="store_true", help="Open dashboard in browser")
    demo_parser.set_defaults(func=cmd_demo)

    # Status command
    status_parser = subparsers.add_parser("status", help="Show system status")
    status_parser.set_defaults(func=cmd_status)

    # Open command
    open_parser = subparsers.add_parser("open", help="Open latest dashboard")
    open_parser.set_defaults(func=cmd_open)

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 0

    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
