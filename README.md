# Kirmani Synthesis Mesh

Elite Signal Synthesis for 48+ Trading Systems

The Synthesis Mesh aggregates signals from all Kirmani trading systems, applies Bayesian fusion, detects contradictions and confirmations, and produces unified actionable intelligence.

## Architecture

```
MACRO LAYER  →  Sornette, Minsky, Murphy, CGMA, BubbleRadar
RISK LAYER   →  SENTINEL-8, FRM Elite, Squeeze, VIX Lab, Greeks
ALPHA LAYER  →  Elite Scanner, StratEngine, Wyckoff, Hurst, Power Hour
                              ↓
                    FUSION ENGINE (Bayesian aggregation)
                              ↓
                    QUICK WINS (Automated rules)
                              ↓
                    DECISION DASHBOARD (Actionable intelligence)
```

## Quick Wins Rules

| Rule | Trigger | Action |
|------|---------|--------|
| Squeeze + Bubble | SENTINEL-8 squeeze + Minsky Phase ≥4 | 75% position reduction |
| Murphy + Sornette | Intermarket divergence + LPPLS ≥0.7 | 50% equity reduction |
| Scanner + Wyckoff | Elite Scanner + Wyckoff confirmation | Full position entry |
| Hurst + Greeks | Cycle trough + favorable Greeks | Buy calls at trough |
| Triple Confirmation | 3+ systems align on direction | Increase position 25% |
| Crisis Exit | Crisis regime detected | 50% reduction + VIX hedge |

## Installation

```bash
pip install kirmani-synthesis-mesh
```

## Usage

### CLI

```bash
# Run synthesis on signals from a file
synthesis run --file signals.json --open-dashboard

# Run demo with sample signals
synthesis demo --open

# Check system status
synthesis status

# Open latest dashboard
synthesis open
```

### Python API

```python
from kirmani_synthesis import UnifiedSynthesisRunner

runner = UnifiedSynthesisRunner()

# Add signals from any system
runner.add_signal({
    "system_id": "sornette-lppls",
    "signal_type": "CRASH_HAZARD",
    "ticker": "SPY",
    "direction": "BEARISH",
    "confidence": 0.72,
    "metadata": {"hazard_rate": 0.72}
})

runner.add_signal({
    "system_id": "elite-scanner",
    "signal_type": "ACCUMULATION",
    "ticker": "NVDA",
    "direction": "BULLISH",
    "confidence": 0.82,
})

# Run synthesis
report = runner.run()

# Print summary
runner.print_summary()

# Save outputs
runner.save_report("synthesis_report.json")
runner.save_dashboard("synthesis_dashboard.html")
```

## Components

### FusionEngine
Bayesian aggregation of signals across all systems with confidence-weighted fusion.

### QuickWinsEngine
Rule-based position recommendations for common multi-system patterns.

### ContradictionDetector
Identifies conflicting signals between systems for the same ticker.

### ConfirmationDetector
Finds cross-system confirmations that increase conviction.

### DashboardGenerator
Real-time HTML dashboard with market state, recommendations, and alerts.

### UnifiedSynthesisRunner
Main orchestrator that ties all components together.

## License

MIT License - Kirmani Partners LP
