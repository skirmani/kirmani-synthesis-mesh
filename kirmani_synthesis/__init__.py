"""
Kirmani Synthesis Mesh

Elite Signal Synthesis for 48+ Trading Systems

The Synthesis Mesh aggregates signals from all Kirmani trading systems,
applies Bayesian fusion, detects contradictions and confirmations,
and produces unified actionable intelligence.

Architecture:
    MACRO LAYER → Sornette, Minsky, Murphy, CGMA, BubbleRadar
    RISK LAYER  → SENTINEL-8, FRM Elite, Squeeze, VIX Lab, Greeks
    ALPHA LAYER → Elite Scanner, StratEngine, Wyckoff, Hurst, Power Hour
                              ↓
                    FUSION ENGINE (Bayesian aggregation)
                              ↓
                    QUICK WINS (Automated rules)
                              ↓
                    DECISION DASHBOARD (Actionable intelligence)
"""

from .fusion import FusionEngine, SynthesizedView
from .rules import QuickWinsEngine, Rule, RuleResult
from .detection import ContradictionDetector, ConfirmationDetector
from .dashboard import DashboardGenerator
from .runners import UnifiedSynthesisRunner
from .models import MarketState, PositionRecommendation, SynthesisReport
from .intelligence import (
    MacroIntelligence,
    MacroIntelligenceReport,
    RiskIntelligence,
    RiskIntelligenceReport,
    AlphaIntelligence,
    AlphaIntelligenceReport,
    CrossSystemIntelligenceEngine,
    UnifiedIntelligenceReport,
)

__version__ = "2.0.0"  # Phase 3 Cross-System Intelligence
__all__ = [
    # Phase 1-2: Fusion & Quick Wins
    "FusionEngine",
    "SynthesizedView",
    "QuickWinsEngine",
    "Rule",
    "RuleResult",
    "ContradictionDetector",
    "ConfirmationDetector",
    "DashboardGenerator",
    "UnifiedSynthesisRunner",
    "MarketState",
    "PositionRecommendation",
    "SynthesisReport",
    # Phase 3: Cross-System Intelligence
    "MacroIntelligence",
    "MacroIntelligenceReport",
    "RiskIntelligence",
    "RiskIntelligenceReport",
    "AlphaIntelligence",
    "AlphaIntelligenceReport",
    "CrossSystemIntelligenceEngine",
    "UnifiedIntelligenceReport",
]
