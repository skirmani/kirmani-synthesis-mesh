"""
Cross-System Intelligence Module

Phase 3 of the Kirmani Synthesis Mesh integration plan.
Synthesizes deeper intelligence from multi-system signal combinations.

Three Intelligence Layers:
- MacroIntelligence: Sornette + Minsky + Murphy + CGMA
- RiskIntelligence: SENTINEL-8 + FRM + Greeks + VixLab
- AlphaIntelligence: Elite-Scanner + Wyckoff + Hurst + StratEngine
"""

from .macro import MacroIntelligence, MacroIntelligenceReport
from .risk import RiskIntelligence, RiskIntelligenceReport
from .alpha import AlphaIntelligence, AlphaIntelligenceReport
from .engine import CrossSystemIntelligenceEngine, UnifiedIntelligenceReport
from .dashboard import IntelligenceDashboardGenerator

__all__ = [
    "MacroIntelligence",
    "MacroIntelligenceReport",
    "RiskIntelligence",
    "RiskIntelligenceReport",
    "AlphaIntelligence",
    "AlphaIntelligenceReport",
    "CrossSystemIntelligenceEngine",
    "UnifiedIntelligenceReport",
    "IntelligenceDashboardGenerator",
]
