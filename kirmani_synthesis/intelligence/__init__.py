"""
Cross-System Intelligence Module
================================

Phase 3 of the Kirmani Synthesis Mesh integration plan.
Synthesizes deeper intelligence from multi-system signal combinations.

Three Intelligence Layers:
- MacroIntelligence: Sornette + Minsky + Murphy + CGMA
- RiskIntelligence: SENTINEL-8 + FRM + Greeks + VixLab
- AlphaIntelligence: Elite-Scanner + Wyckoff + Hurst + StratEngine

S-Tier Components:
- RealDataConnector: Live IBKR + Polygon + FRED data
- SignalProducerRegistry: Standardized signal emission
- LiveValidationTracker: Outcome tracking and accuracy monitoring

S++ Tier Components:
- ConsensusCascade: Multi-system consensus with cascade multipliers
- SPPValidationTracker: 60-day validation, drift detection, calibration
- CrossLayerValidator: MACRO/RISK/ALPHA alignment scoring
"""

from .macro import MacroIntelligence, MacroIntelligenceReport
from .risk import RiskIntelligence, RiskIntelligenceReport
from .alpha import AlphaIntelligence, AlphaIntelligenceReport
from .engine import CrossSystemIntelligenceEngine, UnifiedIntelligenceReport
from .dashboard import IntelligenceDashboardGenerator

# S-Tier modules
from .real_data import (
    RealDataConnector,
    RealSignalGenerator,
    RealDataBundle,
    PortfolioPosition,
    MarketSnapshot,
    run_real_intelligence,
)
from .signal_producers import (
    SignalProducerRegistry,
    StandardSignal,
    SignalDirection,
    VIXLabProducer,
    SornetteProducer,
    MinskyProducer,
    SentinelProducer,
    MurphyProducer,
    WyckoffProducer,
    HurstProducer,
    EliteScannerProducer,
    GreeksProducer,
)
from .validation import (
    LiveValidationTracker,
    TrackedSignal,
    SystemAccuracy,
    ValidationReport,
    run_validation_update,
)

# S++ Tier modules
from .spp_validation import (
    SPPValidationTracker,
    SPPValidationReport,
    DriftAlert,
    DriftSeverity,
    CalibrationResult,
)
from .consensus import (
    ConsensusCascade,
    ConsensusLevel,
    ConsensusResult,
    CrossLayerValidator,
)
from .spp_runner import (
    run_spp_intelligence,
    print_spp_status,
)

__all__ = [
    # Core Intelligence
    "MacroIntelligence",
    "MacroIntelligenceReport",
    "RiskIntelligence",
    "RiskIntelligenceReport",
    "AlphaIntelligence",
    "AlphaIntelligenceReport",
    "CrossSystemIntelligenceEngine",
    "UnifiedIntelligenceReport",
    "IntelligenceDashboardGenerator",
    # S-Tier Real Data
    "RealDataConnector",
    "RealSignalGenerator",
    "RealDataBundle",
    "PortfolioPosition",
    "MarketSnapshot",
    "run_real_intelligence",
    # Signal Producers
    "SignalProducerRegistry",
    "StandardSignal",
    "SignalDirection",
    "VIXLabProducer",
    "SornetteProducer",
    "MinskyProducer",
    "SentinelProducer",
    "MurphyProducer",
    "WyckoffProducer",
    "HurstProducer",
    "EliteScannerProducer",
    "GreeksProducer",
    # Validation
    "LiveValidationTracker",
    "TrackedSignal",
    "SystemAccuracy",
    "ValidationReport",
    "run_validation_update",
    # S++ Tier
    "SPPValidationTracker",
    "SPPValidationReport",
    "DriftAlert",
    "DriftSeverity",
    "CalibrationResult",
    "ConsensusCascade",
    "ConsensusLevel",
    "ConsensusResult",
    "CrossLayerValidator",
    "run_spp_intelligence",
    "print_spp_status",
]
