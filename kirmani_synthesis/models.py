"""
Synthesis Mesh Data Models

Core data structures for cross-system signal synthesis.
"""

from datetime import datetime
from enum import Enum
from typing import Any, Optional
from uuid import uuid4

from pydantic import BaseModel, Field


class MarketRegime(str, Enum):
    """Overall market regime classification"""
    RISK_ON = "RISK_ON"
    RISK_OFF = "RISK_OFF"
    NEUTRAL = "NEUTRAL"
    TRANSITION = "TRANSITION"
    CRISIS = "CRISIS"


class ActionType(str, Enum):
    """Types of recommended actions"""
    REDUCE_LONG = "REDUCE_LONG"
    REDUCE_SHORT = "REDUCE_SHORT"
    INCREASE_LONG = "INCREASE_LONG"
    INCREASE_SHORT = "INCREASE_SHORT"
    HEDGE = "HEDGE"
    EXIT = "EXIT"
    HOLD = "HOLD"
    ENTER_LONG = "ENTER_LONG"
    ENTER_SHORT = "ENTER_SHORT"


class UrgencyLevel(str, Enum):
    """Action urgency"""
    IMMEDIATE = "IMMEDIATE"  # Within minutes
    URGENT = "URGENT"        # Within hours
    NORMAL = "NORMAL"        # Within day
    LOW = "LOW"              # When convenient


class MarketState(BaseModel):
    """
    Synthesized market state from all systems

    Represents the unified view of market conditions
    derived from aggregating signals across all layers.
    """

    timestamp: datetime = Field(default_factory=datetime.utcnow)

    # Macro view
    regime: MarketRegime = MarketRegime.NEUTRAL
    regime_confidence: float = Field(default=0.5, ge=0, le=1)

    # Crash/bubble metrics
    crash_hazard: float = Field(default=0.0, ge=0, le=1)
    bubble_phase: int = Field(default=1, ge=1, le=5)  # Minsky 1-5
    bubble_confidence: float = Field(default=0.5, ge=0, le=1)

    # Risk metrics
    overall_risk_score: float = Field(default=50, ge=0, le=100)
    squeeze_risk: float = Field(default=0.0, ge=0, le=1)
    vol_regime: str = "NORMAL"  # LOW, NORMAL, ELEVATED, CRISIS

    # Intermarket
    intermarket_signal: str = "NEUTRAL"  # BULLISH, BEARISH, NEUTRAL
    bonds_signal: str = "NEUTRAL"
    dollar_signal: str = "NEUTRAL"
    commodities_signal: str = "NEUTRAL"

    # Alpha opportunity
    momentum_opportunity: float = Field(default=0.5, ge=0, le=1)
    mean_reversion_opportunity: float = Field(default=0.5, ge=0, le=1)

    # Aggregated confidence
    signal_count: int = 0
    system_agreement: float = Field(default=0.5, ge=0, le=1)  # How much systems agree

    # Components (raw signals by system)
    components: dict[str, Any] = Field(default_factory=dict)


class PositionRecommendation(BaseModel):
    """
    Specific position recommendation from synthesis

    When the synthesis engine determines an action is needed,
    it outputs a PositionRecommendation with specific details.
    """

    id: str = Field(default_factory=lambda: str(uuid4()))
    timestamp: datetime = Field(default_factory=datetime.utcnow)

    ticker: str
    action: ActionType
    urgency: UrgencyLevel

    # Position sizing
    current_position_pct: float = 0  # Current % of portfolio
    target_position_pct: float = 0   # Target % of portfolio
    change_pct: float = 0            # Amount to change

    # Execution
    execution_algo: str = "VWAP"     # VWAP, TWAP, MARKET, LIMIT
    execution_window: str = "10:30"  # Recommended execution time
    price_limit: Optional[float] = None

    # Rationale
    primary_trigger: str = ""        # What triggered this
    supporting_signals: list[str] = Field(default_factory=list)
    confidence: float = Field(default=0.5, ge=0, le=1)

    # Risk parameters
    stop_loss_pct: Optional[float] = None
    take_profit_pct: Optional[float] = None
    max_loss_dollars: Optional[float] = None


class QuickWinTrigger(BaseModel):
    """
    When a Quick Win rule fires, this captures the details
    """

    id: str = Field(default_factory=lambda: str(uuid4()))
    timestamp: datetime = Field(default_factory=datetime.utcnow)

    rule_name: str
    rule_id: str

    # Conditions that were met
    conditions_met: dict[str, Any] = Field(default_factory=dict)

    # Resulting action
    action: ActionType
    target_ticker: str
    position_change_pct: float

    # Confidence from contributing signals
    confidence: float = Field(default=0.5, ge=0, le=1)

    # Source signals that triggered this
    source_signals: list[str] = Field(default_factory=list)


class ContradictionAlert(BaseModel):
    """
    Alert when two systems emit conflicting signals
    """

    id: str = Field(default_factory=lambda: str(uuid4()))
    timestamp: datetime = Field(default_factory=datetime.utcnow)

    # The conflicting systems
    system_a: str
    system_b: str

    # Their signals
    signal_a_type: str
    signal_a_direction: str
    signal_a_confidence: float

    signal_b_type: str
    signal_b_direction: str
    signal_b_confidence: float

    # Affected ticker
    ticker: str

    # Severity
    severity: float = Field(ge=0, le=1)  # How severe is this conflict

    # Resolution
    recommended_action: str = "WAIT"  # WAIT, FAVOR_A, FAVOR_B, REDUCE
    resolution_rationale: str = ""


class ConfirmationAlert(BaseModel):
    """
    Alert when multiple systems agree on a signal
    """

    id: str = Field(default_factory=lambda: str(uuid4()))
    timestamp: datetime = Field(default_factory=datetime.utcnow)

    # Systems that agree
    systems: list[str] = Field(default_factory=list)
    system_count: int = 0
    layer_count: int = 0  # How many layers (MACRO, RISK, ALPHA) agree

    # The consensus
    ticker: str
    direction: str
    combined_confidence: float = Field(ge=0, le=1)

    # Resulting strength
    confirmation_strength: str = "WEAK"  # WEAK, MODERATE, STRONG, VERY_STRONG


class SynthesisReport(BaseModel):
    """
    Complete synthesis report for a given time

    This is the main output of the Synthesis Mesh,
    containing all synthesized intelligence.
    """

    id: str = Field(default_factory=lambda: str(uuid4()))
    generated_at: datetime = Field(default_factory=datetime.utcnow)

    # Market state
    market_state: MarketState

    # Recommendations
    recommendations: list[PositionRecommendation] = Field(default_factory=list)

    # Quick wins that fired
    quick_wins_triggered: list[QuickWinTrigger] = Field(default_factory=list)

    # Contradictions detected
    contradictions: list[ContradictionAlert] = Field(default_factory=list)

    # Confirmations detected
    confirmations: list[ConfirmationAlert] = Field(default_factory=list)

    # Summary statistics
    total_signals_processed: int = 0
    systems_reporting: int = 0
    critical_alerts: int = 0

    # Overall assessment
    overall_stance: str = "NEUTRAL"  # DEFENSIVE, NEUTRAL, AGGRESSIVE
    recommended_exposure: float = 1.0  # Multiplier for position sizes

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization"""
        return self.model_dump(mode="json")
