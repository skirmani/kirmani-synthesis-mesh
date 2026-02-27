"""
S-Tier Signal Producers
=======================

These producers connect existing trading systems to the
Kirmani Signal Protocol, enabling real-time signal emission.

Each producer adapts a specific system's output format to
the standardized KSP signal format.
"""

import logging
import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional
from enum import Enum

logger = logging.getLogger(__name__)


class SignalDirection(str, Enum):
    BULLISH = "BULLISH"
    BEARISH = "BEARISH"
    NEUTRAL = "NEUTRAL"


@dataclass
class StandardSignal:
    """Standardized signal format for KSP"""
    system_id: str
    signal_type: str
    ticker: str
    direction: SignalDirection
    confidence: float
    magnitude: float
    components: Dict[str, Any]
    timestamp: datetime = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.utcnow()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "system_id": self.system_id,
            "signal_type": self.signal_type,
            "ticker": self.ticker,
            "direction": self.direction.value,
            "confidence": self.confidence,
            "magnitude": self.magnitude,
            "components": self.components,
            "timestamp": self.timestamp.isoformat(),
        }


class SignalProducer(ABC):
    """Base class for signal producers"""

    @property
    @abstractmethod
    def system_id(self) -> str:
        """Unique system identifier"""
        pass

    @abstractmethod
    def produce_signals(self, data: Any) -> List[StandardSignal]:
        """Generate signals from system output"""
        pass


class VIXLabProducer(SignalProducer):
    """
    VIXLab Signal Producer

    Converts VIXLab volatility analysis into standardized signals.
    Emits: VOL_REGIME, VIX_TERM_STRUCTURE, VVIX_SIGNAL
    """

    @property
    def system_id(self) -> str:
        return "vixlab"

    def produce_signals(self, data: Dict) -> List[StandardSignal]:
        signals = []

        # VIX level signal
        vix_level = data.get("vix_level", 18.0)
        vvix = data.get("vvix", 90.0)

        # Determine regime
        if vix_level >= 30:
            regime = "EXTREME"
            direction = SignalDirection.BEARISH
        elif vix_level >= 25:
            regime = "HIGH_VOL"
            direction = SignalDirection.BEARISH
        elif vix_level >= 20:
            regime = "ELEVATED"
            direction = SignalDirection.NEUTRAL
        elif vix_level >= 15:
            regime = "NORMAL"
            direction = SignalDirection.NEUTRAL
        else:
            regime = "LOW_VOL"
            direction = SignalDirection.BULLISH

        signals.append(StandardSignal(
            system_id=self.system_id,
            signal_type="VOL_REGIME",
            ticker="VIX",
            direction=direction,
            confidence=0.85,
            magnitude=vix_level / 50,  # Normalize to 0-1
            components={
                "vix_level": vix_level,
                "vvix": vvix,
                "regime": regime,
                "vix_percentile": min(99, vix_level * 2.5),
            },
        ))

        # Term structure signal
        term_structure = data.get("term_structure", "CONTANGO")

        signals.append(StandardSignal(
            system_id=self.system_id,
            signal_type="VIX_TERM_STRUCTURE",
            ticker="VIX",
            direction=SignalDirection.BEARISH if term_structure == "BACKWARDATION" else SignalDirection.NEUTRAL,
            confidence=0.75,
            magnitude=0.5,
            components={
                "structure": term_structure,
                "roll_yield": data.get("roll_yield", 0.02),
            },
        ))

        return signals


class SornetteProducer(SignalProducer):
    """
    Sornette LPPLS Producer

    Converts crash hazard analysis into standardized signals.
    Emits: CRASH_HAZARD, BUBBLE_PHASE
    """

    @property
    def system_id(self) -> str:
        return "sornette"

    def produce_signals(self, data: Dict) -> List[StandardSignal]:
        signals = []

        hazard = data.get("crash_hazard", 0.3)
        bubble_phase = data.get("bubble_phase", 2)
        time_to_critical = data.get("time_to_critical", "MEDIUM_TERM")

        # Crash hazard signal
        if hazard >= 0.7:
            direction = SignalDirection.BEARISH
        elif hazard >= 0.5:
            direction = SignalDirection.NEUTRAL
        else:
            direction = SignalDirection.BULLISH

        signals.append(StandardSignal(
            system_id=self.system_id,
            signal_type="CRASH_HAZARD",
            ticker=data.get("ticker", "SPY"),
            direction=direction,
            confidence=0.7 + hazard * 0.2,
            magnitude=hazard,
            components={
                "crash_hazard": hazard,
                "bubble_phase": bubble_phase,
                "time_to_critical": time_to_critical,
                "lppl_omega": data.get("lppl_omega"),
                "lppl_tc": data.get("lppl_tc"),
            },
        ))

        return signals


class MinskyProducer(SignalProducer):
    """
    Minsky Financial Instability Producer

    Converts Minsky bubble analysis into signals.
    Emits: MINSKY_PHASE, CREDIT_CONDITION
    """

    @property
    def system_id(self) -> str:
        return "minsky"

    def produce_signals(self, data: Dict) -> List[StandardSignal]:
        signals = []

        phase = data.get("minsky_phase", "HEDGE")  # HEDGE, SPECULATIVE, PONZI
        instability = data.get("instability_score", 0.3)

        phase_map = {
            "HEDGE": (SignalDirection.BULLISH, 0.2),
            "SPECULATIVE": (SignalDirection.NEUTRAL, 0.5),
            "PONZI": (SignalDirection.BEARISH, 0.8),
        }

        direction, base_magnitude = phase_map.get(phase, (SignalDirection.NEUTRAL, 0.5))

        signals.append(StandardSignal(
            system_id=self.system_id,
            signal_type="MINSKY_PHASE",
            ticker="MACRO",
            direction=direction,
            confidence=0.75,
            magnitude=instability,
            components={
                "phase": phase,
                "instability_score": instability,
                "credit_growth": data.get("credit_growth", 0.05),
                "leverage_ratio": data.get("leverage_ratio", 1.5),
            },
        ))

        return signals


class SentinelProducer(SignalProducer):
    """
    SENTINEL-8 Risk Producer

    Converts risk analysis into signals.
    Emits: SQUEEZE_ALERT, PORTFOLIO_RISK, CONCENTRATION_RISK
    """

    @property
    def system_id(self) -> str:
        return "sentinel"

    def produce_signals(self, data: Dict) -> List[StandardSignal]:
        signals = []

        # Squeeze alerts
        for alert in data.get("squeeze_alerts", []):
            ticker = alert.get("ticker", "UNKNOWN")
            score = alert.get("squeeze_score", 50)

            if score >= 80:
                direction = SignalDirection.BEARISH
            elif score >= 60:
                direction = SignalDirection.NEUTRAL
            else:
                direction = SignalDirection.BULLISH

            signals.append(StandardSignal(
                system_id=self.system_id,
                signal_type="SQUEEZE_ALERT",
                ticker=ticker,
                direction=direction,
                confidence=0.8,
                magnitude=score / 100,
                components={
                    "squeeze_score": score,
                    "short_interest_pct": alert.get("short_interest", 20),
                    "days_to_cover": alert.get("days_to_cover", 3),
                    "ctb_rate": alert.get("ctb_rate", 10),
                    "alert_level": "IMMINENT" if score >= 80 else "ELEVATED" if score >= 60 else "WATCH",
                },
            ))

        # Portfolio risk
        risk_score = data.get("risk_score", 30)

        signals.append(StandardSignal(
            system_id=self.system_id,
            signal_type="PORTFOLIO_RISK",
            ticker="PORTFOLIO",
            direction=SignalDirection.BEARISH if risk_score >= 60 else SignalDirection.NEUTRAL,
            confidence=0.85,
            magnitude=risk_score / 100,
            components={
                "risk_score": risk_score,
                "var_95": data.get("var_95", -0.02),
                "max_drawdown_risk": data.get("max_drawdown_risk", 0.15),
            },
        ))

        return signals


class MurphyProducer(SignalProducer):
    """
    Murphy Intermarket Producer

    Converts intermarket analysis into signals.
    Emits: INTERMARKET, ASSET_ROTATION
    """

    @property
    def system_id(self) -> str:
        return "murphy"

    def produce_signals(self, data: Dict) -> List[StandardSignal]:
        signals = []

        regime = data.get("intermarket_regime", "NEUTRAL")  # RISK_ON, NEUTRAL, RISK_OFF
        score = data.get("intermarket_score", 50)

        direction_map = {
            "RISK_ON": SignalDirection.BULLISH,
            "NEUTRAL": SignalDirection.NEUTRAL,
            "RISK_OFF": SignalDirection.BEARISH,
        }

        signals.append(StandardSignal(
            system_id=self.system_id,
            signal_type="INTERMARKET",
            ticker="MACRO",
            direction=direction_map.get(regime, SignalDirection.NEUTRAL),
            confidence=0.75,
            magnitude=score / 100,
            components={
                "regime": regime,
                "intermarket_score": score,
                "dollar_strength": data.get("dollar_strength", 50),
                "bond_equity_correlation": data.get("bond_equity_corr", -0.3),
                "commodity_trend": data.get("commodity_trend", "NEUTRAL"),
            },
        ))

        return signals


class WyckoffProducer(SignalProducer):
    """
    Wyckoff Phase Producer

    Converts Wyckoff analysis into signals.
    Emits: WYCKOFF_PHASE, COMPOSITE_MAN_ACTION
    """

    @property
    def system_id(self) -> str:
        return "wyckoff"

    def produce_signals(self, data: Dict) -> List[StandardSignal]:
        signals = []

        phase = data.get("phase", "MARKUP")  # ACCUMULATION, MARKUP, DISTRIBUTION, MARKDOWN
        ticker = data.get("ticker", "SPY")

        phase_direction = {
            "ACCUMULATION": SignalDirection.BULLISH,
            "MARKUP": SignalDirection.BULLISH,
            "DISTRIBUTION": SignalDirection.BEARISH,
            "MARKDOWN": SignalDirection.BEARISH,
        }

        signals.append(StandardSignal(
            system_id=self.system_id,
            signal_type="WYCKOFF_PHASE",
            ticker=ticker,
            direction=phase_direction.get(phase, SignalDirection.NEUTRAL),
            confidence=0.7,
            magnitude=0.6,
            components={
                "phase": phase,
                "phase_progress": data.get("phase_progress", 50),
                "volume_pattern": data.get("volume_pattern", "NORMAL"),
                "spring_detected": data.get("spring_detected", False),
                "upthrust_detected": data.get("upthrust_detected", False),
            },
        ))

        return signals


class HurstProducer(SignalProducer):
    """
    Hurst Cycles Producer

    Converts cycle analysis into signals.
    Emits: CYCLE_PHASE, DOMINANT_CYCLE
    """

    @property
    def system_id(self) -> str:
        return "hurst"

    def produce_signals(self, data: Dict) -> List[StandardSignal]:
        signals = []

        dominant_cycle = data.get("dominant_cycle", 40)
        phase = data.get("cycle_phase", "MID_CYCLE")  # TROUGH, RISING, PEAK, FALLING
        ticker = data.get("ticker", "SPY")

        phase_direction = {
            "TROUGH": SignalDirection.BULLISH,
            "RISING": SignalDirection.BULLISH,
            "PEAK": SignalDirection.BEARISH,
            "FALLING": SignalDirection.BEARISH,
            "MID_CYCLE": SignalDirection.NEUTRAL,
        }

        signals.append(StandardSignal(
            system_id=self.system_id,
            signal_type="CYCLE",
            ticker=ticker,
            direction=phase_direction.get(phase, SignalDirection.NEUTRAL),
            confidence=0.65,
            magnitude=0.5,
            components={
                "dominant_cycle": dominant_cycle,
                "cycle_phase": phase,
                "trough_distance": data.get("trough_distance", dominant_cycle // 2),
                "next_trough_estimate": data.get("next_trough"),
                "cycle_alignment": data.get("cycle_alignment", 0.6),
            },
        ))

        return signals


class EliteScannerProducer(SignalProducer):
    """
    Elite Scanner Producer

    Converts scanner signals into standardized format.
    Emits: MOMENTUM, BREAKOUT, PATTERN
    """

    @property
    def system_id(self) -> str:
        return "elite_scanner"

    def produce_signals(self, data: Dict) -> List[StandardSignal]:
        signals = []

        for scan_result in data.get("results", []):
            ticker = scan_result.get("ticker")
            signal_type = scan_result.get("type", "MOMENTUM")
            direction = SignalDirection.BULLISH if scan_result.get("direction") == "LONG" else SignalDirection.BEARISH
            confidence = scan_result.get("confidence", 0.7)

            signals.append(StandardSignal(
                system_id=self.system_id,
                signal_type=signal_type,
                ticker=ticker,
                direction=direction,
                confidence=confidence,
                magnitude=scan_result.get("strength", 0.5),
                components={
                    "pattern": scan_result.get("pattern"),
                    "entry_price": scan_result.get("entry_price"),
                    "target_1": scan_result.get("target_1"),
                    "target_2": scan_result.get("target_2"),
                    "stop_loss": scan_result.get("stop_loss"),
                    "risk_reward": scan_result.get("risk_reward", 2.0),
                },
            ))

        return signals


class GreeksProducer(SignalProducer):
    """
    Greeks/Options Producer

    Converts options analytics into signals.
    Emits: GREEKS_EXPOSURE, GAMMA_REGIME, IV_SIGNAL
    """

    @property
    def system_id(self) -> str:
        return "greeks_trader"

    def produce_signals(self, data: Dict) -> List[StandardSignal]:
        signals = []

        # Greeks exposure
        net_delta = data.get("net_delta", 0)
        net_gamma = data.get("net_gamma", 0)
        net_vega = data.get("net_vega", 0)

        if net_delta > 1000:
            delta_bias = "LONG_DELTA_BIAS"
            direction = SignalDirection.BULLISH
        elif net_delta < -1000:
            delta_bias = "SHORT_DELTA_BIAS"
            direction = SignalDirection.BEARISH
        else:
            delta_bias = "BALANCED"
            direction = SignalDirection.NEUTRAL

        signals.append(StandardSignal(
            system_id=self.system_id,
            signal_type="GREEKS",
            ticker="PORTFOLIO",
            direction=direction,
            confidence=0.8,
            magnitude=min(1.0, abs(net_delta) / 10000),
            components={
                "net_delta": net_delta,
                "net_gamma": net_gamma,
                "net_vega": net_vega,
                "delta_bias": delta_bias,
            },
        ))

        # Gamma regime
        gamma_regime = data.get("gamma_regime", "NEUTRAL")  # LONG_GAMMA, NEUTRAL, SHORT_GAMMA

        signals.append(StandardSignal(
            system_id=self.system_id,
            signal_type="GAMMA_REGIME",
            ticker="MARKET",
            direction=SignalDirection.BULLISH if gamma_regime == "LONG_GAMMA" else SignalDirection.BEARISH if gamma_regime == "SHORT_GAMMA" else SignalDirection.NEUTRAL,
            confidence=0.7,
            magnitude=0.5,
            components={
                "gamma_regime": gamma_regime,
                "gex": data.get("gex", 0),
                "dex": data.get("dex", 0),
            },
        ))

        return signals


class SignalProducerRegistry:
    """
    Registry of all signal producers.

    Provides unified interface for generating signals from all systems.
    """

    def __init__(self):
        self.producers: Dict[str, SignalProducer] = {}
        self._register_default_producers()

    def _register_default_producers(self):
        """Register all default producers"""
        default_producers = [
            VIXLabProducer(),
            SornetteProducer(),
            MinskyProducer(),
            SentinelProducer(),
            MurphyProducer(),
            WyckoffProducer(),
            HurstProducer(),
            EliteScannerProducer(),
            GreeksProducer(),
        ]

        for producer in default_producers:
            self.register(producer)

    def register(self, producer: SignalProducer):
        """Register a signal producer"""
        self.producers[producer.system_id] = producer
        logger.info(f"Registered producer: {producer.system_id}")

    def produce_all(self, system_outputs: Dict[str, Dict]) -> List[Dict]:
        """
        Produce signals from all systems.

        Args:
            system_outputs: Dict mapping system_id to their raw output

        Returns:
            List of signal dicts ready for intelligence engine
        """
        all_signals = []

        for system_id, output in system_outputs.items():
            producer = self.producers.get(system_id)
            if producer:
                try:
                    signals = producer.produce_signals(output)
                    all_signals.extend([s.to_dict() for s in signals])
                    logger.debug(f"{system_id}: produced {len(signals)} signals")
                except Exception as e:
                    logger.error(f"{system_id} producer failed: {e}")
            else:
                logger.warning(f"No producer for system: {system_id}")

        return all_signals

    def get_producer(self, system_id: str) -> Optional[SignalProducer]:
        """Get a specific producer"""
        return self.producers.get(system_id)

    @property
    def system_ids(self) -> List[str]:
        """List all registered system IDs"""
        return list(self.producers.keys())
