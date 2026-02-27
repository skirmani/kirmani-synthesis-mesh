"""
Alpha Intelligence Synthesis

Combines signals from:
- Elite-Momentum-Scanner: Minervini/O'Neil momentum patterns, RS rankings
- ATHENA-Wyckoff-Elite: Accumulation/distribution phases, volume analysis
- ATHENA-Hurst-Cycles: Cycle phases, trend persistence, turning points
- StratEngine: "The Strat" patterns, timeframe continuity, combos

Output: High-conviction entry signals with multi-timeframe confirmation
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional
from collections import defaultdict

logger = logging.getLogger(__name__)


class SignalStrength(str, Enum):
    """Signal conviction level"""
    WEAK = "WEAK"
    MODERATE = "MODERATE"
    STRONG = "STRONG"
    VERY_STRONG = "VERY_STRONG"
    EXCEPTIONAL = "EXCEPTIONAL"


class EntryType(str, Enum):
    """Type of entry signal"""
    BREAKOUT = "BREAKOUT"
    PULLBACK = "PULLBACK"
    REVERSAL = "REVERSAL"
    CONTINUATION = "CONTINUATION"
    ACCUMULATION = "ACCUMULATION"
    CYCLE_TURN = "CYCLE_TURN"


class WyckoffPhase(str, Enum):
    """Wyckoff market phase"""
    ACCUMULATION = "ACCUMULATION"
    MARKUP = "MARKUP"
    DISTRIBUTION = "DISTRIBUTION"
    MARKDOWN = "MARKDOWN"
    SPRING = "SPRING"
    UPTHRUST = "UPTHRUST"


@dataclass
class AlphaSignal:
    """Individual alpha signal from synthesis"""
    ticker: str
    direction: str                       # LONG, SHORT
    entry_type: EntryType
    strength: SignalStrength
    confidence: float                    # 0-1

    # Entry parameters
    entry_price: Optional[float]
    stop_loss: Optional[float]
    target_1: Optional[float]
    target_2: Optional[float]
    risk_reward: Optional[float]

    # System confirmations
    systems_confirming: list[str]
    confirmation_count: int

    # Technical context
    pattern: str                         # Pattern description
    timeframe_alignment: str             # FTFC status
    volume_confirmation: bool
    relative_strength: Optional[float]   # RS ranking

    # Cycle context
    hurst_phase: Optional[str]
    cycle_position: Optional[str]        # TROUGH, PEAK, RISING, FALLING

    def to_dict(self) -> dict:
        return {
            "ticker": self.ticker,
            "direction": self.direction,
            "entry_type": self.entry_type.value,
            "strength": self.strength.value,
            "confidence": self.confidence,
            "entry_price": self.entry_price,
            "stop_loss": self.stop_loss,
            "target_1": self.target_1,
            "target_2": self.target_2,
            "risk_reward": self.risk_reward,
            "systems_confirming": self.systems_confirming,
            "confirmation_count": self.confirmation_count,
            "pattern": self.pattern,
            "timeframe_alignment": self.timeframe_alignment,
            "volume_confirmation": self.volume_confirmation,
            "relative_strength": self.relative_strength,
            "hurst_phase": self.hurst_phase,
            "cycle_position": self.cycle_position,
        }


@dataclass
class AlphaIntelligenceReport:
    """Unified alpha intelligence output"""
    # Top signals
    alpha_signals: list[AlphaSignal]
    top_long_signal: Optional[AlphaSignal]
    top_short_signal: Optional[AlphaSignal]

    # Market context
    market_breadth: str                  # STRONG, MODERATE, WEAK
    sector_leaders: list[str]
    sector_laggards: list[str]

    # Wyckoff analysis
    spy_wyckoff_phase: WyckoffPhase
    qqq_wyckoff_phase: WyckoffPhase
    accumulation_names: list[str]
    distribution_names: list[str]

    # Cycle analysis
    dominant_cycle: int                  # Days
    cycle_phase: str                     # Market-wide cycle phase
    next_turn_estimate: Optional[str]    # Date of expected turn

    # Strat analysis
    ftfc_bullish: list[str]              # Tickers in full bullish FTFC
    ftfc_bearish: list[str]              # Tickers in full bearish FTFC
    strat_setups: list[str]              # Active Strat combo setups

    # System contributions
    scanner_signals: int
    wyckoff_signals: int
    hurst_signals: int
    strat_signals: int

    # Meta
    systems_reporting: list[str]
    total_opportunities: int
    high_conviction_count: int
    generated_at: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> dict:
        return {
            "alpha_signals": [s.to_dict() for s in self.alpha_signals],
            "top_long_signal": self.top_long_signal.to_dict() if self.top_long_signal else None,
            "top_short_signal": self.top_short_signal.to_dict() if self.top_short_signal else None,
            "market_breadth": self.market_breadth,
            "sector_leaders": self.sector_leaders,
            "sector_laggards": self.sector_laggards,
            "spy_wyckoff_phase": self.spy_wyckoff_phase.value,
            "qqq_wyckoff_phase": self.qqq_wyckoff_phase.value,
            "accumulation_names": self.accumulation_names,
            "distribution_names": self.distribution_names,
            "dominant_cycle": self.dominant_cycle,
            "cycle_phase": self.cycle_phase,
            "next_turn_estimate": self.next_turn_estimate,
            "ftfc_bullish": self.ftfc_bullish,
            "ftfc_bearish": self.ftfc_bearish,
            "strat_setups": self.strat_setups,
            "scanner_signals": self.scanner_signals,
            "wyckoff_signals": self.wyckoff_signals,
            "hurst_signals": self.hurst_signals,
            "strat_signals": self.strat_signals,
            "systems_reporting": self.systems_reporting,
            "total_opportunities": self.total_opportunities,
            "high_conviction_count": self.high_conviction_count,
            "generated_at": self.generated_at.isoformat(),
        }


class AlphaIntelligence:
    """
    Alpha Intelligence Synthesis Engine

    Combines Elite-Scanner, Wyckoff, Hurst, and StratEngine signals
    into high-conviction entry signals with multi-system confirmation.

    Usage:
        alpha = AlphaIntelligence()
        report = alpha.synthesize(signals)
    """

    SCANNER_SYSTEMS = {"elite-scanner", "elite-momentum", "momentum-scanner", "minervini"}
    WYCKOFF_SYSTEMS = {"wyckoff-elite", "wyckoff", "athena-wyckoff"}
    HURST_SYSTEMS = {"hurst-cycles", "hurst", "athena-hurst", "cycles"}
    STRAT_SYSTEMS = {"stratengine", "strat", "the-strat", "strat-engine"}

    # Additional alpha systems
    POWER_HOUR_SYSTEMS = {"power-hour", "athena-power-hour"}
    STEALTH_SYSTEMS = {"stealth-scanner", "athena-stealth"}

    def __init__(self):
        self._signal_cache = defaultdict(list)

    def synthesize(self, signals: list[dict]) -> AlphaIntelligenceReport:
        """
        Synthesize alpha intelligence from system signals

        Args:
            signals: List of signal dictionaries from alpha systems

        Returns:
            AlphaIntelligenceReport with unified intelligence
        """
        # Extract signals by system
        scanner_signals = self._extract_signals(signals, self.SCANNER_SYSTEMS)
        wyckoff_signals = self._extract_signals(signals, self.WYCKOFF_SYSTEMS)
        hurst_signals = self._extract_signals(signals, self.HURST_SYSTEMS)
        strat_signals = self._extract_signals(signals, self.STRAT_SYSTEMS)

        # Combine into unified signals
        alpha_signals = self._synthesize_signals(
            scanner_signals, wyckoff_signals, hurst_signals, strat_signals
        )

        # Sort by strength and confidence
        alpha_signals.sort(
            key=lambda x: (
                self._strength_value(x.strength),
                x.confidence,
                x.confirmation_count,
            ),
            reverse=True,
        )

        # Get top signals
        top_long = self._get_top_signal(alpha_signals, "LONG")
        top_short = self._get_top_signal(alpha_signals, "SHORT")

        # Market context
        breadth = self._analyze_breadth(scanner_signals)
        leaders, laggards = self._analyze_sectors(scanner_signals)

        # Wyckoff analysis
        spy_phase = self._get_index_wyckoff(wyckoff_signals, "SPY")
        qqq_phase = self._get_index_wyckoff(wyckoff_signals, "QQQ")
        accum_names = self._get_wyckoff_names(wyckoff_signals, "ACCUMULATION")
        dist_names = self._get_wyckoff_names(wyckoff_signals, "DISTRIBUTION")

        # Cycle analysis
        dom_cycle, cycle_phase, next_turn = self._analyze_cycles(hurst_signals)

        # Strat analysis
        ftfc_bull, ftfc_bear = self._analyze_ftfc(strat_signals)
        strat_setups = self._get_strat_setups(strat_signals)

        # Systems reporting
        systems_reporting = []
        if scanner_signals:
            systems_reporting.append("elite-scanner")
        if wyckoff_signals:
            systems_reporting.append("wyckoff-elite")
        if hurst_signals:
            systems_reporting.append("hurst-cycles")
        if strat_signals:
            systems_reporting.append("stratengine")

        # Counts
        high_conviction = len([s for s in alpha_signals if s.strength in (
            SignalStrength.STRONG, SignalStrength.VERY_STRONG, SignalStrength.EXCEPTIONAL
        )])

        return AlphaIntelligenceReport(
            alpha_signals=alpha_signals[:20],  # Top 20 signals
            top_long_signal=top_long,
            top_short_signal=top_short,
            market_breadth=breadth,
            sector_leaders=leaders,
            sector_laggards=laggards,
            spy_wyckoff_phase=spy_phase,
            qqq_wyckoff_phase=qqq_phase,
            accumulation_names=accum_names[:10],
            distribution_names=dist_names[:10],
            dominant_cycle=dom_cycle,
            cycle_phase=cycle_phase,
            next_turn_estimate=next_turn,
            ftfc_bullish=ftfc_bull[:10],
            ftfc_bearish=ftfc_bear[:10],
            strat_setups=strat_setups[:10],
            scanner_signals=len(scanner_signals),
            wyckoff_signals=len(wyckoff_signals),
            hurst_signals=len(hurst_signals),
            strat_signals=len(strat_signals),
            systems_reporting=systems_reporting,
            total_opportunities=len(alpha_signals),
            high_conviction_count=high_conviction,
        )

    def _extract_signals(self, signals: list[dict], system_ids: set) -> list[dict]:
        """Extract signals from specific systems"""
        return [
            s for s in signals
            if s.get("system_id", "").lower() in system_ids
            or any(sid in s.get("system_id", "").lower() for sid in system_ids)
        ]

    def _synthesize_signals(
        self,
        scanner: list[dict],
        wyckoff: list[dict],
        hurst: list[dict],
        strat: list[dict],
    ) -> list[AlphaSignal]:
        """Synthesize signals from all alpha systems"""
        # Group all signals by ticker
        by_ticker = defaultdict(list)

        for sig in scanner + wyckoff + hurst + strat:
            ticker = sig.get("ticker", "SPY")
            by_ticker[ticker].append(sig)

        alpha_signals = []

        for ticker, ticker_signals in by_ticker.items():
            # Get direction consensus
            directions = [s.get("direction", "NEUTRAL") for s in ticker_signals]
            bullish = directions.count("BULLISH") + directions.count("LONG")
            bearish = directions.count("BEARISH") + directions.count("SHORT")

            if bullish == 0 and bearish == 0:
                continue  # No directional signal

            direction = "LONG" if bullish >= bearish else "SHORT"

            # Get confirming systems
            systems = list(set(s.get("system_id", "") for s in ticker_signals))
            confirmation_count = len(systems)

            # Determine entry type
            entry_type = self._determine_entry_type(ticker_signals)

            # Calculate confidence
            confidences = [s.get("confidence", 0.5) for s in ticker_signals]
            base_conf = sum(confidences) / len(confidences) if confidences else 0.5

            # Boost confidence for multi-system confirmation
            if confirmation_count >= 3:
                base_conf = min(1.0, base_conf * 1.2)
            elif confirmation_count >= 2:
                base_conf = min(1.0, base_conf * 1.1)

            # Determine strength
            strength = self._determine_strength(confirmation_count, base_conf)

            # Extract entry parameters
            entry_price, stop_loss, target_1, target_2 = self._extract_levels(ticker_signals)

            # Calculate risk/reward
            rr = None
            if entry_price and stop_loss and target_1:
                risk = abs(entry_price - stop_loss)
                reward = abs(target_1 - entry_price)
                rr = round(reward / risk, 2) if risk > 0 else None

            # Pattern description
            pattern = self._describe_pattern(ticker_signals)

            # FTFC status
            ftfc = self._get_ftfc_status(ticker_signals)

            # Volume confirmation
            vol_confirm = any(
                s.get("metadata", {}).get("volume_confirmation", False)
                or s.get("metadata", {}).get("volume_surge", 0) > 1.5
                for s in ticker_signals
            )

            # RS ranking
            rs = None
            for sig in ticker_signals:
                if "relative_strength" in sig.get("metadata", {}):
                    rs = sig["metadata"]["relative_strength"]
                    break

            # Hurst cycle info
            hurst_phase = None
            cycle_pos = None
            for sig in ticker_signals:
                meta = sig.get("metadata", {})
                if "cycle_phase" in meta:
                    hurst_phase = meta["cycle_phase"]
                if "cycle_position" in meta:
                    cycle_pos = meta["cycle_position"]

            alpha_signals.append(AlphaSignal(
                ticker=ticker,
                direction=direction,
                entry_type=entry_type,
                strength=strength,
                confidence=round(base_conf, 3),
                entry_price=entry_price,
                stop_loss=stop_loss,
                target_1=target_1,
                target_2=target_2,
                risk_reward=rr,
                systems_confirming=systems,
                confirmation_count=confirmation_count,
                pattern=pattern,
                timeframe_alignment=ftfc,
                volume_confirmation=vol_confirm,
                relative_strength=rs,
                hurst_phase=hurst_phase,
                cycle_position=cycle_pos,
            ))

        return alpha_signals

    def _determine_entry_type(self, signals: list[dict]) -> EntryType:
        """Determine entry type from signals"""
        signal_types = [s.get("signal_type", "") for s in signals]

        if "ACCUMULATION" in signal_types or "SPRING" in signal_types:
            return EntryType.ACCUMULATION
        if "BREAKOUT" in signal_types:
            return EntryType.BREAKOUT
        if "PULLBACK" in signal_types:
            return EntryType.PULLBACK
        if "CYCLE_TURN" in signal_types or "TROUGH" in signal_types:
            return EntryType.CYCLE_TURN
        if "REVERSAL" in signal_types:
            return EntryType.REVERSAL

        return EntryType.CONTINUATION

    def _determine_strength(self, confirmations: int, confidence: float) -> SignalStrength:
        """Determine signal strength"""
        if confirmations >= 4 and confidence >= 0.8:
            return SignalStrength.EXCEPTIONAL
        if confirmations >= 3 and confidence >= 0.75:
            return SignalStrength.VERY_STRONG
        if confirmations >= 2 and confidence >= 0.7:
            return SignalStrength.STRONG
        if confidence >= 0.6:
            return SignalStrength.MODERATE
        return SignalStrength.WEAK

    def _strength_value(self, strength: SignalStrength) -> int:
        """Convert strength to numeric value for sorting"""
        return {
            SignalStrength.EXCEPTIONAL: 5,
            SignalStrength.VERY_STRONG: 4,
            SignalStrength.STRONG: 3,
            SignalStrength.MODERATE: 2,
            SignalStrength.WEAK: 1,
        }.get(strength, 0)

    def _extract_levels(self, signals: list[dict]) -> tuple:
        """Extract entry, stop, and target levels"""
        entry = stop = t1 = t2 = None

        for sig in signals:
            meta = sig.get("metadata", {})
            if not entry and "entry_price" in meta:
                entry = meta["entry_price"]
            if not stop and "stop_loss" in meta:
                stop = meta["stop_loss"]
            if not t1 and "target_1" in meta:
                t1 = meta["target_1"]
            if not t2 and "target_2" in meta:
                t2 = meta["target_2"]
            if not t1 and "target" in meta:
                t1 = meta["target"]

        return entry, stop, t1, t2

    def _describe_pattern(self, signals: list[dict]) -> str:
        """Create pattern description from signals"""
        patterns = []

        for sig in signals:
            meta = sig.get("metadata", {})
            if "pattern" in meta:
                patterns.append(meta["pattern"])
            if "wyckoff_phase" in meta:
                patterns.append(f"Wyckoff {meta['wyckoff_phase']}")
            if "strat_combo" in meta:
                patterns.append(f"Strat {meta['strat_combo']}")

        return " + ".join(patterns[:3]) if patterns else "Multi-system confirmation"

    def _get_ftfc_status(self, signals: list[dict]) -> str:
        """Get Full Timeframe Continuity status"""
        for sig in signals:
            meta = sig.get("metadata", {})
            if "ftfc" in meta:
                return meta["ftfc"]
            if "timeframe_continuity" in meta:
                return meta["timeframe_continuity"]

        return "UNKNOWN"

    def _get_top_signal(
        self,
        signals: list[AlphaSignal],
        direction: str,
    ) -> Optional[AlphaSignal]:
        """Get top signal for a direction"""
        dir_signals = [s for s in signals if s.direction == direction]
        return dir_signals[0] if dir_signals else None

    def _analyze_breadth(self, scanner_signals: list[dict]) -> str:
        """Analyze market breadth from scanner signals"""
        if not scanner_signals:
            return "UNKNOWN"

        bullish = sum(1 for s in scanner_signals if s.get("direction") == "BULLISH")
        bearish = sum(1 for s in scanner_signals if s.get("direction") == "BEARISH")
        total = len(scanner_signals)

        if total == 0:
            return "UNKNOWN"

        bull_pct = bullish / total

        if bull_pct >= 0.7:
            return "STRONG"
        elif bull_pct >= 0.5:
            return "MODERATE"
        else:
            return "WEAK"

    def _analyze_sectors(self, scanner_signals: list[dict]) -> tuple[list[str], list[str]]:
        """Analyze sector leadership"""
        sector_scores = defaultdict(list)

        for sig in scanner_signals:
            meta = sig.get("metadata", {})
            sector = meta.get("sector", "Unknown")
            rs = meta.get("relative_strength", 50)
            sector_scores[sector].append(rs)

        # Average RS by sector
        sector_avg = {
            s: sum(scores) / len(scores)
            for s, scores in sector_scores.items()
            if scores
        }

        sorted_sectors = sorted(sector_avg.items(), key=lambda x: x[1], reverse=True)
        leaders = [s[0] for s in sorted_sectors[:3]]
        laggards = [s[0] for s in sorted_sectors[-3:]]

        return leaders, laggards

    def _get_index_wyckoff(
        self,
        wyckoff_signals: list[dict],
        ticker: str,
    ) -> WyckoffPhase:
        """Get Wyckoff phase for index"""
        for sig in wyckoff_signals:
            if sig.get("ticker", "") == ticker:
                phase = sig.get("metadata", {}).get("phase", "")
                try:
                    return WyckoffPhase(phase.upper())
                except ValueError:
                    pass

        return WyckoffPhase.MARKUP  # Default

    def _get_wyckoff_names(
        self,
        wyckoff_signals: list[dict],
        phase: str,
    ) -> list[str]:
        """Get tickers in specific Wyckoff phase"""
        names = []
        for sig in wyckoff_signals:
            sig_phase = sig.get("metadata", {}).get("phase", "")
            if sig_phase.upper() == phase:
                names.append(sig.get("ticker", ""))
        return list(set(names))

    def _analyze_cycles(
        self,
        hurst_signals: list[dict],
    ) -> tuple[int, str, Optional[str]]:
        """Analyze Hurst cycle signals"""
        if not hurst_signals:
            return 40, "UNKNOWN", None

        cycles = []
        phases = []
        turns = []

        for sig in hurst_signals:
            meta = sig.get("metadata", {})
            if "dominant_cycle" in meta:
                cycles.append(meta["dominant_cycle"])
            if "cycle_phase" in meta:
                phases.append(meta["cycle_phase"])
            if "next_turn" in meta:
                turns.append(meta["next_turn"])

        dom_cycle = int(sum(cycles) / len(cycles)) if cycles else 40
        phase = max(set(phases), key=phases.count) if phases else "UNKNOWN"
        next_turn = turns[0] if turns else None

        return dom_cycle, phase, next_turn

    def _analyze_ftfc(self, strat_signals: list[dict]) -> tuple[list[str], list[str]]:
        """Analyze Full Timeframe Continuity"""
        bullish = []
        bearish = []

        for sig in strat_signals:
            ticker = sig.get("ticker", "")
            meta = sig.get("metadata", {})
            ftfc = meta.get("ftfc", meta.get("timeframe_continuity", ""))

            if "BULLISH" in ftfc.upper() or "UP" in ftfc.upper():
                bullish.append(ticker)
            elif "BEARISH" in ftfc.upper() or "DOWN" in ftfc.upper():
                bearish.append(ticker)

        return list(set(bullish)), list(set(bearish))

    def _get_strat_setups(self, strat_signals: list[dict]) -> list[str]:
        """Get active Strat combo setups"""
        setups = []

        for sig in strat_signals:
            ticker = sig.get("ticker", "")
            meta = sig.get("metadata", {})
            combo = meta.get("strat_combo", meta.get("combo", ""))

            if combo:
                setups.append(f"{ticker}: {combo}")

        return setups
