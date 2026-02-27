"""
Risk Intelligence Synthesis

Combines signals from:
- SENTINEL-8: Short squeeze detection, gamma exposure, CTB rates
- ATHENA-FRM: Portfolio risk metrics, systematic/unsystematic risk
- Greeks-Trader: Options Greeks exposure, delta/gamma/vega
- VixLab: VIX regime, term structure, volatility forecasts

Output: Portfolio heat map with squeeze/vol alerts and position sizing
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional
from collections import defaultdict

logger = logging.getLogger(__name__)


class VolRegime(str, Enum):
    """Volatility regime classification"""
    LOW_VOL = "LOW_VOL"           # VIX < 15
    NORMAL = "NORMAL"             # VIX 15-20
    ELEVATED = "ELEVATED"         # VIX 20-25
    HIGH_VOL = "HIGH_VOL"         # VIX 25-35
    EXTREME = "EXTREME"           # VIX > 35


class RiskLevel(str, Enum):
    """Overall portfolio risk level"""
    MINIMAL = "MINIMAL"
    LOW = "LOW"
    MODERATE = "MODERATE"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


@dataclass
class SqueezeAlert:
    """Individual squeeze alert from SENTINEL-8"""
    ticker: str
    squeeze_score: float          # 0-100
    short_interest: float         # Percentage
    days_to_cover: float
    ctb_rate: float               # Cost to borrow
    gamma_exposure: float         # Dealer gamma
    trigger_price: Optional[float]
    alert_level: str              # WATCH, ELEVATED, IMMINENT

    def to_dict(self) -> dict:
        return {
            "ticker": self.ticker,
            "squeeze_score": self.squeeze_score,
            "short_interest": self.short_interest,
            "days_to_cover": self.days_to_cover,
            "ctb_rate": self.ctb_rate,
            "gamma_exposure": self.gamma_exposure,
            "trigger_price": self.trigger_price,
            "alert_level": self.alert_level,
        }


@dataclass
class GreeksExposure:
    """Portfolio-level Greeks exposure"""
    total_delta: float
    total_gamma: float
    total_theta: float
    total_vega: float
    delta_dollars: float
    gamma_dollars: float
    theta_daily: float
    vega_per_point: float

    def to_dict(self) -> dict:
        return {
            "total_delta": self.total_delta,
            "total_gamma": self.total_gamma,
            "total_theta": self.total_theta,
            "total_vega": self.total_vega,
            "delta_dollars": self.delta_dollars,
            "gamma_dollars": self.gamma_dollars,
            "theta_daily": self.theta_daily,
            "vega_per_point": self.vega_per_point,
        }


@dataclass
class RiskIntelligenceReport:
    """Unified risk intelligence output"""
    # Overall assessment
    risk_level: RiskLevel
    risk_score: float                    # 0-100
    risk_confidence: float               # 0-1

    # Volatility
    vol_regime: VolRegime
    vix_level: float
    vix_term_structure: str              # CONTANGO, BACKWARDATION, FLAT
    vol_forecast_1w: float
    vol_forecast_1m: float

    # Squeeze intelligence
    squeeze_alerts: list[SqueezeAlert]
    highest_squeeze_risk: Optional[str]  # Ticker with highest squeeze score
    portfolio_squeeze_exposure: float    # % of portfolio in squeeze names

    # Greeks
    greeks_exposure: GreeksExposure
    greeks_imbalance: str                # Description of any imbalances
    delta_recommendation: str

    # Portfolio risk
    portfolio_var_1d: float              # 1-day VaR
    portfolio_var_5d: float              # 5-day VaR
    max_drawdown_risk: float             # Expected max drawdown
    concentration_risk: float            # 0-100

    # Systematic vs unsystematic
    systematic_risk_pct: float           # Beta-related risk
    unsystematic_risk_pct: float         # Idiosyncratic risk
    sector_concentration: dict[str, float]

    # System contributions
    sentinel_risk: float
    frm_risk: float
    greeks_risk: float
    vixlab_risk: float

    # Meta
    systems_reporting: list[str]

    # Recommendations
    position_size_multiplier: float      # Scale positions by this
    hedge_urgency: str                   # NONE, LOW, MEDIUM, HIGH, CRITICAL
    specific_hedges: list[str]

    # Generated timestamp (with default)
    generated_at: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> dict:
        return {
            "risk_level": self.risk_level.value,
            "risk_score": self.risk_score,
            "risk_confidence": self.risk_confidence,
            "vol_regime": self.vol_regime.value,
            "vix_level": self.vix_level,
            "vix_term_structure": self.vix_term_structure,
            "vol_forecast_1w": self.vol_forecast_1w,
            "vol_forecast_1m": self.vol_forecast_1m,
            "squeeze_alerts": [s.to_dict() for s in self.squeeze_alerts],
            "highest_squeeze_risk": self.highest_squeeze_risk,
            "portfolio_squeeze_exposure": self.portfolio_squeeze_exposure,
            "greeks_exposure": self.greeks_exposure.to_dict(),
            "greeks_imbalance": self.greeks_imbalance,
            "delta_recommendation": self.delta_recommendation,
            "portfolio_var_1d": self.portfolio_var_1d,
            "portfolio_var_5d": self.portfolio_var_5d,
            "max_drawdown_risk": self.max_drawdown_risk,
            "concentration_risk": self.concentration_risk,
            "systematic_risk_pct": self.systematic_risk_pct,
            "unsystematic_risk_pct": self.unsystematic_risk_pct,
            "sector_concentration": self.sector_concentration,
            "sentinel_risk": self.sentinel_risk,
            "frm_risk": self.frm_risk,
            "greeks_risk": self.greeks_risk,
            "vixlab_risk": self.vixlab_risk,
            "systems_reporting": self.systems_reporting,
            "generated_at": self.generated_at.isoformat(),
            "position_size_multiplier": self.position_size_multiplier,
            "hedge_urgency": self.hedge_urgency,
            "specific_hedges": self.specific_hedges,
        }


class RiskIntelligence:
    """
    Risk Intelligence Synthesis Engine

    Combines SENTINEL-8, FRM, Greeks, and VixLab signals into unified
    risk intelligence with squeeze alerts and position sizing.

    Usage:
        risk = RiskIntelligence()
        report = risk.synthesize(signals)
    """

    SENTINEL_SYSTEMS = {"sentinel-8", "sentinel", "squeeze", "short-squeeze"}
    FRM_SYSTEMS = {"athena-frm", "frm", "frm-elite", "risk-manager"}
    GREEKS_SYSTEMS = {"greeks-trader", "greeks", "options-greeks", "natenberg"}
    VIXLAB_SYSTEMS = {"vixlab", "vix", "volatility", "vol-regime"}

    DEFAULT_WEIGHTS = {
        "sentinel": 0.30,
        "frm": 0.25,
        "greeks": 0.25,
        "vixlab": 0.20,
    }

    def __init__(self, weights: Optional[dict] = None):
        self.weights = weights or self.DEFAULT_WEIGHTS.copy()

    def synthesize(self, signals: list[dict]) -> RiskIntelligenceReport:
        """
        Synthesize risk intelligence from system signals

        Args:
            signals: List of signal dictionaries from risk systems

        Returns:
            RiskIntelligenceReport with unified intelligence
        """
        # Extract signals by system
        sentinel_signals = self._extract_signals(signals, self.SENTINEL_SYSTEMS)
        frm_signals = self._extract_signals(signals, self.FRM_SYSTEMS)
        greeks_signals = self._extract_signals(signals, self.GREEKS_SYSTEMS)
        vixlab_signals = self._extract_signals(signals, self.VIXLAB_SYSTEMS)

        # Extract metrics
        squeeze_alerts = self._get_squeeze_alerts(sentinel_signals)
        frm_metrics = self._get_frm_metrics(frm_signals)
        greeks_exposure = self._get_greeks_exposure(greeks_signals)
        vol_metrics = self._get_vol_metrics(vixlab_signals)

        # Calculate individual risk scores
        sentinel_risk = self._calculate_sentinel_risk(squeeze_alerts)
        frm_risk = frm_metrics.get("overall_risk", 50)
        greeks_risk = self._calculate_greeks_risk(greeks_exposure)
        vixlab_risk = self._calculate_vol_risk(vol_metrics)

        # Unified risk score
        risk_score = self._calculate_unified_risk(
            sentinel_risk, frm_risk, greeks_risk, vixlab_risk
        )

        # Classify risk level
        risk_level = self._classify_risk_level(risk_score)

        # Vol regime
        vol_regime = self._classify_vol_regime(vol_metrics.get("vix", 20))

        # Greeks analysis
        greeks_imbalance = self._analyze_greeks_imbalance(greeks_exposure)
        delta_rec = self._recommend_delta_adjustment(greeks_exposure, vol_regime)

        # Position sizing
        position_multiplier = self._calculate_position_multiplier(risk_score, vol_regime)

        # Hedge recommendations
        hedge_urgency, specific_hedges = self._recommend_hedges(
            risk_score, squeeze_alerts, vol_regime, greeks_exposure
        )

        # Highest squeeze risk
        highest_squeeze = None
        if squeeze_alerts:
            highest = max(squeeze_alerts, key=lambda x: x.squeeze_score)
            highest_squeeze = highest.ticker

        # Systems reporting
        systems_reporting = []
        if sentinel_signals:
            systems_reporting.append("sentinel-8")
        if frm_signals:
            systems_reporting.append("athena-frm")
        if greeks_signals:
            systems_reporting.append("greeks-trader")
        if vixlab_signals:
            systems_reporting.append("vixlab")

        return RiskIntelligenceReport(
            risk_level=risk_level,
            risk_score=risk_score,
            risk_confidence=0.8 if len(systems_reporting) >= 3 else 0.6,
            vol_regime=vol_regime,
            vix_level=vol_metrics.get("vix", 20),
            vix_term_structure=vol_metrics.get("term_structure", "CONTANGO"),
            vol_forecast_1w=vol_metrics.get("forecast_1w", 20),
            vol_forecast_1m=vol_metrics.get("forecast_1m", 20),
            squeeze_alerts=squeeze_alerts,
            highest_squeeze_risk=highest_squeeze,
            portfolio_squeeze_exposure=self._calc_squeeze_exposure(squeeze_alerts),
            greeks_exposure=greeks_exposure,
            greeks_imbalance=greeks_imbalance,
            delta_recommendation=delta_rec,
            portfolio_var_1d=frm_metrics.get("var_1d", 0),
            portfolio_var_5d=frm_metrics.get("var_5d", 0),
            max_drawdown_risk=frm_metrics.get("max_dd_risk", 0),
            concentration_risk=frm_metrics.get("concentration", 0),
            systematic_risk_pct=frm_metrics.get("systematic", 60),
            unsystematic_risk_pct=frm_metrics.get("unsystematic", 40),
            sector_concentration=frm_metrics.get("sectors", {}),
            sentinel_risk=sentinel_risk,
            frm_risk=frm_risk,
            greeks_risk=greeks_risk,
            vixlab_risk=vixlab_risk,
            systems_reporting=systems_reporting,
            position_size_multiplier=position_multiplier,
            hedge_urgency=hedge_urgency,
            specific_hedges=specific_hedges,
        )

    def _extract_signals(self, signals: list[dict], system_ids: set) -> list[dict]:
        """Extract signals from specific systems"""
        return [
            s for s in signals
            if s.get("system_id", "").lower() in system_ids
            or any(sid in s.get("system_id", "").lower() for sid in system_ids)
        ]

    def _get_squeeze_alerts(self, signals: list[dict]) -> list[SqueezeAlert]:
        """Extract squeeze alerts from SENTINEL signals"""
        alerts = []

        for sig in signals:
            meta = sig.get("metadata", {})
            ticker = sig.get("ticker", meta.get("ticker", "UNKNOWN"))

            score = meta.get("squeeze_score", 0)
            if score < 50:  # Only alert on elevated scores
                continue

            alert_level = "IMMINENT" if score >= 80 else "ELEVATED" if score >= 65 else "WATCH"

            alerts.append(SqueezeAlert(
                ticker=ticker,
                squeeze_score=score,
                short_interest=meta.get("short_interest", 0),
                days_to_cover=meta.get("days_to_cover", 0),
                ctb_rate=meta.get("ctb", meta.get("ctb_rate", 0)),
                gamma_exposure=meta.get("gamma_exposure", 0),
                trigger_price=meta.get("trigger_price"),
                alert_level=alert_level,
            ))

        return sorted(alerts, key=lambda x: x.squeeze_score, reverse=True)

    def _get_frm_metrics(self, signals: list[dict]) -> dict:
        """Extract risk metrics from FRM signals"""
        if not signals:
            return {}

        metrics = {}
        for sig in signals:
            meta = sig.get("metadata", {})
            metrics.update({
                "overall_risk": meta.get("overall_risk", meta.get("risk_score", 50)),
                "var_1d": meta.get("var_1d", meta.get("daily_var", 0)),
                "var_5d": meta.get("var_5d", 0),
                "max_dd_risk": meta.get("max_drawdown_risk", 0),
                "concentration": meta.get("concentration_risk", 0),
                "systematic": meta.get("systematic_risk", 60),
                "unsystematic": meta.get("unsystematic_risk", 40),
                "sectors": meta.get("sector_exposure", {}),
            })

        return metrics

    def _get_greeks_exposure(self, signals: list[dict]) -> GreeksExposure:
        """Extract Greeks from options signals"""
        delta = gamma = theta = vega = 0
        delta_d = gamma_d = theta_d = vega_p = 0

        for sig in signals:
            meta = sig.get("metadata", {})
            delta += meta.get("portfolio_delta", meta.get("delta", 0))
            gamma += meta.get("portfolio_gamma", meta.get("gamma", 0))
            theta += meta.get("portfolio_theta", meta.get("theta", 0))
            vega += meta.get("portfolio_vega", meta.get("vega", 0))
            delta_d = meta.get("delta_dollars", delta_d)
            gamma_d = meta.get("gamma_dollars", gamma_d)
            theta_d = meta.get("theta_bleed", meta.get("theta_daily", theta_d))
            vega_p = meta.get("vega_exposure", meta.get("vega_per_point", vega_p))

        return GreeksExposure(
            total_delta=delta,
            total_gamma=gamma,
            total_theta=theta,
            total_vega=vega,
            delta_dollars=delta_d,
            gamma_dollars=gamma_d,
            theta_daily=theta_d,
            vega_per_point=vega_p,
        )

    def _get_vol_metrics(self, signals: list[dict]) -> dict:
        """Extract volatility metrics from VixLab signals"""
        if not signals:
            return {"vix": 20, "term_structure": "CONTANGO"}

        metrics = {}
        for sig in signals:
            meta = sig.get("metadata", {})
            metrics.update({
                "vix": meta.get("vix_level", meta.get("vix", 20)),
                "term_structure": meta.get("term_structure", "CONTANGO"),
                "forecast_1w": meta.get("vol_forecast_1w", 20),
                "forecast_1m": meta.get("vol_forecast_1m", 20),
                "vol_regime": meta.get("vol_regime", "NORMAL"),
            })

        return metrics

    def _calculate_sentinel_risk(self, alerts: list[SqueezeAlert]) -> float:
        """Calculate risk contribution from squeeze alerts"""
        if not alerts:
            return 0

        # Weight by squeeze score
        total_risk = sum(a.squeeze_score for a in alerts)
        max_risk = max(a.squeeze_score for a in alerts)

        # Normalize: max single squeeze is 40 risk, total portfolio squeeze is 60
        return min(100, (max_risk * 0.6) + (total_risk / len(alerts) * 0.4))

    def _calculate_greeks_risk(self, greeks: GreeksExposure) -> float:
        """Calculate risk from Greeks exposure"""
        risk = 0

        # Delta risk (too long or too short)
        if abs(greeks.total_delta) > 0.8:
            risk += 30
        elif abs(greeks.total_delta) > 0.5:
            risk += 15

        # Gamma risk (negative gamma = exposed to gaps)
        if greeks.total_gamma < -0.1:
            risk += 25
        elif greeks.total_gamma < 0:
            risk += 10

        # Theta bleed
        if greeks.theta_daily < -1000:
            risk += 20
        elif greeks.theta_daily < -500:
            risk += 10

        # Vega exposure in high vol
        if abs(greeks.total_vega) > 10000:
            risk += 15

        return min(100, risk)

    def _calculate_vol_risk(self, vol_metrics: dict) -> float:
        """Calculate risk from volatility regime"""
        vix = vol_metrics.get("vix", 20)
        term = vol_metrics.get("term_structure", "CONTANGO")

        risk = 0

        # VIX level risk
        if vix >= 35:
            risk = 80
        elif vix >= 25:
            risk = 60
        elif vix >= 20:
            risk = 40
        else:
            risk = 20

        # Backwardation adds risk (market stressed)
        if term == "BACKWARDATION":
            risk += 15

        return min(100, risk)

    def _calculate_unified_risk(
        self,
        sentinel: float,
        frm: float,
        greeks: float,
        vixlab: float,
    ) -> float:
        """Calculate unified risk score"""
        risk = (
            self.weights["sentinel"] * sentinel +
            self.weights["frm"] * frm +
            self.weights["greeks"] * greeks +
            self.weights["vixlab"] * vixlab
        )

        # Cross-system amplification
        high_risk_count = sum([
            sentinel >= 60,
            frm >= 60,
            greeks >= 60,
            vixlab >= 60,
        ])

        if high_risk_count >= 3:
            risk = min(100, risk * 1.2)

        return round(risk, 1)

    def _classify_risk_level(self, score: float) -> RiskLevel:
        """Classify risk level from score"""
        if score >= 80:
            return RiskLevel.CRITICAL
        elif score >= 60:
            return RiskLevel.HIGH
        elif score >= 40:
            return RiskLevel.MODERATE
        elif score >= 20:
            return RiskLevel.LOW
        return RiskLevel.MINIMAL

    def _classify_vol_regime(self, vix: float) -> VolRegime:
        """Classify volatility regime"""
        if vix >= 35:
            return VolRegime.EXTREME
        elif vix >= 25:
            return VolRegime.HIGH_VOL
        elif vix >= 20:
            return VolRegime.ELEVATED
        elif vix >= 15:
            return VolRegime.NORMAL
        return VolRegime.LOW_VOL

    def _analyze_greeks_imbalance(self, greeks: GreeksExposure) -> str:
        """Analyze and describe Greeks imbalances"""
        issues = []

        if greeks.total_delta > 0.7:
            issues.append("VERY LONG DELTA")
        elif greeks.total_delta > 0.4:
            issues.append("LONG DELTA BIAS")
        elif greeks.total_delta < -0.4:
            issues.append("SHORT DELTA BIAS")

        if greeks.total_gamma < -0.1:
            issues.append("NEGATIVE GAMMA (gap risk)")

        if greeks.theta_daily < -1000:
            issues.append(f"HIGH THETA BLEED (${abs(greeks.theta_daily):.0f}/day)")

        if abs(greeks.total_vega) > 5000:
            vega_dir = "LONG" if greeks.total_vega > 0 else "SHORT"
            issues.append(f"{vega_dir} VEGA EXPOSURE")

        return "; ".join(issues) if issues else "BALANCED"

    def _recommend_delta_adjustment(
        self,
        greeks: GreeksExposure,
        vol_regime: VolRegime,
    ) -> str:
        """Recommend delta adjustments"""
        if vol_regime in (VolRegime.HIGH_VOL, VolRegime.EXTREME):
            if greeks.total_delta > 0.5:
                return "REDUCE DELTA: High vol regime, reduce long exposure"
            elif greeks.total_delta < -0.3:
                return "COVER SHORTS: High vol can cause short squeezes"

        if greeks.total_delta > 0.8:
            return "REDUCE DELTA: Over-exposed to downside, hedge or reduce"

        if greeks.total_gamma < -0.15:
            return "BUY GAMMA: Negative gamma creates gap risk, buy options"

        return "NO CHANGE: Greeks are balanced"

    def _calc_squeeze_exposure(self, alerts: list[SqueezeAlert]) -> float:
        """Calculate portfolio exposure to squeeze names"""
        if not alerts:
            return 0
        # Simplified: assume each alert is ~5% of portfolio risk
        return min(100, len(alerts) * 5)

    def _calculate_position_multiplier(
        self,
        risk_score: float,
        vol_regime: VolRegime,
    ) -> float:
        """Calculate position size multiplier based on risk"""
        base = 1.0

        # Risk score adjustment
        if risk_score >= 80:
            base = 0.5
        elif risk_score >= 60:
            base = 0.7
        elif risk_score >= 40:
            base = 0.85

        # Vol regime adjustment
        vol_adj = {
            VolRegime.EXTREME: 0.5,
            VolRegime.HIGH_VOL: 0.7,
            VolRegime.ELEVATED: 0.85,
            VolRegime.NORMAL: 1.0,
            VolRegime.LOW_VOL: 1.1,
        }.get(vol_regime, 1.0)

        return round(base * vol_adj, 2)

    def _recommend_hedges(
        self,
        risk_score: float,
        squeeze_alerts: list[SqueezeAlert],
        vol_regime: VolRegime,
        greeks: GreeksExposure,
    ) -> tuple[str, list[str]]:
        """Generate hedge recommendations"""
        hedges = []

        # Urgency based on risk score
        if risk_score >= 80:
            urgency = "CRITICAL"
        elif risk_score >= 60:
            urgency = "HIGH"
        elif risk_score >= 40:
            urgency = "MEDIUM"
        elif risk_score >= 25:
            urgency = "LOW"
        else:
            urgency = "NONE"

        # Specific hedges
        if vol_regime in (VolRegime.HIGH_VOL, VolRegime.EXTREME):
            hedges.append("Consider VIX put spreads to benefit from vol decline")

        if greeks.total_delta > 0.5:
            hedges.append("Buy SPY puts or put spreads to reduce delta")

        if greeks.total_gamma < -0.1:
            hedges.append("Buy ATM straddles to add gamma protection")

        for alert in squeeze_alerts[:3]:  # Top 3 squeeze risks
            if alert.squeeze_score >= 70:
                hedges.append(f"Close/hedge {alert.ticker} short position (squeeze risk)")

        if not hedges and urgency != "NONE":
            hedges.append("Maintain standard hedges per risk policy")

        return urgency, hedges
