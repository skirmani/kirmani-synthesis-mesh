"""
S++ Tier Validation Framework
=============================

Elite-grade validation for institutional quant operations:
- 1d, 5d, 20d, 60d outcome tracking
- Model drift detection with alerts
- Confidence calibration and recalibration
- Cross-system consensus tracking
- Backtest vs live reconciliation

This is the validation layer that separates S++ from S tier.
"""

import json
import logging
import sqlite3
import statistics
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from enum import Enum

logger = logging.getLogger(__name__)


class DriftSeverity(str, Enum):
    NONE = "NONE"
    MINOR = "MINOR"       # <5% degradation
    MODERATE = "MODERATE" # 5-15% degradation
    SEVERE = "SEVERE"     # 15-30% degradation
    CRITICAL = "CRITICAL" # >30% degradation


@dataclass
class DriftAlert:
    """Model drift detection alert"""
    system_id: str
    severity: DriftSeverity
    historical_win_rate: float
    recent_win_rate: float
    degradation_pct: float
    sample_size: int
    detected_at: datetime
    recommendation: str


@dataclass
class CalibrationResult:
    """Confidence calibration assessment"""
    system_id: str
    expected_win_rate: float  # Based on avg confidence
    actual_win_rate: float
    calibration_error: float  # abs(expected - actual)
    is_overconfident: bool
    is_underconfident: bool
    adjustment_factor: float  # Multiply confidence by this
    sample_size: int


@dataclass
class ConsensusSignal:
    """Multi-system consensus tracking"""
    ticker: str
    direction: str
    systems_agreeing: List[str]
    avg_confidence: float
    consensus_strength: int  # Number of systems
    timestamp: datetime
    outcome: Optional[str] = None
    return_realized: Optional[float] = None


@dataclass
class SPPValidationReport:
    """S++ Tier Comprehensive Validation Report"""
    # Summary
    total_signals_tracked: int
    total_validated: int
    validation_rate: float

    # Win rates by horizon
    win_rate_1d: float
    win_rate_5d: float
    win_rate_20d: float
    win_rate_60d: float

    # Average returns by horizon
    avg_return_1d: float
    avg_return_5d: float
    avg_return_20d: float
    avg_return_60d: float

    # System rankings
    system_rankings: List[Dict[str, Any]]
    top_systems: List[str]
    degraded_systems: List[str]

    # Drift alerts
    drift_alerts: List[DriftAlert]
    systems_with_drift: int

    # Calibration
    calibration_scores: Dict[str, CalibrationResult]
    overall_calibration: float

    # Consensus performance
    consensus_win_rate: float
    consensus_signals_count: int
    consensus_premium: float  # Win rate lift from consensus

    # Quality metrics
    confidence_correlation: float  # Correlation: confidence → accuracy
    layer_alignment_score: float   # Cross-layer agreement

    generated_at: datetime = field(default_factory=datetime.utcnow)


class SPPValidationTracker:
    """
    S++ Tier Validation Tracker

    Extends standard validation with:
    - 60-day outcome tracking
    - Model drift detection
    - Confidence calibration
    - Consensus signal tracking
    - Real-time degradation alerts
    """

    HORIZONS = [1, 5, 20, 60]  # Days
    DRIFT_WINDOW = 30  # Days for recent performance
    MIN_SAMPLES_DRIFT = 20  # Minimum signals for drift detection

    def __init__(self, db_path: str = None):
        if db_path is None:
            db_dir = Path.home() / ".kirmani_reports"
            db_dir.mkdir(exist_ok=True)
            db_path = str(db_dir / "spp_validation.db")

        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        """Initialize S++ validation database schema"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Extended signals table with 60d tracking
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS spp_signals (
                signal_id TEXT PRIMARY KEY,
                system_id TEXT NOT NULL,
                signal_type TEXT NOT NULL,
                ticker TEXT NOT NULL,
                direction TEXT NOT NULL,
                confidence REAL NOT NULL,
                magnitude REAL NOT NULL,
                layer TEXT,  -- MACRO, RISK, ALPHA

                entry_price REAL NOT NULL,
                entry_timestamp TEXT NOT NULL,

                -- Outcomes at each horizon
                price_1d REAL, return_1d REAL, outcome_1d TEXT,
                price_5d REAL, return_5d REAL, outcome_5d TEXT,
                price_20d REAL, return_20d REAL, outcome_20d TEXT,
                price_60d REAL, return_60d REAL, outcome_60d TEXT,

                validated_1d INTEGER DEFAULT 0,
                validated_5d INTEGER DEFAULT 0,
                validated_20d INTEGER DEFAULT 0,
                validated_60d INTEGER DEFAULT 0,
                fully_validated INTEGER DEFAULT 0,

                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Consensus signals table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS consensus_signals (
                consensus_id TEXT PRIMARY KEY,
                ticker TEXT NOT NULL,
                direction TEXT NOT NULL,
                systems_json TEXT NOT NULL,  -- JSON array of system_ids
                consensus_strength INTEGER NOT NULL,
                avg_confidence REAL NOT NULL,
                entry_price REAL,
                entry_timestamp TEXT NOT NULL,

                outcome TEXT,
                return_realized REAL,
                validated INTEGER DEFAULT 0,

                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Drift detection history
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS drift_alerts (
                alert_id INTEGER PRIMARY KEY AUTOINCREMENT,
                system_id TEXT NOT NULL,
                severity TEXT NOT NULL,
                historical_wr REAL,
                recent_wr REAL,
                degradation_pct REAL,
                sample_size INTEGER,
                detected_at TEXT,
                resolved INTEGER DEFAULT 0,
                resolved_at TEXT
            )
        """)

        # Calibration history
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS calibration_history (
                calibration_id INTEGER PRIMARY KEY AUTOINCREMENT,
                system_id TEXT NOT NULL,
                expected_wr REAL,
                actual_wr REAL,
                calibration_error REAL,
                adjustment_factor REAL,
                sample_size INTEGER,
                calculated_at TEXT
            )
        """)

        # Create indexes
        for idx in [
            "CREATE INDEX IF NOT EXISTS idx_spp_system ON spp_signals(system_id)",
            "CREATE INDEX IF NOT EXISTS idx_spp_ticker ON spp_signals(ticker)",
            "CREATE INDEX IF NOT EXISTS idx_spp_timestamp ON spp_signals(entry_timestamp)",
            "CREATE INDEX IF NOT EXISTS idx_spp_validated ON spp_signals(fully_validated)",
            "CREATE INDEX IF NOT EXISTS idx_consensus_ticker ON consensus_signals(ticker)",
        ]:
            cursor.execute(idx)

        conn.commit()
        conn.close()

    def record_signal(
        self,
        signal: Dict[str, Any],
        entry_price: float,
        layer: str = None,
    ) -> str:
        """Record a signal for S++ validation"""
        import uuid

        signal_id = f"spp_{signal.get('system_id', 'unknown')}_{uuid.uuid4().hex[:8]}"

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            INSERT INTO spp_signals (
                signal_id, system_id, signal_type, ticker, direction,
                confidence, magnitude, layer, entry_price, entry_timestamp
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            signal_id,
            signal.get("system_id", "unknown"),
            signal.get("signal_type", "UNKNOWN"),
            signal.get("ticker", "UNKNOWN"),
            signal.get("direction", "NEUTRAL"),
            signal.get("confidence", 0.5),
            signal.get("magnitude", 0.5),
            layer or self._infer_layer(signal.get("system_id", "")),
            entry_price,
            signal.get("timestamp", datetime.utcnow().isoformat()),
        ))

        conn.commit()
        conn.close()

        return signal_id

    def _infer_layer(self, system_id: str) -> str:
        """Infer signal layer from system ID"""
        macro_systems = {"sornette", "minsky", "murphy", "cgma", "gkhy", "nations"}
        risk_systems = {"sentinel", "frm", "greeks", "vixlab", "squeeze", "stress"}

        system_lower = system_id.lower()

        for s in macro_systems:
            if s in system_lower:
                return "MACRO"

        for s in risk_systems:
            if s in system_lower:
                return "RISK"

        return "ALPHA"

    def record_consensus(
        self,
        ticker: str,
        direction: str,
        systems: List[str],
        avg_confidence: float,
        entry_price: float = None,
    ) -> str:
        """Record a consensus signal (multiple systems agreeing)"""
        import uuid

        consensus_id = f"consensus_{ticker}_{uuid.uuid4().hex[:8]}"

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            INSERT INTO consensus_signals (
                consensus_id, ticker, direction, systems_json,
                consensus_strength, avg_confidence, entry_price, entry_timestamp
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            consensus_id,
            ticker,
            direction,
            json.dumps(systems),
            len(systems),
            avg_confidence,
            entry_price,
            datetime.utcnow().isoformat(),
        ))

        conn.commit()
        conn.close()

        logger.info(f"Recorded consensus: {len(systems)} systems on {ticker} {direction}")
        return consensus_id

    def update_outcomes(
        self,
        prices: Dict[str, float],
        as_of_date: datetime = None,
    ) -> Dict[str, int]:
        """Update outcomes for all pending signals"""
        if as_of_date is None:
            as_of_date = datetime.utcnow()

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        updates = {h: 0 for h in self.HORIZONS}

        # Get pending signals
        cursor.execute("""
            SELECT signal_id, ticker, direction, entry_price, entry_timestamp,
                   validated_1d, validated_5d, validated_20d, validated_60d
            FROM spp_signals
            WHERE fully_validated = 0
        """)

        pending = cursor.fetchall()

        for row in pending:
            (signal_id, ticker, direction, entry_price, entry_ts,
             v1, v5, v20, v60) = row

            current_price = prices.get(ticker)
            if current_price is None or entry_price == 0:
                continue

            entry_time = datetime.fromisoformat(entry_ts)
            days_since = (as_of_date - entry_time).days

            # Calculate return
            raw_return = (current_price - entry_price) / entry_price

            if direction == "BEARISH":
                adj_return = -raw_return
            elif direction == "BULLISH":
                adj_return = raw_return
            else:
                adj_return = abs(raw_return)

            # Determine outcome
            if adj_return > 0.005:  # 0.5% threshold
                outcome = "WIN"
            elif adj_return < -0.005:
                outcome = "LOSS"
            else:
                outcome = "NEUTRAL"

            # Update each horizon as it becomes available
            for horizon in self.HORIZONS:
                if days_since >= horizon:
                    col_validated = f"validated_{horizon}d"
                    if not locals().get(f"v{horizon}", 0):
                        cursor.execute(f"""
                            UPDATE spp_signals SET
                                price_{horizon}d = ?,
                                return_{horizon}d = ?,
                                outcome_{horizon}d = ?,
                                {col_validated} = 1
                            WHERE signal_id = ? AND {col_validated} = 0
                        """, (current_price, round(adj_return, 4), outcome, signal_id))

                        if cursor.rowcount > 0:
                            updates[horizon] += 1

            # Check if fully validated
            if days_since >= 60:
                cursor.execute("""
                    UPDATE spp_signals SET fully_validated = 1
                    WHERE signal_id = ?
                """, (signal_id,))

        conn.commit()
        conn.close()

        logger.info(f"Updated outcomes: {updates}")
        return updates

    def detect_drift(self) -> List[DriftAlert]:
        """Detect model drift for all systems"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        alerts = []

        # Get all systems
        cursor.execute("SELECT DISTINCT system_id FROM spp_signals")
        systems = [r[0] for r in cursor.fetchall()]

        thirty_days_ago = (datetime.utcnow() - timedelta(days=self.DRIFT_WINDOW)).isoformat()

        for system_id in systems:
            # Historical win rate (all time)
            cursor.execute("""
                SELECT COUNT(*), SUM(CASE WHEN outcome_20d = 'WIN' THEN 1 ELSE 0 END)
                FROM spp_signals
                WHERE system_id = ? AND validated_20d = 1
            """, (system_id,))

            hist_total, hist_wins = cursor.fetchone()
            if hist_total < self.MIN_SAMPLES_DRIFT:
                continue

            hist_wr = (hist_wins or 0) / hist_total

            # Recent win rate (last 30 days)
            cursor.execute("""
                SELECT COUNT(*), SUM(CASE WHEN outcome_20d = 'WIN' THEN 1 ELSE 0 END)
                FROM spp_signals
                WHERE system_id = ? AND validated_20d = 1 AND entry_timestamp > ?
            """, (system_id, thirty_days_ago))

            recent_total, recent_wins = cursor.fetchone()
            if recent_total < 5:  # Need at least 5 recent signals
                continue

            recent_wr = (recent_wins or 0) / recent_total

            # Calculate degradation
            if hist_wr > 0:
                degradation = (hist_wr - recent_wr) / hist_wr * 100
            else:
                degradation = 0

            # Determine severity
            if degradation >= 30:
                severity = DriftSeverity.CRITICAL
                recommendation = f"CRITICAL: {system_id} degraded {degradation:.0f}%. Consider disabling."
            elif degradation >= 15:
                severity = DriftSeverity.SEVERE
                recommendation = f"SEVERE: {system_id} degraded {degradation:.0f}%. Reduce weight significantly."
            elif degradation >= 5:
                severity = DriftSeverity.MODERATE
                recommendation = f"MODERATE: {system_id} degraded {degradation:.0f}%. Monitor closely."
            elif degradation > 0:
                severity = DriftSeverity.MINOR
                recommendation = f"MINOR: {system_id} slightly degraded. Continue monitoring."
            else:
                severity = DriftSeverity.NONE
                continue  # No alert needed

            alert = DriftAlert(
                system_id=system_id,
                severity=severity,
                historical_win_rate=round(hist_wr, 3),
                recent_win_rate=round(recent_wr, 3),
                degradation_pct=round(degradation, 1),
                sample_size=recent_total,
                detected_at=datetime.utcnow(),
                recommendation=recommendation,
            )

            alerts.append(alert)

            # Record alert
            cursor.execute("""
                INSERT INTO drift_alerts (
                    system_id, severity, historical_wr, recent_wr,
                    degradation_pct, sample_size, detected_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                system_id, severity.value, hist_wr, recent_wr,
                degradation, recent_total, datetime.utcnow().isoformat()
            ))

        conn.commit()
        conn.close()

        return alerts

    def calculate_calibration(self, system_id: str = None) -> Dict[str, CalibrationResult]:
        """Calculate confidence calibration for systems"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        if system_id:
            systems = [system_id]
        else:
            cursor.execute("SELECT DISTINCT system_id FROM spp_signals WHERE validated_20d = 1")
            systems = [r[0] for r in cursor.fetchall()]

        results = {}

        for sys_id in systems:
            cursor.execute("""
                SELECT confidence, outcome_20d
                FROM spp_signals
                WHERE system_id = ? AND validated_20d = 1
            """, (sys_id,))

            rows = cursor.fetchall()
            if len(rows) < 10:
                continue

            confidences = [r[0] for r in rows]
            wins = [1 if r[1] == "WIN" else 0 for r in rows]

            expected_wr = statistics.mean(confidences)
            actual_wr = statistics.mean(wins)

            calibration_error = abs(expected_wr - actual_wr)

            # Calculate adjustment factor
            if expected_wr > 0:
                adjustment = actual_wr / expected_wr
            else:
                adjustment = 1.0

            results[sys_id] = CalibrationResult(
                system_id=sys_id,
                expected_win_rate=round(expected_wr, 3),
                actual_win_rate=round(actual_wr, 3),
                calibration_error=round(calibration_error, 3),
                is_overconfident=expected_wr > actual_wr + 0.05,
                is_underconfident=actual_wr > expected_wr + 0.05,
                adjustment_factor=round(adjustment, 2),
                sample_size=len(rows),
            )

            # Record calibration
            cursor.execute("""
                INSERT INTO calibration_history (
                    system_id, expected_wr, actual_wr, calibration_error,
                    adjustment_factor, sample_size, calculated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                sys_id, expected_wr, actual_wr, calibration_error,
                adjustment, len(rows), datetime.utcnow().isoformat()
            ))

        conn.commit()
        conn.close()

        return results

    def get_consensus_performance(self) -> Dict[str, float]:
        """Calculate consensus signal performance"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Overall consensus performance
        cursor.execute("""
            SELECT
                COUNT(*) as total,
                SUM(CASE WHEN outcome = 'WIN' THEN 1 ELSE 0 END) as wins,
                AVG(return_realized) as avg_return
            FROM consensus_signals
            WHERE validated = 1
        """)

        total, wins, avg_ret = cursor.fetchone()

        # Compare to non-consensus
        cursor.execute("""
            SELECT AVG(CASE WHEN outcome_20d = 'WIN' THEN 1.0 ELSE 0.0 END)
            FROM spp_signals
            WHERE validated_20d = 1
        """)

        baseline_wr = cursor.fetchone()[0] or 0.5

        conn.close()

        consensus_wr = (wins or 0) / (total or 1)
        premium = consensus_wr - baseline_wr

        return {
            "consensus_total": total or 0,
            "consensus_wins": wins or 0,
            "consensus_win_rate": round(consensus_wr, 3),
            "baseline_win_rate": round(baseline_wr, 3),
            "consensus_premium": round(premium, 3),
            "consensus_avg_return": round(avg_ret or 0, 4),
        }

    def generate_spp_report(self) -> SPPValidationReport:
        """Generate comprehensive S++ validation report"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Overall stats
        cursor.execute("""
            SELECT
                COUNT(*) as total,
                SUM(fully_validated) as validated,
                AVG(CASE WHEN outcome_1d = 'WIN' THEN 1.0 ELSE 0.0 END) as wr_1d,
                AVG(CASE WHEN outcome_5d = 'WIN' THEN 1.0 ELSE 0.0 END) as wr_5d,
                AVG(CASE WHEN outcome_20d = 'WIN' THEN 1.0 ELSE 0.0 END) as wr_20d,
                AVG(CASE WHEN outcome_60d = 'WIN' THEN 1.0 ELSE 0.0 END) as wr_60d,
                AVG(return_1d) as ret_1d,
                AVG(return_5d) as ret_5d,
                AVG(return_20d) as ret_20d,
                AVG(return_60d) as ret_60d
            FROM spp_signals
        """)

        row = cursor.fetchone()

        total = row[0] or 0
        validated = row[1] or 0

        # System rankings
        cursor.execute("""
            SELECT
                system_id,
                COUNT(*) as signals,
                AVG(CASE WHEN outcome_20d = 'WIN' THEN 1.0 ELSE 0.0 END) as win_rate,
                AVG(return_20d) as avg_return,
                AVG(confidence) as avg_conf
            FROM spp_signals
            WHERE validated_20d = 1
            GROUP BY system_id
            HAVING COUNT(*) >= 5
            ORDER BY win_rate DESC
        """)

        rankings = []
        top_systems = []
        degraded_systems = []

        for r in cursor.fetchall():
            sys_id, signals, wr, ret, conf = r
            rankings.append({
                "system_id": sys_id,
                "signals": signals,
                "win_rate": round(wr or 0, 3),
                "avg_return": round(ret or 0, 4),
                "avg_confidence": round(conf or 0, 3),
            })

            if (wr or 0) >= 0.6:
                top_systems.append(sys_id)
            elif (wr or 0) < 0.4:
                degraded_systems.append(sys_id)

        conn.close()

        # Get drift alerts
        drift_alerts = self.detect_drift()

        # Get calibration
        calibration = self.calculate_calibration()

        overall_cal = 0
        if calibration:
            overall_cal = 1 - statistics.mean(
                [c.calibration_error for c in calibration.values()]
            )

        # Get consensus performance
        consensus = self.get_consensus_performance()

        return SPPValidationReport(
            total_signals_tracked=total,
            total_validated=validated,
            validation_rate=validated / total if total > 0 else 0,
            win_rate_1d=row[2] or 0,
            win_rate_5d=row[3] or 0,
            win_rate_20d=row[4] or 0,
            win_rate_60d=row[5] or 0,
            avg_return_1d=row[6] or 0,
            avg_return_5d=row[7] or 0,
            avg_return_20d=row[8] or 0,
            avg_return_60d=row[9] or 0,
            system_rankings=rankings,
            top_systems=top_systems,
            degraded_systems=degraded_systems,
            drift_alerts=drift_alerts,
            systems_with_drift=len([a for a in drift_alerts if a.severity != DriftSeverity.NONE]),
            calibration_scores=calibration,
            overall_calibration=round(overall_cal, 3),
            consensus_win_rate=consensus["consensus_win_rate"],
            consensus_signals_count=consensus["consensus_total"],
            consensus_premium=consensus["consensus_premium"],
            confidence_correlation=0,  # TODO: Calculate
            layer_alignment_score=0,   # TODO: Calculate
        )

    def print_report(self):
        """Print S++ validation report"""
        report = self.generate_spp_report()

        print("\n" + "=" * 80)
        print("KIRMANI S++ TIER VALIDATION REPORT")
        print("=" * 80)

        print(f"\nSignals: {report.total_signals_tracked} tracked, {report.total_validated} validated ({report.validation_rate:.0%})")

        print("\n" + "-" * 80)
        print("WIN RATES BY HORIZON")
        print("-" * 80)
        print(f"  1-day:  {report.win_rate_1d:.1%}  (avg return: {report.avg_return_1d:+.2%})")
        print(f"  5-day:  {report.win_rate_5d:.1%}  (avg return: {report.avg_return_5d:+.2%})")
        print(f"  20-day: {report.win_rate_20d:.1%}  (avg return: {report.avg_return_20d:+.2%})")
        print(f"  60-day: {report.win_rate_60d:.1%}  (avg return: {report.avg_return_60d:+.2%})")

        if report.system_rankings:
            print("\n" + "-" * 80)
            print("SYSTEM RANKINGS (by 20d win rate)")
            print("-" * 80)
            for r in report.system_rankings[:10]:
                print(f"  {r['system_id']:25s} | WR: {r['win_rate']:.1%} | Ret: {r['avg_return']:+.2%} | N={r['signals']}")

        if report.drift_alerts:
            print("\n" + "-" * 80)
            print(f"DRIFT ALERTS ({len(report.drift_alerts)})")
            print("-" * 80)
            for alert in report.drift_alerts:
                print(f"  [{alert.severity.value}] {alert.system_id}: {alert.degradation_pct:+.1f}% degradation")
                print(f"      {alert.recommendation}")

        print("\n" + "-" * 80)
        print("CONSENSUS PERFORMANCE")
        print("-" * 80)
        print(f"  Consensus Win Rate: {report.consensus_win_rate:.1%}")
        print(f"  Consensus Premium:  {report.consensus_premium:+.1%} vs baseline")
        print(f"  Consensus Signals:  {report.consensus_signals_count}")

        print("\n" + "-" * 80)
        print("CALIBRATION")
        print("-" * 80)
        print(f"  Overall Calibration: {report.overall_calibration:.1%}")

        if report.calibration_scores:
            overconf = [s for s, c in report.calibration_scores.items() if c.is_overconfident]
            underconf = [s for s, c in report.calibration_scores.items() if c.is_underconfident]

            if overconf:
                print(f"  Overconfident: {', '.join(overconf[:5])}")
            if underconf:
                print(f"  Underconfident: {', '.join(underconf[:5])}")

        print("\n")


if __name__ == "__main__":
    tracker = SPPValidationTracker()
    tracker.print_report()
