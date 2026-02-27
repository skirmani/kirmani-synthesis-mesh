"""
S-Tier Live Validation Tracker
==============================

Tracks signal predictions vs outcomes for continuous accuracy monitoring.
This is essential for S-tier status - validates that the intelligence
engine actually works with real data.

Features:
- Records all signals with entry conditions
- Tracks outcomes at multiple time horizons (1d, 5d, 20d)
- Calculates accuracy by system, signal type, and ticker
- Monitors model drift (backtest vs live performance)
- Generates validation reports
"""

import json
import logging
import sqlite3
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class TrackedSignal:
    """A signal tracked for validation"""
    signal_id: str
    system_id: str
    signal_type: str
    ticker: str
    direction: str  # BULLISH, BEARISH, NEUTRAL
    confidence: float
    magnitude: float

    # Entry conditions
    entry_price: float
    entry_timestamp: datetime

    # Predicted outcomes
    target_1: Optional[float] = None
    target_2: Optional[float] = None
    stop_loss: Optional[float] = None

    # Actual outcomes (filled in later)
    price_1d: Optional[float] = None
    price_5d: Optional[float] = None
    price_20d: Optional[float] = None

    return_1d: Optional[float] = None
    return_5d: Optional[float] = None
    return_20d: Optional[float] = None

    outcome_1d: Optional[str] = None  # WIN, LOSS, NEUTRAL
    outcome_5d: Optional[str] = None
    outcome_20d: Optional[str] = None

    validated: bool = False
    validation_timestamp: Optional[datetime] = None


@dataclass
class SystemAccuracy:
    """Accuracy metrics for a system"""
    system_id: str
    total_signals: int
    validated_signals: int

    win_rate_1d: float
    win_rate_5d: float
    win_rate_20d: float

    avg_return_1d: float
    avg_return_5d: float
    avg_return_20d: float

    avg_confidence: float
    confidence_correlation: float  # Correlation between confidence and accuracy

    last_updated: datetime


@dataclass
class ValidationReport:
    """Full validation report"""
    total_signals: int
    validated_signals: int
    pending_signals: int

    overall_win_rate: float
    overall_avg_return: float

    system_accuracy: Dict[str, SystemAccuracy]
    signal_type_accuracy: Dict[str, float]
    ticker_accuracy: Dict[str, float]

    model_drift_score: float  # 0 = no drift, 1 = significant drift
    confidence_calibration: float  # How well confidence predicts accuracy

    generated_at: datetime


class LiveValidationTracker:
    """
    S-Tier Validation Tracker

    Tracks signals and their outcomes for continuous accuracy monitoring.
    Uses SQLite for persistent storage across sessions.
    """

    def __init__(self, db_path: str = None):
        if db_path is None:
            db_dir = Path.home() / ".kirmani_reports"
            db_dir.mkdir(exist_ok=True)
            db_path = str(db_dir / "validation_tracker.db")

        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        """Initialize database schema"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS signals (
                signal_id TEXT PRIMARY KEY,
                system_id TEXT NOT NULL,
                signal_type TEXT NOT NULL,
                ticker TEXT NOT NULL,
                direction TEXT NOT NULL,
                confidence REAL NOT NULL,
                magnitude REAL NOT NULL,
                entry_price REAL NOT NULL,
                entry_timestamp TEXT NOT NULL,
                target_1 REAL,
                target_2 REAL,
                stop_loss REAL,
                price_1d REAL,
                price_5d REAL,
                price_20d REAL,
                return_1d REAL,
                return_5d REAL,
                return_20d REAL,
                outcome_1d TEXT,
                outcome_5d TEXT,
                outcome_20d TEXT,
                validated INTEGER DEFAULT 0,
                validation_timestamp TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_signals_system ON signals(system_id)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_signals_ticker ON signals(ticker)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_signals_validated ON signals(validated)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_signals_timestamp ON signals(entry_timestamp)
        """)

        conn.commit()
        conn.close()

    def record_signal(
        self,
        signal: Dict[str, Any],
        entry_price: float,
    ) -> str:
        """
        Record a new signal for tracking.

        Args:
            signal: Signal dict from intelligence engine
            entry_price: Current price at signal generation

        Returns:
            signal_id for later reference
        """
        import uuid

        signal_id = f"{signal.get('system_id', 'unknown')}_{uuid.uuid4().hex[:8]}"

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        components = signal.get("components", {})

        cursor.execute("""
            INSERT INTO signals (
                signal_id, system_id, signal_type, ticker, direction,
                confidence, magnitude, entry_price, entry_timestamp,
                target_1, target_2, stop_loss
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            signal_id,
            signal.get("system_id", "unknown"),
            signal.get("signal_type", "UNKNOWN"),
            signal.get("ticker", "UNKNOWN"),
            signal.get("direction", "NEUTRAL"),
            signal.get("confidence", 0.5),
            signal.get("magnitude", 0.5),
            entry_price,
            signal.get("timestamp", datetime.utcnow().isoformat()),
            components.get("target_1"),
            components.get("target_2"),
            components.get("stop_loss"),
        ))

        conn.commit()
        conn.close()

        logger.debug(f"Recorded signal: {signal_id}")
        return signal_id

    def record_signals_batch(
        self,
        signals: List[Dict[str, Any]],
        prices: Dict[str, float],
    ) -> List[str]:
        """Record multiple signals at once"""
        signal_ids = []

        for signal in signals:
            ticker = signal.get("ticker", "UNKNOWN")
            price = prices.get(ticker, 0.0)

            if price > 0:
                sid = self.record_signal(signal, price)
                signal_ids.append(sid)

        logger.info(f"Recorded {len(signal_ids)} signals for validation")
        return signal_ids

    def update_outcomes(
        self,
        prices: Dict[str, float],
        as_of_date: datetime = None,
    ) -> int:
        """
        Update outcomes for pending signals based on current prices.

        Args:
            prices: Dict of ticker -> current price
            as_of_date: Date to use for calculations (default: now)

        Returns:
            Number of signals updated
        """
        if as_of_date is None:
            as_of_date = datetime.utcnow()

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Get pending signals
        cursor.execute("""
            SELECT signal_id, ticker, direction, entry_price, entry_timestamp
            FROM signals
            WHERE validated = 0
        """)

        pending = cursor.fetchall()
        updated = 0

        for signal_id, ticker, direction, entry_price, entry_ts in pending:
            current_price = prices.get(ticker)
            if current_price is None or entry_price == 0:
                continue

            entry_time = datetime.fromisoformat(entry_ts)
            days_since = (as_of_date - entry_time).days

            # Calculate returns
            raw_return = (current_price - entry_price) / entry_price

            # Adjust for direction
            if direction == "BEARISH":
                adj_return = -raw_return
            elif direction == "BULLISH":
                adj_return = raw_return
            else:
                adj_return = abs(raw_return)  # NEUTRAL: magnitude matters

            # Determine outcome
            if adj_return > 0.01:  # 1% threshold for win
                outcome = "WIN"
            elif adj_return < -0.01:
                outcome = "LOSS"
            else:
                outcome = "NEUTRAL"

            # Update appropriate time horizon
            if days_since >= 1 and days_since < 5:
                cursor.execute("""
                    UPDATE signals SET
                        price_1d = ?, return_1d = ?, outcome_1d = ?
                    WHERE signal_id = ?
                """, (current_price, round(adj_return, 4), outcome, signal_id))
                updated += 1

            elif days_since >= 5 and days_since < 20:
                cursor.execute("""
                    UPDATE signals SET
                        price_5d = ?, return_5d = ?, outcome_5d = ?
                    WHERE signal_id = ?
                """, (current_price, round(adj_return, 4), outcome, signal_id))
                updated += 1

            elif days_since >= 20:
                cursor.execute("""
                    UPDATE signals SET
                        price_20d = ?, return_20d = ?, outcome_20d = ?,
                        validated = 1, validation_timestamp = ?
                    WHERE signal_id = ?
                """, (current_price, round(adj_return, 4), outcome,
                      as_of_date.isoformat(), signal_id))
                updated += 1

        conn.commit()
        conn.close()

        logger.info(f"Updated {updated} signal outcomes")
        return updated

    def get_system_accuracy(self, system_id: str) -> Optional[SystemAccuracy]:
        """Get accuracy metrics for a specific system"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            SELECT
                COUNT(*) as total,
                SUM(CASE WHEN validated = 1 THEN 1 ELSE 0 END) as validated,
                AVG(CASE WHEN outcome_1d = 'WIN' THEN 1.0 ELSE 0.0 END) as wr_1d,
                AVG(CASE WHEN outcome_5d = 'WIN' THEN 1.0 ELSE 0.0 END) as wr_5d,
                AVG(CASE WHEN outcome_20d = 'WIN' THEN 1.0 ELSE 0.0 END) as wr_20d,
                AVG(return_1d) as avg_ret_1d,
                AVG(return_5d) as avg_ret_5d,
                AVG(return_20d) as avg_ret_20d,
                AVG(confidence) as avg_conf
            FROM signals
            WHERE system_id = ?
        """, (system_id,))

        row = cursor.fetchone()
        conn.close()

        if not row or row[0] == 0:
            return None

        return SystemAccuracy(
            system_id=system_id,
            total_signals=row[0],
            validated_signals=row[1] or 0,
            win_rate_1d=row[2] or 0.0,
            win_rate_5d=row[3] or 0.0,
            win_rate_20d=row[4] or 0.0,
            avg_return_1d=row[5] or 0.0,
            avg_return_5d=row[6] or 0.0,
            avg_return_20d=row[7] or 0.0,
            avg_confidence=row[8] or 0.5,
            confidence_correlation=self._calc_confidence_correlation(system_id),
            last_updated=datetime.utcnow(),
        )

    def _calc_confidence_correlation(self, system_id: str) -> float:
        """Calculate correlation between confidence and win rate"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            SELECT confidence, outcome_20d
            FROM signals
            WHERE system_id = ? AND validated = 1
        """, (system_id,))

        rows = cursor.fetchall()
        conn.close()

        if len(rows) < 10:
            return 0.0

        # Simple correlation: high conf should correlate with wins
        high_conf_wins = sum(1 for c, o in rows if c > 0.7 and o == "WIN")
        high_conf_total = sum(1 for c, o in rows if c > 0.7)
        low_conf_wins = sum(1 for c, o in rows if c <= 0.7 and o == "WIN")
        low_conf_total = sum(1 for c, o in rows if c <= 0.7)

        if high_conf_total == 0 or low_conf_total == 0:
            return 0.0

        high_wr = high_conf_wins / high_conf_total
        low_wr = low_conf_wins / low_conf_total

        # Correlation: positive if high conf has higher win rate
        return round(high_wr - low_wr, 3)

    def generate_validation_report(self) -> ValidationReport:
        """Generate full validation report"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Overall stats
        cursor.execute("""
            SELECT
                COUNT(*) as total,
                SUM(CASE WHEN validated = 1 THEN 1 ELSE 0 END) as validated,
                AVG(CASE WHEN outcome_20d = 'WIN' THEN 1.0 ELSE 0.0 END) as overall_wr,
                AVG(return_20d) as overall_ret
            FROM signals
        """)

        overall = cursor.fetchone()

        # System accuracy
        cursor.execute("SELECT DISTINCT system_id FROM signals")
        systems = [r[0] for r in cursor.fetchall()]

        system_accuracy = {}
        for sys_id in systems:
            acc = self.get_system_accuracy(sys_id)
            if acc:
                system_accuracy[sys_id] = acc

        # Signal type accuracy
        cursor.execute("""
            SELECT signal_type,
                   AVG(CASE WHEN outcome_20d = 'WIN' THEN 1.0 ELSE 0.0 END) as wr
            FROM signals
            WHERE validated = 1
            GROUP BY signal_type
        """)
        signal_type_accuracy = {r[0]: r[1] for r in cursor.fetchall()}

        # Ticker accuracy
        cursor.execute("""
            SELECT ticker,
                   AVG(CASE WHEN outcome_20d = 'WIN' THEN 1.0 ELSE 0.0 END) as wr
            FROM signals
            WHERE validated = 1
            GROUP BY ticker
            HAVING COUNT(*) >= 3
        """)
        ticker_accuracy = {r[0]: r[1] for r in cursor.fetchall()}

        conn.close()

        # Model drift score (compare recent vs historical)
        drift_score = self._calculate_drift_score()

        # Confidence calibration
        conf_calibration = self._calculate_confidence_calibration()

        return ValidationReport(
            total_signals=overall[0] or 0,
            validated_signals=overall[1] or 0,
            pending_signals=(overall[0] or 0) - (overall[1] or 0),
            overall_win_rate=overall[2] or 0.0,
            overall_avg_return=overall[3] or 0.0,
            system_accuracy=system_accuracy,
            signal_type_accuracy=signal_type_accuracy,
            ticker_accuracy=ticker_accuracy,
            model_drift_score=drift_score,
            confidence_calibration=conf_calibration,
            generated_at=datetime.utcnow(),
        )

    def _calculate_drift_score(self) -> float:
        """Calculate model drift (recent performance vs historical)"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Recent (last 30 days) vs historical
        thirty_days_ago = (datetime.utcnow() - timedelta(days=30)).isoformat()

        cursor.execute("""
            SELECT
                AVG(CASE WHEN outcome_20d = 'WIN' THEN 1.0 ELSE 0.0 END) as recent_wr
            FROM signals
            WHERE validated = 1 AND entry_timestamp > ?
        """, (thirty_days_ago,))
        recent = cursor.fetchone()[0] or 0.5

        cursor.execute("""
            SELECT
                AVG(CASE WHEN outcome_20d = 'WIN' THEN 1.0 ELSE 0.0 END) as hist_wr
            FROM signals
            WHERE validated = 1 AND entry_timestamp <= ?
        """, (thirty_days_ago,))
        historical = cursor.fetchone()[0] or 0.5

        conn.close()

        # Drift = absolute difference
        return round(abs(recent - historical), 3)

    def _calculate_confidence_calibration(self) -> float:
        """Calculate how well confidence predicts accuracy"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Group by confidence buckets and compare expected vs actual
        cursor.execute("""
            SELECT
                ROUND(confidence, 1) as conf_bucket,
                AVG(CASE WHEN outcome_20d = 'WIN' THEN 1.0 ELSE 0.0 END) as actual_wr,
                COUNT(*) as cnt
            FROM signals
            WHERE validated = 1
            GROUP BY ROUND(confidence, 1)
            HAVING COUNT(*) >= 5
        """)

        rows = cursor.fetchall()
        conn.close()

        if not rows:
            return 0.5

        # Perfect calibration: confidence = win rate
        # Calculate mean squared error
        mse = sum((r[0] - r[1]) ** 2 for r in rows) / len(rows)

        # Convert to calibration score (0 = poor, 1 = perfect)
        calibration = max(0, 1 - mse * 4)

        return round(calibration, 3)

    def print_summary(self):
        """Print validation summary"""
        report = self.generate_validation_report()

        print("\n" + "=" * 70)
        print("KIRMANI S-TIER VALIDATION REPORT")
        print("=" * 70)

        print(f"\nTotal Signals: {report.total_signals}")
        print(f"Validated: {report.validated_signals}")
        print(f"Pending: {report.pending_signals}")

        print(f"\nOverall Win Rate: {report.overall_win_rate:.1%}")
        print(f"Overall Avg Return: {report.overall_avg_return:.2%}")

        print(f"\nModel Drift Score: {report.model_drift_score:.3f} (0=stable, 1=drifted)")
        print(f"Confidence Calibration: {report.confidence_calibration:.3f} (1=perfect)")

        if report.system_accuracy:
            print("\n" + "-" * 70)
            print("SYSTEM ACCURACY")
            print("-" * 70)

            for sys_id, acc in sorted(
                report.system_accuracy.items(),
                key=lambda x: x[1].win_rate_20d,
                reverse=True
            ):
                print(f"  {sys_id:20s} | WR: {acc.win_rate_20d:.1%} | "
                      f"Ret: {acc.avg_return_20d:+.2%} | "
                      f"Conf Corr: {acc.confidence_correlation:+.3f} | "
                      f"N={acc.validated_signals}")

        print("\n")


def run_validation_update():
    """
    Run validation update with current prices.

    This should be run daily to update signal outcomes.
    """
    from .real_data import RealDataConnector

    print("=" * 70)
    print("KIRMANI VALIDATION UPDATE")
    print("=" * 70)

    # Get current prices
    connector = RealDataConnector()
    tracker = LiveValidationTracker()

    print("\n[1] Loading pending signals...")

    # Get all unique tickers from pending signals
    conn = sqlite3.connect(tracker.db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT DISTINCT ticker FROM signals WHERE validated = 0")
    tickers = [r[0] for r in cursor.fetchall()]
    conn.close()

    if not tickers:
        print("    No pending signals to validate")
        tracker.print_summary()
        return

    print(f"    {len(tickers)} tickers to price")

    # Get prices
    print("\n[2] Fetching current prices...")
    snapshots = connector.get_market_snapshot(tickers)
    prices = {t: s.price for t, s in snapshots.items()}
    print(f"    Got prices for {len(prices)} tickers")

    # Update outcomes
    print("\n[3] Updating signal outcomes...")
    updated = tracker.update_outcomes(prices)
    print(f"    Updated {updated} signals")

    # Print summary
    print("\n[4] Generating validation report...")
    tracker.print_summary()


if __name__ == "__main__":
    run_validation_update()
