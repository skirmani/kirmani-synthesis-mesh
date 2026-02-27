"""Shared test fixtures for kirmani_synthesis_mesh."""
import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta


@pytest.fixture
def sample_ohlcv():
    """Generate sample OHLCV data for testing."""
    dates = pd.date_range(end=datetime.now(), periods=252, freq="B")
    np.random.seed(42)
    close = 100 * np.cumprod(1 + np.random.normal(0.0003, 0.015, len(dates)))
    return pd.DataFrame({
        "date": dates,
        "open": close * (1 + np.random.uniform(-0.005, 0.005, len(dates))),
        "high": close * (1 + np.abs(np.random.normal(0, 0.01, len(dates)))),
        "low": close * (1 - np.abs(np.random.normal(0, 0.01, len(dates)))),
        "close": close,
        "volume": np.random.randint(100000, 10000000, len(dates)),
    }).set_index("date")


@pytest.fixture
def sample_symbols():
    """Standard test symbol universe."""
    return ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "JPM", "V", "SPY"]


@pytest.fixture
def sample_signals():
    """Generate sample trading signals for testing."""
    return [
        {"symbol": "AAPL", "direction": "long", "confidence": 0.85, "source": "kirmani_synthesis_mesh"},
        {"symbol": "MSFT", "direction": "long", "confidence": 0.72, "source": "kirmani_synthesis_mesh"},
        {"symbol": "TSLA", "direction": "short", "confidence": 0.65, "source": "kirmani_synthesis_mesh"},
    ]


@pytest.fixture
def tmp_duckdb(tmp_path):
    """Temporary DuckDB database for testing."""
    return str(tmp_path / "test.duckdb")
