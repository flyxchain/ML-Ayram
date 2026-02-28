"""
tests/conftest.py
Fixtures compartidos para toda la suite de tests.
"""

import numpy as np
import pandas as pd
import pytest
from datetime import datetime, timezone, timedelta


# ── Generador de velas OHLCV sintéticas ──────────────────────────────────────

def _make_ohlcv(
    n: int = 300,
    pair: str = "EURUSD",
    tf: str = "H1",
    start_price: float = 1.1000,
    volatility: float = 0.0005,
    start_ts: datetime = None,
) -> pd.DataFrame:
    """Genera n velas OHLCV con random walk realista."""
    rng = np.random.default_rng(42)
    if start_ts is None:
        start_ts = datetime(2025, 1, 1, tzinfo=timezone.utc)

    tf_delta = {"M15": timedelta(minutes=15), "H1": timedelta(hours=1),
                "H4": timedelta(hours=4), "D1": timedelta(days=1)}.get(tf, timedelta(hours=1))

    timestamps = [start_ts + tf_delta * i for i in range(n)]
    closes = [start_price]
    for _ in range(n - 1):
        closes.append(closes[-1] + rng.normal(0, volatility))
    closes = np.array(closes)

    highs = closes + rng.uniform(0, volatility * 2, n)
    lows = closes - rng.uniform(0, volatility * 2, n)
    opens = closes + rng.normal(0, volatility * 0.5, n)

    return pd.DataFrame({
        "pair": pair,
        "timeframe": tf,
        "timestamp": timestamps,
        "open": opens,
        "high": highs,
        "low": lows,
        "close": closes,
        "volume": rng.integers(100, 5000, n).astype(float),
    })


@pytest.fixture
def ohlcv_eurusd_h1():
    """300 velas H1 de EURUSD para calcular todos los indicadores."""
    return _make_ohlcv(300, "EURUSD", "H1", 1.1000)


@pytest.fixture
def ohlcv_xauusd_h1():
    """300 velas H1 de XAUUSD."""
    return _make_ohlcv(300, "XAUUSD", "H1", 2650.0, volatility=5.0)


@pytest.fixture
def ohlcv_short():
    """Solo 50 velas — insuficientes para EMA200."""
    return _make_ohlcv(50, "EURUSD", "H1", 1.1000)


@pytest.fixture
def ohlcv_eurusd_h4():
    """300 velas H4 de EURUSD para HTF features."""
    return _make_ohlcv(300, "EURUSD", "H4", 1.1000)


@pytest.fixture
def sample_signal():
    """Señal simulada como pd.Series (imitando una fila de la BD)."""
    return pd.Series({
        "id": 1,
        "pair": "EURUSD",
        "timeframe": "H1",
        "timestamp": pd.Timestamp("2025-06-15 12:00:00", tz="UTC"),
        "direction": 1,
        "confidence": 0.72,
        "entry_price": 1.1000,
        "tp_price": 1.1050,
        "sl_price": 1.0975,
        "tp_pips": 50.0,
        "sl_pips": 25.0,
        "rr_ratio": 2.0,
        "adx": 28.5,
        "session": "london",
    })


@pytest.fixture
def sample_signal_short():
    """Señal SHORT simulada."""
    return pd.Series({
        "id": 2,
        "pair": "XAUUSD",
        "timeframe": "H4",
        "timestamp": pd.Timestamp("2025-06-15 16:00:00", tz="UTC"),
        "direction": -1,
        "confidence": 0.65,
        "entry_price": 2650.0,
        "tp_price": 2640.0,
        "sl_price": 2655.0,
        "tp_pips": 100.0,
        "sl_pips": 50.0,
        "rr_ratio": 2.0,
        "adx": 32.0,
        "session": "newyork",
    })


@pytest.fixture
def forward_bars_tp_hit():
    """5 barras forward donde el TP se alcanza en la barra 3 (LONG, TP=1.1050)."""
    return pd.DataFrame({
        "timestamp": pd.date_range("2025-06-15 13:00", periods=5, freq="1h", tz="UTC"),
        "open":  [1.1005, 1.1015, 1.1030, 1.1040, 1.1055],
        "high":  [1.1020, 1.1035, 1.1055, 1.1060, 1.1065],   # barra 2 (idx=2) toca 1.1055 >= TP
        "low":   [1.0995, 1.1010, 1.1025, 1.1035, 1.1050],
        "close": [1.1015, 1.1030, 1.1045, 1.1055, 1.1060],
    })


@pytest.fixture
def forward_bars_sl_hit():
    """5 barras forward donde el SL se alcanza en la barra 1 (LONG, SL=1.0975)."""
    return pd.DataFrame({
        "timestamp": pd.date_range("2025-06-15 13:00", periods=5, freq="1h", tz="UTC"),
        "open":  [1.0990, 1.0980, 1.0970, 1.0965, 1.0960],
        "high":  [1.0995, 1.0985, 1.0975, 1.0970, 1.0965],
        "low":   [1.0970, 1.0960, 1.0950, 1.0940, 1.0935],   # barra 0 (idx=0) low=1.0970 <= SL
        "close": [1.0980, 1.0970, 1.0960, 1.0955, 1.0950],
    })


@pytest.fixture
def forward_bars_timeout():
    """5 barras forward laterales — ni TP ni SL se alcanzan."""
    return pd.DataFrame({
        "timestamp": pd.date_range("2025-06-15 13:00", periods=5, freq="1h", tz="UTC"),
        "open":  [1.1000, 1.1002, 1.0998, 1.1001, 1.0999],
        "high":  [1.1010, 1.1008, 1.1005, 1.1007, 1.1006],
        "low":   [1.0995, 1.0997, 1.0993, 1.0996, 1.0994],
        "close": [1.1002, 1.0998, 1.1001, 1.0999, 1.1000],
    })
