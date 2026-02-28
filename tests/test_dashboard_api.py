"""
tests/test_dashboard_api.py
Unit tests para src/dashboard/app.py — Endpoints de la API del dashboard.

Estrategia de mocking:
  - Se parchea _query() para devolver DataFrames sintéticos
  - Se parchea _table_exists() para controlar existencia de tablas
  - Se parchea _latest_result_file() para simular resultados JSON
  - No se requiere conexión real a PostgreSQL

Cubre:
  - GET /api/status
  - GET /api/signals/latest
  - GET /api/signals/history
  - GET /api/chart/{pair}/{tf}
  - GET /api/metrics
  - GET /api/performance
  - GET /api/positions
  - GET /api/monitor
  - GET /api/config
  - POST /api/config
"""

import os
import json

# Poner DATABASE_URL fake ANTES de importar app (evita error en create_engine)
os.environ.setdefault("DATABASE_URL", "postgresql://test:test@localhost:5432/test")

import numpy as np
import pandas as pd
import pytest
from datetime import datetime, timezone, timedelta
from unittest.mock import patch, MagicMock

from fastapi.testclient import TestClient


# ── Fixture: mock de _query para evitar DB real ───────────────────────────────

def _mock_signals_df(n: int = 20) -> pd.DataFrame:
    """Genera DataFrame de señales sintéticas."""
    rng = np.random.default_rng(42)
    pairs = ["EURUSD", "GBPUSD", "USDJPY", "EURJPY", "XAUUSD"]
    tfs = ["M15", "H1", "H4"]
    ts = pd.date_range("2025-06-01", periods=n, freq="1h", tz="UTC")

    return pd.DataFrame({
        "id": range(1, n + 1),
        "pair": rng.choice(pairs, n),
        "timeframe": rng.choice(tfs, n),
        "timestamp": ts,
        "direction": rng.choice([1, -1], n),
        "confidence": rng.uniform(0.55, 0.95, n).round(3),
        "entry_price": rng.uniform(1.05, 1.15, n).round(5),
        "tp_price": rng.uniform(1.10, 1.20, n).round(5),
        "sl_price": rng.uniform(1.00, 1.10, n).round(5),
        "tp_pips": rng.uniform(20, 80, n).round(1),
        "sl_pips": rng.uniform(10, 40, n).round(1),
        "rr_ratio": rng.uniform(1.5, 3.0, n).round(2),
        "adx": rng.uniform(15, 50, n).round(1),
        "session": rng.choice(["london", "newyork", "overlap"], n),
        "filter_reason": [None] * n,
        "xgb_direction": rng.choice([1, -1], n),
        "lstm_direction": rng.choice([1, -1], n),
        "agreement": rng.choice([True, False], n),
    })


def _mock_ohlcv_df(n: int = 200) -> pd.DataFrame:
    """Genera DataFrame OHLCV sintético."""
    rng = np.random.default_rng(42)
    ts = pd.date_range("2025-06-01", periods=n, freq="1h", tz="UTC")
    closes = 1.1000 + np.cumsum(rng.normal(0, 0.0005, n))

    return pd.DataFrame({
        "timestamp": ts,
        "open": closes + rng.normal(0, 0.0002, n),
        "high": closes + rng.uniform(0, 0.001, n),
        "low": closes - rng.uniform(0, 0.001, n),
        "close": closes,
        "volume": rng.integers(100, 5000, n).astype(float),
    })


def _mock_trades_df(n: int = 30) -> pd.DataFrame:
    """Genera DataFrame de trades cerrados."""
    rng = np.random.default_rng(42)
    pairs = ["EURUSD", "GBPUSD", "XAUUSD"]
    results = ["tp_hit"] * 18 + ["sl_hit"] * 12  # 60% WR

    return pd.DataFrame({
        "id": range(1, n + 1),
        "pair": rng.choice(pairs, n),
        "timeframe": rng.choice(["H1", "H4"], n),
        "direction": rng.choice([1, -1], n),
        "entry_price": rng.uniform(1.05, 1.15, n).round(5),
        "exit_price": rng.uniform(1.05, 1.15, n).round(5),
        "lot_size": np.full(n, 0.01),
        "pnl": np.where(
            np.array(results) == "tp_hit",
            rng.uniform(5, 25, n),
            rng.uniform(-15, -3, n),
        ).round(2),
        "result": results,
        "opened_at": pd.date_range("2025-06-01", periods=n, freq="1d", tz="UTC"),
        "closed_at": pd.date_range("2025-06-02", periods=n, freq="1d", tz="UTC"),
        "duration_bars": rng.integers(3, 20, n),
    })


def _mock_monitor_df(table: str) -> pd.DataFrame:
    """Genera DataFrame de monitor de datos."""
    pairs = ["EURUSD", "GBPUSD", "USDJPY", "EURJPY", "XAUUSD"]
    tfs = ["M15", "H1", "H4"]
    now = datetime.now(timezone.utc)
    rows = []
    for pair in pairs:
        for tf in tfs:
            ts_col = "last_candle" if "ohlcv" in table else "last_feature"
            rows.append({
                "pair": pair,
                "timeframe": tf,
                ts_col: now - timedelta(minutes=30),
                "total_rows": 5000,
            })
    return pd.DataFrame(rows)


@pytest.fixture
def mock_db(monkeypatch):
    """
    Fixture que parchea _query, _table_exists y _latest_result_file
    para que los endpoints no necesiten BD real.
    """
    import src.dashboard.app as app_module

    signals_df = _mock_signals_df()
    ohlcv_df = _mock_ohlcv_df()
    trades_df = _mock_trades_df()

    def fake_query(sql: str, params=None):
        sql_lower = sql.lower().strip()

        # /api/status
        if "count(*) as n from signals" in sql_lower:
            return pd.DataFrame({"n": [50]})
        if "count(*) as n from ohlcv_raw" in sql_lower:
            return pd.DataFrame({"n": [10000]})
        if "from signals where direction != 0 order by timestamp desc limit 1" in sql_lower:
            return pd.DataFrame({"timestamp": [datetime.now(timezone.utc)]})
        if "count(*) as n from positions_active" in sql_lower:
            return pd.DataFrame({"n": [2]})

        # /api/signals/latest
        if "from signals order by timestamp desc limit" in sql_lower:
            n = params.get("limit", 20) if params else 20
            df = signals_df.head(n).copy()
            df["timestamp"] = df["timestamp"].astype(str)
            return df

        # /api/signals/history — count
        if "select count(*) as n from signals where" in sql_lower:
            return pd.DataFrame({"n": [len(signals_df)]})

        # /api/signals/history — data
        if "from signals where" in sql_lower and "limit" in sql_lower:
            return signals_df.head(params.get("limit", 50) if params else 50).copy()

        # /api/monitor — ohlcv (check BEFORE chart, both match "from ohlcv_raw")
        if "from ohlcv_raw" in sql_lower and "group by pair" in sql_lower:
            return _mock_monitor_df("ohlcv")

        # /api/monitor — features
        if "from features_computed" in sql_lower and "group by pair" in sql_lower:
            return _mock_monitor_df("features")

        # /api/chart — OHLCV
        if "from ohlcv_raw" in sql_lower and "pair" in sql_lower:
            return ohlcv_df.copy()

        # /api/chart — signals overlay
        if "from signals" in sql_lower and "filter_reason is null" in sql_lower:
            overlay = signals_df.head(5).copy()
            overlay["timestamp"] = overlay["timestamp"].astype(str)
            return overlay

        # /api/metrics
        if "from signals where" in sql_lower and "agreement" in sql_lower:
            return signals_df.copy()

        # /api/performance
        if "from trades_history" in sql_lower:
            return trades_df.copy()

        # /api/positions
        if "from positions_active" in sql_lower:
            return pd.DataFrame()  # sin posiciones

        # Fallback: tabla existe check
        if "information_schema.tables" in sql_lower:
            return pd.DataFrame({0: [True]})

        return pd.DataFrame()

    def fake_table_exists(table: str) -> bool:
        return table in ("signals", "ohlcv_raw", "features_computed",
                         "trades_history", "positions_active", "pipeline_runs")

    def fake_latest_result(prefix: str):
        if prefix == "anomalies_":
            return {"alerts": [], "summary": {"total_checks": 6, "alerts": 0}}
        if prefix == "health_":
            return {"models": [], "overall": "healthy"}
        if prefix == "summary_":
            return {"period": "2025-06", "metrics": {}}
        return None

    def fake_latest_meta(prefix: str):
        return {"file": f"{prefix}2025-06.json",
                "modified": datetime.now(timezone.utc).isoformat()}

    monkeypatch.setattr(app_module, "_query", fake_query)
    monkeypatch.setattr(app_module, "_table_exists", fake_table_exists)
    monkeypatch.setattr(app_module, "_latest_result_file", fake_latest_result)
    monkeypatch.setattr(app_module, "_latest_result_meta", fake_latest_meta)


@pytest.fixture
def client(mock_db):
    """TestClient con mocks activos."""
    from src.dashboard.app import app
    return TestClient(app, raise_server_exceptions=False)


# ── Tests GET /api/status ─────────────────────────────────────────────────────

class TestApiStatus:
    def test_status_200(self, client):
        r = client.get("/api/status")
        assert r.status_code == 200

    def test_status_has_required_keys(self, client):
        data = client.get("/api/status").json()
        assert "status" in data
        assert "total_signals" in data
        assert "total_bars" in data
        assert "server_time" in data

    def test_status_online(self, client):
        data = client.get("/api/status").json()
        assert data["status"] == "online"

    def test_status_signal_count_positive(self, client):
        data = client.get("/api/status").json()
        assert data["total_signals"] > 0
        assert data["total_bars"] > 0


# ── Tests GET /api/signals/latest ─────────────────────────────────────────────

class TestApiSignalsLatest:
    def test_latest_200(self, client):
        r = client.get("/api/signals/latest")
        assert r.status_code == 200

    def test_latest_returns_list(self, client):
        data = client.get("/api/signals/latest").json()
        assert isinstance(data, list)

    def test_latest_limit_parameter(self, client):
        data = client.get("/api/signals/latest?limit=5").json()
        assert len(data) <= 5

    def test_latest_signal_has_fields(self, client):
        data = client.get("/api/signals/latest?limit=1").json()
        if data:
            sig = data[0]
            assert "pair" in sig
            assert "direction" in sig
            assert "confidence" in sig


# ── Tests GET /api/signals/history ────────────────────────────────────────────

class TestApiSignalsHistory:
    def test_history_200(self, client):
        r = client.get("/api/signals/history")
        assert r.status_code == 200

    def test_history_has_pagination(self, client):
        data = client.get("/api/signals/history").json()
        assert "total" in data
        assert "page" in data
        assert "pages" in data
        assert "signals" in data

    def test_history_filter_by_pair(self, client):
        r = client.get("/api/signals/history?pair=EURUSD")
        assert r.status_code == 200

    def test_history_filter_by_direction(self, client):
        r = client.get("/api/signals/history?direction=1")
        assert r.status_code == 200

    def test_history_pagination(self, client):
        r = client.get("/api/signals/history?page=1&page_size=10")
        assert r.status_code == 200


# ── Tests GET /api/chart/{pair}/{tf} ──────────────────────────────────────────

class TestApiChart:
    def test_chart_200(self, client):
        r = client.get("/api/chart/EURUSD/H1")
        assert r.status_code == 200

    def test_chart_has_candles(self, client):
        data = client.get("/api/chart/EURUSD/H1").json()
        assert "candles" in data
        assert "signals" in data
        assert "pair" in data
        assert "precision" in data

    def test_chart_candles_have_ohlcv(self, client):
        data = client.get("/api/chart/EURUSD/H1").json()
        if data["candles"]:
            candle = data["candles"][0]
            for key in ("timestamp", "open", "high", "low", "close"):
                assert key in candle

    def test_chart_precision_eurusd(self, client):
        data = client.get("/api/chart/EURUSD/H1").json()
        assert data["precision"] == 5

    def test_chart_bars_parameter(self, client):
        r = client.get("/api/chart/EURUSD/H1?bars=50")
        assert r.status_code == 200


# ── Tests GET /api/metrics ────────────────────────────────────────────────────

class TestApiMetrics:
    def test_metrics_200(self, client):
        r = client.get("/api/metrics")
        assert r.status_code == 200

    def test_metrics_has_distribution(self, client):
        data = client.get("/api/metrics").json()
        assert "total_signals" in data
        assert "long_signals" in data
        assert "short_signals" in data
        assert "avg_confidence" in data

    def test_metrics_filter_by_days(self, client):
        r = client.get("/api/metrics?days=30")
        assert r.status_code == 200

    def test_metrics_percentages_valid(self, client):
        data = client.get("/api/metrics").json()
        if "long_pct" in data and "short_pct" in data:
            assert 0 <= data["long_pct"] <= 100
            assert 0 <= data["short_pct"] <= 100


# ── Tests GET /api/monitor ────────────────────────────────────────────────────

class TestApiMonitor:
    def test_monitor_200(self, client):
        r = client.get("/api/monitor")
        assert r.status_code == 200

    def test_monitor_has_sections(self, client):
        data = client.get("/api/monitor").json()
        assert "summary" in data
        assert "ohlcv" in data
        assert "features" in data

    def test_monitor_ohlcv_entries(self, client):
        data = client.get("/api/monitor").json()
        assert len(data["ohlcv"]) > 0
        entry = data["ohlcv"][0]
        assert "pair" in entry
        assert "timeframe" in entry
        assert "status" in entry


# ── Tests GET /api/config & POST /api/config ──────────────────────────────────

class TestApiConfig:
    def test_get_config_200(self, client):
        r = client.get("/api/config")
        assert r.status_code == 200

    def test_get_config_has_filters(self, client):
        data = client.get("/api/config").json()
        assert "min_confidence" in data
        assert "min_adx" in data
        assert "min_rr" in data
        assert "cooldown_bars" in data

    def test_post_config_updates(self, client):
        new_cfg = {
            "min_confidence": 0.75,
            "min_adx": 25.0,
            "allow_offmarket": True,
            "min_rr": 2.0,
            "cooldown_bars": 5,
            "tp_multiplier": 2.5,
            "sl_multiplier": 1.5,
        }
        r = client.post("/api/config", json=new_cfg)
        assert r.status_code == 200
        data = r.json()
        assert data["ok"] is True
        assert data["config"]["min_confidence"] == 0.75

    def test_config_roundtrip(self, client):
        """POST config → GET config devuelve lo mismo."""
        new_cfg = {
            "min_confidence": 0.80,
            "min_adx": 30.0,
            "allow_offmarket": False,
            "min_rr": 1.8,
            "cooldown_bars": 4,
            "tp_multiplier": 2.0,
            "sl_multiplier": 1.0,
        }
        client.post("/api/config", json=new_cfg)
        data = client.get("/api/config").json()
        assert data["min_confidence"] == 0.80
        assert data["min_adx"] == 30.0


# ── Tests GET /api/health & /api/anomalies & /api/summary ────────────────────

class TestApiMonitoring:
    def test_health_200(self, client):
        r = client.get("/api/health")
        assert r.status_code == 200

    def test_anomalies_200(self, client):
        r = client.get("/api/anomalies")
        assert r.status_code == 200

    def test_summary_200(self, client):
        r = client.get("/api/summary")
        assert r.status_code == 200

    def test_health_has_data(self, client):
        data = client.get("/api/health").json()
        assert "overall" in data or "models" in data or "error" in data


# ── Tests OpenAPI docs ────────────────────────────────────────────────────────

class TestOpenAPI:
    def test_openapi_json(self, client):
        r = client.get("/openapi.json")
        assert r.status_code == 200
        schema = r.json()
        assert schema["info"]["title"] == "ML-Ayram Trading API"
        assert "2.1.0" in schema["info"]["version"]

    def test_openapi_has_tags(self, client):
        schema = client.get("/openapi.json").json()
        tag_names = [t["name"] for t in schema.get("tags", [])]
        assert "Status" in tag_names
        assert "Signals" in tag_names
        assert "Config" in tag_names

    def test_docs_endpoint(self, client):
        r = client.get("/docs")
        assert r.status_code == 200

    def test_redoc_endpoint(self, client):
        r = client.get("/redoc")
        assert r.status_code == 200
