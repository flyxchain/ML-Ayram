"""
src/dashboard/app.py
Dashboard web de ML-Ayram — FastAPI backend

Endpoints:
  GET  /                        → SPA (index.html)
  GET  /api/signals/latest      → señales recientes
  GET  /api/signals/history     → historial paginado
  GET  /api/metrics             → métricas de acierto del modelo
  GET  /api/chart/{pair}/{tf}   → velas OHLCV para el gráfico
  GET  /api/config              → configuración actual de filtros
  POST /api/config              → actualizar filtros del generador
  GET  /api/status              → estado del sistema

Arranque:
  uvicorn src.dashboard.app:app --host 0.0.0.0 --port 8000 --reload
"""

import os
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from sqlalchemy import create_engine, text
from loguru import logger

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")
STATIC_DIR   = Path(__file__).parent / "static"
STATIC_DIR.mkdir(exist_ok=True)

app    = FastAPI(title="ML-Ayram Dashboard", version="1.0.0")
engine = create_engine(DATABASE_URL, pool_pre_ping=True)

# ── Modelos Pydantic ──────────────────────────────────────────────────────────

class FilterConfig(BaseModel):
    min_confidence:  float = 0.54
    min_adx:         float = 20.0
    allow_offmarket: bool  = False
    min_rr:          float = 1.5
    cooldown_bars:   int   = 3
    tp_multiplier:   float = 2.0
    sl_multiplier:   float = 1.0


# Config en memoria (se puede persistir en BD si se quiere)
_current_config = FilterConfig()


# ── Helpers BD ────────────────────────────────────────────────────────────────

def _query(sql: str, params: dict = None) -> pd.DataFrame:
    with engine.connect() as conn:
        return pd.read_sql(text(sql), conn, params=params)


# ── Endpoints API ─────────────────────────────────────────────────────────────

@app.get("/api/status")
def get_status():
    """Estado general del sistema."""
    try:
        total_signals = _query("SELECT COUNT(*) as n FROM signals").iloc[0]["n"]
        total_bars    = _query("SELECT COUNT(*) as n FROM ohlcv_raw").iloc[0]["n"]
        last_signal   = _query(
            "SELECT timestamp FROM signals WHERE direction != 0 ORDER BY timestamp DESC LIMIT 1"
        )
        last_ts = str(last_signal.iloc[0]["timestamp"]) if not last_signal.empty else None

        return {
            "status":         "online",
            "total_signals":  int(total_signals),
            "total_bars":     int(total_bars),
            "last_signal_at": last_ts,
            "server_time":    datetime.now(timezone.utc).isoformat(),
        }
    except Exception as e:
        return {"status": "error", "detail": str(e)}


@app.get("/api/signals/latest")
def get_latest_signals(limit: int = Query(20, le=100)):
    """Señales más recientes (válidas + filtradas)."""
    try:
        df = _query(
            """SELECT * FROM signals
               ORDER BY timestamp DESC
               LIMIT :limit""",
            {"limit": limit},
        )
        df["timestamp"] = df["timestamp"].astype(str)
        return df.to_dict(orient="records")
    except Exception as e:
        raise HTTPException(500, str(e))


@app.get("/api/signals/history")
def get_signal_history(
    pair:      Optional[str] = None,
    timeframe: Optional[str] = None,
    direction: Optional[int] = None,
    days:      int = Query(30, le=365),
    page:      int = Query(1, ge=1),
    page_size: int = Query(50, le=200),
):
    """Historial paginado con filtros opcionales."""
    try:
        where = ["timestamp >= :from_ts", "direction != 0"]
        params = {
            "from_ts": (datetime.now(timezone.utc) - timedelta(days=days)).isoformat(),
            "offset":  (page - 1) * page_size,
            "limit":   page_size,
        }
        if pair:
            where.append("pair = :pair")
            params["pair"] = pair
        if timeframe:
            where.append("timeframe = :timeframe")
            params["timeframe"] = timeframe
        if direction is not None:
            where.append("direction = :direction")
            params["direction"] = direction

        where_sql = " AND ".join(where)

        total = _query(
            f"SELECT COUNT(*) as n FROM signals WHERE {where_sql}",
            {k: v for k, v in params.items() if k not in ("offset", "limit")},
        ).iloc[0]["n"]

        df = _query(
            f"""SELECT * FROM signals WHERE {where_sql}
                ORDER BY timestamp DESC
                LIMIT :limit OFFSET :offset""",
            params,
        )
        df["timestamp"] = df["timestamp"].astype(str)

        return {
            "total":    int(total),
            "page":     page,
            "pages":    max(1, -(-int(total) // page_size)),
            "signals":  df.to_dict(orient="records"),
        }
    except Exception as e:
        raise HTTPException(500, str(e))


@app.get("/api/metrics")
def get_metrics(
    pair:      Optional[str] = None,
    timeframe: Optional[str] = None,
    days:      int = Query(90, le=365),
):
    """Métricas de acierto del modelo."""
    try:
        where = ["s.timestamp >= :from_ts", "s.direction != 0", "s.filter_reason IS NULL"]
        params = {
            "from_ts": (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
        }
        if pair:
            where.append("s.pair = :pair")
            params["pair"] = pair
        if timeframe:
            where.append("s.timeframe = :timeframe")
            params["timeframe"] = timeframe

        where_sql = " AND ".join(where)

        df = _query(
            f"""SELECT s.pair, s.timeframe, s.direction, s.confidence,
                       s.rr_ratio, s.tp_pips, s.sl_pips,
                       s.xgb_direction, s.lstm_direction, s.agreement,
                       s.timestamp
                FROM signals s
                WHERE {where_sql}
                ORDER BY s.timestamp""",
            params,
        )

        if df.empty:
            return {"error": "Sin señales en el período seleccionado"}

        total = len(df)
        longs  = (df["direction"] == 1).sum()
        shorts = (df["direction"] == -1).sum()

        # Acierto de cada modelo individualmente
        xgb_match  = (df["xgb_direction"]  == df["direction"]).sum()
        lstm_match = (df["lstm_direction"] == df["direction"]).sum()
        both_agree = df["agreement"].sum()

        # Distribución por par y timeframe
        by_pair = df.groupby("pair").size().to_dict()
        by_tf   = df.groupby("timeframe").size().to_dict()

        # Confianza promedio
        avg_conf = float(df["confidence"].mean())
        avg_rr   = float(df["rr_ratio"].mean())

        # Señales por día
        df["date"] = pd.to_datetime(df["timestamp"]).dt.date.astype(str)
        by_day = df.groupby("date").size().reset_index(name="count")
        by_day = by_day.tail(30).to_dict(orient="records")

        return {
            "period_days":       days,
            "total_signals":     int(total),
            "long_signals":      int(longs),
            "short_signals":     int(shorts),
            "long_pct":          round(longs  / total * 100, 1),
            "short_pct":         round(shorts / total * 100, 1),
            "avg_confidence":    round(avg_conf * 100, 1),
            "avg_rr":            round(avg_rr, 2),
            "xgb_agreement_pct": round(xgb_match  / total * 100, 1),
            "lstm_agreement_pct":round(lstm_match / total * 100, 1),
            "both_agree_pct":    round(both_agree / total * 100, 1),
            "by_pair":           by_pair,
            "by_timeframe":      by_tf,
            "signals_by_day":    by_day,
        }
    except Exception as e:
        raise HTTPException(500, str(e))


@app.get("/api/chart/{pair}/{timeframe}")
def get_chart_data(
    pair:      str,
    timeframe: str,
    bars:      int = Query(200, le=1000),
):
    """Velas OHLCV para el gráfico + señales superpuestas."""
    try:
        df = _query(
            """SELECT timestamp, open, high, low, close, volume
               FROM ohlcv_raw
               WHERE pair = :pair AND timeframe = :tf
               ORDER BY timestamp DESC
               LIMIT :bars""",
            {"pair": pair, "tf": timeframe, "bars": bars},
        ).sort_values("timestamp")

        if df.empty:
            raise HTTPException(404, f"Sin datos para {pair} {timeframe}")

        df["timestamp"] = pd.to_datetime(df["timestamp"]).astype("int64") // 10**9

        # Señales en ese rango
        min_ts = pd.to_datetime(df["timestamp"], unit="s").min().isoformat()
        signals = _query(
            """SELECT timestamp, direction, entry_price, tp_price, sl_price,
                      confidence, filter_reason
               FROM signals
               WHERE pair = :pair AND timeframe = :tf
               AND timestamp >= :min_ts
               AND direction != 0
               AND filter_reason IS NULL
               ORDER BY timestamp""",
            {"pair": pair, "tf": timeframe, "min_ts": min_ts},
        )
        signals["timestamp"] = (
            pd.to_datetime(signals["timestamp"]).astype("int64") // 10**9
        )

        return {
            "pair":      pair,
            "timeframe": timeframe,
            "candles":   df[["timestamp","open","high","low","close","volume"]].to_dict(orient="records"),
            "signals":   signals.to_dict(orient="records"),
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, str(e))


@app.get("/api/config")
def get_config():
    """Devuelve la configuración actual de filtros."""
    return _current_config.dict()


@app.post("/api/config")
def update_config(cfg: FilterConfig):
    """Actualiza la configuración de filtros en memoria."""
    global _current_config
    _current_config = cfg
    logger.info(f"Configuración actualizada: {cfg}")
    return {"ok": True, "config": cfg.dict()}


# ── Frontend SPA ──────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
def serve_dashboard():
    html_path = STATIC_DIR / "index.html"
    if not html_path.exists():
        return HTMLResponse("<h1>index.html no encontrado en src/dashboard/static/</h1>", 500)
    return HTMLResponse(html_path.read_text(encoding="utf-8"))


# Archivos estáticos (CSS, JS, imágenes)
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
