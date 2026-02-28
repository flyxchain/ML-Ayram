"""
src/dashboard/app.py
Dashboard web de ML-Ayram — FastAPI backend

Endpoints:
  GET  /                          → SPA (index.html)
  GET  /api/status                → estado del sistema
  GET  /api/signals/latest        → señales recientes
  GET  /api/signals/history       → historial paginado
  GET  /api/chart/{pair}/{tf}     → velas OHLCV + señales superpuestas
  GET  /api/metrics               → distribución y stats de señales
  GET  /api/performance           → rendimiento real de trades cerrados
  GET  /api/positions             → posiciones abiertas actualmente
  GET  /api/monitor               → estado de frescura de datos (monitor)
  GET  /api/config                → configuración actual de filtros
  POST /api/config                → actualizar filtros del generador

Arranque:
  uvicorn src.dashboard.app:app --host 0.0.0.0 --port 8000 --workers 1
"""

import os
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse
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


_current_config = FilterConfig()


# ── Helper BD ─────────────────────────────────────────────────────────────────

def _query(sql: str, params: dict = None) -> pd.DataFrame:
    with engine.connect() as conn:
        return pd.read_sql(text(sql), conn, params=params or {})


def _table_exists(table: str) -> bool:
    result = _query(
        "SELECT EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = :t)",
        {"t": table},
    )
    return bool(result.iloc[0, 0])


# ── /api/status ───────────────────────────────────────────────────────────────

@app.get("/api/status")
def get_status():
    try:
        total_signals = int(_query("SELECT COUNT(*) AS n FROM signals").iloc[0]["n"])
        total_bars    = int(_query("SELECT COUNT(*) AS n FROM ohlcv_raw").iloc[0]["n"])

        last_signal = _query(
            "SELECT timestamp FROM signals WHERE direction != 0 ORDER BY timestamp DESC LIMIT 1"
        )
        last_ts = str(last_signal.iloc[0]["timestamp"]) if not last_signal.empty else None

        open_positions = 0
        if _table_exists("positions_active"):
            open_positions = int(
                _query("SELECT COUNT(*) AS n FROM positions_active WHERE status = 'open'").iloc[0]["n"]
            )

        return {
            "status":          "online",
            "total_signals":   total_signals,
            "total_bars":      total_bars,
            "open_positions":  open_positions,
            "last_signal_at":  last_ts,
            "server_time":     datetime.now(timezone.utc).isoformat(),
        }
    except Exception as e:
        logger.error(f"/api/status error: {e}")
        return {"status": "error", "detail": str(e)}


# ── /api/signals/latest ───────────────────────────────────────────────────────

@app.get("/api/signals/latest")
def get_latest_signals(limit: int = Query(20, le=100)):
    try:
        df = _query(
            "SELECT * FROM signals ORDER BY timestamp DESC LIMIT :limit",
            {"limit": limit},
        )
        df["timestamp"] = df["timestamp"].astype(str)
        return df.to_dict(orient="records")
    except Exception as e:
        raise HTTPException(500, str(e))


# ── /api/signals/history ──────────────────────────────────────────────────────

@app.get("/api/signals/history")
def get_signal_history(
    pair:      Optional[str] = None,
    timeframe: Optional[str] = None,
    direction: Optional[int] = None,
    days:      int = Query(30, le=365),
    page:      int = Query(1, ge=1),
    page_size: int = Query(50, le=200),
):
    try:
        from_ts = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
        where   = ["timestamp >= :from_ts", "direction != 0"]
        params  = {"from_ts": from_ts, "offset": (page - 1) * page_size, "limit": page_size}

        if pair:
            where.append("pair = :pair");      params["pair"] = pair
        if timeframe:
            where.append("timeframe = :timeframe"); params["timeframe"] = timeframe
        if direction is not None:
            where.append("direction = :direction"); params["direction"] = direction

        where_sql = " AND ".join(where)
        count_params = {k: v for k, v in params.items() if k not in ("offset", "limit")}

        total = int(_query(f"SELECT COUNT(*) AS n FROM signals WHERE {where_sql}", count_params).iloc[0]["n"])
        df    = _query(
            f"SELECT * FROM signals WHERE {where_sql} ORDER BY timestamp DESC LIMIT :limit OFFSET :offset",
            params,
        )
        df["timestamp"] = df["timestamp"].astype(str)

        return {
            "total":   total,
            "page":    page,
            "pages":   max(1, -(-total // page_size)),
            "signals": df.to_dict(orient="records"),
        }
    except Exception as e:
        raise HTTPException(500, str(e))


# ── /api/chart/{pair}/{timeframe} ─────────────────────────────────────────────

@app.get("/api/chart/{pair}/{timeframe}")
def get_chart_data(
    pair:      str,
    timeframe: str,
    bars:      int = Query(200, le=1000),
):
    try:
        df = _query(
            """SELECT timestamp, open, high, low, close, volume
               FROM ohlcv_raw
               WHERE pair = :pair AND timeframe = :tf
               ORDER BY timestamp DESC LIMIT :bars""",
            {"pair": pair, "tf": timeframe, "bars": bars},
        ).sort_values("timestamp")

        if df.empty:
            raise HTTPException(404, f"Sin datos para {pair} {timeframe}")

        # Guardar rango de timestamps ANTES de convertir a epoch
        min_ts_str = str(df["timestamp"].min())
        df["timestamp"] = (
            pd.to_datetime(df["timestamp"]).astype("int64") // 10**9
        )

        # Señales en ese rango temporal
        signals = _query(
            """SELECT timestamp, direction, entry_price, tp_price, sl_price, confidence
               FROM signals
               WHERE pair = :pair AND timeframe = :tf
                 AND timestamp >= :min_ts
                 AND direction != 0
                 AND filter_reason IS NULL
               ORDER BY timestamp""",
            {"pair": pair, "tf": timeframe, "min_ts": min_ts_str},
        )
        if not signals.empty:
            signals["timestamp"] = (
                pd.to_datetime(signals["timestamp"]).astype("int64") // 10**9
            )

        return {
            "pair":      pair,
            "timeframe": timeframe,
            "candles":   df[["timestamp","open","high","low","close","volume"]].to_dict(orient="records"),
            "signals":   signals.to_dict(orient="records") if not signals.empty else [],
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, str(e))


# ── /api/metrics ──────────────────────────────────────────────────────────────

@app.get("/api/metrics")
def get_metrics(
    pair:      Optional[str] = None,
    timeframe: Optional[str] = None,
    days:      int = Query(90, le=365),
):
    """Distribución y stats descriptivas de señales generadas."""
    try:
        from_ts = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
        where   = ["timestamp >= :from_ts", "direction != 0", "filter_reason IS NULL"]
        params  = {"from_ts": from_ts}

        if pair:
            where.append("pair = :pair");           params["pair"] = pair
        if timeframe:
            where.append("timeframe = :timeframe"); params["timeframe"] = timeframe

        df = _query(
            f"""SELECT pair, timeframe, direction, confidence, rr_ratio,
                       tp_pips, sl_pips, xgb_direction, lstm_direction,
                       agreement, timestamp
                FROM signals WHERE {" AND ".join(where)} ORDER BY timestamp""",
            params,
        )

        if df.empty:
            return {"error": "Sin señales en el período seleccionado"}

        total  = len(df)
        longs  = int((df["direction"] ==  1).sum())
        shorts = int((df["direction"] == -1).sum())

        # Señales por día (últimos 30)
        df["date"] = pd.to_datetime(df["timestamp"]).dt.date.astype(str)
        by_day = df.groupby("date").size().reset_index(name="count").tail(30).to_dict(orient="records")

        return {
            "period_days":    days,
            "total_signals":  total,
            "long_signals":   longs,
            "short_signals":  shorts,
            "long_pct":       round(longs  / total * 100, 1),
            "short_pct":      round(shorts / total * 100, 1),
            "avg_confidence": round(float(df["confidence"].mean()) * 100, 1),
            "avg_rr":         round(float(df["rr_ratio"].mean()), 2),
            "agreement_pct":  round(float(df["agreement"].mean()) * 100, 1),
            "by_pair":        df.groupby("pair").size().to_dict(),
            "by_timeframe":   df.groupby("timeframe").size().to_dict(),
            "signals_by_day": by_day,
        }
    except Exception as e:
        raise HTTPException(500, str(e))


# ── /api/performance ──────────────────────────────────────────────────────────

@app.get("/api/performance")
def get_performance(
    pair:      Optional[str] = None,
    timeframe: Optional[str] = None,
    days:      int = Query(30, le=365),
):
    """Rendimiento real basado en trades cerrados (trades_history)."""
    try:
        if not _table_exists("trades_history"):
            return {"error": "Aún no hay trades cerrados", "trades": 0}

        from_ts = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
        where   = ["closed_at >= :from_ts"]
        params  = {"from_ts": from_ts}

        if pair:
            where.append("pair = :pair");           params["pair"] = pair
        if timeframe:
            where.append("timeframe = :timeframe"); params["timeframe"] = timeframe

        df = _query(
            f"SELECT * FROM trades_history WHERE {' AND '.join(where)} ORDER BY closed_at DESC",
            params,
        )

        if df.empty:
            return {"error": f"Sin trades cerrados en los últimos {days} días", "trades": 0}

        total  = len(df)
        wins   = df[df["result"] == "tp_hit"]
        losses = df[df["result"] == "sl_hit"]

        win_rate  = round(len(wins) / total * 100, 1)
        total_pnl = round(float(df["pnl"].sum()), 2)

        # Profit factor: suma ganancias / suma pérdidas absolutas
        gross_profit = float(wins["pnl"].sum())   if not wins.empty   else 0.0
        gross_loss   = abs(float(losses["pnl"].sum())) if not losses.empty else 0.0
        profit_factor = round(gross_profit / gross_loss, 2) if gross_loss > 0 else float("inf")

        # Equity curve diaria
        df["date"] = pd.to_datetime(df["closed_at"]).dt.date.astype(str)
        daily_pnl  = df.groupby("date")["pnl"].sum().reset_index(name="pnl")
        daily_pnl["cumulative"] = daily_pnl["pnl"].cumsum().round(2)
        equity_curve = daily_pnl.to_dict(orient="records")

        # Max drawdown
        cumulative   = daily_pnl["cumulative"].values
        running_max  = np.maximum.accumulate(cumulative)
        drawdown     = cumulative - running_max
        max_drawdown = round(float(drawdown.min()), 2)

        # Por par
        by_pair = (
            df.groupby("pair")
            .agg(trades=("pnl","count"), pnl=("pnl","sum"), win_rate=("result", lambda x: (x=="tp_hit").mean()*100))
            .round(2)
            .reset_index()
            .to_dict(orient="records")
        )

        # Últimos 10 trades
        recent = df.head(10).copy()
        recent["opened_at"] = recent["opened_at"].astype(str)
        recent["closed_at"] = recent["closed_at"].astype(str)

        return {
            "period_days":    days,
            "trades":         total,
            "wins":           len(wins),
            "losses":         len(losses),
            "win_rate":       win_rate,
            "total_pnl":      total_pnl,
            "avg_win":        round(float(wins["pnl"].mean()), 2)   if not wins.empty   else 0,
            "avg_loss":       round(float(losses["pnl"].mean()), 2) if not losses.empty else 0,
            "profit_factor":  profit_factor,
            "max_drawdown":   max_drawdown,
            "avg_duration_bars": round(float(df["duration_bars"].mean()), 1),
            "best_trade":     round(float(df["pnl"].max()), 2),
            "worst_trade":    round(float(df["pnl"].min()), 2),
            "by_pair":        by_pair,
            "equity_curve":   equity_curve,
            "recent_trades":  recent[["pair","timeframe","direction","entry_price",
                                      "exit_price","pnl","result","opened_at","closed_at",
                                      "duration_bars"]].to_dict(orient="records"),
        }
    except Exception as e:
        raise HTTPException(500, str(e))


# ── /api/positions ────────────────────────────────────────────────────────────

@app.get("/api/positions")
def get_open_positions():
    """Posiciones abiertas actualmente."""
    try:
        if not _table_exists("positions_active"):
            return []

        df = _query("SELECT * FROM positions_active WHERE status = 'open' ORDER BY opened_at DESC")
        if df.empty:
            return []

        df["opened_at"] = df["opened_at"].astype(str)

        # Calcular PnL flotante aproximado desde features_computed
        rows = df.to_dict(orient="records")
        for row in rows:
            try:
                last = _query(
                    """SELECT close FROM features_computed
                       WHERE pair = :pair AND timeframe = :tf
                       ORDER BY timestamp DESC LIMIT 1""",
                    {"pair": row["pair"], "tf": row["timeframe"]},
                )
                if not last.empty:
                    current = float(last.iloc[0]["close"])
                    pip_sizes = {"EURUSD":0.0001,"GBPUSD":0.0001,"USDJPY":0.01,"EURJPY":0.01,"XAUUSD":0.10}
                    pip_vals  = {"EURUSD":10.0,"GBPUSD":10.0,"USDJPY":9.1,"EURJPY":9.1,"XAUUSD":10.0}
                    ps  = pip_sizes.get(row["pair"], 0.0001)
                    pv  = pip_vals.get(row["pair"], 10.0)
                    direction = int(row["direction"])
                    pnl_pips  = (current - row["entry_price"]) / ps * direction
                    row["floating_pnl"] = round(pnl_pips * pv * row["lot_size"], 2)
                    row["current_price"] = round(current, 5)
            except Exception:
                row["floating_pnl"]  = None
                row["current_price"] = None

        return rows
    except Exception as e:
        raise HTTPException(500, str(e))


# ── /api/monitor ──────────────────────────────────────────────────────────────

@app.get("/api/monitor")
def get_monitor():
    """Estado de frescura de datos para monitoreo sin SSH."""
    try:
        now = datetime.now(timezone.utc)

        # Última vela por par/timeframe
        ohlcv = _query("""
            SELECT pair, timeframe,
                   MAX(timestamp) AS last_candle,
                   COUNT(*)       AS total_rows
            FROM ohlcv_raw
            GROUP BY pair, timeframe
            ORDER BY pair, timeframe
        """)

        # Últimos features por par/timeframe
        features = _query("""
            SELECT pair, timeframe,
                   MAX(timestamp) AS last_feature,
                   COUNT(*)       AS total_rows
            FROM features_computed
            GROUP BY pair, timeframe
            ORDER BY pair, timeframe
        """)

        # Umbrales de alerta (minutos sin actualizar)
        STALE_THRESHOLDS = {"M15": 60, "H1": 180, "H4": 480, "D1": 1800}

        def build_rows(df, ts_col):
            rows = []
            for _, r in df.iterrows():
                ts = pd.to_datetime(r[ts_col], utc=True) if pd.notna(r[ts_col]) else None
                age_min = (now - ts).total_seconds() / 60 if ts else None
                threshold = STALE_THRESHOLDS.get(r["timeframe"], 180)
                status = "ok"
                if age_min is None:
                    status = "no_data"
                elif age_min > threshold * 3:
                    status = "critical"
                elif age_min > threshold:
                    status = "stale"
                rows.append({
                    "pair":       r["pair"],
                    "timeframe":  r["timeframe"],
                    "last_update": str(ts) if ts else None,
                    "age_minutes": round(age_min, 1) if age_min else None,
                    "total_rows":  int(r["total_rows"]),
                    "status":      status,
                })
            return rows

        ohlcv_rows    = build_rows(ohlcv, "last_candle")
        features_rows = build_rows(features, "last_feature")

        # Resumen global
        ohlcv_ok    = sum(1 for r in ohlcv_rows if r["status"] == "ok")
        feat_ok     = sum(1 for r in features_rows if r["status"] == "ok")
        ohlcv_total = len(ohlcv_rows)
        feat_total  = len(features_rows)

        # Total velas y features en BD
        total_candles  = sum(r["total_rows"] for r in ohlcv_rows)
        total_features = sum(r["total_rows"] for r in features_rows)

        # Última actividad global
        last_candle_global  = ohlcv["last_candle"].max()  if not ohlcv.empty else None
        last_feature_global = features["last_feature"].max() if not features.empty else None

        return {
            "server_time":     now.isoformat(),
            "summary": {
                "collector_health":  f"{ohlcv_ok}/{ohlcv_total} OK",
                "features_health":   f"{feat_ok}/{feat_total} OK",
                "total_candles":     total_candles,
                "total_features":    total_features,
                "last_candle":       str(last_candle_global) if last_candle_global else None,
                "last_feature":      str(last_feature_global) if last_feature_global else None,
            },
            "ohlcv":    ohlcv_rows,
            "features": features_rows,
        }
    except Exception as e:
        logger.error(f"/api/monitor error: {e}")
        raise HTTPException(500, str(e))


# ── /api/config ───────────────────────────────────────────────────────────────

@app.get("/api/config")
def get_config():
    return _current_config.dict()


@app.post("/api/config")
def update_config(cfg: FilterConfig):
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


if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
