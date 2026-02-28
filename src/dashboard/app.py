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
  GET  /api/health                → salud de modelos (model_health)
  GET  /api/anomalies             → alertas anomalías (anomaly_detector)
  GET  /api/summary               → resumen mensual IA
  GET  /api/services              → estado de servicios systemd
  GET  /api/config                → configuración actual de filtros
  POST /api/config                → actualizar filtros del generador

Arranque:
  uvicorn src.dashboard.app:app --host 0.0.0.0 --port 8000 --workers 1
"""

import json
import math
import os
import glob
import subprocess
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional, List

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from sqlalchemy import create_engine, text
from loguru import logger

from src.utils.pipeline_logger import (
    ensure_table as ensure_pipeline_table,
    is_forex_market_open,
)


class NaNSafeEncoder(json.JSONEncoder):
    """JSON encoder que convierte NaN/Inf a null en vez de fallar."""
    def default(self, obj):
        if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
            return None
        return super().default(obj)

    def iterencode(self, o, _one_shot=False):
        return super().iterencode(self._sanitize(o), _one_shot)

    def _sanitize(self, obj):
        if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
            return None
        if isinstance(obj, dict):
            return {k: self._sanitize(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [self._sanitize(v) for v in obj]
        return obj

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")
STATIC_DIR   = Path(__file__).parent / "static"
STATIC_DIR.mkdir(exist_ok=True)
RESULTS_DIR  = Path(__file__).resolve().parent.parent.parent / "results"

app    = FastAPI(title="ML-Ayram Dashboard", version="2.0.0")
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


class BotConfig(BaseModel):
    active_pairs:               List[str] = ["EURUSD","GBPUSD","USDJPY","EURJPY","XAUUSD"]
    active_timeframes:          List[str] = ["M15","H1","H4"]
    risk_per_trade_pct:         float = 1.0
    max_lot_size:               float = 0.10
    max_simultaneous_positions: int   = 3
    max_daily_loss_eur:         float = 50.0
    weekend_pause:              bool  = True
    training_day:               str   = "sunday"
    training_hour_utc:          int   = 2
    notes:                      str   = ""


BOT_CONFIG_PATH = Path(__file__).resolve().parent.parent.parent / "config" / "bot_config.json"


def _load_bot_config() -> BotConfig:
    if BOT_CONFIG_PATH.exists():
        try:
            with open(BOT_CONFIG_PATH, "r", encoding="utf-8") as f:
                return BotConfig(**json.load(f))
        except Exception as e:
            logger.warning(f"Error leyendo bot_config.json: {e}")
    return BotConfig()


def _save_bot_config(cfg: BotConfig):
    BOT_CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(BOT_CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(cfg.dict(), f, indent=4, ensure_ascii=False)
    logger.info(f"Bot config guardado en {BOT_CONFIG_PATH}")


# ── Constantes de precisión FOREX ──────────────────────────────────────────────────

PAIR_PRECISION = {
    "EURUSD": 5, "GBPUSD": 5,   # Pares mayores: 5 decimales (pip = 0.0001)
    "USDJPY": 3, "EURJPY": 3,   # Pares JPY: 3 decimales (pip = 0.01)
    "XAUUSD": 2,                 # Oro: 2 decimales
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _query(sql: str, params: dict = None) -> pd.DataFrame:
    with engine.connect() as conn:
        return pd.read_sql(text(sql), conn, params=params or {})


def _safe_json(data):
    """Devuelve JSONResponse limpiando NaN/Inf → null."""
    clean = json.loads(json.dumps(data, cls=NaNSafeEncoder, default=str))
    return JSONResponse(content=clean)


def _table_exists(table: str) -> bool:
    result = _query(
        "SELECT EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = :t)",
        {"t": table},
    )
    return bool(result.iloc[0, 0])


def _latest_result_file(prefix: str) -> Optional[dict]:
    """Lee el JSON más reciente de results/ con el prefijo dado."""
    pattern = str(RESULTS_DIR / f"{prefix}*.json")
    files = sorted(glob.glob(pattern), reverse=True)
    if not files:
        return None
    try:
        with open(files[0], "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Error leyendo {files[0]}: {e}")
        return None


def _latest_result_meta(prefix: str) -> dict:
    """Devuelve metadatos del último fichero result (nombre, fecha mod)."""
    pattern = str(RESULTS_DIR / f"{prefix}*.json")
    files = sorted(glob.glob(pattern), reverse=True)
    if not files:
        return {"file": None, "modified": None}
    p = Path(files[0])
    return {
        "file": p.name,
        "modified": datetime.fromtimestamp(p.stat().st_mtime, tz=timezone.utc).isoformat(),
    }


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
                _query("SELECT COUNT(*) AS n FROM positions_active").iloc[0]["n"]
            )

        # Estado rápido de anomalías
        anomaly_meta = _latest_result_meta("anomalies_")
        health_meta  = _latest_result_meta("health_")

        return {
            "status":           "online",
            "total_signals":    total_signals,
            "total_bars":       total_bars,
            "open_positions":   open_positions,
            "last_signal_at":   last_ts,
            "last_anomaly_check": anomaly_meta["modified"],
            "last_health_check":  health_meta["modified"],
            "server_time":      datetime.now(timezone.utc).isoformat(),
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
        return _safe_json(df.to_dict(orient="records"))
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

        return _safe_json({
            "total":   total,
            "page":    page,
            "pages":   max(1, -(-total // page_size)),
            "signals": df.to_dict(orient="records"),
        })
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

        df = df.dropna(subset=["open", "high", "low", "close"])
        if df.empty:
            raise HTTPException(404, f"Sin datos válidos para {pair} {timeframe}")

        min_ts_str = str(df["timestamp"].min())
        df["timestamp"] = (
            pd.to_datetime(df["timestamp"]).astype("int64") // 10**9
        )

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

        # Precisión decimal según par
        precision = PAIR_PRECISION.get(pair, 5)

        return _safe_json({
            "pair":      pair,
            "timeframe": timeframe,
            "precision": precision,
            "candles":   df[["timestamp","open","high","low","close","volume"]].to_dict(orient="records"),
            "signals":   signals.to_dict(orient="records") if not signals.empty else [],
        })
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

        df["date"] = pd.to_datetime(df["timestamp"]).dt.date.astype(str)
        by_day = df.groupby("date").size().reset_index(name="count").tail(30).to_dict(orient="records")

        return _safe_json({
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
        })
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

        gross_profit = float(wins["pnl"].sum())   if not wins.empty   else 0.0
        gross_loss   = abs(float(losses["pnl"].sum())) if not losses.empty else 0.0
        profit_factor = round(gross_profit / gross_loss, 2) if gross_loss > 0 else float("inf")

        df["date"] = pd.to_datetime(df["closed_at"]).dt.date.astype(str)
        daily_pnl  = df.groupby("date")["pnl"].sum().reset_index(name="pnl")
        daily_pnl["cumulative"] = daily_pnl["pnl"].cumsum().round(2)
        equity_curve = daily_pnl.to_dict(orient="records")

        cumulative   = daily_pnl["cumulative"].values
        running_max  = np.maximum.accumulate(cumulative)
        drawdown     = cumulative - running_max
        max_drawdown = round(float(drawdown.min()), 2)

        by_pair = (
            df.groupby("pair")
            .agg(trades=("pnl","count"), pnl=("pnl","sum"), win_rate=("result", lambda x: (x=="tp_hit").mean()*100))
            .round(2)
            .reset_index()
            .to_dict(orient="records")
        )

        recent = df.head(10).copy()
        recent["opened_at"] = recent["opened_at"].astype(str)
        recent["closed_at"] = recent["closed_at"].astype(str)

        return _safe_json({
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
        })
    except Exception as e:
        raise HTTPException(500, str(e))


# ── /api/positions ────────────────────────────────────────────────────────────

@app.get("/api/positions")
def get_open_positions():
    """Posiciones abiertas actualmente."""
    try:
        if not _table_exists("positions_active"):
            return []

        df = _query("SELECT * FROM positions_active ORDER BY opened_at DESC")
        if df.empty:
            return []

        df["opened_at"] = df["opened_at"].astype(str)

        pip_sizes = {"EURUSD":0.0001,"GBPUSD":0.0001,"USDJPY":0.01,"EURJPY":0.01,"XAUUSD":0.10}
        pip_vals  = {"EURUSD":10.0,"GBPUSD":10.0,"USDJPY":9.1,"EURJPY":9.1,"XAUUSD":10.0}

        rows = df.to_dict(orient="records")
        for row in rows:
            row["tp_price"] = row.get("tp1_price")
            row["sl_price"] = row.get("current_sl")
            row["timeframe"] = "—"
            row["risk_amount"] = None

            try:
                last = _query(
                    """SELECT close FROM ohlcv_raw
                       WHERE pair = :pair AND timeframe = 'H1'
                       ORDER BY timestamp DESC LIMIT 1""",
                    {"pair": row["pair"]},
                )
                if not last.empty:
                    current = float(last.iloc[0]["close"])
                    ps  = pip_sizes.get(row["pair"], 0.0001)
                    pv  = pip_vals.get(row["pair"], 10.0)
                    direction = int(row["direction"])
                    pnl_pips  = (current - float(row["entry_price"])) / ps * direction
                    row["floating_pnl"] = round(pnl_pips * pv * float(row["lot_size"]), 2)
                    row["current_price"] = round(current, 5)
                else:
                    row["floating_pnl"]  = None
                    row["current_price"] = None
            except Exception:
                row["floating_pnl"]  = None
                row["current_price"] = None

        return _safe_json(rows)
    except Exception as e:
        raise HTTPException(500, str(e))


# ── /api/monitor ──────────────────────────────────────────────────────────────

@app.get("/api/monitor")
def get_monitor():
    """Estado de frescura de datos para monitoreo sin SSH."""
    try:
        now = datetime.now(timezone.utc)

        ohlcv = _query("""
            SELECT pair, timeframe,
                   MAX(timestamp) AS last_candle,
                   COUNT(*)       AS total_rows
            FROM ohlcv_raw
            GROUP BY pair, timeframe
            ORDER BY pair, timeframe
        """)

        features = _query("""
            SELECT pair, timeframe,
                   MAX(timestamp) AS last_feature,
                   COUNT(*)       AS total_rows
            FROM features_computed
            GROUP BY pair, timeframe
            ORDER BY pair, timeframe
        """)

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

        ohlcv_ok    = sum(1 for r in ohlcv_rows if r["status"] == "ok")
        feat_ok     = sum(1 for r in features_rows if r["status"] == "ok")
        ohlcv_total = len(ohlcv_rows)
        feat_total  = len(features_rows)

        total_candles  = sum(r["total_rows"] for r in ohlcv_rows)
        total_features = sum(r["total_rows"] for r in features_rows)

        last_candle_global  = ohlcv["last_candle"].max()  if not ohlcv.empty else None
        last_feature_global = features["last_feature"].max() if not features.empty else None

        return _safe_json({
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
        })
    except Exception as e:
        logger.error(f"/api/monitor error: {e}")
        raise HTTPException(500, str(e))


# ── /api/health — Salud de modelos ────────────────────────────────────────────

@app.get("/api/health")
def get_model_health():
    """Lee el último informe de salud de modelos generado por model_health.py."""
    try:
        data = _latest_result_file("health_")
        if data is None:
            # Intentar leer de la tabla model_performance si existe
            if _table_exists("model_performance"):
                df = _query("""
                    SELECT pair, timeframe, win_rate, profit_factor, total_pnl,
                           max_drawdown_pct, expectancy, total_trades, status,
                           checked_at
                    FROM model_performance
                    ORDER BY checked_at DESC
                    LIMIT 20
                """)
                if not df.empty:
                    df["checked_at"] = df["checked_at"].astype(str)
                    return _safe_json({
                        "source": "database",
                        "models": df.to_dict(orient="records"),
                    })
            return {"error": "Sin datos de salud. Ejecuta: python -m src.monitoring.model_health"}

        meta = _latest_result_meta("health_")
        data["_meta"] = meta
        return _safe_json(data)
    except Exception as e:
        raise HTTPException(500, str(e))


# ── /api/anomalies — Alertas de anomalías ─────────────────────────────────────

@app.get("/api/anomalies")
def get_anomalies():
    """Lee el último informe de anomalías generado por anomaly_detector.py."""
    try:
        data = _latest_result_file("anomalies_")
        if data is None:
            return {"error": "Sin datos de anomalías. Ejecuta: python -m src.monitoring.anomaly_detector"}

        meta = _latest_result_meta("anomalies_")
        data["_meta"] = meta
        return _safe_json(data)
    except Exception as e:
        raise HTTPException(500, str(e))


# ── /api/summary — Resumen mensual IA ─────────────────────────────────────────

@app.get("/api/summary")
def get_monthly_summary():
    """Lee el último resumen mensual generado por monthly_summary.py."""
    try:
        data = _latest_result_file("summary_")
        if data is None:
            return {"error": "Sin resumen mensual. Ejecuta: python -m src.analysis.monthly_summary"}

        meta = _latest_result_meta("summary_")

        # También buscar el prompt IA generado
        prompt_pattern = str(RESULTS_DIR / "ai_prompt_*.md")
        prompt_files = sorted(glob.glob(prompt_pattern), reverse=True)
        prompt_text = None
        if prompt_files:
            try:
                with open(prompt_files[0], "r", encoding="utf-8") as f:
                    prompt_text = f.read()
            except Exception:
                pass

        return _safe_json({
            "summary": data,
            "ai_prompt": prompt_text,
            "_meta": meta,
        })
    except Exception as e:
        raise HTTPException(500, str(e))


# ── /api/pipeline — Estado del pipeline (barras de progreso) ──────────────────

@app.get("/api/pipeline")
def get_pipeline(hours: int = Query(24, le=168)):
    """Devuelve historial de ejecuciones del pipeline para visualización."""
    try:
        if not _table_exists("pipeline_runs"):
            ensure_pipeline_table()
            return _safe_json({"server_time": datetime.now(timezone.utc).isoformat(),
                               "market_open": is_forex_market_open(),
                               "runs": [], "errors": [], "stats": {}})

        runs_df = _query(
            f"SELECT id, task, started_at, finished_at, status, error_message, "
            f"       rows_processed, duration_seconds "
            f"FROM pipeline_runs "
            f"WHERE started_at >= NOW() - INTERVAL '{int(hours)} hours' "
            f"ORDER BY started_at DESC LIMIT 1000"
        )

        errors_df = _query(
            "SELECT id, task, started_at, error_message, duration_seconds "
            "FROM pipeline_runs WHERE status = 'error' "
            "ORDER BY started_at DESC LIMIT 50"
        )

        runs = []
        if not runs_df.empty:
            runs_df["started_at"]  = runs_df["started_at"].astype(str)
            runs_df["finished_at"] = runs_df["finished_at"].astype(str)
            runs = runs_df.to_dict(orient="records")

        errors = []
        if not errors_df.empty:
            errors_df["started_at"] = errors_df["started_at"].astype(str)
            errors = errors_df.to_dict(orient="records")

        # Stats por tarea
        stats = {}
        if not runs_df.empty:
            for task, grp in runs_df.groupby("task"):
                total    = len(grp)
                ok       = int((grp["status"] == "success").sum())
                err      = int((grp["status"] == "error").sum())
                skipped  = int((grp["status"] == "skipped").sum())
                avg_dur  = float(grp["duration_seconds"].dropna().mean()) if not grp["duration_seconds"].dropna().empty else 0
                stats[task] = {
                    "total": total, "success": ok, "errors": err,
                    "skipped": skipped, "avg_duration": round(avg_dur, 1),
                }

        return _safe_json({
            "server_time": datetime.now(timezone.utc).isoformat(),
            "market_open": is_forex_market_open(),
            "period_hours": hours,
            "runs":   runs,
            "errors": errors,
            "stats":  stats,
        })
    except Exception as e:
        logger.error(f"/api/pipeline error: {e}")
        raise HTTPException(500, str(e))


# ── /api/bot — Configuración del bot ──────────────────────────────────────────

@app.get("/api/bot")
def get_bot_config():
    return _load_bot_config().dict()


@app.post("/api/bot")
def update_bot_config(cfg: BotConfig):
    _save_bot_config(cfg)
    return {"ok": True, "config": cfg.dict()}


# ── /api/services — Estado de servicios systemd ──────────────────────────────

@app.get("/api/services")
def get_services_status():
    """Consulta el estado de los servicios y timers systemd de ML-Ayram."""
    services = [
        {"name": "ayram-dashboard",    "type": "service", "desc": "Dashboard FastAPI"},
        {"name": "ayram-signals",      "type": "service", "desc": "Generador de señales (continuo)"},
        {"name": "ayram-collector",    "type": "timer",   "desc": "Descarga velas EODHD (cada 15min)"},
        {"name": "ayram-features",     "type": "timer",   "desc": "Recálculo features (cada 3h)"},
        {"name": "ayram-positions",    "type": "timer",   "desc": "Gestión posiciones (cada 5min)"},
        {"name": "ayram-anomaly",      "type": "timer",   "desc": "Detección anomalías (cada 6h)"},
        {"name": "ayram-train",        "type": "timer",   "desc": "Reentrenamiento semanal (dom 02:00)"},
        {"name": "ayram-walkforward",  "type": "timer",   "desc": "Walk-forward mensual (1er dom 04:00)"},
    ]

    results = []
    for svc in services:
        unit = f"{svc['name']}.{svc['type']}"
        info = {
            "name":   svc["name"],
            "unit":   unit,
            "type":   svc["type"],
            "desc":   svc["desc"],
            "active": "unknown",
            "enabled": "unknown",
            "since":  None,
            "next_run": None,
        }

        try:
            # is-active
            r = subprocess.run(
                ["systemctl", "is-active", unit],
                capture_output=True, text=True, timeout=5,
            )
            info["active"] = r.stdout.strip()

            # is-enabled
            r = subprocess.run(
                ["systemctl", "is-enabled", unit],
                capture_output=True, text=True, timeout=5,
            )
            info["enabled"] = r.stdout.strip()

            # Para timers: cuándo fue la última ejecución y la próxima
            if svc["type"] == "timer":
                r = subprocess.run(
                    ["systemctl", "show", unit,
                     "--property=LastTriggerUSec,NextElapseUSecRealtime"],
                    capture_output=True, text=True, timeout=5,
                )
                for line in r.stdout.strip().split("\n"):
                    if line.startswith("LastTriggerUSec="):
                        val = line.split("=", 1)[1]
                        if val and val != "n/a":
                            info["since"] = val
                    elif line.startswith("NextElapseUSecRealtime="):
                        val = line.split("=", 1)[1]
                        if val and val != "n/a":
                            info["next_run"] = val

            # Para services: desde cuándo está activo
            if svc["type"] == "service":
                r = subprocess.run(
                    ["systemctl", "show", unit, "--property=ActiveEnterTimestamp"],
                    capture_output=True, text=True, timeout=5,
                )
                for line in r.stdout.strip().split("\n"):
                    if line.startswith("ActiveEnterTimestamp="):
                        val = line.split("=", 1)[1]
                        if val and val != "n/a":
                            info["since"] = val

        except (subprocess.TimeoutExpired, FileNotFoundError, Exception) as e:
            info["active"] = "error"
            info["error"]  = str(e)

        results.append(info)

    return _safe_json({
        "server_time": datetime.now(timezone.utc).isoformat(),
        "services":    results,
    })


# ── /api/correlations ─────────────────────────────────────────────────────────

@app.get("/api/correlations")
def get_correlations(
    timeframe: str = Query("H1"),
    days: int = Query(90, le=365),
):
    """
    Matriz de correlaciones entre pares basada en retornos logarítmicos.
    Incluye correlaciones rolling de 20 periodos para ver tendencia reciente.
    """
    try:
        from_ts = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
        df = _query(
            "SELECT pair, timestamp, close FROM ohlcv_raw "
            "WHERE timeframe = :tf AND timestamp >= :from_ts "
            "ORDER BY timestamp",
            {"tf": timeframe, "from_ts": from_ts},
        )
        if df.empty:
            return {"error": "Sin datos suficientes"}

        pivot = df.pivot_table(index="timestamp", columns="pair", values="close")
        pivot = pivot.dropna()

        if len(pivot) < 30:
            return {"error": "Datos insuficientes para calcular correlaciones"}

        # Retornos logarítmicos
        log_returns = np.log(pivot / pivot.shift(1)).dropna()

        # Matriz de correlación completa
        corr = log_returns.corr().round(4)
        pairs_list = list(corr.columns)
        matrix = {}
        for p in pairs_list:
            matrix[p] = {p2: float(corr.loc[p, p2]) for p2 in pairs_list}

        # Correlaciones rolling recientes (últimos 20 periodos)
        rolling_window = min(20, len(log_returns) - 1)
        recent_corr = log_returns.tail(rolling_window).corr().round(4)
        recent_matrix = {}
        for p in pairs_list:
            recent_matrix[p] = {p2: float(recent_corr.loc[p, p2]) for p2 in pairs_list}

        # Top correlaciones (positivas y negativas, excluyendo autocorrelación)
        top_pairs = []
        seen = set()
        for i, p1 in enumerate(pairs_list):
            for p2 in pairs_list[i+1:]:
                key = tuple(sorted([p1, p2]))
                if key not in seen:
                    seen.add(key)
                    val = float(corr.loc[p1, p2])
                    recent_val = float(recent_corr.loc[p1, p2])
                    top_pairs.append({
                        "pair1": p1, "pair2": p2,
                        "correlation": val,
                        "recent": recent_val,
                        "change": round(recent_val - val, 4),
                    })
        top_pairs.sort(key=lambda x: abs(x["correlation"]), reverse=True)

        return _safe_json({
            "timeframe":     timeframe,
            "period_days":   days,
            "data_points":   len(log_returns),
            "pairs":         pairs_list,
            "matrix":        matrix,
            "recent_matrix": recent_matrix,
            "top_correlations": top_pairs,
        })
    except Exception as e:
        logger.error(f"/api/correlations error: {e}")
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
