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

# ── OpenAPI tags ──────────────────────────────────────────────────────────────
TAGS_META = [
    {"name": "Status",        "description": "Estado general del sistema, servicios y pipeline."},
    {"name": "Signals",       "description": "Señales de trading generadas por el ensemble ML."},
    {"name": "Charts",        "description": "Datos OHLCV y señales para gráficos interactivos."},
    {"name": "Performance",   "description": "Métricas de rendimiento, posiciones abiertas y backtest."},
    {"name": "Monitoring",    "description": "Frescura de datos, salud de modelos y anomalías."},
    {"name": "Training",      "description": "Estado en tiempo real del entrenamiento ML."},
    {"name": "Notifications", "description": "Historial de notificaciones Telegram y reglas de alerta."},
    {"name": "Config",        "description": "Configuración de filtros del generador y bot."},
    {"name": "Docs",          "description": "Documentación del proyecto en formato Markdown."},
    {"name": "Models",        "description": "Comparación side-by-side de modelos XGBoost vs LSTM."},
]

app = FastAPI(
    title="ML-Ayram Trading API",
    version="2.1.0",
    description=(
        "API del sistema de trading algorítmico ML-Ayram.\n\n"
        "Combina modelos XGBoost + LSTM con filtros técnicos para generar "
        "señales de forex (EURUSD, GBPUSD, USDJPY, EURJPY, XAUUSD) "
        "en timeframes M15, H1 y H4.\n\n"
        "**Componentes principales:**\n"
        "- 📡 Generación de señales con ensemble ML\n"
        "- 📊 Backtest sobre señales históricas\n"
        "- 🚨 Monitoreo y detección de anomalías\n"
        "- 🔔 Notificaciones Telegram con reglas configurables\n"
        "- 🏋️ Entrenamiento automático semanal"
    ),
    openapi_tags=TAGS_META,
    docs_url="/docs",
    redoc_url="/redoc",
)
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

@app.get("/api/status", tags=["Status"], summary="Estado general del sistema",
         description="Resumen de señales (24h/7d), último modelo entrenado, última señal y estado del mercado forex.")
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

@app.get("/api/signals/latest", tags=["Signals"], summary="Señales más recientes",
         description="Devuelve las últimas N señales ordenadas por timestamp descendente, incluyendo filtradas y válidas.")
def get_latest_signals(limit: int = Query(20, le=100, description="Máximo de señales a devolver")):
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

@app.get("/api/signals/history", tags=["Signals"], summary="Historial de señales paginado",
         description="Historial completo de señales con filtros por par, timeframe, dirección y período. Soporta paginación con offset/limit.")
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

@app.get("/api/chart/{pair}/{timeframe}", tags=["Charts"], summary="Datos para gráfico de velas",
         description="Devuelve velas OHLCV y señales superpuestas para un par/timeframe específico. Incluye marcadores de TP/SL.")
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

@app.get("/api/metrics", tags=["Performance"], summary="Métricas y distribución de señales",
         description="Distribución de señales por par, timeframe, dirección y sesión. Incluye confianza media y ADX medio.")
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

@app.get("/api/performance", tags=["Performance"], summary="Rendimiento de trades cerrados",
         description="Métricas reales de PnL, win rate, profit factor y equity curve basadas en posiciones cerradas.")
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

@app.get("/api/positions", tags=["Performance"], summary="Posiciones abiertas",
         description="Lista de posiciones actualmente abiertas con PnL no realizado, par, dirección y duración.")
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

@app.get("/api/monitor", tags=["Monitoring"], summary="Monitor de frescura de datos",
         description="Verifica la antigüedad de datos OHLCV, features calculados, señales generadas y modelos por par/timeframe.")
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

@app.get("/api/health", tags=["Monitoring"], summary="Salud de modelos ML",
         description="Lee el último informe JSON de model_health_check: estado de cada modelo XGBoost/LSTM, F1 scores y antigüedad.")
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

@app.get("/api/anomalies", tags=["Monitoring"], summary="Alertas de anomalías",
         description="Lee el último informe del anomaly_detector: alertas de sequía de señales, drawdown, datos obsoletos, etc.")
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

@app.get("/api/summary", tags=["Monitoring"], summary="Resumen mensual IA",
         description="Lee el último resumen mensual generado por el análisis IA (Claude/ChatGPT) con recomendaciones estratégicas.")
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

@app.get("/api/pipeline", tags=["Status"], summary="Logs del pipeline",
         description="Últimos eventos del pipeline de datos: recolección, features, labels, señales. Filtrable por horas.")
def get_pipeline(hours: int = Query(24, le=168, description="Horas hacia atrás a consultar")):
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

@app.get("/api/bot", tags=["Config"], summary="Configuración del bot",
         description="Devuelve la configuración actual del bot de trading: pares activos, timeframes y estado de auto-trading.")
def get_bot_config():
    return _load_bot_config().dict()


@app.post("/api/bot", tags=["Config"], summary="Actualizar configuración del bot",
          description="Modifica los pares activos, timeframes y estado de auto-trading del bot.")
def update_bot_config(cfg: BotConfig):
    _save_bot_config(cfg)
    return {"ok": True, "config": cfg.dict()}


# ── /api/services — Estado de servicios systemd ──────────────────────────────

@app.get("/api/services", tags=["Status"], summary="Estado de servicios systemd",
         description="Consulta el estado de todos los servicios y timers systemd de ML-Ayram: activo, último run, próximo run.")
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

@app.get("/api/correlations", tags=["Performance"], summary="Correlaciones entre pares",
         description="Matriz de correlación de log-returns entre los pares monitoreados para un timeframe y período dados.")
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


# ── /api/train/status — Estado del entrenamiento ─────────────────────────────

MODELS_DIR = Path(__file__).resolve().parent.parent.parent / "models" / "saved"


@app.get("/api/train/status", tags=["Training"], summary="Estado del entrenamiento en curso",
         description="Lee logs del watchdog/nohup y journalctl para extraer progreso, modelo actual, épocas, F1 scores y archivos generados.")
def get_train_status(lines: int = Query(150, le=500, description="Líneas de log a analizar")):
    """
    Estado en tiempo real del entrenamiento.
    Lee logs del watchdog nohup y systemd journalctl.
    """
    try:
        LOGS_DIR = Path(__file__).resolve().parent.parent.parent / "logs"

        # 1. Estado systemd
        svc_status = "unknown"
        svc_since = None
        try:
            r = subprocess.run(
                ["systemctl", "is-active", "ayram-train.service"],
                capture_output=True, text=True, timeout=5,
            )
            svc_status = r.stdout.strip()
            if svc_status == "active":
                r2 = subprocess.run(
                    ["systemctl", "show", "ayram-train.service",
                     "--property=ActiveEnterTimestamp"],
                    capture_output=True, text=True, timeout=5,
                )
                for line in r2.stdout.strip().split("\n"):
                    if line.startswith("ActiveEnterTimestamp="):
                        val = line.split("=", 1)[1]
                        if val and val != "n/a":
                            svc_since = val
        except Exception:
            pass

        # 2. Leer logs: primero watchdog/train nohup, luego journalctl como fallback
        log_lines = []
        log_source = "none"

        # Buscar log de watchdog más reciente
        watchdog_logs = sorted(LOGS_DIR.glob("watchdog_*.log"), key=lambda f: f.stat().st_mtime, reverse=True) if LOGS_DIR.exists() else []
        train_logs    = sorted(LOGS_DIR.glob("train_*.log"),    key=lambda f: f.stat().st_mtime, reverse=True) if LOGS_DIR.exists() else []

        # Determinar si el watchdog está activo (modificado en últimos 30 min)
        active_watchdog = None
        for wlog in watchdog_logs:
            age_min = (datetime.now(timezone.utc) - datetime.fromtimestamp(wlog.stat().st_mtime, tz=timezone.utc)).total_seconds() / 60
            if age_min < 30:  # modificado en últimos 30 min = activo
                active_watchdog = wlog
                break

        if active_watchdog or watchdog_logs:
            # Leer el watchdog log más reciente
            log_file = active_watchdog or watchdog_logs[0]
            try:
                all_lines = log_file.read_text(encoding="utf-8", errors="replace").splitlines()
                log_lines = all_lines[-lines:]
                log_source = f"watchdog: {log_file.name}"
                if svc_status == "unknown" or svc_status not in ("active",):
                    age_min = (datetime.now(timezone.utc) - datetime.fromtimestamp(log_file.stat().st_mtime, tz=timezone.utc)).total_seconds() / 60
                    svc_status = "active" if age_min < 10 else "inactive"
            except Exception:
                pass
        elif train_logs:
            # Leer el train log más reciente
            try:
                all_lines = train_logs[0].read_text(encoding="utf-8", errors="replace").splitlines()
                log_lines = all_lines[-lines:]
                log_source = f"train: {train_logs[0].name}"
            except Exception:
                pass

        # Fallback: journalctl
        if not log_lines:
            try:
                r = subprocess.run(
                    ["journalctl", "-u", "ayram-train.service",
                     "--no-pager", "-n", str(lines), "--output=short-iso"],
                    capture_output=True, text=True, timeout=10,
                )
                log_lines = r.stdout.strip().split("\n") if r.stdout.strip() else []
                log_source = "journalctl"
            except Exception:
                pass

        # Timestamps de logs disponibles para el frontend
        available_logs = []
        for f in (watchdog_logs[:5] + train_logs[:5]):
            available_logs.append({
                "name": f.name,
                "modified": datetime.fromtimestamp(f.stat().st_mtime, tz=timezone.utc).isoformat(),
                "size_kb": round(f.stat().st_size / 1024, 1),
            })

        # 3. Parsear logs para extraer progreso
        import re

        pairs = ["EURUSD", "GBPUSD", "USDJPY", "EURJPY", "XAUUSD"]
        timeframes = ["M15", "H1", "H4"]
        total_models = len(pairs) * len(timeframes) * 2  # XGB + LSTM = 30

        completed_models = []   # {type, pair, tf, f1, timestamp}
        current_model = None    # {type, pair, tf}
        current_epoch = None    # {epoch, total, val_f1}
        last_cv_f1 = None
        training_started = None
        errors = []

        for line in log_lines:
            # Inicio de entrenamiento
            m = re.search(r"ML-Ayram.*inicio de entrenamiento", line)
            if m:
                training_started = line[:25].strip()  # timestamp aprox
                completed_models = []
                current_model = None
                current_epoch = None

            # XGB iniciado
            m = re.search(r"\[XGB\]\s+(\w+)\s+(\w+)\s+\((\d+) filas", line)
            if m:
                current_model = {"type": "XGBoost", "pair": m.group(1),
                                 "tf": m.group(2), "rows": int(m.group(3))}
                current_epoch = None

            # LSTM iniciado
            m = re.search(r"\[LSTM\]\s+(\w+)\s+(\w+)\s+\((\d+) filas", line)
            if m:
                current_model = {"type": "LSTM", "pair": m.group(1),
                                 "tf": m.group(2), "rows": int(m.group(3))}
                current_epoch = None

            # CV F1 (XGBoost completado)
            m = re.search(r"CV F1 promedio:\s+([\d.]+)", line)
            if m:
                last_cv_f1 = float(m.group(1))

            # Modelo XGB guardado
            m = re.search(r"Modelo guardado:.*xgb_(\w+)_(\w+)_", line)
            if m:
                completed_models.append({
                    "type": "XGBoost", "pair": m.group(1), "tf": m.group(2),
                    "f1": last_cv_f1, "timestamp": line[:25].strip(),
                })
                current_model = None
                last_cv_f1 = None

            # Modelo LSTM guardado
            m = re.search(r"Modelo guardado:.*lstm_(\w+)_(\w+)_", line)
            if m:
                completed_models.append({
                    "type": "LSTM", "pair": m.group(1), "tf": m.group(2),
                    "f1": last_cv_f1, "timestamp": line[:25].strip(),
                })
                current_model = None
                last_cv_f1 = None

            # Época LSTM
            m = re.search(r"Epoch\s+(\d+)/(\d+).*val_f1[=:]\s*([\d.]+)", line)
            if m:
                current_epoch = {
                    "epoch": int(m.group(1)),
                    "total": int(m.group(2)),
                    "val_f1": float(m.group(3)),
                }

            # Fold XGB
            m = re.search(r"Fold\s+(\d+)/(\d+).*F1.*?([\d.]+)", line)
            if m:
                current_epoch = {
                    "fold": int(m.group(1)),
                    "total": int(m.group(2)),
                    "f1": float(m.group(3)),
                }

            # Errores
            if "ERROR" in line or "Traceback" in line or "Exception" in line:
                errors.append(line.strip())

        # 4. Modelos recientes en disco
        recent_files = []
        if MODELS_DIR.exists():
            model_files = sorted(MODELS_DIR.glob("*.*"), key=lambda f: f.stat().st_mtime, reverse=True)
            for mf in model_files[:20]:
                recent_files.append({
                    "name": mf.name,
                    "size_kb": round(mf.stat().st_size / 1024, 1),
                    "modified": datetime.fromtimestamp(
                        mf.stat().st_mtime, tz=timezone.utc
                    ).isoformat(),
                })

        # 5. Progreso global
        done = len(completed_models)
        progress_pct = round(done / total_models * 100, 1) if total_models > 0 else 0

        # Elapsed time
        elapsed = None
        if svc_status == "active" and svc_since:
            try:
                # Parse systemd timestamp
                start = pd.to_datetime(svc_since, utc=True)
                elapsed_sec = (datetime.now(timezone.utc) - start).total_seconds()
                hours = int(elapsed_sec // 3600)
                mins = int((elapsed_sec % 3600) // 60)
                elapsed = f"{hours}h {mins}m"
            except Exception:
                pass

        return _safe_json({
            "server_time":     datetime.now(timezone.utc).isoformat(),
            "service_status":  svc_status,
            "service_since":   svc_since,
            "elapsed":         elapsed,
            "log_source":      log_source,
            "available_logs":  available_logs,
            "training_started": training_started,
            "total_models":    total_models,
            "completed":       done,
            "progress_pct":    progress_pct,
            "current_model":   current_model,
            "current_epoch":   current_epoch,
            "completed_models": completed_models,
            "recent_files":    recent_files,
            "errors":          errors[-10:],
            "log_tail":        log_lines[-50:],
        })
    except Exception as e:
        logger.error(f"/api/train/status error: {e}")
        raise HTTPException(500, str(e))


# ── /api/backtest — Backtesting visual ───────────────────────────────

class BacktestRequest(BaseModel):
    pairs: List[str] = ["EURUSD", "GBPUSD", "USDJPY", "EURJPY", "XAUUSD"]
    timeframes: List[str] = ["M15", "H1", "H4"]
    days: int = 90
    min_confidence: float = 0.54


@app.post("/api/backtest/run", tags=["Performance"], summary="Ejecutar backtest",
          description="Ejecuta un backtest completo sobre señales históricas con simulación realista (spread, slippage, position sizing).")
def api_backtest_run(req: BacktestRequest):
    """
    Ejecuta backtest sobre señales históricas y devuelve el informe completo.
    """
    from dataclasses import asdict
    try:
        from src.backtest.engine import run_backtest

        report = run_backtest(
            pairs=req.pairs,
            timeframes=req.timeframes,
            days=req.days,
            min_confidence=req.min_confidence,
        )
        data = asdict(report)
        # Convertir inf a null para JSON válido
        if isinstance(data.get("profit_factor"), float) and math.isinf(data["profit_factor"]):
            data["profit_factor"] = None
        return JSONResponse(content=json.loads(json.dumps(data, cls=NaNSafeEncoder, default=str)))

    except Exception as e:
        logger.error(f"/api/backtest/run error: {e}")
        raise HTTPException(500, str(e))


@app.get("/api/backtest/quick-stats", tags=["Performance"], summary="Stats rápidos de último backtest",
         description="Devuelve métricas resumidas del último backtest ejecutado (total trades, win rate, PnL, drawdown).")
def api_backtest_quick_stats():
    """
    Estadísticas rápidas: cuántas señales hay disponibles para backtest por par/TF.
    """
    try:
        with create_engine(DATABASE_URL).connect() as conn:
            rows = conn.execute(text("""
                SELECT pair, timeframe,
                       COUNT(*) as total,
                       MIN(timestamp) as first_signal,
                       MAX(timestamp) as last_signal
                FROM signals
                WHERE direction != 0 AND filter_reason IS NULL
                GROUP BY pair, timeframe
                ORDER BY pair, timeframe
            """)).fetchall()
        stats = []
        for r in rows:
            stats.append({
                "pair": r[0], "timeframe": r[1], "total": r[2],
                "first": str(r[3]) if r[3] else None,
                "last": str(r[4]) if r[4] else None,
            })
        return {"stats": stats, "has_signals": len(stats) > 0}
    except Exception as e:
        logger.error(f"/api/backtest/quick-stats error: {e}")
        return {"stats": [], "has_signals": False, "error": str(e)}


# ── /api/models/compare — Comparador XGBoost vs LSTM ─────────────────────

@app.get("/api/models/compare", tags=["Models"],
         summary="Comparar modelos XGBoost vs LSTM",
         description="Escanea models/saved/ y devuelve métricas lado a lado por par/timeframe.")
def compare_models():
    """Devuelve métricas de todos los modelos guardados, agrupados por par/TF."""
    try:
        comparisons = {}  # key = "PAIR_TF"

        # --- XGBoost: leer *_meta.json ---
        for meta_file in sorted(MODELS_DIR.glob("xgb_*_meta.json")):
            try:
                meta = json.loads(meta_file.read_text())
                pair = meta.get("pair", "")
                tf   = meta.get("timeframe", "")
                key  = f"{pair}_{tf}"
                ts   = meta.get("trained_at", "")
                m    = meta.get("metrics", {})

                # Quedarnos solo con el más reciente por par/TF
                prev = comparisons.get(key, {}).get("xgb", {})
                if not prev or ts > prev.get("trained_at", ""):
                    comparisons.setdefault(key, {"pair": pair, "timeframe": tf})
                    model_path = meta_file.with_suffix(".ubj")
                    comparisons[key]["xgb"] = {
                        "trained_at":  ts,
                        "cv_f1_mean":  m.get("cv_f1_mean"),
                        "cv_f1_std":   m.get("cv_f1_std"),
                        "cv_f1_folds": m.get("cv_f1_folds", []),
                        "n_features":  len(meta.get("features", [])),
                        "file_size_kb": round(model_path.stat().st_size / 1024, 1) if model_path.exists() else None,
                    }
            except Exception as e:
                logger.warning(f"Error leyendo {meta_file.name}: {e}")

        # --- LSTM: intentar cargar checkpoints .pt ---
        try:
            import torch
            _has_torch = True
        except ImportError:
            _has_torch = False

        for pt_file in sorted(MODELS_DIR.glob("lstm_*.pt")):
            try:
                # Extraer pair/tf del nombre: lstm_PAIR_TF_YYYYMMDD_HHMM.pt
                parts = pt_file.stem.split("_")
                if len(parts) < 4:
                    continue
                pair = parts[1]
                tf   = parts[2]
                ts   = "_".join(parts[3:])  # YYYYMMDD_HHMM
                key  = f"{pair}_{tf}"

                prev = comparisons.get(key, {}).get("lstm", {})
                if not prev or ts > prev.get("trained_at", ""):
                    comparisons.setdefault(key, {"pair": pair, "timeframe": tf})
                    lstm_info = {
                        "trained_at":   ts,
                        "file_size_kb": round(pt_file.stat().st_size / 1024, 1),
                    }

                    if _has_torch:
                        ckpt = torch.load(str(pt_file), map_location="cpu", weights_only=False)
                        m = ckpt.get("metrics", {})
                        lstm_info["best_val_f1"] = m.get("best_val_f1")
                        cfg = ckpt.get("model_config", {})
                        lstm_info["hidden_size"] = cfg.get("hidden_size")
                        lstm_info["num_layers"]  = cfg.get("num_layers")
                        lstm_info["n_features"]  = len(ckpt.get("feature_cols", []))

                    comparisons[key]["lstm"] = lstm_info
            except Exception as e:
                logger.warning(f"Error leyendo {pt_file.name}: {e}")

        # Ordenar por par y TF
        result = sorted(comparisons.values(), key=lambda x: (x["pair"], x["timeframe"]))

        return _safe_json({
            "models_dir":  str(MODELS_DIR),
            "total_pairs":  len(result),
            "has_torch":    _has_torch if '_has_torch' in dir() else False,
            "comparisons":  result,
        })
    except Exception as e:
        logger.error(f"/api/models/compare error: {e}")
        raise HTTPException(500, str(e))





# ── /api/docs — Documentación dinámica ──────────────────────────────────

DOCS_DIR = Path(__file__).resolve().parent.parent.parent / "docs"


@app.get("/api/docs-list", tags=["Docs"], summary="Listar documentación",
         description="Escanea los archivos .md del directorio docs/ y devuelve sus nombres y tamaños.")
def list_docs():
    """Lista todos los .md de la carpeta docs/"""
    if not DOCS_DIR.exists():
        return {"files": []}
    files = []
    for f in sorted(DOCS_DIR.glob("*.md")):
        stat = f.stat()
        files.append({
            "name": f.stem,               # sin extensión
            "filename": f.name,           # con extensión
            "size_kb": round(stat.st_size / 1024, 1),
            "modified": datetime.fromtimestamp(
                stat.st_mtime, tz=timezone.utc
            ).isoformat(),
        })
    return {"files": files}


@app.get("/api/docs-content/{filename}", tags=["Docs"], summary="Leer documento",
         description="Devuelve el contenido raw de un archivo .md del directorio docs/.")
def get_doc(filename: str):
    """Devuelve el contenido markdown de un archivo"""
    # Seguridad: solo .md, sin path traversal
    if not filename.endswith(".md") or "/" in filename or "\\" in filename:
        raise HTTPException(400, "Solo se permiten archivos .md")
    filepath = DOCS_DIR / filename
    if not filepath.exists():
        raise HTTPException(404, f"{filename} no encontrado")
    content = filepath.read_text(encoding="utf-8")
    return {"filename": filename, "name": filepath.stem, "content": content}


# ── /api/notifications ─ Historial de notificaciones ────────────────────

ALERT_RULES_PATH = Path("config/alert_rules.json")

@app.get("/api/notifications")
def get_notifications(
    notif_type: Optional[str] = None,
    severity: Optional[str] = None,
    days: int = Query(default=30, ge=1, le=365),
    limit: int = Query(default=100, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
):
    """Historial de notificaciones enviadas."""
    engine = create_engine(DATABASE_URL)
    # Asegurar que la tabla existe
    try:
        with engine.connect() as conn:
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS notification_log (
                    id          BIGSERIAL       PRIMARY KEY,
                    created_at  TIMESTAMPTZ     DEFAULT NOW(),
                    notif_type  VARCHAR(30)     NOT NULL,
                    severity    VARCHAR(10)     DEFAULT 'info',
                    title       VARCHAR(200),
                    message     TEXT,
                    pair        VARCHAR(10),
                    timeframe   VARCHAR(5),
                    delivered   BOOLEAN         DEFAULT TRUE,
                    metadata    JSONB
                )
            """))
            conn.commit()
    except Exception:
        pass

    cutoff = datetime.now(timezone.utc) - timedelta(days=days)
    conditions = ["created_at >= :cutoff"]
    params = {"cutoff": cutoff, "lim": limit, "off": offset}

    if notif_type:
        conditions.append("notif_type = :ntype")
        params["ntype"] = notif_type
    if severity:
        conditions.append("severity = :sev")
        params["sev"] = severity

    where = " AND ".join(conditions)

    try:
        df = pd.read_sql(
            text(f"""
                SELECT id, created_at, notif_type, severity, title, pair, timeframe, delivered, metadata
                FROM notification_log
                WHERE {where}
                ORDER BY created_at DESC
                LIMIT :lim OFFSET :off
            """),
            engine, params=params,
        )
        # Contar total
        with engine.connect() as conn:
            total = conn.execute(
                text(f"SELECT COUNT(*) FROM notification_log WHERE {where}"),
                {k: v for k, v in params.items() if k not in ("lim", "off")},
            ).scalar()

        # Stats por tipo
        stats_df = pd.read_sql(
            text(f"""
                SELECT notif_type, COUNT(*) as cnt
                FROM notification_log WHERE {where}
                GROUP BY notif_type ORDER BY cnt DESC
            """),
            engine,
            params={k: v for k, v in params.items() if k not in ("lim", "off")},
        )

        records = []
        for _, row in df.iterrows():
            r = {
                "id": int(row["id"]),
                "created_at": row["created_at"].isoformat() if pd.notna(row["created_at"]) else None,
                "notif_type": row["notif_type"],
                "severity": row["severity"],
                "title": row["title"],
                "pair": row["pair"],
                "timeframe": row["timeframe"],
                "delivered": bool(row["delivered"]),
                "metadata": row["metadata"] if pd.notna(row.get("metadata")) else None,
            }
            records.append(r)

        return {
            "notifications": records,
            "total": int(total),
            "stats_by_type": {row["notif_type"]: int(row["cnt"]) for _, row in stats_df.iterrows()},
        }
    except Exception as e:
        return {"notifications": [], "total": 0, "stats_by_type": {}, "error": str(e)}


# ── /api/alert-rules ─ Reglas configurables ──────────────────────

@app.get("/api/alert-rules")
def get_alert_rules():
    """Devuelve las reglas de alerta configuradas."""
    if not ALERT_RULES_PATH.exists():
        return {"rules": []}
    data = json.loads(ALERT_RULES_PATH.read_text(encoding="utf-8"))
    return {"rules": data.get("rules", [])}


class AlertRuleUpdate(BaseModel):
    id: str
    enabled: Optional[bool] = None
    threshold: Optional[float] = None
    severity: Optional[str] = None
    cooldown_hours: Optional[int] = None


class AlertRuleCreate(BaseModel):
    name: str
    check_type: str       # signal_drought | drawdown | low_winrate | stale_data | stale_models
    pair: Optional[str] = None
    timeframe: Optional[str] = None
    threshold: float
    unit: str             # days | percent | hours
    severity: str = "warning"
    cooldown_hours: int = 24


@app.put("/api/alert-rules")
def update_alert_rule(update: AlertRuleUpdate):
    """Actualiza una regla existente (toggle, umbral, severidad)."""
    if not ALERT_RULES_PATH.exists():
        raise HTTPException(404, "Config no encontrada")
    data = json.loads(ALERT_RULES_PATH.read_text(encoding="utf-8"))
    found = False
    for rule in data.get("rules", []):
        if rule["id"] == update.id:
            if update.enabled is not None:
                rule["enabled"] = update.enabled
            if update.threshold is not None:
                rule["threshold"] = update.threshold
            if update.severity is not None:
                rule["severity"] = update.severity
            if update.cooldown_hours is not None:
                rule["cooldown_hours"] = update.cooldown_hours
            found = True
            break
    if not found:
        raise HTTPException(404, f"Regla {update.id} no encontrada")
    ALERT_RULES_PATH.write_text(json.dumps(data, indent=4, ensure_ascii=False), encoding="utf-8")
    return {"ok": True, "rule": next(r for r in data["rules"] if r["id"] == update.id)}


@app.post("/api/alert-rules")
def create_alert_rule(rule: AlertRuleCreate):
    """Crea una nueva regla de alerta."""
    if not ALERT_RULES_PATH.exists():
        data = {"rules": [], "last_fired": {}}
    else:
        data = json.loads(ALERT_RULES_PATH.read_text(encoding="utf-8"))

    # Generar ID
    rule_id = f"{rule.check_type}_{rule.pair or 'all'}_{len(data['rules'])+1}"
    new_rule = {
        "id": rule_id,
        "enabled": True,
        "name": rule.name,
        "description": "",
        "check_type": rule.check_type,
        "pair": rule.pair,
        "timeframe": rule.timeframe,
        "threshold": rule.threshold,
        "unit": rule.unit,
        "severity": rule.severity,
        "cooldown_hours": rule.cooldown_hours,
    }
    data["rules"].append(new_rule)
    ALERT_RULES_PATH.write_text(json.dumps(data, indent=4, ensure_ascii=False), encoding="utf-8")
    return {"ok": True, "rule": new_rule}


@app.delete("/api/alert-rules/{rule_id}")
def delete_alert_rule(rule_id: str):
    """Elimina una regla de alerta."""
    if not ALERT_RULES_PATH.exists():
        raise HTTPException(404, "Config no encontrada")
    data = json.loads(ALERT_RULES_PATH.read_text(encoding="utf-8"))
    original_len = len(data.get("rules", []))
    data["rules"] = [r for r in data.get("rules", []) if r["id"] != rule_id]
    if len(data["rules"]) == original_len:
        raise HTTPException(404, f"Regla {rule_id} no encontrada")
    ALERT_RULES_PATH.write_text(json.dumps(data, indent=4, ensure_ascii=False), encoding="utf-8")
    return {"ok": True}


@app.post("/api/alert-rules/test/{rule_id}")
def test_alert_rule(rule_id: str):
    """Envía un mensaje de prueba a Telegram para una regla."""
    if not ALERT_RULES_PATH.exists():
        raise HTTPException(404, "Config no encontrada")
    data = json.loads(ALERT_RULES_PATH.read_text(encoding="utf-8"))
    rule = next((r for r in data.get("rules", []) if r["id"] == rule_id), None)
    if not rule:
        raise HTTPException(404, f"Regla {rule_id} no encontrada")
    try:
        from src.notifications.telegram import send_message, log_notification
        msg = (
            f"🔔 <b>Test de regla de alerta</b>\n"
            f"━━━━━━━━━━━━━━━━━━━━\n"
            f"📌 <b>{rule['name']}</b>\n"
            f"🛠️ Tipo: {rule['check_type']}\n"
            f"🎯 Umbral: {rule['threshold']} {rule['unit']}\n"
            f"💱 Par: {rule.get('pair') or 'Todos'}\n"
            f"\n<i>Este es un mensaje de prueba.</i>"
        )
        ok = send_message(msg)
        if ok:
            log_notification(
                notif_type="alert_rule", severity="info",
                title=f"Test: {rule['name']}", message=msg,
                pair=rule.get("pair"), delivered=True,
                metadata={"rule_id": rule_id, "test": True},
            )
        return {"ok": ok, "message": "Notificación enviada" if ok else "Fallo al enviar"}
    except Exception as e:
        return {"ok": False, "message": str(e)}


# ── /api/config ───────────────────────────────────────────────────────────────

@app.get("/api/config", tags=["Config"], summary="Filtros del generador de señales",
         description="Devuelve los filtros actuales: min_confidence, min_adx, min_rr, cooldown_bars, allow_offmarket.")
def get_config():
    return _current_config.dict()


@app.post("/api/config", tags=["Config"], summary="Actualizar filtros del generador",
          description="Modifica los filtros de calidad del generador de señales en caliente.")
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
