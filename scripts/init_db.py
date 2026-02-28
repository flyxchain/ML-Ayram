"""
scripts/init_db.py
Crea todas las tablas necesarias en la base de datos PostgreSQL.

Ejecutar UNA sola vez en el servidor antes de arrancar el bot:
    python -m scripts.init_db

Es idempotente: usa CREATE TABLE IF NOT EXISTS, no borra datos existentes.
"""

import os
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from loguru import logger

load_dotenv()
DATABASE_URL = os.getenv("DATABASE_URL")


SCHEMA = """
-- ── Velas OHLCV crudas ─────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS ohlcv_raw (
    id          BIGSERIAL PRIMARY KEY,
    pair        TEXT        NOT NULL,
    timeframe   TEXT        NOT NULL,
    timestamp   TIMESTAMPTZ NOT NULL,
    open        REAL        NOT NULL,
    high        REAL        NOT NULL,
    low         REAL        NOT NULL,
    close       REAL        NOT NULL,
    volume      REAL        DEFAULT 0,
    UNIQUE (pair, timeframe, timestamp)
);
CREATE INDEX IF NOT EXISTS idx_ohlcv_pair_tf_ts
    ON ohlcv_raw (pair, timeframe, timestamp DESC);

-- ── Features calculados ────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS features_computed (
    id              BIGSERIAL PRIMARY KEY,
    pair            TEXT        NOT NULL,
    timeframe       TEXT        NOT NULL,
    timestamp       TIMESTAMPTZ NOT NULL,
    -- precio de cierre (para referencia rápida)
    close           REAL,
    high            REAL,
    low             REAL,
    -- Tendencia
    ema_20          REAL, ema_50 REAL, ema_200 REAL,
    macd_line       REAL, macd_signal REAL, macd_hist REAL,
    adx             REAL, adx_pos REAL, adx_neg REAL,
    rsi_14          REAL, rsi_7 REAL,
    stoch_k         REAL, stoch_d REAL,
    williams_r      REAL, roc_10 REAL, cci_20 REAL,
    -- Volatilidad
    atr_14          REAL, atr_7 REAL,
    bb_upper        REAL, bb_middle REAL, bb_lower REAL,
    bb_width        REAL, bb_pct REAL,
    kc_upper        REAL, kc_lower REAL, kc_width REAL,
    dc_upper        REAL, dc_lower REAL, dc_width REAL,
    -- Estructura
    volume_ratio_20 REAL,
    swing_high_20   REAL, swing_low_20 REAL,
    price_vs_sh     REAL, price_vs_sl REAL,
    trend_direction SMALLINT,
    body_size       REAL, upper_wick REAL, lower_wick REAL,
    is_bullish      SMALLINT,
    log_return_1    REAL, log_return_5 REAL, log_return_10 REAL,
    close_vs_ema20  REAL, close_vs_ema50 REAL, close_vs_ema200 REAL,
    -- Temporales
    hour_of_day     SMALLINT, day_of_week SMALLINT,
    week_of_year    SMALLINT, month SMALLINT,
    session_name    TEXT,
    is_london       SMALLINT, is_newyork SMALLINT,
    is_overlap      SMALLINT, is_offmarket SMALLINT,
    -- HTF features
    htf_trend       SMALLINT, htf_rsi REAL, htf_adx REAL, htf_atr REAL,
    -- Etiquetas (Triple Barrier)
    label           SMALLINT,
    label_return    REAL,
    bars_to_exit    SMALLINT,
    tp_price        REAL,
    sl_price        REAL,
    UNIQUE (pair, timeframe, timestamp)
);
CREATE INDEX IF NOT EXISTS idx_features_pair_tf_ts
    ON features_computed (pair, timeframe, timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_features_label
    ON features_computed (pair, timeframe, label)
    WHERE label IS NOT NULL;

-- ── Señales generadas ──────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS signals (
    id              BIGSERIAL PRIMARY KEY,
    pair            TEXT        NOT NULL,
    timeframe       TEXT        NOT NULL,
    timestamp       TIMESTAMPTZ NOT NULL,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    direction       SMALLINT    NOT NULL,   -- +1 long / -1 short / 0 neutral
    confidence      REAL,
    prob_long       REAL,
    prob_neutral    REAL,
    prob_short      REAL,
    xgb_direction   SMALLINT,
    lstm_direction  SMALLINT,
    agreement       BOOLEAN,
    entry_price     REAL,
    tp_price        REAL,
    sl_price        REAL,
    tp_pips         REAL,
    sl_pips         REAL,
    rr_ratio        REAL,
    atr_14          REAL,
    adx             REAL,
    session         TEXT,
    filter_reason   TEXT,       -- NULL si pasó todos los filtros
    UNIQUE (pair, timeframe, timestamp)
);
CREATE INDEX IF NOT EXISTS idx_signals_ts   ON signals (timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_signals_dir  ON signals (direction) WHERE direction != 0;

-- ── Posiciones abiertas ────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS positions_active (
    id          SERIAL PRIMARY KEY,
    signal_id   INT         UNIQUE NOT NULL,
    pair        TEXT        NOT NULL,
    timeframe   TEXT        NOT NULL,
    direction   SMALLINT    NOT NULL,
    entry_price REAL        NOT NULL,
    tp_price    REAL        NOT NULL,
    sl_price    REAL        NOT NULL,
    lot_size    REAL        NOT NULL,
    risk_amount REAL        NOT NULL,
    opened_at   TIMESTAMPTZ NOT NULL,
    status      TEXT        NOT NULL DEFAULT 'open',
    closed_at   TIMESTAMPTZ,
    pnl         REAL
);

-- ── Historial de trades cerrados ───────────────────────────────────────────
CREATE TABLE IF NOT EXISTS trades_history (
    id              SERIAL PRIMARY KEY,
    signal_id       INT         NOT NULL,
    pair            TEXT        NOT NULL,
    timeframe       TEXT        NOT NULL,
    direction       SMALLINT    NOT NULL,
    entry_price     REAL        NOT NULL,
    exit_price      REAL        NOT NULL,
    tp_price        REAL        NOT NULL,
    sl_price        REAL        NOT NULL,
    lot_size        REAL        NOT NULL,
    risk_amount     REAL        NOT NULL,
    pnl             REAL        NOT NULL,
    result          TEXT        NOT NULL,   -- 'tp_hit' | 'sl_hit' | 'expired'
    opened_at       TIMESTAMPTZ NOT NULL,
    closed_at       TIMESTAMPTZ NOT NULL,
    duration_bars   INT
);
CREATE INDEX IF NOT EXISTS idx_trades_closed ON trades_history (closed_at DESC);
"""


def init_db() -> None:
    engine = create_engine(DATABASE_URL)
    with engine.connect() as conn:
        conn.execute(text("SET statement_timeout = 0"))  # índices pueden tardar
        for statement in SCHEMA.split(";"):
            stmt = statement.strip()
            if stmt:
                conn.execute(text(stmt))
        conn.commit()
    logger.success("✅ Base de datos inicializada correctamente")


if __name__ == "__main__":
    init_db()
