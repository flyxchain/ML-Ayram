-- ============================================================
-- ML-Ayram — Schema de Base de Datos PostgreSQL
-- TimescaleDB eliminado (no disponible en Supabase free)
-- ============================================================

-- ============================================================
-- TABLA 1: Velas OHLCV históricas y en tiempo real
-- ============================================================
CREATE TABLE IF NOT EXISTS ohlcv_raw (
    id              BIGSERIAL,
    pair            VARCHAR(10)     NOT NULL,   -- 'EURUSD', 'GBPUSD', etc.
    timeframe       VARCHAR(5)      NOT NULL,   -- 'M15', 'H1', 'H4', 'D1'
    timestamp       TIMESTAMPTZ     NOT NULL,
    open            DOUBLE PRECISION NOT NULL,
    high            DOUBLE PRECISION NOT NULL,
    low             DOUBLE PRECISION NOT NULL,
    close           DOUBLE PRECISION NOT NULL,
    volume          DOUBLE PRECISION,
    tick_volume     INTEGER,
    spread          DOUBLE PRECISION,
    created_at      TIMESTAMPTZ     DEFAULT NOW(),
    PRIMARY KEY (pair, timeframe, timestamp)
);

-- Índices para consultas frecuentes
CREATE INDEX IF NOT EXISTS idx_ohlcv_pair_tf ON ohlcv_raw (pair, timeframe, timestamp DESC);

-- ============================================================
-- TABLA 2: Features calculados por vela
-- ============================================================
CREATE TABLE IF NOT EXISTS features_computed (
    id              BIGSERIAL,
    pair            VARCHAR(10)     NOT NULL,
    timeframe       VARCHAR(5)      NOT NULL,
    timestamp       TIMESTAMPTZ     NOT NULL,
    -- Indicadores de tendencia
    ema_20          DOUBLE PRECISION,
    ema_50          DOUBLE PRECISION,
    ema_200         DOUBLE PRECISION,
    macd_line       DOUBLE PRECISION,
    macd_signal     DOUBLE PRECISION,
    macd_hist       DOUBLE PRECISION,
    adx             DOUBLE PRECISION,
    -- Osciladores
    rsi_14          DOUBLE PRECISION,
    stoch_k         DOUBLE PRECISION,
    stoch_d         DOUBLE PRECISION,
    cci_20          DOUBLE PRECISION,
    -- Volatilidad
    atr_14          DOUBLE PRECISION,
    bb_upper        DOUBLE PRECISION,
    bb_middle       DOUBLE PRECISION,
    bb_lower        DOUBLE PRECISION,
    bb_width        DOUBLE PRECISION,
    -- Volumen
    volume_ratio_20 DOUBLE PRECISION,   -- volumen actual / media 20 periodos
    -- Estructura de mercado
    swing_high_20   DOUBLE PRECISION,
    swing_low_20    DOUBLE PRECISION,
    trend_direction SMALLINT,           -- 1 (alcista), -1 (bajista), 0 (lateral)
    -- Sesión
    session_name    VARCHAR(10),        -- 'tokyo', 'london', 'newyork', 'overlap'
    hour_of_day     SMALLINT,
    day_of_week     SMALLINT,
    -- Features multi-timeframe (del TF superior)
    htf_trend       SMALLINT,           -- tendencia en H4 cuando estamos en H1
    htf_rsi         DOUBLE PRECISION,
    -- Label (generado por Triple-Barrier)
    label           SMALLINT,           -- 1=BUY, -1=SELL, 0=NEUTRAL
    label_return    DOUBLE PRECISION,   -- retorno real alcanzado
    sample_weight   DOUBLE PRECISION,  -- peso para el entrenamiento
    PRIMARY KEY (pair, timeframe, timestamp)
);

CREATE INDEX IF NOT EXISTS idx_features_pair_tf ON features_computed (pair, timeframe, timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_features_label ON features_computed (label) WHERE label IS NOT NULL;

-- ============================================================
-- TABLA 3: Señales generadas por el motor
-- ============================================================
CREATE TABLE IF NOT EXISTS signals_log (
    id              BIGSERIAL       PRIMARY KEY,
    created_at      TIMESTAMPTZ     DEFAULT NOW(),
    pair            VARCHAR(10)     NOT NULL,
    timeframe       VARCHAR(5)      NOT NULL,
    direction       VARCHAR(4)      NOT NULL,   -- 'BUY' o 'SELL'
    entry_price     DOUBLE PRECISION NOT NULL,
    sl_price        DOUBLE PRECISION NOT NULL,
    tp1_price       DOUBLE PRECISION NOT NULL,
    tp2_price       DOUBLE PRECISION NOT NULL,
    atr_at_signal   DOUBLE PRECISION,
    model_confidence DOUBLE PRECISION,          -- 0 a 1
    regime          VARCHAR(20),                -- 'trending', 'ranging', 'high_vol'
    timeframes_aligned VARCHAR(50),             -- ej: 'H1,H4'
    lot_size        DOUBLE PRECISION,
    risk_percent    DOUBLE PRECISION,
    mode            VARCHAR(20)     DEFAULT 'paper',  -- 'paper' o 'live'
    -- Resultado
    status          VARCHAR(20)     DEFAULT 'open',   -- 'open', 'tp1', 'tp2', 'sl', 'expired'
    closed_at       TIMESTAMPTZ,
    pnl_pips        DOUBLE PRECISION,
    pnl_usd         DOUBLE PRECISION,
    ctrader_order_id BIGINT
);

CREATE INDEX IF NOT EXISTS idx_signals_pair ON signals_log (pair, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_signals_status ON signals_log (status, created_at DESC);

-- ============================================================
-- TABLA 4: Rendimiento de los modelos por período
-- ============================================================
CREATE TABLE IF NOT EXISTS model_performance (
    id              BIGSERIAL       PRIMARY KEY,
    recorded_at     TIMESTAMPTZ     DEFAULT NOW(),
    model_version   VARCHAR(50),
    pair            VARCHAR(10),
    timeframe       VARCHAR(5),
    period_start    DATE,
    period_end      DATE,
    -- Métricas de clasificación
    accuracy        DOUBLE PRECISION,
    f1_buy          DOUBLE PRECISION,
    f1_sell         DOUBLE PRECISION,
    -- Métricas de trading
    hit_rate        DOUBLE PRECISION,
    profit_factor   DOUBLE PRECISION,
    sharpe_ratio    DOUBLE PRECISION,
    max_drawdown    DOUBLE PRECISION,
    total_trades    INTEGER,
    is_oos          BOOLEAN         DEFAULT TRUE  -- out-of-sample
);

-- ============================================================
-- TABLA 5: Posiciones activas (para el auto-trader)
-- ============================================================
CREATE TABLE IF NOT EXISTS positions_active (
    id              BIGSERIAL       PRIMARY KEY,
    signal_id       BIGINT          REFERENCES signals_log(id),
    opened_at       TIMESTAMPTZ     DEFAULT NOW(),
    pair            VARCHAR(10)     NOT NULL,
    direction       VARCHAR(4)      NOT NULL,
    lot_size        DOUBLE PRECISION,
    entry_price     DOUBLE PRECISION,
    current_sl      DOUBLE PRECISION,       -- SL puede moverse (trailing)
    tp1_price       DOUBLE PRECISION,
    tp2_price       DOUBLE PRECISION,
    tp1_hit         BOOLEAN         DEFAULT FALSE,
    ctrader_order_id BIGINT,
    ctrader_position_id BIGINT
);

-- ============================================================
-- VISTA: Resumen de rendimiento de los últimos 30 días
-- ============================================================
CREATE OR REPLACE VIEW performance_summary_30d AS
SELECT
    pair,
    COUNT(*)                                        AS total_signals,
    COUNT(*) FILTER (WHERE status IN ('tp1','tp2')) AS winners,
    COUNT(*) FILTER (WHERE status = 'sl')           AS losers,
    ROUND(
        COUNT(*) FILTER (WHERE status IN ('tp1','tp2'))::NUMERIC
        / NULLIF(COUNT(*) FILTER (WHERE status != 'open'), 0) * 100,
        1
    )                                               AS hit_rate_pct,
    ROUND(SUM(pnl_pips)::NUMERIC, 1)               AS total_pips,
    ROUND(SUM(pnl_usd)::NUMERIC, 2)                AS total_usd,
    mode
FROM signals_log
WHERE created_at >= NOW() - INTERVAL '30 days'
  AND status != 'open'
GROUP BY pair, mode
ORDER BY total_usd DESC;

-- ============================================================
-- TABLA 6: Log de notificaciones enviadas
-- ============================================================
CREATE TABLE IF NOT EXISTS notification_log (
    id              BIGSERIAL       PRIMARY KEY,
    created_at      TIMESTAMPTZ     DEFAULT NOW(),
    notif_type      VARCHAR(30)     NOT NULL,    -- 'signal', 'training', 'anomaly', 'heartbeat', 'error', 'summary', 'alert_rule'
    severity        VARCHAR(10)     DEFAULT 'info',  -- 'info', 'warning', 'high', 'critical'
    title           VARCHAR(200),
    message         TEXT,
    pair            VARCHAR(10),
    timeframe       VARCHAR(5),
    delivered       BOOLEAN         DEFAULT TRUE,
    metadata        JSONB
);

CREATE INDEX IF NOT EXISTS idx_notif_type ON notification_log (notif_type, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_notif_created ON notification_log (created_at DESC);

-- ============================================================
-- Mensaje de confirmación
-- ============================================================
DO $
BEGIN
    RAISE NOTICE 'Schema ML-Ayram creado correctamente. Tablas: ohlcv_raw, features_computed, signals_log, model_performance, positions_active, notification_log';
END $;
