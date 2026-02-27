"""
src/signals/generator.py
Genera señales de trading accionables combinando el ensemble ML con filtros técnicos.

Flujo por vela:
  1. Carga las últimas N velas con features calculados
  2. Pide predicción al ensemble (XGBoost + LSTM)
  3. Aplica filtros de calidad (ADX, sesión, confianza mínima, cooldown)
  4. Calcula TP / SL en pips y en precio
  5. Guarda la señal en la BD (tabla signals)
  6. Devuelve un objeto SignalResult listo para notificaciones / ejecución

Tabla esperada en BD:
  CREATE TABLE IF NOT EXISTS signals (
      id            SERIAL PRIMARY KEY,
      pair          TEXT        NOT NULL,
      timeframe     TEXT        NOT NULL,
      timestamp     TIMESTAMPTZ NOT NULL,
      direction     SMALLINT    NOT NULL,   -- +1 long / -1 short / 0 neutral
      confidence    REAL,
      prob_long     REAL,
      prob_neutral  REAL,
      prob_short    REAL,
      xgb_direction SMALLINT,
      lstm_direction SMALLINT,
      agreement     BOOLEAN,
      entry_price   REAL,
      tp_price      REAL,
      sl_price      REAL,
      tp_pips       REAL,
      sl_pips       REAL,
      rr_ratio      REAL,
      atr_14        REAL,
      adx           REAL,
      session       TEXT,
      filter_reason TEXT,       -- NULL si pasó todos los filtros
      created_at    TIMESTAMPTZ DEFAULT NOW(),
      UNIQUE (pair, timeframe, timestamp)
  );
"""

import os
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Optional

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from loguru import logger
from sqlalchemy import create_engine, text

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")

# ── Configuración ─────────────────────────────────────────────────────────────

PAIRS      = ["EURUSD", "GBPUSD", "USDJPY", "EURJPY", "XAUUSD"]
TIMEFRAMES = ["M15", "H1", "H4"]

# Pip size por par (para convertir ATR a pips)
PIP_SIZE = {
    "EURUSD": 0.0001,
    "GBPUSD": 0.0001,
    "USDJPY": 0.01,
    "EURJPY": 0.01,
    "XAUUSD": 0.1,
}

# Mínimo de velas a cargar para que el ensemble tenga contexto suficiente
MIN_BARS = 300

# Filtros de calidad — ajustables sin tocar el ensemble
FILTERS = {
    "min_confidence":   0.54,   # confianza ensemble mínima
    "min_adx":          20.0,   # tendencia mínima (ADX)
    "allow_offmarket":  False,  # rechazar señales fuera de sesión
    "min_rr":           1.5,    # ratio riesgo/beneficio mínimo
    "cooldown_bars":    3,      # mínimo de velas entre señales del mismo par/tf
}

# Multiplicadores TP/SL sobre ATR (igual que labels.py para consistencia)
TP_MULTIPLIER = 2.0
SL_MULTIPLIER = 1.0


# ── Dataclass de resultado ────────────────────────────────────────────────────

@dataclass
class SignalResult:
    pair:           str
    timeframe:      str
    timestamp:      datetime
    direction:      int           # +1 / -1 / 0
    confidence:     float
    prob_long:      float
    prob_neutral:   float
    prob_short:     float
    xgb_direction:  int
    lstm_direction: int
    agreement:      bool
    entry_price:    float
    tp_price:       float
    sl_price:       float
    tp_pips:        float
    sl_pips:        float
    rr_ratio:       float
    atr_14:         float
    adx:            float
    session:        str
    filter_reason:  Optional[str] = None   # None = señal válida

    @property
    def is_valid(self) -> bool:
        return self.filter_reason is None and self.direction != 0

    @property
    def direction_label(self) -> str:
        return {1: "LONG", -1: "SHORT", 0: "NEUTRAL"}.get(self.direction, "?")

    def summary(self) -> str:
        if not self.is_valid:
            return (f"[{self.pair} {self.timeframe}] "
                    f"Sin señal — {self.filter_reason or 'neutral'}")
        return (
            f"[{self.pair} {self.timeframe}] {self.direction_label} "
            f"@ {self.entry_price:.5f} | "
            f"TP {self.tp_price:.5f} (+{self.tp_pips:.1f} pips) | "
            f"SL {self.sl_price:.5f} (-{self.sl_pips:.1f} pips) | "
            f"R:R {self.rr_ratio:.2f} | "
            f"Conf {self.confidence:.2%} | "
            f"ADX {self.adx:.1f} | "
            f"Sesión {self.session}"
        )


# ── Helpers ───────────────────────────────────────────────────────────────────

def _pip_size(pair: str) -> float:
    return PIP_SIZE.get(pair, 0.0001)


def _price_to_pips(price_diff: float, pair: str) -> float:
    return abs(price_diff) / _pip_size(pair)


def _get_session(hour: int) -> str:
    """Devuelve la sesión activa en UTC."""
    london  = 7  <= hour < 16
    newyork = 12 <= hour < 21
    tokyo   = 0  <= hour < 9
    if london and newyork:
        return "overlap"
    if london:
        return "london"
    if newyork:
        return "newyork"
    if tokyo:
        return "tokyo"
    return "offmarket"


def _load_bars(pair: str, timeframe: str, engine, n: int = MIN_BARS) -> pd.DataFrame:
    """Carga las últimas N velas con features desde la BD."""
    df = pd.read_sql(
        text("""
            SELECT * FROM features_computed
            WHERE pair = :pair AND timeframe = :tf
            ORDER BY timestamp DESC
            LIMIT :n
        """),
        engine,
        params={"pair": pair, "tf": timeframe, "n": n},
    )
    if df.empty:
        return df
    return df.sort_values("timestamp").reset_index(drop=True)


def _last_signal_bars_ago(pair: str, timeframe: str, engine) -> int:
    """Cuántas velas han pasado desde la última señal válida (direction != 0)."""
    row = pd.read_sql(
        text("""
            SELECT timestamp FROM signals
            WHERE pair = :pair AND timeframe = :tf AND direction != 0
            ORDER BY timestamp DESC
            LIMIT 1
        """),
        engine,
        params={"pair": pair, "tf": timeframe},
    )
    if row.empty:
        return 9999   # nunca ha habido señal → sin restricción

    last_ts = pd.to_datetime(row["timestamp"].iloc[0])
    now     = datetime.now(timezone.utc)
    if last_ts.tzinfo is None:
        last_ts = last_ts.replace(tzinfo=timezone.utc)

    # Aproximar barras por minutos del timeframe
    tf_minutes = {"M15": 15, "H1": 60, "H4": 240, "D1": 1440}
    mins = tf_minutes.get(timeframe, 60)
    elapsed_bars = int((now - last_ts).total_seconds() / 60 / mins)
    return elapsed_bars


def _save_signal(signal: SignalResult, engine) -> None:
    """Inserta o actualiza la señal en la tabla signals."""
    row = asdict(signal)
    row["timestamp"] = signal.timestamp.isoformat() if hasattr(signal.timestamp, "isoformat") else str(signal.timestamp)

    sql = text("""
        INSERT INTO signals (
            pair, timeframe, timestamp, direction,
            confidence, prob_long, prob_neutral, prob_short,
            xgb_direction, lstm_direction, agreement,
            entry_price, tp_price, sl_price, tp_pips, sl_pips, rr_ratio,
            atr_14, adx, session, filter_reason
        ) VALUES (
            :pair, :timeframe, :timestamp, :direction,
            :confidence, :prob_long, :prob_neutral, :prob_short,
            :xgb_direction, :lstm_direction, :agreement,
            :entry_price, :tp_price, :sl_price, :tp_pips, :sl_pips, :rr_ratio,
            :atr_14, :adx, :session, :filter_reason
        )
        ON CONFLICT (pair, timeframe, timestamp) DO UPDATE SET
            direction      = EXCLUDED.direction,
            confidence     = EXCLUDED.confidence,
            filter_reason  = EXCLUDED.filter_reason
    """)

    with engine.connect() as conn:
        conn.execute(sql, row)
        conn.commit()


# ── Generador principal ───────────────────────────────────────────────────────

def generate_signal(
    pair:      str,
    timeframe: str,
    engine,
    xgb_model=None,
    lstm_model=None,
    lstm_scaler=None,
    lstm_features=None,
    filters:   dict = None,
    save:      bool = True,
) -> SignalResult:
    """
    Genera una señal para la vela más reciente de un par/timeframe.

    Parámetros
    ----------
    pair, timeframe : identificadores del mercado
    engine          : SQLAlchemy engine conectado a la BD
    xgb_model, ...  : modelos precargados (si None los carga del disco)
    filters         : diccionario para sobreescribir FILTERS por defecto
    save            : si True guarda la señal en BD

    Devuelve
    --------
    SignalResult con filter_reason=None si la señal pasó todos los filtros.
    """
    cfg = {**FILTERS, **(filters or {})}

    # 1. Cargar datos
    df = _load_bars(pair, timeframe, engine)
    if df.empty or len(df) < 60:
        logger.warning(f"{pair} {timeframe}: insuficientes velas ({len(df)})")
        return None

    last = df.iloc[-1]
    entry_price = float(last["close"])
    atr         = float(last.get("atr_14", np.nan))
    adx         = float(last.get("adx",    np.nan))
    hour        = int(pd.to_datetime(last["timestamp"]).hour)
    session     = _get_session(hour)
    ts          = pd.to_datetime(last["timestamp"])
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)

    # 2. Ensemble → predicción
    try:
        from src.models.ensemble import ensemble_predict, load_models
        if xgb_model is None:
            xgb_model, lstm_model, lstm_scaler, lstm_features = load_models(pair, timeframe)

        result_df = ensemble_predict(
            df, pair, timeframe,
            xgb_model, lstm_model, lstm_scaler, lstm_features
        )
    except Exception as e:
        logger.error(f"Ensemble falló para {pair} {timeframe}: {e}")
        return None

    last_pred    = result_df.iloc[-1]
    direction    = int(last_pred["signal"])
    confidence   = float(last_pred["confidence"])
    prob_long    = float(last_pred["prob_long"])
    prob_neutral = float(last_pred["prob_neutral"])
    prob_short   = float(last_pred["prob_short"])
    xgb_dir      = int(last_pred["xgb_direction"])
    lstm_dir     = int(last_pred.get("lstm_direction", 0) or 0)
    agreement    = bool(last_pred["agreement"])

    # 3. Calcular TP / SL
    if not np.isnan(atr) and atr > 0:
        if direction == 1:    # LONG
            tp_price = entry_price + TP_MULTIPLIER * atr
            sl_price = entry_price - SL_MULTIPLIER * atr
        elif direction == -1: # SHORT
            tp_price = entry_price - TP_MULTIPLIER * atr
            sl_price = entry_price + SL_MULTIPLIER * atr
        else:
            tp_price = sl_price = entry_price
    else:
        tp_price = sl_price = entry_price

    tp_pips  = _price_to_pips(tp_price - entry_price, pair)
    sl_pips  = _price_to_pips(sl_price - entry_price, pair)
    rr_ratio = round(tp_pips / sl_pips, 2) if sl_pips > 0 else 0.0

    # 4. Filtros de calidad
    filter_reason = None

    if direction == 0:
        filter_reason = "neutral"
    elif confidence < cfg["min_confidence"]:
        filter_reason = f"confianza baja ({confidence:.2%} < {cfg['min_confidence']:.2%})"
    elif not agreement:
        filter_reason = "XGBoost y LSTM no coinciden"
    elif not np.isnan(adx) and adx < cfg["min_adx"]:
        filter_reason = f"ADX débil ({adx:.1f} < {cfg['min_adx']})"
    elif not cfg["allow_offmarket"] and session == "offmarket":
        filter_reason = "fuera de sesión de mercado"
    elif rr_ratio < cfg["min_rr"]:
        filter_reason = f"R:R insuficiente ({rr_ratio:.2f} < {cfg['min_rr']})"
    else:
        # Cooldown: evitar señales consecutivas del mismo par/tf
        bars_ago = _last_signal_bars_ago(pair, timeframe, engine)
        if bars_ago < cfg["cooldown_bars"]:
            filter_reason = f"cooldown ({bars_ago} velas desde última señal)"

    signal = SignalResult(
        pair           = pair,
        timeframe      = timeframe,
        timestamp      = ts,
        direction      = direction,
        confidence     = confidence,
        prob_long      = prob_long,
        prob_neutral   = prob_neutral,
        prob_short     = prob_short,
        xgb_direction  = xgb_dir,
        lstm_direction = lstm_dir,
        agreement      = agreement,
        entry_price    = entry_price,
        tp_price       = round(tp_price, 5),
        sl_price       = round(sl_price, 5),
        tp_pips        = round(tp_pips, 1),
        sl_pips        = round(sl_pips, 1),
        rr_ratio       = rr_ratio,
        atr_14         = round(atr, 6) if not np.isnan(atr) else 0.0,
        adx            = round(adx, 2) if not np.isnan(adx) else 0.0,
        session        = session,
        filter_reason  = filter_reason,
    )

    # 5. Log
    if signal.is_valid:
        logger.success(signal.summary())
    else:
        logger.debug(signal.summary())

    # 6. Guardar en BD
    if save:
        try:
            _save_signal(signal, engine)
        except Exception as e:
            logger.error(f"Error guardando señal {pair} {timeframe}: {e}")

    # Notificación Telegram
    try:
        from src.notifications.telegram import notify_if_valid
        notify_if_valid(signal)
    except Exception as e:
        logger.warning(f"Telegram no disponible: {e}")

    return signal


def run_once(
    pairs:      list = PAIRS,
    timeframes: list = TIMEFRAMES,
    filters:    dict = None,
    save:       bool = True,
) -> list[SignalResult]:
    """
    Genera señales para todos los pares y timeframes ahora mismo.
    Devuelve solo las señales válidas (que pasaron los filtros).
    """
    engine  = create_engine(DATABASE_URL)
    valid   = []
    models_cache = {}   # evitar cargar el mismo modelo N veces

    for tf in timeframes:
        for pair in pairs:
            key = (pair, tf)
            if key not in models_cache:
                try:
                    from src.models.ensemble import load_models
                    models_cache[key] = load_models(pair, tf)
                except Exception as e:
                    logger.warning(f"No se pudo cargar modelo {pair} {tf}: {e}")
                    models_cache[key] = (None, None, None, None)

            xgb_m, lstm_m, lstm_s, lstm_f = models_cache[key]
            signal = generate_signal(
                pair, tf, engine,
                xgb_m, lstm_m, lstm_s, lstm_f,
                filters=filters,
                save=save,
            )
            if signal and signal.is_valid:
                valid.append(signal)

    logger.info(f"Señales válidas generadas: {len(valid)}")
    return valid


def run_loop(
    interval_seconds: int  = 60,
    pairs:            list = PAIRS,
    timeframes:       list = TIMEFRAMES,
    filters:          dict = None,
) -> None:
    """
    Bucle infinito que genera señales cada `interval_seconds`.
    Pensado para ejecutarse como servicio en el servidor.

    Uso:
        python -m src.signals.generator
    """
    logger.info(f"Iniciando generador de señales — intervalo {interval_seconds}s")
    while True:
        try:
            signals = run_once(pairs, timeframes, filters)
            for s in signals:
                logger.success(s.summary())
        except Exception as e:
            logger.error(f"Error en ciclo de señales: {e}")

        logger.debug(f"Durmiendo {interval_seconds}s...")
        time.sleep(interval_seconds)


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Por defecto arranca el bucle cada 60 segundos
    # Para un test puntual: run_once()
    run_loop(interval_seconds=60)
