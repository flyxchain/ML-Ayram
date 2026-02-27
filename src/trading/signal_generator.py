"""
src/trading/signal_generator.py
Convierte predicciones del ensemble en señales de trading concretas.
Calcula TP/SL, tamaño de posición y filtra señales de baja calidad.
Modo simulado: registra en BD sin enviar órdenes reales.
"""

import os
import uuid
from datetime import datetime
from dataclasses import dataclass, asdict

import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text
from dotenv import load_dotenv
from loguru import logger

from src.models.ensemble import ensemble_predict, get_latest_signal, Signal
from src.data.collector import get_latest_candles

load_dotenv()
DATABASE_URL = os.getenv("DATABASE_URL")

# Gestión de riesgo
RISK_PER_TRADE   = 0.01    # 1% del capital por operación
DEFAULT_CAPITAL  = 10_000  # capital simulado inicial
TP_ATR_MULT      = 2.0
SL_ATR_MULT      = 1.0

# Pares y timeframes activos
ACTIVE_PAIRS      = ["EURUSD", "GBPUSD", "USDJPY", "EURJPY", "XAUUSD"]
ACTIVE_TIMEFRAMES = ["H1"]   # comenzamos solo con H1


@dataclass
class TradeSignal:
    signal_id:    str
    pair:         str
    timeframe:    str
    timestamp:    str
    direction:    int      # +1 long, -1 short
    entry_price:  float
    tp_price:     float
    sl_price:     float
    lot_size:     float    # tamaño posición calculado
    risk_amount:  float    # € en riesgo
    confidence:   float
    prob_long:    float
    prob_short:   float
    prob_neutral: float
    agreement:    bool
    status:       str = "open"   # open, tp_hit, sl_hit, expired


def pip_value(pair: str) -> float:
    """Valor aproximado de 1 pip por lote estándar en USD."""
    pip_values = {
        "EURUSD": 10.0,
        "GBPUSD": 10.0,
        "USDJPY":  9.1,
        "EURJPY":  9.1,
        "XAUUSD": 10.0,
    }
    return pip_values.get(pair, 10.0)


def calculate_lot_size(
    pair:        str,
    sl_distance: float,   # en precio (no pips)
    capital:     float = DEFAULT_CAPITAL,
    risk_pct:    float = RISK_PER_TRADE,
) -> float:
    """
    Calcula el tamaño de posición basado en riesgo fijo.
    Fórmula: lot_size = (capital * risk_pct) / (sl_distance * pip_value * 10000)
    """
    if sl_distance <= 0:
        return 0.01  # mínimo

    risk_amount = capital * risk_pct
    pv          = pip_value(pair)
    pips        = sl_distance * 10_000 if "JPY" not in pair else sl_distance * 100
    lot         = risk_amount / (pips * pv)
    lot         = max(0.01, min(round(lot, 2), 10.0))  # entre 0.01 y 10 lotes
    return lot


def generate_signal(
    pair:        str,
    timeframe:   str,
    capital:     float = DEFAULT_CAPITAL,
    xgb_model   = None,
    lstm_model   = None,
    lstm_scaler  = None,
    lstm_features= None,
) -> TradeSignal | None:
    """
    Genera una señal de trading para el par/timeframe dado.
    Devuelve None si no hay señal o no supera los filtros de calidad.
    """
    df = get_latest_candles(pair, timeframe, n=300)
    if len(df) < 100:
        logger.warning(f"Insuficientes velas para {pair} {timeframe}")
        return None

    signal = get_latest_signal(
        df, pair, timeframe,
        xgb_model, lstm_model, lstm_scaler, lstm_features
    )

    if signal.direction == 0:
        return None   # sin señal clara

    # Precio actual y ATR
    last_row    = df.iloc[-1]
    entry_price = float(last_row["close"])
    atr         = float(last_row["atr_14"]) if "atr_14" in df.columns else entry_price * 0.001

    if signal.direction == 1:   # long
        tp_price = entry_price + TP_ATR_MULT * atr
        sl_price = entry_price - SL_ATR_MULT * atr
    else:                        # short
        tp_price = entry_price - TP_ATR_MULT * atr
        sl_price = entry_price + SL_ATR_MULT * atr

    sl_distance = abs(entry_price - sl_price)
    lot_size    = calculate_lot_size(pair, sl_distance, capital)
    risk_amount = round(capital * RISK_PER_TRADE, 2)

    ts = TradeSignal(
        signal_id    = str(uuid.uuid4())[:8],
        pair         = pair,
        timeframe    = timeframe,
        timestamp    = str(signal.timestamp or datetime.utcnow()),
        direction    = signal.direction,
        entry_price  = round(entry_price, 5),
        tp_price     = round(tp_price, 5),
        sl_price     = round(sl_price, 5),
        lot_size     = lot_size,
        risk_amount  = risk_amount,
        confidence   = round(signal.confidence, 4),
        prob_long    = round(signal.prob_long, 4),
        prob_short   = round(signal.prob_short, 4),
        prob_neutral = round(signal.prob_neutral, 4),
        agreement    = signal.agreement,
    )

    return ts


def save_signal(signal: TradeSignal, engine=None) -> None:
    """Guarda la señal en signals_log y la posición en positions_active."""
    if engine is None:
        engine = create_engine(DATABASE_URL)

    d = asdict(signal)

    sql_signal = text("""
        INSERT INTO signals_log
            (signal_id, pair, timeframe, timestamp, direction,
             entry_price, tp_price, sl_price, lot_size, risk_amount,
             confidence, prob_long, prob_short, prob_neutral, agreement, status)
        VALUES
            (:signal_id, :pair, :timeframe, :timestamp, :direction,
             :entry_price, :tp_price, :sl_price, :lot_size, :risk_amount,
             :confidence, :prob_long, :prob_short, :prob_neutral, :agreement, :status)
        ON CONFLICT (signal_id) DO NOTHING
    """)

    sql_position = text("""
        INSERT INTO positions_active
            (signal_id, pair, timeframe, direction, entry_price,
             tp_price, sl_price, lot_size, opened_at, status)
        VALUES
            (:signal_id, :pair, :timeframe, :direction, :entry_price,
             :tp_price, :sl_price, :lot_size, :timestamp, 'open')
        ON CONFLICT (signal_id) DO NOTHING
    """)

    with engine.connect() as conn:
        conn.execute(sql_signal, d)
        conn.execute(sql_position, d)
        conn.commit()

    logger.info(
        f"✅ Señal guardada: {signal.pair} {signal.timeframe} "
        f"{'LONG' if signal.direction == 1 else 'SHORT'} "
        f"@ {signal.entry_price}  TP: {signal.tp_price}  SL: {signal.sl_price}  "
        f"Conf: {signal.confidence:.2%}"
    )


def update_positions(engine=None) -> list[dict]:
    """
    Revisa las posiciones abiertas y actualiza las que tocaron TP/SL.
    Devuelve lista de posiciones cerradas.
    Debe ejecutarse en cada ciclo del bot.
    """
    if engine is None:
        engine = create_engine(DATABASE_URL)

    positions = pd.read_sql(
        "SELECT * FROM positions_active WHERE status = 'open'",
        engine
    )
    closed = []

    for _, pos in positions.iterrows():
        pair      = pos["pair"]
        timeframe = pos["timeframe"]
        direction = int(pos["direction"])
        tp_price  = float(pos["tp_price"])
        sl_price  = float(pos["sl_price"])
        signal_id = pos["signal_id"]
        lot_size  = float(pos["lot_size"])
        entry     = float(pos["entry_price"])

        # Obtener última vela
        df = get_latest_candles(pair, timeframe, n=5)
        if df.empty:
            continue

        last       = df.iloc[-1]
        high_price = float(last["high"])
        low_price  = float(last["low"])
        status     = None
        pnl        = 0.0

        if direction == 1:   # long
            if high_price >= tp_price:
                status = "tp_hit"
                pnl    = (tp_price - entry) * lot_size * 100_000
            elif low_price <= sl_price:
                status = "sl_hit"
                pnl    = (sl_price - entry) * lot_size * 100_000
        else:                # short
            if low_price <= tp_price:
                status = "tp_hit"
                pnl    = (entry - tp_price) * lot_size * 100_000
            elif high_price >= sl_price:
                status = "sl_hit"
                pnl    = (entry - sl_price) * lot_size * 100_000

        if status:
            closed_at = str(last["timestamp"])
            with engine.connect() as conn:
                conn.execute(text("""
                    UPDATE positions_active
                    SET status = :status, closed_at = :closed_at, pnl = :pnl
                    WHERE signal_id = :signal_id
                """), {"status": status, "closed_at": closed_at,
                       "pnl": round(pnl, 2), "signal_id": signal_id})
                conn.execute(text("""
                    UPDATE signals_log
                    SET status = :status, result_pnl = :pnl, closed_at = :closed_at
                    WHERE signal_id = :signal_id
                """), {"status": status, "pnl": round(pnl, 2),
                       "closed_at": closed_at, "signal_id": signal_id})
                conn.commit()

            closed.append({
                "signal_id": signal_id,
                "pair":      pair,
                "status":    status,
                "pnl":       round(pnl, 2),
            })
            logger.info(f"Posición cerrada: {pair} {status}  PnL: {pnl:.2f}€")

    return closed


def run_signal_scan(capital: float = DEFAULT_CAPITAL) -> list[TradeSignal]:
    """
    Escanea todos los pares activos y genera señales.
    Punto de entrada principal del bot en cada ciclo.
    """
    engine  = create_engine(DATABASE_URL)
    signals = []

    # Primero actualizar posiciones abiertas
    closed = update_positions(engine)
    if closed:
        logger.info(f"Posiciones cerradas en este ciclo: {len(closed)}")

    # Generar nuevas señales
    for pair in ACTIVE_PAIRS:
        for tf in ACTIVE_TIMEFRAMES:
            try:
                signal = generate_signal(pair, tf, capital)
                if signal:
                    save_signal(signal, engine)
                    signals.append(signal)
            except Exception as e:
                logger.error(f"Error generando señal {pair} {tf}: {e}")

    logger.info(f"Ciclo completado. Señales generadas: {len(signals)}")
    return signals


if __name__ == "__main__":
    signals = run_signal_scan()
    for s in signals:
        direction = "LONG" if s.direction == 1 else "SHORT"
        logger.info(
            f"{s.pair} {s.timeframe} {direction} "
            f"@ {s.entry_price}  TP:{s.tp_price}  SL:{s.sl_price}  "
            f"Conf:{s.confidence:.2%}  Lotes:{s.lot_size}"
        )
