"""
src/execution/position_manager.py
Gestión de posiciones simuladas: sizing, apertura, seguimiento y cierre.

Consume señales de src/signals/generator.py (tabla `signals`)
y mantiene el estado en las tablas `positions_active` y `trades_history`.

Las tablas se crean automáticamente si no existen.
"""

import os
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from typing import Optional

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from loguru import logger
from sqlalchemy import create_engine, text

load_dotenv()
DATABASE_URL = os.getenv("DATABASE_URL")

# ── Configuración de riesgo ───────────────────────────────────────────────────

RISK_PER_TRADE  = 0.01      # 1% del capital por operación
DEFAULT_CAPITAL = 10_000.0  # capital simulado inicial (€)
MAX_LOT         = 10.0
MIN_LOT         = 0.01

# Pip size por par (precio de 1 pip)
PIP_SIZE = {
    "EURUSD": 0.0001,
    "GBPUSD": 0.0001,
    "USDJPY": 0.01,
    "EURJPY": 0.01,
    "XAUUSD": 0.10,     # oro: 1 pip = 0.10 USD
}

# Valor de 1 pip por lote estándar en USD
PIP_VALUE_PER_LOT = {
    "EURUSD": 10.0,
    "GBPUSD": 10.0,
    "USDJPY":  9.1,
    "EURJPY":  9.1,
    "XAUUSD": 10.0,
}


# ── Dataclass de posición ─────────────────────────────────────────────────────

@dataclass
class Position:
    signal_id:   int       # FK a tabla signals
    pair:        str
    timeframe:   str
    direction:   int       # +1 long / -1 short
    entry_price: float
    tp_price:    float
    sl_price:    float
    lot_size:    float
    risk_amount: float     # € en riesgo
    opened_at:   str
    status:      str = "open"    # open | tp_hit | sl_hit | expired
    closed_at:   Optional[str] = None
    pnl:         Optional[float] = None


# ── SQL ───────────────────────────────────────────────────────────────────────

CREATE_POSITIONS_TABLE = """
CREATE TABLE IF NOT EXISTS positions_active (
    id          SERIAL PRIMARY KEY,
    signal_id   INT UNIQUE NOT NULL,
    pair        TEXT NOT NULL,
    timeframe   TEXT NOT NULL,
    direction   SMALLINT NOT NULL,
    entry_price REAL NOT NULL,
    tp_price    REAL NOT NULL,
    sl_price    REAL NOT NULL,
    lot_size    REAL NOT NULL,
    risk_amount REAL NOT NULL,
    opened_at   TIMESTAMPTZ NOT NULL,
    status      TEXT NOT NULL DEFAULT 'open',
    closed_at   TIMESTAMPTZ,
    pnl         REAL
);
"""

CREATE_TRADES_TABLE = """
CREATE TABLE IF NOT EXISTS trades_history (
    id          SERIAL PRIMARY KEY,
    signal_id   INT NOT NULL UNIQUE,
    pair        TEXT NOT NULL,
    timeframe   TEXT NOT NULL,
    direction   SMALLINT NOT NULL,
    entry_price REAL NOT NULL,
    exit_price  REAL NOT NULL,
    tp_price    REAL NOT NULL,
    sl_price    REAL NOT NULL,
    lot_size    REAL NOT NULL,
    risk_amount REAL NOT NULL,
    pnl         REAL NOT NULL,
    result      TEXT NOT NULL,   -- 'tp_hit' | 'sl_hit' | 'expired'
    opened_at   TIMESTAMPTZ NOT NULL,
    closed_at   TIMESTAMPTZ NOT NULL,
    duration_bars INT
);
"""


def ensure_tables(engine) -> None:
    """Crea las tablas si no existen."""
    with engine.connect() as conn:
        conn.execute(text(CREATE_POSITIONS_TABLE))
        conn.execute(text(CREATE_TRADES_TABLE))
        conn.commit()


# ── Sizing ────────────────────────────────────────────────────────────────────

def calculate_lot_size(
    pair:        str,
    sl_distance: float,   # en precio (no pips)
    capital:     float = DEFAULT_CAPITAL,
    risk_pct:    float = RISK_PER_TRADE,
) -> tuple[float, float]:
    """
    Calcula el tamaño de posición basado en riesgo fijo.

    Fórmula: lot = (capital × risk_pct) / (sl_pips × pip_value_per_lot)

    Devuelve (lot_size, risk_amount).
    """
    if sl_distance <= 0:
        return MIN_LOT, round(capital * risk_pct, 2)

    pip_size = PIP_SIZE.get(pair, 0.0001)
    pip_val  = PIP_VALUE_PER_LOT.get(pair, 10.0)

    sl_pips     = sl_distance / pip_size
    risk_amount = capital * risk_pct
    lot         = risk_amount / (sl_pips * pip_val)
    lot         = max(MIN_LOT, min(round(lot, 2), MAX_LOT))

    return lot, round(risk_amount, 2)


# ── Apertura de posición ──────────────────────────────────────────────────────

def open_position(signal_row: dict, engine, capital: float = DEFAULT_CAPITAL) -> Optional[Position]:
    """
    Abre una posición para una señal válida de la tabla `signals`.

    signal_row: dict con las columnas de la tabla signals
    """
    signal_id   = signal_row["id"]
    pair        = signal_row["pair"]
    timeframe   = signal_row["timeframe"]
    direction   = int(signal_row["direction"])
    entry_price = float(signal_row["entry_price"])
    tp_price    = float(signal_row["tp_price"])
    sl_price    = float(signal_row["sl_price"])
    opened_at   = str(signal_row.get("timestamp", datetime.now(timezone.utc).isoformat()))

    sl_distance      = abs(entry_price - sl_price)
    lot_size, risk_€ = calculate_lot_size(pair, sl_distance, capital)

    position = Position(
        signal_id   = signal_id,
        pair        = pair,
        timeframe   = timeframe,
        direction   = direction,
        entry_price = entry_price,
        tp_price    = tp_price,
        sl_price    = sl_price,
        lot_size    = lot_size,
        risk_amount = risk_€,
        opened_at   = opened_at,
    )

    sql = text("""
        INSERT INTO positions_active
            (signal_id, pair, timeframe, direction, entry_price,
             tp_price, sl_price, lot_size, risk_amount, opened_at, status)
        VALUES
            (:signal_id, :pair, :timeframe, :direction, :entry_price,
             :tp_price, :sl_price, :lot_size, :risk_amount, :opened_at, 'open')
        ON CONFLICT (signal_id) DO NOTHING
    """)

    with engine.connect() as conn:
        conn.execute(sql, asdict(position))
        conn.commit()

    logger.info(
        f"📂 Posición abierta: {pair} {timeframe} "
        f"{'LONG' if direction == 1 else 'SHORT'} "
        f"@ {entry_price}  TP:{tp_price}  SL:{sl_price}  "
        f"Lotes:{lot_size}  Riesgo:{risk_€}€"
    )
    return position


# ── Actualización de posiciones ───────────────────────────────────────────────

def update_positions(engine) -> list[dict]:
    """
    Compara las posiciones abiertas con los precios recientes en features_computed.
    Cierra las que tocaron TP o SL.
    Devuelve lista de posiciones cerradas con su PnL.
    """
    positions = pd.read_sql(
        "SELECT * FROM positions_active WHERE status = 'open'",
        engine
    )
    if positions.empty:
        return []

    closed = []

    for _, pos in positions.iterrows():
        pair      = pos["pair"]
        timeframe = pos["timeframe"]
        direction = int(pos["direction"])
        tp_price  = float(pos["tp_price"])
        sl_price  = float(pos["sl_price"])
        entry     = float(pos["entry_price"])
        lot_size  = float(pos["lot_size"])
        signal_id = int(pos["signal_id"])
        opened_at = pd.Timestamp(pos["opened_at"])

        # Velas posteriores a la apertura
        recent = pd.read_sql(
            text("""
                SELECT timestamp, high, low, close FROM features_computed
                WHERE pair = :pair AND timeframe = :tf
                  AND timestamp > :opened_at
                ORDER BY timestamp ASC
                LIMIT 50
            """),
            engine,
            params={"pair": pair, "tf": timeframe, "opened_at": str(opened_at)},
        )

        if recent.empty:
            continue

        status     = None
        exit_price = None
        closed_at  = None
        bars_count = 0

        for bars_count, (_, bar) in enumerate(recent.iterrows(), 1):
            high = float(bar["high"])
            low  = float(bar["low"])

            if direction == 1:    # long
                if high >= tp_price:
                    status, exit_price = "tp_hit", tp_price
                    break
                if low <= sl_price:
                    status, exit_price = "sl_hit", sl_price
                    break
            else:                  # short
                if low <= tp_price:
                    status, exit_price = "tp_hit", tp_price
                    break
                if high >= sl_price:
                    status, exit_price = "sl_hit", sl_price
                    break

        if not status:
            continue

        closed_at = str(bar["timestamp"])

        # PnL en €
        pip_size = PIP_SIZE.get(pair, 0.0001)
        pip_val  = PIP_VALUE_PER_LOT.get(pair, 10.0)
        if direction == 1:
            pnl_pips = (exit_price - entry) / pip_size
        else:
            pnl_pips = (entry - exit_price) / pip_size
        pnl = round(pnl_pips * pip_val * lot_size, 2)

        # Actualizar BD
        with engine.connect() as conn:
            conn.execute(text("""
                UPDATE positions_active
                SET status = :status, closed_at = :closed_at, pnl = :pnl
                WHERE signal_id = :signal_id
            """), {"status": status, "closed_at": closed_at,
                   "pnl": pnl, "signal_id": signal_id})

            conn.execute(text("""
                INSERT INTO trades_history
                    (signal_id, pair, timeframe, direction, entry_price, exit_price,
                     tp_price, sl_price, lot_size, risk_amount, pnl, result,
                     opened_at, closed_at, duration_bars)
                VALUES
                    (:signal_id, :pair, :timeframe, :direction, :entry_price, :exit_price,
                     :tp_price, :sl_price, :lot_size, :risk_amount, :pnl, :result,
                     :opened_at, :closed_at, :duration_bars)
                ON CONFLICT (signal_id) DO NOTHING
            """), {
                "signal_id":    signal_id,
                "pair":         pair,
                "timeframe":    timeframe,
                "direction":    direction,
                "entry_price":  entry,
                "exit_price":   exit_price,
                "tp_price":     tp_price,
                "sl_price":     sl_price,
                "lot_size":     lot_size,
                "risk_amount":  float(pos["risk_amount"]),
                "pnl":          pnl,
                "result":       status,
                "opened_at":    str(opened_at),
                "closed_at":    closed_at,
                "duration_bars": bars_count,
            })
            conn.commit()

        emoji = "✅" if status == "tp_hit" else "❌"
        logger.info(f"{emoji} Posición cerrada: {pair} {timeframe} {status}  PnL: {pnl:+.2f}€  ({bars_count} velas)")

        closed.append({
            "signal_id": signal_id,
            "pair":      pair,
            "timeframe": timeframe,
            "status":    status,
            "pnl":       pnl,
        })

    return closed


# ── Stats de rendimiento ──────────────────────────────────────────────────────

def performance_stats(engine, days: int = 30) -> dict:
    """Estadísticas de rendimiento de los últimos N días."""
    df = pd.read_sql(
        text("""
            SELECT * FROM trades_history
            WHERE closed_at >= NOW() - (INTERVAL '1 day' * :days)
            ORDER BY closed_at DESC
        """),
        engine,
        params={"days": days},
    )

    if df.empty:
        return {"trades": 0, "message": "Sin trades en el período"}

    wins   = (df["result"] == "tp_hit").sum()
    losses = (df["result"] == "sl_hit").sum()
    total  = len(df)

    return {
        "trades":       total,
        "wins":         int(wins),
        "losses":       int(losses),
        "win_rate":     round(wins / total * 100, 1) if total else 0,
        "total_pnl":    round(df["pnl"].sum(), 2),
        "avg_win":      round(df[df["result"] == "tp_hit"]["pnl"].mean(), 2) if wins else 0,
        "avg_loss":     round(df[df["result"] == "sl_hit"]["pnl"].mean(), 2) if losses else 0,
        "best_trade":   round(df["pnl"].max(), 2),
        "worst_trade":  round(df["pnl"].min(), 2),
        "avg_duration": round(df["duration_bars"].mean(), 1),
    }


# ── Runner ────────────────────────────────────────────────────────────────────

def run_cycle(capital: float = DEFAULT_CAPITAL) -> dict:
    """
    Un ciclo completo del position manager:
    1. Crea tablas si no existen
    2. Cierra posiciones que tocaron TP/SL
    3. Abre posiciones para señales válidas sin posición abierta
    Devuelve resumen del ciclo.
    """
    engine = create_engine(DATABASE_URL)
    ensure_tables(engine)

    # 1. Cerrar posiciones maduras
    closed = update_positions(engine)

    # 2. Buscar señales válidas sin posición abierta
    new_signals = pd.read_sql(
        text("""
            SELECT s.* FROM signals s
            LEFT JOIN positions_active p ON p.signal_id = s.id
            WHERE s.direction != 0
              AND s.filter_reason IS NULL
              AND p.signal_id IS NULL
            ORDER BY s.created_at DESC
            LIMIT 20
        """),
        engine,
    )

    opened = []
    for _, row in new_signals.iterrows():
        pos = open_position(row.to_dict(), engine, capital)
        if pos:
            opened.append(pos)

    summary = {
        "positions_closed": len(closed),
        "positions_opened": len(opened),
        "pnl_this_cycle":   round(sum(c["pnl"] for c in closed), 2),
    }
    logger.info(f"Ciclo position manager: {summary}")
    return summary


if __name__ == "__main__":
    run_cycle()
