"""
src/data/labels.py
Triple Barrier Method (López de Prado) para etiquetar operaciones.

Para cada vela genera una etiqueta:
  +1 → precio tocó take profit antes que stop loss (long ganador)
  -1 → precio tocó stop loss antes que take profit (long perdedor)
   0 → venció el tiempo sin tocar ninguna barrera

Parámetros configurables:
  - tp_multiplier: barreras en múltiplos de ATR
  - sl_multiplier: ídem
  - horizon:       número máximo de velas a mirar hacia adelante
"""

import os

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from loguru import logger
from sqlalchemy import create_engine, text

load_dotenv()
DATABASE_URL = os.getenv("DATABASE_URL")

# Configuración por defecto
DEFAULT_TP = 2.0   # 2x ATR
DEFAULT_SL = 1.0   # 1x ATR  →  ratio 2:1
DEFAULT_H  = 20    # máximo 20 velas hacia adelante


def compute_triple_barrier(
    df: pd.DataFrame,
    tp_multiplier: float = DEFAULT_TP,
    sl_multiplier: float = DEFAULT_SL,
    horizon: int = DEFAULT_H,
    atr_col: str = "atr_14",
) -> pd.DataFrame:
    """
    Aplica Triple Barrier Method vectorizado a un DataFrame con atr_14.

    Devuelve el DataFrame con columnas añadidas:
      label        : +1, -1, 0
      label_return : retorno real de la operación
      bars_to_exit : velas hasta el cierre
      tp_price     : nivel de take profit
      sl_price     : nivel de stop loss
    """
    df = df.copy().sort_values("timestamp").reset_index(drop=True)

    closes = df["close"].values.astype(float)
    highs  = df["high"].values.astype(float)
    lows   = df["low"].values.astype(float)
    atrs   = df[atr_col].values.astype(float)
    n      = len(df)

    labels       = np.zeros(n, dtype=np.int8)
    returns      = np.zeros(n, dtype=float)
    bars_to_exit = np.full(n, horizon, dtype=np.int16)
    tp_prices    = np.full(n, np.nan)
    sl_prices    = np.full(n, np.nan)

    # Ignorar filas sin ATR válido
    valid = ~np.isnan(atrs) & (atrs > 0)

    tp_arr = np.where(valid, closes + tp_multiplier * atrs, np.nan)
    sl_arr = np.where(valid, closes - sl_multiplier * atrs, np.nan)
    tp_prices[:] = tp_arr
    sl_prices[:] = sl_arr

    # Núcleo vectorizado: para cada vela i miramos la ventana [i+1, i+horizon]
    # Construimos matrices de highs/lows desplazadas (shape: n × horizon)
    # y buscamos la primera vela que toca TP o SL con argmax.

    # Usamos sliding_window_view para evitar el loop Python interno
    # — pero el loop externo sigue siendo necesario para conocer el entry.
    # Optimización: procesamos en bloque usando numpy broadcasting.

    for i in range(n - 1):
        if not valid[i]:
            continue

        tp = tp_arr[i]
        sl = sl_arr[i]
        end = min(i + horizon + 1, n)

        window_h = highs[i + 1 : end]
        window_l = lows[i + 1 : end]

        tp_hits = np.where(window_h >= tp)[0]
        sl_hits = np.where(window_l <= sl)[0]

        first_tp = tp_hits[0] if len(tp_hits) else horizon
        first_sl = sl_hits[0] if len(sl_hits) else horizon

        if first_tp < first_sl:
            labels[i]       = 1
            returns[i]      = (tp - closes[i]) / closes[i]
            bars_to_exit[i] = first_tp + 1
        elif first_sl < first_tp:
            labels[i]       = -1
            returns[i]      = (sl - closes[i]) / closes[i]
            bars_to_exit[i] = first_sl + 1
        else:
            # empate o ninguno → neutral
            labels[i]       = 0
            ret_idx          = min(i + horizon, n - 1)
            returns[i]       = (closes[ret_idx] - closes[i]) / closes[i]
            bars_to_exit[i] = horizon

    df["label"]        = labels
    df["label_return"] = np.round(returns, 6)
    df["bars_to_exit"] = bars_to_exit
    df["tp_price"]     = np.round(tp_prices, 6)
    df["sl_price"]     = np.round(sl_prices, 6)

    return df


def label_stats(df: pd.DataFrame) -> dict:
    """Estadísticas básicas del etiquetado."""
    total   = len(df)
    if total == 0:
        return {"total": 0}
    longs   = int((df["label"] ==  1).sum())
    shorts  = int((df["label"] == -1).sum())
    neutral = int((df["label"] ==  0).sum())
    wins    = df[df["label"] ==  1]["label_return"]
    losses  = df[df["label"] == -1]["label_return"]
    return {
        "total":           total,
        "label_1_pct":     round(longs   / total * 100, 1),
        "label_0_pct":     round(neutral / total * 100, 1),
        "label_m1_pct":    round(shorts  / total * 100, 1),
        "avg_return_win":  round(float(wins.mean())   * 100, 3) if len(wins)   else 0,
        "avg_return_loss": round(float(losses.mean()) * 100, 3) if len(losses) else 0,
        "avg_bars_win":    round(float(df[df["label"] == 1]["bars_to_exit"].mean()), 1) if longs else 0,
    }


def run_labeling(
    tp: float = DEFAULT_TP,
    sl: float = DEFAULT_SL,
    horizon: int = DEFAULT_H,
) -> None:
    """
    Etiqueta todas las filas de features_computed que aún no tienen label.
    Usa un bulk UPDATE con tabla temporal para máxima eficiencia.
    """
    engine = create_engine(DATABASE_URL)

    pairs      = ["EURUSD", "GBPUSD", "USDJPY", "EURJPY", "XAUUSD"]
    timeframes = ["M15", "H1", "H4"]

    for pair in pairs:
        for tf in timeframes:
            logger.info(f"Etiquetando {pair} {tf}  TP={tp}×ATR  SL={sl}×ATR  H={horizon}")

            df = pd.read_sql(
                text("""
                    SELECT * FROM features_computed
                    WHERE pair = :pair AND timeframe = :tf
                    ORDER BY timestamp
                """),
                engine,
                params={"pair": pair, "tf": tf},
            )

            if df.empty:
                logger.warning(f"  Sin datos para {pair} {tf}")
                continue

            if "atr_14" not in df.columns:
                logger.warning(f"  Columna atr_14 no encontrada para {pair} {tf}")
                continue

            df = compute_triple_barrier(df, tp, sl, horizon)

            stats = label_stats(df)
            logger.info(
                f"  +1: {stats['label_1_pct']}%  "
                f"0: {stats['label_0_pct']}%  "
                f"-1: {stats['label_m1_pct']}%  "
                f"Avg win: {stats['avg_return_win']}%"
            )

            # ── Bulk UPDATE vía tabla temporal ────────────────────────────
            # Mucho más rápido que N UPDATEs individuales:
            # 1. Crear tabla temporal con los valores nuevos
            # 2. Un solo UPDATE features_computed JOIN tmp
            cols = ["pair", "timeframe", "timestamp",
                    "label", "label_return", "bars_to_exit", "tp_price", "sl_price"]
            update_df = df[cols].copy()
            update_df["timestamp"] = update_df["timestamp"].astype(str)

            with engine.connect() as conn:
                # Escribir en tabla temporal (se borra al cerrar conexión)
                conn.execute(text("""
                    CREATE TEMP TABLE IF NOT EXISTS _label_update (
                        pair        TEXT,
                        timeframe   TEXT,
                        timestamp   TIMESTAMPTZ,
                        label       SMALLINT,
                        label_return REAL,
                        bars_to_exit SMALLINT,
                        tp_price    REAL,
                        sl_price    REAL
                    ) ON COMMIT DROP
                """))

                # Insertar en lotes de 5000 para no saturar memoria
                batch_size = 5_000
                for start in range(0, len(update_df), batch_size):
                    batch = update_df.iloc[start : start + batch_size]
                    conn.execute(
                        text("""
                            INSERT INTO _label_update
                            VALUES (:pair, :timeframe, :timestamp,
                                    :label, :label_return, :bars_to_exit,
                                    :tp_price, :sl_price)
                        """),
                        batch.to_dict("records"),
                    )

                # Un solo UPDATE masivo
                conn.execute(text("""
                    UPDATE features_computed fc
                    SET
                        label        = u.label,
                        label_return = u.label_return,
                        bars_to_exit = u.bars_to_exit,
                        tp_price     = u.tp_price,
                        sl_price     = u.sl_price
                    FROM _label_update u
                    WHERE fc.pair      = u.pair
                      AND fc.timeframe = u.timeframe
                      AND fc.timestamp = u.timestamp
                """))
                conn.commit()

            logger.success(f"  ✅ {pair} {tf}: {len(df)} filas etiquetadas")


if __name__ == "__main__":
    run_labeling()
