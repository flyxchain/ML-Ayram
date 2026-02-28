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
    Etiqueta filas de features_computed que aún no tienen label.
    Modo incremental: solo procesa filas con label IS NULL.
    Updates directos en batches pequeños (óptimo para Supabase remoto).
    """
    engine = create_engine(DATABASE_URL)

    pairs      = ["EURUSD", "GBPUSD", "USDJPY", "EURJPY", "XAUUSD"]
    timeframes = ["M15", "H1", "H4"]

    for pair in pairs:
        for tf in timeframes:
            logger.info(f"Etiquetando {pair} {tf}  TP={tp}×ATR  SL={sl}×ATR  H={horizon}")

            # Contar filas sin etiquetar
            with engine.connect() as conn:
                pending = conn.execute(
                    text("SELECT COUNT(*) FROM features_computed WHERE pair = :p AND timeframe = :tf AND label IS NULL"),
                    {"p": pair, "tf": tf},
                ).scalar()

            if pending == 0:
                logger.info(f"  Sin filas pendientes para {pair} {tf}")
                continue

            logger.info(f"  {pending} filas pendientes de etiquetar")

            # Cargar TODO el par/tf (necesitamos lookforward para Triple Barrier)
            # pero solo actualizaremos las filas sin label
            df = pd.read_sql(
                text("""
                    SELECT fc.pair, fc.timeframe, fc.timestamp, fc.atr_14, fc.label,
                           r.open, r.high, r.low, r.close
                    FROM features_computed fc
                    JOIN ohlcv_raw r
                      ON r.pair = fc.pair
                     AND r.timeframe = fc.timeframe
                     AND r.timestamp = fc.timestamp
                    WHERE fc.pair = :pair AND fc.timeframe = :tf
                    ORDER BY fc.timestamp
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

            # Recordar cuáles no tenían label antes del cálculo
            unlabeled_mask = df["label"].isna()

            df = compute_triple_barrier(df, tp, sl, horizon)

            # Solo las filas que eran NULL
            new_labels = df[unlabeled_mask].copy()
            if new_labels.empty:
                logger.info(f"  Nada nuevo que etiquetar para {pair} {tf}")
                continue

            stats = label_stats(new_labels)
            logger.info(
                f"  +1: {stats['label_1_pct']}%  "
                f"0: {stats['label_0_pct']}%  "
                f"-1: {stats['label_m1_pct']}%  "
                f"Avg win: {stats['avg_return_win']}%"
            )

            # ── Updates directos en batches pequeños ──────────────────────
            update_sql = text("""
                UPDATE features_computed
                SET label        = :label,
                    label_return = :label_return,
                    bars_to_exit = :bars_to_exit,
                    tp_price     = :tp_price,
                    sl_price     = :sl_price
                WHERE pair = :pair AND timeframe = :timeframe AND timestamp = :timestamp
            """)

            records = new_labels[["pair", "timeframe", "timestamp",
                                  "label", "label_return", "bars_to_exit",
                                  "tp_price", "sl_price"]].copy()
            records["timestamp"] = records["timestamp"].astype(str)

            # Convertir tipos numpy a Python nativos
            rows = records.to_dict("records")
            for row in rows:
                for k, v in row.items():
                    if isinstance(v, (np.integer,)):
                        row[k] = int(v)
                    elif isinstance(v, (np.floating,)):
                        row[k] = float(v) if not np.isnan(v) else None

            batch_size = 200
            total_rows = len(rows)
            updated = 0

            for start in range(0, total_rows, batch_size):
                batch = rows[start : start + batch_size]
                with engine.connect() as conn:
                    conn.execute(update_sql, batch)
                    conn.commit()
                updated += len(batch)
                if total_rows > batch_size:
                    logger.info(f"    Progreso: {updated}/{total_rows} ({updated*100//total_rows}%)")

            logger.success(f"  ✅ {pair} {tf}: {updated} filas etiquetadas")


if __name__ == "__main__":
    run_labeling()
