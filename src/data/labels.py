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

import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text
from dotenv import load_dotenv
from loguru import logger
import os

load_dotenv()
DATABASE_URL = os.getenv("DATABASE_URL")

# Configuración por defecto
DEFAULT_TP  = 2.0   # 2x ATR
DEFAULT_SL  = 1.0   # 1x ATR (ratio 2:1)
DEFAULT_H   = 20    # máximo 20 velas hacia adelante


def compute_triple_barrier(
    df: pd.DataFrame,
    tp_multiplier: float = DEFAULT_TP,
    sl_multiplier: float = DEFAULT_SL,
    horizon: int = DEFAULT_H,
    atr_col: str = "atr_14",
) -> pd.DataFrame:
    """
    Aplica Triple Barrier Method a un DataFrame que ya tiene la columna atr_14.
    Devuelve el DataFrame con columnas añadidas:
      - label:       +1, -1, 0
      - label_return: retorno real de la operación
      - bars_to_exit: velas hasta que se cerró
      - tp_price:    nivel de take profit
      - sl_price:    nivel de stop loss
    """
    df = df.copy().sort_values("timestamp").reset_index(drop=True)

    labels       = np.zeros(len(df), dtype=int)
    returns      = np.zeros(len(df), dtype=float)
    bars_to_exit = np.full(len(df), horizon, dtype=int)
    tp_prices    = np.full(len(df), np.nan)
    sl_prices    = np.full(len(df), np.nan)

    closes = df["close"].values
    highs  = df["high"].values
    lows   = df["low"].values
    atrs   = df[atr_col].values

    n = len(df)

    for i in range(n - 1):
        atr = atrs[i]
        if np.isnan(atr) or atr <= 0:
            continue

        entry = closes[i]
        tp    = entry + tp_multiplier * atr
        sl    = entry - sl_multiplier * atr

        tp_prices[i] = tp
        sl_prices[i] = sl

        # Buscar en las siguientes `horizon` velas
        end = min(i + horizon + 1, n)
        hit_tp = False
        hit_sl = False

        for j in range(i + 1, end):
            if highs[j] >= tp:
                hit_tp = True
                bars_to_exit[i] = j - i
                break
            if lows[j] <= sl:
                hit_sl = True
                bars_to_exit[i] = j - i
                break

        if hit_tp:
            labels[i]  = 1
            returns[i] = (tp - entry) / entry
        elif hit_sl:
            labels[i]  = -1
            returns[i] = (sl - entry) / entry
        else:
            labels[i]  = 0
            ret_close   = closes[min(i + horizon, n - 1)]
            returns[i]  = (ret_close - entry) / entry

    df["label"]        = labels
    df["label_return"] = returns
    df["bars_to_exit"] = bars_to_exit
    df["tp_price"]     = tp_prices
    df["sl_price"]     = sl_prices

    return df


def label_stats(df: pd.DataFrame) -> dict:
    """Estadísticas básicas del etiquetado."""
    total = len(df)
    longs = (df["label"] == 1).sum()
    shorts = (df["label"] == -1).sum()
    neutral = (df["label"] == 0).sum()
    return {
        "total":        total,
        "label_1_pct":  round(longs   / total * 100, 1),
        "label_0_pct":  round(neutral / total * 100, 1),
        "label_m1_pct": round(shorts  / total * 100, 1),
        "avg_return_win":  round(df[df["label"]==1]["label_return"].mean() * 100, 3),
        "avg_return_loss": round(df[df["label"]==-1]["label_return"].mean() * 100, 3),
        "avg_bars_win":    round(df[df["label"]==1]["bars_to_exit"].mean(), 1),
    }


def run_labeling(
    tp: float = DEFAULT_TP,
    sl: float = DEFAULT_SL,
    horizon: int = DEFAULT_H,
):
    """
    Añade labels a todas las filas de features_computed que aún no las tienen.
    Se puede relanzar con distintos parámetros para experimentar.
    """
    engine = create_engine(DATABASE_URL)

    pairs = ["EURUSD", "GBPUSD", "USDJPY", "EURJPY", "XAUUSD"]
    timeframes = ["M15", "H1", "H4"]  # D1 no se etiqueta (muy largo plazo)

    for pair in pairs:
        for tf in timeframes:
            logger.info(f"Etiquetando: {pair} {tf}  TP={tp}xATR  SL={sl}xATR  H={horizon}")

            df = pd.read_sql(
                f"""SELECT * FROM features_computed
                    WHERE pair='{pair}' AND timeframe='{tf}'
                    ORDER BY timestamp""",
                engine
            )

            if df.empty:
                logger.warning(f"  Sin datos en features_computed para {pair} {tf}")
                continue

            if "atr_14" not in df.columns:
                logger.warning(f"  Columna atr_14 no encontrada para {pair} {tf}")
                continue

            df = compute_triple_barrier(df, tp, sl, horizon)

            stats = label_stats(df)
            logger.info(
                f"  Distribución → +1: {stats['label_1_pct']}%  "
                f"0: {stats['label_0_pct']}%  "
                f"-1: {stats['label_m1_pct']}%  "
                f"Avg win: {stats['avg_return_win']}%"
            )

            # Actualizar labels en BD
            rows = df[["pair", "timeframe", "timestamp", "label", "label_return",
                        "bars_to_exit", "tp_price", "sl_price"]].to_dict("records")

            sql = text("""
                UPDATE features_computed
                SET
                    label        = :label,
                    label_return = :label_return,
                    bars_to_exit = :bars_to_exit,
                    tp_price     = :tp_price,
                    sl_price     = :sl_price
                WHERE pair = :pair AND timeframe = :timeframe AND timestamp = :timestamp
            """)

            with engine.connect() as conn:
                conn.execute(sql, rows)
                conn.commit()

            logger.success(f"  ✅ {pair} {tf}: {len(df)} filas etiquetadas")


if __name__ == "__main__":
    run_labeling()
