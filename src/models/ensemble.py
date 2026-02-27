"""
src/models/ensemble.py
Combina las predicciones de XGBoost y LSTM con votación ponderada.
Solo genera señal cuando ambos modelos coinciden con suficiente confianza.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from loguru import logger

from src.models.xgboost_model import (
    load_model as load_xgb,
    predict_proba as xgb_proba,
    FEATURE_COLS as XGB_FEATURES,
)
from src.models.lstm_model import (
    load_model as load_lstm,
    predict_proba as lstm_proba,
)

# Peso de cada modelo en la votación final (deben sumar 1.0)
XGB_WEIGHT  = 0.55
LSTM_WEIGHT = 0.45

# Umbrales mínimos para emitir señal
MIN_CONFIDENCE_LONG  = 0.52
MIN_CONFIDENCE_SHORT = 0.52
MIN_AGREEMENT        = 0.15   # diferencia mínima entre mejor clase y segunda


@dataclass
class Signal:
    pair:          str
    timeframe:     str
    timestamp:     object
    direction:     int
    confidence:    float
    prob_long:     float
    prob_neutral:  float
    prob_short:    float
    xgb_direction: int
    lstm_direction: int
    agreement:     bool


def load_models(pair: str, timeframe: str) -> tuple:
    """Carga XGBoost y LSTM para un par/timeframe."""
    xgb_model = load_xgb(pair, timeframe)
    lstm_model, lstm_scaler, lstm_features = load_lstm(pair, timeframe)
    return xgb_model, lstm_model, lstm_scaler, lstm_features


def ensemble_predict(
    df:            pd.DataFrame,
    pair:          str,
    timeframe:     str,
    xgb_model=None,
    lstm_model=None,
    lstm_scaler=None,
    lstm_features=None,
    xgb_weight:    float = XGB_WEIGHT,
    lstm_weight:   float = LSTM_WEIGHT,
) -> pd.DataFrame:
    """
    Combina predicciones de ambos modelos con votación ponderada vectorizada.
    Devuelve DataFrame con columnas de probabilidad y señal final.
    """
    if xgb_model is None:
        xgb_model, lstm_model, lstm_scaler, lstm_features = load_models(pair, timeframe)

    # Probabilidades de cada modelo
    xgb_p  = xgb_proba(xgb_model, df).reset_index(drop=True)
    lstm_p = lstm_proba(lstm_model, lstm_scaler, lstm_features, df).reset_index(drop=True)

    n_xgb  = len(xgb_p)
    n_lstm = len(lstm_p)
    offset = n_xgb - n_lstm   # primeras filas sin predicción LSTM (warm-up)

    # ── Combinación vectorizada ───────────────────────────────────────────
    # Arrays XGB completos
    xgb_long    = xgb_p["prob_long"].values
    xgb_neutral = xgb_p["prob_neutral"].values
    xgb_short   = xgb_p["prob_short"].values

    # Arrays de resultado inicializados a NaN
    prob_long    = np.full(n_xgb, np.nan)
    prob_neutral = np.full(n_xgb, np.nan)
    prob_short   = np.full(n_xgb, np.nan)

    # Solo las filas donde LSTM tiene predicción
    lstm_long    = lstm_p["prob_long"].values
    lstm_neutral = lstm_p["prob_neutral"].values
    lstm_short   = lstm_p["prob_short"].values

    prob_long[offset:]    = xgb_weight * xgb_long[offset:]    + lstm_weight * lstm_long
    prob_neutral[offset:] = xgb_weight * xgb_neutral[offset:] + lstm_weight * lstm_neutral
    prob_short[offset:]   = xgb_weight * xgb_short[offset:]   + lstm_weight * lstm_short

    # Dirección de cada modelo individual (argmax sobre [short, neutral, long] → -1, 0, 1)
    xgb_dirs  = xgb_p[["prob_short", "prob_neutral", "prob_long"]].values.argmax(axis=1) - 1
    lstm_dirs = lstm_p[["prob_short", "prob_neutral", "prob_long"]].values.argmax(axis=1) - 1

    lstm_dirs_full = np.full(n_xgb, np.nan)
    lstm_dirs_full[offset:] = lstm_dirs

    # ── Señal final vectorizada ───────────────────────────────────────────
    # Confianza = probabilidad de la clase más probable
    stack = np.column_stack([prob_long, prob_neutral, prob_short])
    confidence = np.nanmax(stack, axis=1)

    # Margen = diferencia entre primera y segunda clase más probable
    sorted_stack = np.sort(stack, axis=1)[:, ::-1]   # desc
    margin = sorted_stack[:, 0] - sorted_stack[:, 1]

    # Acuerdo: ambos modelos predicen la misma dirección y LSTM tiene predicción
    agreement = (xgb_dirs == lstm_dirs_full) & ~np.isnan(lstm_dirs_full)

    signals = np.zeros(n_xgb, dtype=int)
    valid   = ~np.isnan(prob_long)

    long_mask  = valid & (prob_long  >= MIN_CONFIDENCE_LONG)  & (margin >= MIN_AGREEMENT) & agreement
    short_mask = valid & (prob_short >= MIN_CONFIDENCE_SHORT) & (margin >= MIN_AGREEMENT) & agreement

    signals[long_mask]  =  1
    signals[short_mask] = -1

    # ── Construir resultado ───────────────────────────────────────────────
    result = pd.DataFrame({
        "prob_long":      prob_long,
        "prob_neutral":   prob_neutral,
        "prob_short":     prob_short,
        "xgb_direction":  xgb_dirs,
        "lstm_direction": lstm_dirs_full,
        "signal":         signals,
        "confidence":     confidence,
        "agreement":      agreement,
        "pair":           pair,
        "timeframe":      timeframe,
    })

    if "timestamp" in df.columns:
        result.insert(0, "timestamp", df["timestamp"].reset_index(drop=True))

    return result


def get_latest_signal(
    df:          pd.DataFrame,
    pair:        str,
    timeframe:   str,
    xgb_model=None,
    lstm_model=None,
    lstm_scaler=None,
    lstm_features=None,
) -> Signal:
    """Genera la señal para la vela más reciente del DataFrame."""
    result = ensemble_predict(
        df, pair, timeframe,
        xgb_model, lstm_model, lstm_scaler, lstm_features,
    )
    last = result.iloc[-1]

    # lstm_direction puede ser NaN en las primeras seq_len velas
    _lstm_raw = last["lstm_direction"]
    lstm_dir  = int(_lstm_raw) if not np.isnan(_lstm_raw) else 0

    return Signal(
        pair           = pair,
        timeframe      = timeframe,
        timestamp      = last.get("timestamp") if "timestamp" in last else None,
        direction      = int(last["signal"]),
        confidence     = float(last["confidence"]),
        prob_long      = float(last["prob_long"]),
        prob_neutral   = float(last["prob_neutral"]),
        prob_short     = float(last["prob_short"]),
        xgb_direction  = int(last["xgb_direction"]),
        lstm_direction = lstm_dir,
        agreement      = bool(last["agreement"]),
    )


def signal_stats(result: pd.DataFrame) -> dict:
    """Estadísticas de calidad de señales en un período."""
    valid   = result.dropna(subset=["signal"])
    total   = len(valid)
    if total == 0:
        return {"total_bars": 0}
    longs   = int((valid["signal"] ==  1).sum())
    shorts  = int((valid["signal"] == -1).sum())
    agreed  = int(valid["agreement"].sum())
    return {
        "total_bars":      total,
        "signals_long":    longs,
        "signals_short":   shorts,
        "signals_neutral": total - longs - shorts,
        "signal_rate":     round((longs + shorts) / total * 100, 2),
        "agreement_rate":  round(agreed / total * 100, 2),
        "avg_confidence":  round(float(valid["confidence"].mean()), 4),
    }


if __name__ == "__main__":
    from sqlalchemy import create_engine
    from dotenv import load_dotenv
    import os
    load_dotenv()

    engine = create_engine(os.getenv("DATABASE_URL"))
    df = pd.read_sql(
        text("""
            SELECT * FROM features_computed
            WHERE pair = 'EURUSD' AND timeframe = 'H1'
              AND label IS NOT NULL
            ORDER BY timestamp DESC LIMIT 200
        """),
        engine,
    ).sort_values("timestamp").reset_index(drop=True)

    signal = get_latest_signal(df, "EURUSD", "H1")
    logger.info(f"Señal actual EURUSD H1: {signal}")
