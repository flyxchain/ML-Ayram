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
MIN_CONFIDENCE_LONG  = 0.52   # prob_long  mínima para señal +1
MIN_CONFIDENCE_SHORT = 0.52   # prob_short mínima para señal -1
MIN_AGREEMENT        = 0.15   # diferencia mínima entre mejor clase y segunda


@dataclass
class Signal:
    pair:          str
    timeframe:     str
    timestamp:     object
    direction:     int      # +1 long, -1 short, 0 neutral
    confidence:    float    # probabilidad de la clase predicha
    prob_long:     float
    prob_neutral:  float
    prob_short:    float
    xgb_direction: int
    lstm_direction: int
    agreement:     bool     # True si ambos modelos coinciden


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
    Combina predicciones de ambos modelos con votación ponderada.
    Devuelve DataFrame con columnas de probabilidad y señal final.

    Si los modelos no están cargados, los carga automáticamente.
    """
    if xgb_model is None:
        xgb_model, lstm_model, lstm_scaler, lstm_features = load_models(pair, timeframe)

    # Probabilidades XGBoost
    xgb_p = xgb_proba(xgb_model, df).reset_index(drop=True)

    # Probabilidades LSTM (primeras seq_len filas serán NaN)
    lstm_p = lstm_proba(lstm_model, lstm_scaler, lstm_features, df).reset_index(drop=True)

    # Alinear longitudes (LSTM devuelve menos filas)
    n_xgb  = len(xgb_p)
    n_lstm = len(lstm_p)
    offset = n_xgb - n_lstm   # filas sin predicción LSTM al inicio

    result = df[["timestamp"]].copy().reset_index(drop=True) if "timestamp" in df.columns else pd.DataFrame()

    # Probabilidades ponderadas
    prob_long    = np.full(n_xgb, np.nan)
    prob_neutral = np.full(n_xgb, np.nan)
    prob_short   = np.full(n_xgb, np.nan)

    for i in range(offset, n_xgb):
        j = i - offset
        prob_long[i]    = xgb_weight * xgb_p["prob_long"].iloc[i]    + lstm_weight * lstm_p["prob_long"].iloc[j]
        prob_neutral[i] = xgb_weight * xgb_p["prob_neutral"].iloc[i] + lstm_weight * lstm_p["prob_neutral"].iloc[j]
        prob_short[i]   = xgb_weight * xgb_p["prob_short"].iloc[i]   + lstm_weight * lstm_p["prob_short"].iloc[j]

    result["prob_long"]    = prob_long
    result["prob_neutral"] = prob_neutral
    result["prob_short"]   = prob_short

    # Dirección individual de cada modelo
    result["xgb_direction"]  = xgb_p[["prob_short","prob_neutral","prob_long"]].values.argmax(axis=1) - 1
    lstm_dirs = np.full(n_xgb, np.nan)
    lstm_dirs[offset:] = lstm_p[["prob_short","prob_neutral","prob_long"]].values.argmax(axis=1) - 1
    result["lstm_direction"] = lstm_dirs

    # Señal final con filtros de confianza
    signals   = np.zeros(n_xgb, dtype=int)
    agreement = np.zeros(n_xgb, dtype=bool)

    for i in range(n_xgb):
        pl, pn, ps = prob_long[i], prob_neutral[i], prob_short[i]
        if np.isnan(pl):
            continue

        best  = max(pl, pn, ps)
        probs = sorted([pl, pn, ps], reverse=True)
        margin = probs[0] - probs[1]

        xgb_dir  = result["xgb_direction"].iloc[i]
        lstm_dir = result["lstm_direction"].iloc[i]
        agrees   = (xgb_dir == lstm_dir) and not np.isnan(lstm_dir)
        agreement[i] = agrees

        if pl >= MIN_CONFIDENCE_LONG and margin >= MIN_AGREEMENT and agrees:
            signals[i] = 1
        elif ps >= MIN_CONFIDENCE_SHORT and margin >= MIN_AGREEMENT and agrees:
            signals[i] = -1
        else:
            signals[i] = 0

    result["signal"]      = signals
    result["confidence"]  = np.nanmax(
        np.column_stack([prob_long, prob_neutral, prob_short]), axis=1
    )
    result["agreement"]   = agreement
    result["pair"]        = pair
    result["timeframe"]   = timeframe

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
    """
    Genera la señal para la vela más reciente del DataFrame.
    Devuelve un objeto Signal.
    """
    result = ensemble_predict(
        df, pair, timeframe,
        xgb_model, lstm_model, lstm_scaler, lstm_features
    )
    last = result.iloc[-1]

    return Signal(
        pair           = pair,
        timeframe      = timeframe,
        timestamp      = last.get("timestamp"),
        direction      = int(last["signal"]),
        confidence     = float(last["confidence"]),
        prob_long      = float(last["prob_long"]),
        prob_neutral   = float(last["prob_neutral"]),
        prob_short     = float(last["prob_short"]),
        xgb_direction  = int(last["xgb_direction"]),
        lstm_direction = int(last.get("lstm_direction", 0) or 0),
        agreement      = bool(last["agreement"]),
    )


def signal_stats(result: pd.DataFrame) -> dict:
    """Estadísticas de calidad de señales en un período."""
    total   = len(result.dropna(subset=["signal"]))
    longs   = (result["signal"] == 1).sum()
    shorts  = (result["signal"] == -1).sum()
    neutral = (result["signal"] == 0).sum()
    agreed  = result["agreement"].sum()

    return {
        "total_bars":      total,
        "signals_long":    int(longs),
        "signals_short":   int(shorts),
        "signals_neutral": int(neutral),
        "signal_rate":     round((longs + shorts) / total * 100, 2) if total else 0,
        "agreement_rate":  round(agreed / total * 100, 2) if total else 0,
        "avg_confidence":  round(result["confidence"].mean(), 4),
    }


if __name__ == "__main__":
    # Test rápido: genera señal para la última vela de EURUSD H1
    from sqlalchemy import create_engine
    from dotenv import load_dotenv
    import os
    load_dotenv()

    engine = create_engine(os.getenv("DATABASE_URL"))
    df = pd.read_sql(
        """SELECT * FROM features_computed
           WHERE pair='EURUSD' AND timeframe='H1'
           AND label IS NOT NULL
           ORDER BY timestamp DESC LIMIT 200""",
        engine
    ).sort_values("timestamp").reset_index(drop=True)

    signal = get_latest_signal(df, "EURUSD", "H1")
    logger.info(f"Señal actual EURUSD H1: {signal}")
