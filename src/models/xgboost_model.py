"""
src/models/xgboost_model.py
Modelo XGBoost con optimización de hiperparámetros via Optuna.
Clasifica: +1 (long ganador), 0 (neutral), -1 (long perdedor)
"""

import os
import json
import numpy as np
import pandas as pd
import xgboost as xgb
import optuna
try:
    import mlflow
    import mlflow.xgboost
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import f1_score
from sqlalchemy import text
from loguru import logger
from pathlib import Path
from datetime import datetime
from src.utils.db import engine
MLFLOW_URI      = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
MODELS_DIR      = Path("models/saved")
MODELS_DIR.mkdir(parents=True, exist_ok=True)

FEATURE_COLS = [
    "ema_20", "ema_50", "ema_200",
    "macd_line", "macd_signal", "macd_hist",
    "adx", "adx_pos", "adx_neg",
    "rsi_14", "rsi_7",
    "stoch_k", "stoch_d",
    "williams_r", "roc_10", "cci_20",
    "atr_14", "atr_7",
    "bb_width", "bb_pct",
    "kc_width",
    "dc_width",
    "volume_ratio_20",
    "price_vs_sh", "price_vs_sl",
    "trend_direction",
    "body_size", "upper_wick", "lower_wick", "is_bullish",
    "log_return_1", "log_return_5", "log_return_10",
    "close_vs_ema20", "close_vs_ema50", "close_vs_ema200",
    "hour_of_day", "day_of_week", "week_of_year", "month",
    "is_london", "is_newyork", "is_overlap",
    "htf_trend", "htf_rsi",
    # htf_adx excluido: columna añadida tras el histórico, registros anteriores son NULL
]

LABEL_COL = "label"
# XGBoost requiere etiquetas 0,1,2 → mapeamos -1→0, 0→1, +1→2
LABEL_MAP     = {-1: 0, 0: 1, 1: 2}
LABEL_MAP_INV = {0: -1, 1: 0, 2: 1}


def load_dataset(pair: str, timeframe: str) -> tuple[pd.DataFrame, pd.Series]:
    """Carga features + labels desde la BD."""
    df = pd.read_sql(
        text("""
            SELECT * FROM features_computed
            WHERE pair = :pair AND timeframe = :tf
              AND label IS NOT NULL
            ORDER BY timestamp
        """),
        engine,
        params={"pair": pair, "tf": timeframe},
    )
    if df.empty:
        raise ValueError(f"Sin datos etiquetados para {pair} {timeframe}")

    # Filtrar features disponibles, descartar columnas 100% NaN
    available = [c for c in FEATURE_COLS if c in df.columns]
    X = df[available].copy()
    all_nan = X.columns[X.isna().all()].tolist()
    if all_nan:
        logger.warning(f"Columnas 100% NaN eliminadas: {all_nan}")
        X = X.drop(columns=all_nan)
    y = df[LABEL_COL].map(LABEL_MAP)

    # Imputar NaN con mediana
    X = X.fillna(X.median())

    logger.info(f"Dataset {pair} {timeframe}: {len(X)} filas, {len(available)} features")
    logger.info(f"Distribución labels: {y.value_counts().to_dict()}")
    return X, y


def train_xgboost(
    X: pd.DataFrame,
    y: pd.Series,
    params: dict | None = None,
    n_splits: int = 5,
) -> tuple[xgb.XGBClassifier, dict]:
    """
    Entrena XGBoost con validación cruzada temporal (TimeSeriesSplit).
    Devuelve el modelo entrenado y las métricas.
    """
    if params is None:
        params = {
            "n_estimators":     500,
            "max_depth":        6,
            "learning_rate":    0.05,
            "subsample":        0.8,
            "colsample_bytree": 0.8,
            "min_child_weight": 5,
            "gamma":            0.1,
            "reg_alpha":        0.1,
            "reg_lambda":       1.0,
            "use_label_encoder": False,
            "eval_metric":      "mlogloss",
            "random_state":     42,
            "n_jobs":           -1,
        }

    tscv = TimeSeriesSplit(n_splits=n_splits)
    fold_f1s = []

    for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        model = xgb.XGBClassifier(
            num_class=3,
            objective="multi:softprob",
            **{k: v for k, v in params.items() if k not in ("use_label_encoder",)}
        )
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )

        preds = model.predict(X_val)
        f1 = f1_score(y_val, preds, average="weighted")
        fold_f1s.append(f1)
        logger.info(f"  Fold {fold+1}/{n_splits} — F1 weighted: {f1:.4f}")

    avg_f1 = np.mean(fold_f1s)
    logger.info(f"  CV F1 promedio: {avg_f1:.4f}")

    # Reentrenar con todos los datos
    final_model = xgb.XGBClassifier(
        num_class=3,
        objective="multi:softprob",
        **{k: v for k, v in params.items() if k not in ("use_label_encoder",)}
    )
    final_model.fit(X, y, verbose=False)

    metrics = {
        "cv_f1_mean":   float(avg_f1),
        "cv_f1_std":    float(np.std(fold_f1s)),
        "cv_f1_folds":  [float(f) for f in fold_f1s],
    }
    return final_model, metrics


def optimize_hyperparams(
    X: pd.DataFrame,
    y: pd.Series,
    n_trials: int = 50,
) -> dict:
    """
    Búsqueda de hiperparámetros con Optuna.
    Devuelve los mejores parámetros encontrados.
    """
    def objective(trial):
        params = {
            "n_estimators":     trial.suggest_int("n_estimators", 200, 1000),
            "max_depth":        trial.suggest_int("max_depth", 3, 10),
            "learning_rate":    trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "subsample":        trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 20),
            "gamma":            trial.suggest_float("gamma", 0, 1.0),
            "reg_alpha":        trial.suggest_float("reg_alpha", 0, 2.0),
            "reg_lambda":       trial.suggest_float("reg_lambda", 0.5, 3.0),
            "eval_metric":      "mlogloss",
            "random_state":     42,
            "n_jobs":           1,   # 1 por trial; Optuna paraleliza a nivel de trial
        }
        tscv = TimeSeriesSplit(n_splits=3)
        f1s  = []
        for train_idx, val_idx in tscv.split(X):
            X_tr, X_v = X.iloc[train_idx], X.iloc[val_idx]
            y_tr, y_v = y.iloc[train_idx], y.iloc[val_idx]
            m = xgb.XGBClassifier(num_class=3, objective="multi:softprob", **params)
            m.fit(X_tr, y_tr, eval_set=[(X_v, y_v)], verbose=False)
            f1s.append(f1_score(y_v, m.predict(X_v), average="weighted"))
        return np.mean(f1s)

    logger.info(f"Iniciando Optuna: {n_trials} trials...")
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    logger.info(f"Mejor F1: {study.best_value:.4f}")
    logger.info(f"Mejores params: {study.best_params}")
    return study.best_params


def save_model(model: xgb.XGBClassifier, pair: str, timeframe: str, metrics: dict) -> Path:
    """Guarda el modelo y sus métricas en disco."""
    ts   = datetime.utcnow().strftime("%Y%m%d_%H%M")
    name = f"xgb_{pair}_{timeframe}_{ts}"
    path = MODELS_DIR / f"{name}.ubj"

    model.save_model(str(path))

    meta = {
        "pair":      pair,
        "timeframe": timeframe,
        "trained_at": ts,
        "metrics":   metrics,
        "features":  FEATURE_COLS,
        "label_map": LABEL_MAP,
    }
    with open(MODELS_DIR / f"{name}_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    logger.success(f"Modelo guardado: {path}")
    return path


def load_model(pair: str, timeframe: str) -> xgb.XGBClassifier:
    """Carga el modelo más reciente para un par/timeframe."""
    pattern = f"xgb_{pair}_{timeframe}_*.ubj"
    files   = sorted(MODELS_DIR.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No hay modelo guardado para {pair} {timeframe}")
    latest = files[-1]
    model  = xgb.XGBClassifier()
    model.load_model(str(latest))
    logger.info(f"Modelo cargado: {latest}")
    return model


def predict(model: xgb.XGBClassifier, X: pd.DataFrame) -> np.ndarray:
    """
    Genera predicciones. Devuelve array con valores -1, 0, +1.
    """
    available = [c for c in FEATURE_COLS if c in X.columns]
    X_clean   = X[available].fillna(X[available].median())
    raw_preds = model.predict(X_clean)
    return np.vectorize(LABEL_MAP_INV.get)(raw_preds)


def predict_proba(model: xgb.XGBClassifier, X: pd.DataFrame) -> pd.DataFrame:
    """
    Devuelve probabilidades por clase como DataFrame con columnas:
    prob_short (-1), prob_neutral (0), prob_long (+1)
    """
    available = [c for c in FEATURE_COLS if c in X.columns]
    X_clean   = X[available].fillna(X[available].median())
    proba     = model.predict_proba(X_clean)
    return pd.DataFrame(proba, columns=["prob_short", "prob_neutral", "prob_long"])


def train_and_save(
    pair: str,
    timeframe: str,
    optimize: bool = False,
    n_trials: int = 50,
    use_mlflow: bool = True,
) -> xgb.XGBClassifier:
    """
    Pipeline completo: carga datos → (opcional) optimiza → entrena → guarda → loguea en MLflow.
    """
    X, y = load_dataset(pair, timeframe)

    params = None
    if optimize:
        params = optimize_hyperparams(X, y, n_trials=n_trials)

    if use_mlflow and MLFLOW_AVAILABLE:
        mlflow.set_tracking_uri(MLFLOW_URI)
        mlflow.set_experiment(f"xgboost_{pair}_{timeframe}")
        with mlflow.start_run():
            model, metrics = train_xgboost(X, y, params)
            mlflow.log_params(params or {})
            mlflow.log_metrics({
                "cv_f1_mean": metrics["cv_f1_mean"],
                "cv_f1_std":  metrics["cv_f1_std"],
            })
            path = save_model(model, pair, timeframe, metrics)
            mlflow.xgboost.log_model(model, "model")
    else:
        model, metrics = train_xgboost(X, y, params)
        save_model(model, pair, timeframe, metrics)

    return model


if __name__ == "__main__":
    pairs      = ["EURUSD", "GBPUSD", "USDJPY", "EURJPY", "XAUUSD"]
    timeframes = ["M15", "H1", "H4"]
    for tf in timeframes:
        for pair in pairs:
            try:
                train_and_save(pair, tf, optimize=False, use_mlflow=False)
            except Exception as e:
                logger.error(f"Error {pair} {tf}: {e}")
