"""
src/backtest/walk_forward.py
Validación Walk-Forward para ML-Ayram.

Metodología:
  - Ventana deslizante o expandida sobre features_computed
  - Cada fold: entrena XGBoost en IS, evalúa en OOS con simulación de trades
  - Repite para todos los pares/timeframes configurados
  - Resultado: métricas OOS agregadas sin lookahead bias

Uso:
  python -m src.backtest.walk_forward
  python -m src.backtest.walk_forward --pairs EURUSD GBPUSD --tf H1
  python -m src.backtest.walk_forward --folds 6 --is-months 6 --oos-months 1
  python -m src.backtest.walk_forward --expanding          # ventana expansiva (default: rolling)
  python -m src.backtest.walk_forward --output results/wf_20260227.json
"""

import argparse
import json
import os
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from loguru import logger
from sqlalchemy import create_engine, text

load_dotenv()
DATABASE_URL = os.getenv("DATABASE_URL")

# ── Constantes ────────────────────────────────────────────────────────────────

ALL_PAIRS      = ["EURUSD", "GBPUSD", "USDJPY", "EURJPY", "XAUUSD"]
ALL_TIMEFRAMES = ["M15", "H1", "H4"]

PIP_SIZE = {
    "EURUSD": 0.0001,
    "GBPUSD": 0.0001,
    "USDJPY": 0.01,
    "EURJPY": 0.01,
    "XAUUSD": 0.10,
}
PIP_VALUE_PER_LOT = {
    "EURUSD": 10.0,
    "GBPUSD": 10.0,
    "USDJPY": 6.5,
    "EURJPY": 10.0,
    "XAUUSD": 10.0,
}
SPREAD_PIPS = {
    "EURUSD": 0.5, "GBPUSD": 0.8,
    "USDJPY": 0.5, "EURJPY": 0.8, "XAUUSD": 3.0,
}
SLIPPAGE_PIPS = 0.5
MAX_HORIZON   = 40    # barras máximas para que un trade llegue a TP/SL
ACCOUNT_SIZE  = 10_000.0
RISK_PCT      = 0.01
MIN_LOT, MAX_LOT = 0.01, 5.0

# Filtros de señal (mismos que generator.py)
MIN_CONFIDENCE = 0.54
MIN_ADX        = 20.0
MIN_RR         = 1.5
TP_MULT        = 2.0
SL_MULT        = 1.0

# Columnas de features — sincronizadas con xgboost_model.py
FEATURE_COLS = [
    "ema_20", "ema_50", "ema_200",
    "macd_line", "macd_signal", "macd_hist",
    "adx", "adx_pos", "adx_neg",
    "rsi_14", "rsi_7",
    "stoch_k", "stoch_d",
    "williams_r", "roc_10", "cci_20",
    "atr_14", "atr_7",
    "bb_width", "bb_pct", "kc_width", "dc_width",
    "volume_ratio_20",
    "price_vs_sh", "price_vs_sl",
    "trend_direction",
    "body_size", "upper_wick", "lower_wick", "is_bullish",
    "log_return_1", "log_return_5", "log_return_10",
    "close_vs_ema20", "close_vs_ema50", "close_vs_ema200",
    "hour_of_day", "day_of_week", "week_of_year", "month",
    "is_london", "is_newyork", "is_overlap",
    "htf_trend", "htf_rsi", "htf_adx",
]
LABEL_MAP     = {-1: 0, 0: 1, 1: 2}
LABEL_MAP_INV = {0: -1, 1: 0, 2: 1}


# ── Dataclasses ───────────────────────────────────────────────────────────────

@dataclass
class FoldResult:
    fold:          int
    pair:          str
    timeframe:     str
    is_start:      str
    is_end:        str
    oos_start:     str
    oos_end:       str
    is_rows:       int
    oos_rows:      int
    train_f1:      float   # XGBoost CV F1 en IS
    trades:        int
    wins:          int
    losses:        int
    timeouts:      int
    win_rate:      float
    pnl_eur:       float
    pnl_pips:      float
    profit_factor: float
    max_drawdown:  float
    expectancy:    float
    avg_rr:        float


@dataclass
class WalkForwardReport:
    # Configuración
    pairs:          list
    timeframes:     list
    n_folds:        int
    is_months:      int
    oos_months:     int
    expanding:      bool
    min_confidence: float

    # Métricas globales agregadas (OOS únicamente)
    total_folds_run:  int   = 0
    total_trades:     int   = 0
    total_wins:       int   = 0
    total_losses:     int   = 0
    total_timeouts:   int   = 0
    total_pnl_eur:    float = 0.0
    total_pnl_pips:   float = 0.0
    global_win_rate:  float = 0.0
    global_pf:        float = 0.0
    global_expectancy:float = 0.0
    global_max_dd:    float = 0.0
    avg_train_f1:     float = 0.0

    # Estabilidad: desviación estándar del PnL entre folds
    pnl_std_across_folds: float = 0.0

    # Folds individuales
    folds:            list  = field(default_factory=list)

    # Equity curve OOS concatenada
    equity_curve:     list  = field(default_factory=list)


# ── Carga de datos ────────────────────────────────────────────────────────────

def _load_features(engine, pair: str, tf: str) -> pd.DataFrame:
    """Carga todos los features etiquetados para un par/tf, ordenados por tiempo."""
    df = pd.read_sql(
        text("""
            SELECT *
            FROM features_computed
            WHERE pair = :pair AND timeframe = :tf
              AND label IS NOT NULL
            ORDER BY timestamp
        """),
        engine,
        params={"pair": pair, "tf": tf},
    )
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df.reset_index(drop=True)


def _load_ohlcv_range(engine, pair: str, tf: str, from_ts, to_ts) -> pd.DataFrame:
    df = pd.read_sql(
        text("""
            SELECT timestamp, open, high, low, close
            FROM ohlcv_raw
            WHERE pair = :pair AND timeframe = :tf
              AND timestamp >= :from_ts AND timestamp <= :to_ts
            ORDER BY timestamp
        """),
        engine,
        params={"pair": pair, "tf": tf, "from_ts": from_ts, "to_ts": to_ts},
    )
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df.reset_index(drop=True)


# ── Construcción de folds ─────────────────────────────────────────────────────

def _build_folds(
    df:          pd.DataFrame,
    n_folds:     int,
    is_months:   int,
    oos_months:  int,
    expanding:   bool,
) -> list[dict]:
    """
    Genera los índices de cada fold IS/OOS sobre el DataFrame ordenado por timestamp.

    Rolling:   ventana IS fija de is_months, se desliza junto con OOS
    Expanding: IS empieza siempre desde el principio y crece fold a fold
    """
    ts = df["timestamp"]
    t_start = ts.min()
    t_end   = ts.max()

    # Punto de partida: el primero que tiene al menos is_months de historia
    oos_start_base = t_start + pd.DateOffset(months=is_months)
    if oos_start_base >= t_end:
        raise ValueError(
            f"Datos insuficientes: {(t_end-t_start).days} días < "
            f"{is_months} meses IS requeridos"
        )

    folds = []
    for i in range(n_folds):
        oos_start = oos_start_base + pd.DateOffset(months=i * oos_months)
        oos_end   = oos_start + pd.DateOffset(months=oos_months)

        if oos_end > t_end:
            break   # sin datos suficientes para este fold

        if expanding:
            is_start = t_start
        else:
            is_start = oos_start - pd.DateOffset(months=is_months)

        is_end = oos_start

        is_mask  = (ts >= is_start) & (ts <  is_end)
        oos_mask = (ts >= oos_start) & (ts <  oos_end)

        if is_mask.sum() < 200 or oos_mask.sum() < 20:
            continue   # fold demasiado pequeño

        folds.append({
            "fold":      i + 1,
            "is_mask":   is_mask,
            "oos_mask":  oos_mask,
            "is_start":  str(is_start.date()),
            "is_end":    str(is_end.date()),
            "oos_start": str(oos_start.date()),
            "oos_end":   str(oos_end.date()),
        })

    return folds


# ── Entrenamiento XGBoost inline (sin MLflow, sin I/O de archivos) ────────────

def _train_xgb_inline(X_train: pd.DataFrame, y_train: pd.Series) -> tuple:
    """
    Entrena XGBoost con 3-fold TimeSeriesSplit y devuelve (model, cv_f1).
    No guarda en disco — sólo para evaluación interna del fold.
    """
    import xgboost as xgb
    from sklearn.model_selection import TimeSeriesSplit
    from sklearn.metrics import f1_score

    params = {
        "n_estimators": 400, "max_depth": 6, "learning_rate": 0.05,
        "subsample": 0.8, "colsample_bytree": 0.8, "min_child_weight": 5,
        "eval_metric": "mlogloss", "random_state": 42, "n_jobs": -1,
    }
    tscv = TimeSeriesSplit(n_splits=3)
    f1s  = []
    for tr_idx, val_idx in tscv.split(X_train):
        m = xgb.XGBClassifier(num_class=3, objective="multi:softprob", **params)
        m.fit(
            X_train.iloc[tr_idx], y_train.iloc[tr_idx],
            eval_set=[(X_train.iloc[val_idx], y_train.iloc[val_idx])],
            verbose=False,
        )
        f1s.append(f1_score(y_train.iloc[val_idx], m.predict(X_train.iloc[val_idx]),
                            average="weighted"))

    model = xgb.XGBClassifier(num_class=3, objective="multi:softprob", **params)
    model.fit(X_train, y_train, verbose=False)
    return model, float(np.mean(f1s))


# ── Filtros de señal ──────────────────────────────────────────────────────────

def _apply_signal_filters(row: pd.Series, direction: int, confidence: float) -> Optional[str]:
    """
    Replica los filtros de generator.py.
    Devuelve None si la señal es válida, o el motivo de rechazo.
    """
    if direction == 0:
        return "neutral"
    if confidence < MIN_CONFIDENCE:
        return f"confidence {confidence:.2%} < {MIN_CONFIDENCE:.2%}"
    adx = row.get("adx", 0)
    if pd.isna(adx) or adx < MIN_ADX:
        return f"adx {adx:.1f} < {MIN_ADX}"
    return None


def _build_signal(row: pd.Series, direction: int, probs: np.ndarray) -> Optional[dict]:
    """
    Construye un dict de señal a partir de una fila de features y la predicción.
    Devuelve None si no pasa filtros o RR insuficiente.
    """
    pair     = row["pair"]
    pip      = PIP_SIZE.get(pair, 0.0001)
    atr      = float(row.get("atr_14", 0) or 0)

    prob_long, prob_neutral, prob_short = float(probs[2]), float(probs[1]), float(probs[0])
    confidence = max(prob_long, prob_short)

    reason = _apply_signal_filters(row, direction, confidence)
    if reason:
        return None

    close = float(row.get("close", 0) or 0)
    if close == 0 or atr == 0:
        return None

    if direction == 1:
        tp = close + atr * TP_MULT
        sl = close - atr * SL_MULT
    else:
        tp = close - atr * TP_MULT
        sl = close + atr * SL_MULT

    tp_pips = abs(tp - close) / pip
    sl_pips = abs(sl - close) / pip
    rr      = tp_pips / sl_pips if sl_pips > 0 else 0

    if rr < MIN_RR:
        return None

    return {
        "pair":      pair,
        "timeframe": row["timeframe"],
        "timestamp": row["timestamp"],
        "direction": direction,
        "confidence": confidence,
        "prob_long":  prob_long,
        "prob_neutral": prob_neutral,
        "prob_short": prob_short,
        "entry_price": close,
        "tp_price":   tp,
        "sl_price":   sl,
        "tp_pips":    tp_pips,
        "sl_pips":    sl_pips,
        "rr_ratio":   rr,
        "adx":        float(row.get("adx", 0) or 0),
    }


# ── Simulación de trade (reutiliza lógica de engine.py) ──────────────────────

def _simulate_trade(sig: dict, ohlcv: pd.DataFrame) -> dict:
    """Simula un trade buscando la primera barra que toca TP o SL."""
    pair      = sig["pair"]
    pip       = PIP_SIZE.get(pair, 0.0001)
    pip_val   = PIP_VALUE_PER_LOT.get(pair, 10.0)
    spread    = SPREAD_PIPS.get(pair, 1.0) * pip
    slippage  = SLIPPAGE_PIPS * pip
    direction = sig["direction"]

    actual_entry = (sig["entry_price"] + spread + slippage if direction == 1
                    else sig["entry_price"] - spread - slippage)
    tp, sl = sig["tp_price"], sig["sl_price"]

    result_str   = "timeout"
    exit_price   = float(ohlcv.iloc[-1]["close"]) if len(ohlcv) else actual_entry
    bars_to_exit = len(ohlcv)

    for i, row in ohlcv.iterrows():
        h, l = float(row["high"]), float(row["low"])
        if direction == 1:
            if h >= tp:
                result_str, exit_price, bars_to_exit = "tp_hit", tp, i + 1; break
            if l <= sl:
                result_str, exit_price, bars_to_exit = "sl_hit", sl, i + 1; break
        else:
            if l <= tp:
                result_str, exit_price, bars_to_exit = "tp_hit", tp, i + 1; break
            if h >= sl:
                result_str, exit_price, bars_to_exit = "sl_hit", sl, i + 1; break

    pnl_pips = ((exit_price - actual_entry) / pip if direction == 1
                else (actual_entry - exit_price) / pip)

    sl_pips  = max(sig["sl_pips"], 1.0)
    risk_eur = ACCOUNT_SIZE * RISK_PCT
    lot      = round(max(MIN_LOT, min(MAX_LOT, risk_eur / (sl_pips * pip_val))), 2)
    pnl_eur  = pnl_pips * pip_val * lot

    return {
        "result":        result_str,
        "pnl_pips":      round(pnl_pips, 2),
        "pnl_eur":       round(pnl_eur, 2),
        "bars_to_exit":  bars_to_exit,
        "rr_ratio":      sig["rr_ratio"],
        "opened_at":     str(sig["timestamp"]),
    }


def _tf_timedelta(tf: str) -> timedelta:
    return {"M15": timedelta(minutes=15), "H1": timedelta(hours=1),
            "H4": timedelta(hours=4),     "D1": timedelta(days=1)}.get(tf, timedelta(hours=1))


# ── Métricas de un conjunto de trades ────────────────────────────────────────

def _compute_metrics(trade_results: list[dict]) -> dict:
    """Calcula las métricas clave de una lista de trades simulados."""
    if not trade_results:
        return {
            "trades": 0, "wins": 0, "losses": 0, "timeouts": 0,
            "win_rate": 0, "pnl_eur": 0, "pnl_pips": 0,
            "profit_factor": 0, "max_drawdown": 0,
            "expectancy": 0, "avg_rr": 0,
        }

    wins     = [t for t in trade_results if t["result"] == "tp_hit"]
    losses   = [t for t in trade_results if t["result"] == "sl_hit"]
    timeouts = [t for t in trade_results if t["result"] == "timeout"]

    pnl_eur   = sum(t["pnl_eur"] for t in trade_results)
    gross_p   = sum(t["pnl_eur"] for t in trade_results if t["pnl_eur"] > 0)
    gross_l   = abs(sum(t["pnl_eur"] for t in trade_results if t["pnl_eur"] < 0))
    pf        = round(gross_p / gross_l, 2) if gross_l > 0 else float("inf")
    wr        = len(wins) / len(trade_results)
    avg_win   = np.mean([t["pnl_eur"] for t in wins])   if wins   else 0
    avg_loss  = np.mean([t["pnl_eur"] for t in losses]) if losses else 0
    expectancy= wr * avg_win + (1 - wr) * avg_loss
    avg_rr    = np.mean([t["rr_ratio"] for t in trade_results])

    # Drawdown en la serie de equity
    equity = ACCOUNT_SIZE
    peak   = ACCOUNT_SIZE
    max_dd = 0.0
    for t in trade_results:
        equity += t["pnl_eur"]
        peak    = max(peak, equity)
        max_dd  = max(max_dd, peak - equity)

    return {
        "trades":        len(trade_results),
        "wins":          len(wins),
        "losses":        len(losses),
        "timeouts":      len(timeouts),
        "win_rate":      round(wr * 100, 1),
        "pnl_eur":       round(pnl_eur, 2),
        "pnl_pips":      round(sum(t["pnl_pips"] for t in trade_results), 2),
        "profit_factor": pf,
        "max_drawdown":  round(max_dd, 2),
        "expectancy":    round(expectancy, 2),
        "avg_rr":        round(avg_rr, 2),
    }


# ── Motor walk-forward para un par/tf ────────────────────────────────────────

def _run_pair_tf(
    engine,
    pair:       str,
    tf:         str,
    n_folds:    int,
    is_months:  int,
    oos_months: int,
    expanding:  bool,
) -> list[FoldResult]:
    """
    Ejecuta el walk-forward completo para un par/tf.
    Devuelve lista de FoldResult (uno por fold OOS).
    """
    logger.info(f"  [{pair} {tf}] Cargando features...")
    df = _load_features(engine, pair, tf)

    if df.empty:
        logger.warning(f"  [{pair} {tf}] Sin datos etiquetados — omitiendo")
        return []

    available_feats = [c for c in FEATURE_COLS if c in df.columns]
    folds = _build_folds(df, n_folds, is_months, oos_months, expanding)

    if not folds:
        logger.warning(f"  [{pair} {tf}] Sin folds válidos (datos insuficientes)")
        return []

    logger.info(f"  [{pair} {tf}] {len(folds)} folds — {len(df)} filas totales")
    fold_results = []

    for fold_def in folds:
        fold_n    = fold_def["fold"]
        is_mask   = fold_def["is_mask"]
        oos_mask  = fold_def["oos_mask"]

        df_is  = df[is_mask].copy()
        df_oos = df[oos_mask].copy()

        # ── Entrenar en IS ──────────────────────────────────────────────
        X_is = df_is[available_feats].fillna(df_is[available_feats].median())
        y_is = df_is["label"].map(LABEL_MAP)
        valid_mask = y_is.notna()
        X_is, y_is = X_is[valid_mask], y_is[valid_mask].astype(int)

        if len(X_is) < 100:
            logger.debug(f"    Fold {fold_n}: IS demasiado pequeño ({len(X_is)} filas)")
            continue

        logger.info(
            f"    Fold {fold_n}: IS {fold_def['is_start']}→{fold_def['is_end']} "
            f"({len(X_is)} filas) | OOS {fold_def['oos_start']}→{fold_def['oos_end']} "
            f"({len(df_oos)} filas)"
        )

        try:
            model, train_f1 = _train_xgb_inline(X_is, y_is)
        except Exception as e:
            logger.error(f"    Fold {fold_n}: Error entrenando — {e}")
            continue

        logger.info(f"    Fold {fold_n}: CV F1 IS = {train_f1:.4f}")

        # ── Predecir en OOS ─────────────────────────────────────────────
        X_oos = df_oos[available_feats].fillna(df_oos[available_feats].median())
        proba_raw = model.predict_proba(X_oos)   # (n, 3) → short, neutral, long

        directions_raw = np.argmax(proba_raw, axis=1)
        directions     = np.vectorize(LABEL_MAP_INV.get)(directions_raw)

        # ── Construir señales válidas ───────────────────────────────────
        signals = []
        for i, (idx, row) in enumerate(df_oos.iterrows()):
            sig = _build_signal(row, int(directions[i]), proba_raw[i])
            if sig:
                signals.append(sig)

        logger.info(f"    Fold {fold_n}: {len(signals)} señales válidas de {len(df_oos)} barras OOS")

        if not signals:
            fold_results.append(FoldResult(
                fold=fold_n, pair=pair, timeframe=tf,
                is_start=fold_def["is_start"], is_end=fold_def["is_end"],
                oos_start=fold_def["oos_start"], oos_end=fold_def["oos_end"],
                is_rows=len(X_is), oos_rows=len(df_oos),
                train_f1=train_f1, trades=0, wins=0, losses=0, timeouts=0,
                win_rate=0, pnl_eur=0, pnl_pips=0, profit_factor=0,
                max_drawdown=0, expectancy=0, avg_rr=0,
            ))
            continue

        # ── Simular trades ──────────────────────────────────────────────
        tf_delta = _tf_timedelta(tf)
        trade_results = []

        for sig in signals:
            sig_ts   = pd.Timestamp(sig["timestamp"])
            ohlcv_fwd = _load_ohlcv_range(
                engine, pair, tf,
                from_ts = sig_ts,
                to_ts   = sig_ts + tf_delta * (MAX_HORIZON + 1),
            )
            ohlcv_fwd = ohlcv_fwd[ohlcv_fwd["timestamp"] > sig_ts].reset_index(drop=True)

            if not ohlcv_fwd.empty:
                t = _simulate_trade(sig, ohlcv_fwd)
                trade_results.append(t)

        m = _compute_metrics(trade_results)
        logger.info(
            f"    Fold {fold_n}: {m['trades']} trades | "
            f"WR {m['win_rate']}% | PnL {m['pnl_eur']:+.2f}€ | PF {m['profit_factor']}"
        )

        fold_results.append(FoldResult(
            fold=fold_n, pair=pair, timeframe=tf,
            is_start=fold_def["is_start"], is_end=fold_def["is_end"],
            oos_start=fold_def["oos_start"], oos_end=fold_def["oos_end"],
            is_rows=len(X_is), oos_rows=len(df_oos),
            train_f1=train_f1,
            trades=m["trades"], wins=m["wins"], losses=m["losses"],
            timeouts=m["timeouts"], win_rate=m["win_rate"],
            pnl_eur=m["pnl_eur"], pnl_pips=m["pnl_pips"],
            profit_factor=m["profit_factor"],
            max_drawdown=m["max_drawdown"],
            expectancy=m["expectancy"],
            avg_rr=m["avg_rr"],
        ))

    return fold_results


# ── Motor principal ───────────────────────────────────────────────────────────

def run_walk_forward(
    pairs:       list       = None,
    timeframes:  list       = None,
    n_folds:     int        = 6,
    is_months:   int        = 6,
    oos_months:  int        = 1,
    expanding:   bool       = False,
    min_confidence: float   = MIN_CONFIDENCE,
    output_path: Path       = None,
) -> WalkForwardReport:

    pairs      = pairs      or ALL_PAIRS
    timeframes = timeframes or ALL_TIMEFRAMES
    engine     = create_engine(DATABASE_URL)

    logger.info("═" * 60)
    logger.info("ML-Ayram — Walk-Forward Validation")
    logger.info(f"Pares:      {pairs}")
    logger.info(f"Timeframes: {timeframes}")
    logger.info(f"Folds:      {n_folds}  |  IS: {is_months}m  OOS: {oos_months}m")
    logger.info(f"Ventana:    {'expansiva' if expanding else 'rolling'}")
    logger.info("═" * 60)

    report = WalkForwardReport(
        pairs=pairs, timeframes=timeframes,
        n_folds=n_folds, is_months=is_months, oos_months=oos_months,
        expanding=expanding, min_confidence=min_confidence,
    )

    all_folds: list[FoldResult] = []

    for tf in timeframes:
        for pair in pairs:
            fold_results = _run_pair_tf(
                engine, pair, tf, n_folds, is_months, oos_months, expanding
            )
            all_folds.extend(fold_results)

    if not all_folds:
        logger.warning("Sin resultados. Verifica que features y labels estén calculados.")
        return report

    # ── Agregar métricas globales ─────────────────────────────────────────
    report.total_folds_run  = len(all_folds)
    report.total_trades     = sum(f.trades for f in all_folds)
    report.total_wins       = sum(f.wins   for f in all_folds)
    report.total_losses     = sum(f.losses for f in all_folds)
    report.total_timeouts   = sum(f.timeouts for f in all_folds)
    report.total_pnl_eur    = round(sum(f.pnl_eur for f in all_folds), 2)
    report.total_pnl_pips   = round(sum(f.pnl_pips for f in all_folds), 2)
    report.avg_train_f1     = round(np.mean([f.train_f1 for f in all_folds]), 4)

    n = report.total_trades
    report.global_win_rate  = round(report.total_wins / n * 100, 1) if n else 0

    gross_p = sum(f.pnl_eur for f in all_folds if f.pnl_eur > 0)
    gross_l = abs(sum(f.pnl_eur for f in all_folds if f.pnl_eur < 0))
    report.global_pf = round(gross_p / gross_l, 2) if gross_l > 0 else float("inf")

    report.global_max_dd    = round(max((f.max_drawdown for f in all_folds), default=0), 2)
    report.global_expectancy= round(np.mean([f.expectancy for f in all_folds if f.trades > 0]), 2) if any(f.trades > 0 for f in all_folds) else 0

    # Estabilidad: std del PnL entre folds con trades
    pnls_per_fold = [f.pnl_eur for f in all_folds if f.trades > 0]
    report.pnl_std_across_folds = round(float(np.std(pnls_per_fold)), 2) if pnls_per_fold else 0

    # ── Equity curve OOS concatenada ──────────────────────────────────────
    equity = ACCOUNT_SIZE
    for fold in sorted(all_folds, key=lambda x: (x.oos_start, x.pair, x.timeframe)):
        equity += fold.pnl_eur
        report.equity_curve.append({
            "fold":       fold.fold,
            "pair":       fold.pair,
            "timeframe":  fold.timeframe,
            "oos_end":    fold.oos_end,
            "fold_pnl":   fold.pnl_eur,
            "equity":     round(equity, 2),
        })

    report.folds = [asdict(f) for f in all_folds]

    _print_report(report)

    if output_path:
        _save_report(report, output_path)

    return report


# ── Impresión y guardado ──────────────────────────────────────────────────────

def _print_report(r: WalkForwardReport) -> None:
    pf_str = str(r.global_pf) if r.global_pf != float("inf") else "∞"
    logger.info("")
    logger.info("═" * 68)
    logger.info("RESULTADOS WALK-FORWARD (Out-of-Sample únicamente)")
    logger.info("═" * 68)
    logger.info(f"Folds ejecutados:  {r.total_folds_run}")
    logger.info(f"Trades OOS totales:{r.total_trades:>8}")
    logger.info(f"Wins / Losses:     {r.total_wins} / {r.total_losses}  (timeout: {r.total_timeouts})")
    logger.info(f"Win rate:          {r.global_win_rate}%")
    logger.info(f"PnL total OOS:     {r.total_pnl_eur:+.2f}€  ({r.total_pnl_pips:+.1f} pips)")
    logger.info(f"Profit factor:     {pf_str}")
    logger.info(f"Expectancy:        {r.global_expectancy:+.2f}€ / trade")
    logger.info(f"Max drawdown fold: {r.global_max_dd:.2f}€")
    logger.info(f"Avg F1 IS:         {r.avg_train_f1:.4f}")
    logger.info(f"PnL std (folds):   {r.pnl_std_across_folds:.2f}€")
    logger.info("─" * 68)

    # Tabla de folds individuales
    logger.info(f"{'#':<3} {'Par':<8} {'TF':<5} {'OOS':<11} {'Trades':>6} {'WR%':>6} {'PnL€':>8} {'PF':>6} {'F1':>6}")
    logger.info("─" * 68)
    for f in sorted(r.folds, key=lambda x: (x["pair"], x["timeframe"], x["fold"])):
        pf = f["profit_factor"]
        pf_s = f"{pf:.2f}" if pf != float("inf") else "∞"
        logger.info(
            f"{f['fold']:<3} {f['pair']:<8} {f['timeframe']:<5} "
            f"{f['oos_start']:<11} {f['trades']:>6} {f['win_rate']:>6.1f} "
            f"{f['pnl_eur']:>+8.2f} {pf_s:>6} {f['train_f1']:>6.4f}"
        )
    logger.info("═" * 68)

    # Interpretación automática
    if r.total_trades < 10:
        verdict = "⚠️  POCOS TRADES — aumenta el período o baja el umbral de confianza"
    elif r.global_pf < 1.0:
        verdict = "❌ SISTEMA PERDEDOR — revisar features, filtros o hiperparámetros"
    elif r.global_pf < 1.3:
        verdict = "⚠️  SISTEMA MARGINAL — operar en demo primero"
    elif r.global_win_rate < 40 and r.global_pf < 1.5:
        verdict = "⚠️  WIN RATE BAJO — depende del RR, monitorear de cerca"
    else:
        verdict = "✅ SISTEMA PROMETEDOR — validar en demo antes de real"

    logger.info(f"\n  Veredicto: {verdict}\n")


def _save_report(report: WalkForwardReport, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = asdict(report)
    if data.get("global_pf") == float("inf"):
        data["global_pf"] = "inf"
    path.write_text(json.dumps(data, indent=2, default=str))
    logger.success(f"Informe WF guardado: {path}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="ML-Ayram Walk-Forward Validation")
    parser.add_argument("--pairs",      nargs="+", default=None)
    parser.add_argument("--tf",         nargs="+", default=None, dest="timeframes")
    parser.add_argument("--folds",      type=int,  default=6)
    parser.add_argument("--is-months",  type=int,  default=6,  dest="is_months")
    parser.add_argument("--oos-months", type=int,  default=1,  dest="oos_months")
    parser.add_argument("--expanding",  action="store_true")
    parser.add_argument("--min-confidence", type=float, default=MIN_CONFIDENCE, dest="min_confidence")
    parser.add_argument("--output",     type=str, default=None)
    args = parser.parse_args()

    output = Path(args.output) if args.output else Path(
        f"results/walkforward_{datetime.now().strftime('%Y%m%d_%H%M')}.json"
    )

    run_walk_forward(
        pairs          = args.pairs,
        timeframes     = args.timeframes,
        n_folds        = args.folds,
        is_months      = args.is_months,
        oos_months     = args.oos_months,
        expanding      = args.expanding,
        min_confidence = args.min_confidence,
        output_path    = output,
    )


if __name__ == "__main__":
    main()
