"""
src/train.py
Orquestador de entrenamiento para XGBoost + LSTM.

Uso:
  python -m src.train                        # entrena todo
  python -m src.train --pairs EURUSD GBPUSD  # solo esos pares
  python -m src.train --timeframes H1 H4     # solo esos TF
  python -m src.train --xgb-only             # solo XGBoost
  python -m src.train --lstm-only            # solo LSTM
  python -m src.train --optimize --trials 30 # con búsqueda Optuna
"""

import argparse
import sys
import time
from dataclasses import dataclass, field
from typing import Optional

import pandas as pd
from dotenv import load_dotenv
from loguru import logger
from sqlalchemy import create_engine, text
import os

load_dotenv()
DATABASE_URL = os.getenv("DATABASE_URL")

# ── Telegram helpers ──────────────────────────────────────────────────────────

def _tg(text: str, silent: bool = False) -> None:
    """Envía notificación a Telegram (no bloquea si falla)."""
    try:
        from src.notifications.telegram import send_message
        send_message(text, silent=silent)
    except Exception as e:
        logger.debug(f"Telegram notify skip: {e}")

# ── Constantes ────────────────────────────────────────────────────────────────

ALL_PAIRS      = ["EURUSD", "GBPUSD", "USDJPY", "EURJPY", "XAUUSD"]
ALL_TIMEFRAMES = ["M15", "H1", "H4"]
MIN_ROWS       = 500   # mínimo de filas etiquetadas para entrenar


# ── Resultado por combinación ─────────────────────────────────────────────────

@dataclass
class TrainResult:
    pair:       str
    timeframe:  str
    model:      str          # "xgb" | "lstm"
    status:     str          # "ok" | "skip" | "error"
    metric:     float = 0.0  # F1 weighted CV (XGB) o best val F1 (LSTM)
    rows:       int   = 0
    elapsed_s:  float = 0.0
    error_msg:  str   = ""


# ── Helpers ───────────────────────────────────────────────────────────────────

def _labeled_row_count(engine, pair: str, tf: str) -> int:
    """Cuenta filas etiquetadas disponibles para un par/timeframe."""
    with engine.connect() as conn:
        row = conn.execute(
            text("""
                SELECT COUNT(*) FROM features_computed
                WHERE pair = :pair AND timeframe = :tf
                  AND label IS NOT NULL
            """),
            {"pair": pair, "tf": tf},
        ).fetchone()
    return int(row[0]) if row else 0


def _tg_model_done(r: TrainResult, completed: int, total: int) -> None:
    """Notifica a Telegram el resultado de un modelo individual."""
    if r.status == "ok":
        icon = "✅"
        detail = f"F1: <b>{r.metric:.4f}</b>"
    elif r.status == "skip":
        icon = "⏭️"
        detail = f"Omitido ({r.error_msg})"
    else:
        icon = "❌"
        detail = f"Error: {r.error_msg}"

    model_upper = r.model.upper().replace("XGB", "XGBoost")
    elapsed = f"{r.elapsed_s/60:.1f}min" if r.elapsed_s > 60 else f"{r.elapsed_s:.0f}s"

    _tg(
        f"{icon} <b>{model_upper} {r.pair} {r.timeframe}</b>  [{completed}/{total}]\n"
        f"   {detail}  |  {r.rows} filas  |  {elapsed}",
        silent=True,
    )


def _tg_training_summary(results: list["TrainResult"], total_time: float) -> None:
    """Notificación final de resumen completo."""
    ok     = [r for r in results if r.status == "ok"]
    errors = [r for r in results if r.status == "error"]
    skips  = [r for r in results if r.status == "skip"]

    icon = "🎉" if not errors else "⚠️"
    lines = [
        f"{icon} <b>Entrenamiento finalizado</b>",
        f"━━━━━━━━━━━━━━━━━━━━",
        f"✅ Completados: <b>{len(ok)}</b>",
    ]
    if skips:
        lines.append(f"⏭️ Omitidos: {len(skips)}")
    if errors:
        lines.append(f"❌ Errores: <b>{len(errors)}</b>")
    lines.append(f"⏱️ Tiempo total: <b>{total_time/60:.1f} min</b>")

    # Top modelos por F1
    if ok:
        lines.append("")
        lines.append("🏆 <b>Mejores modelos:</b>")
        for r in sorted(ok, key=lambda x: -x.metric)[:5]:
            model_name = r.model.upper().replace("XGB", "XGBoost")
            lines.append(f"   {model_name} {r.pair} {r.timeframe}: <b>{r.metric:.4f}</b>")

    if errors:
        lines.append("")
        lines.append("❌ <b>Errores:</b>")
        for r in errors:
            model_name = r.model.upper().replace("XGB", "XGBoost")
            lines.append(f"   {model_name} {r.pair} {r.timeframe}: {r.error_msg}")

    _tg("\n".join(lines))


def _print_summary(results: list[TrainResult]) -> None:
    """Imprime tabla resumen al final del entrenamiento."""
    ok     = [r for r in results if r.status == "ok"]
    skip   = [r for r in results if r.status == "skip"]
    errors = [r for r in results if r.status == "error"]

    logger.info("\n" + "═" * 72)
    logger.info("RESUMEN DE ENTRENAMIENTO")
    logger.info("═" * 72)
    logger.info(f"{'Par':<8} {'TF':<5} {'Modelo':<6} {'Estado':<7} {'Filas':>7} {'F1':>7} {'Tiempo':>8}")
    logger.info("─" * 72)

    for r in sorted(results, key=lambda x: (x.pair, x.timeframe, x.model)):
        estado = {"ok": "✅ OK", "skip": "⏭  SKIP", "error": "❌ ERR"}[r.status]
        f1_str = f"{r.metric:.4f}" if r.status == "ok" else "—"
        t_str  = f"{r.elapsed_s:.0f}s" if r.elapsed_s else "—"
        logger.info(
            f"{r.pair:<8} {r.timeframe:<5} {r.model:<6} {estado:<10} "
            f"{r.rows:>7} {f1_str:>7} {t_str:>8}"
        )
        if r.error_msg:
            logger.info(f"         ↳ {r.error_msg}")

    logger.info("─" * 72)
    logger.info(
        f"Total: {len(ok)} OK · {len(skip)} omitidos · {len(errors)} errores  "
        f"({sum(r.elapsed_s for r in ok)/60:.1f} min)"
    )
    logger.info("═" * 72)


# ── Entrenadores individuales ─────────────────────────────────────────────────

def train_xgb(
    pair:      str,
    timeframe: str,
    rows:      int,
    optimize:  bool = False,
    n_trials:  int  = 50,
) -> TrainResult:
    from src.models.xgboost_model import train_and_save

    t0 = time.time()
    logger.info(f"[XGB] {pair} {timeframe}  ({rows} filas etiquetadas)")
    try:
        model = train_and_save(
            pair, timeframe,
            optimize=optimize,
            n_trials=n_trials,
            use_mlflow=False,
        )
        # Recuperar métrica del último guardado
        from pathlib import Path
        import json
        metas = sorted(Path("models/saved").glob(f"xgb_{pair}_{timeframe}_*_meta.json"))
        metric = 0.0
        if metas:
            meta = json.loads(metas[-1].read_text())
            metric = meta.get("metrics", {}).get("cv_f1_mean", 0.0)

        return TrainResult(pair, timeframe, "xgb", "ok", metric, rows, time.time() - t0)

    except Exception as e:
        logger.error(f"[XGB] {pair} {timeframe} — {e}")
        return TrainResult(pair, timeframe, "xgb", "error", elapsed_s=time.time()-t0, error_msg=str(e)[:120])


def train_lstm(
    pair:      str,
    timeframe: str,
    rows:      int,
    epochs:    int = 50,
    patience:  int = 10,
) -> TrainResult:
    from src.models.lstm_model import train_and_save

    t0 = time.time()
    logger.info(f"[LSTM] {pair} {timeframe}  ({rows} filas etiquetadas)")
    try:
        model = train_and_save(pair, timeframe, epochs=epochs, patience=patience)

        # Recuperar métrica del último guardado
        import torch
        from pathlib import Path
        pts = sorted(Path("models/saved").glob(f"lstm_{pair}_{timeframe}_*.pt"))
        metric = 0.0
        if pts:
            ckpt   = torch.load(str(pts[-1]), map_location="cpu", weights_only=False)
            metric = ckpt.get("metrics", {}).get("best_val_f1", 0.0)

        return TrainResult(pair, timeframe, "lstm", "ok", metric, rows, time.time() - t0)

    except Exception as e:
        logger.error(f"[LSTM] {pair} {timeframe} — {e}")
        return TrainResult(pair, timeframe, "lstm", "error", elapsed_s=time.time()-t0, error_msg=str(e)[:120])


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Entrenamiento ML-Ayram")
    parser.add_argument("--pairs",       nargs="+", default=ALL_PAIRS,      metavar="PAIR")
    parser.add_argument("--timeframes",  nargs="+", default=ALL_TIMEFRAMES, metavar="TF")
    parser.add_argument("--xgb-only",   action="store_true", help="Solo entrenar XGBoost")
    parser.add_argument("--lstm-only",  action="store_true", help="Solo entrenar LSTM")
    parser.add_argument("--optimize",   action="store_true", help="Búsqueda Optuna para XGBoost")
    parser.add_argument("--trials",     type=int, default=50,  help="Trials Optuna (default: 50)")
    parser.add_argument("--epochs",     type=int, default=50,  help="Epochs LSTM (default: 50)")
    parser.add_argument("--patience",   type=int, default=10,  help="Early stopping patience (default: 10)")
    parser.add_argument("--min-rows",   type=int, default=MIN_ROWS, help=f"Mínimo filas etiquetadas (default: {MIN_ROWS})")
    args = parser.parse_args()

    train_xgb_flag  = not args.lstm_only
    train_lstm_flag = not args.xgb_only

    engine  = create_engine(DATABASE_URL)
    results = []
    total_start = time.time()

    logger.info("═" * 72)
    logger.info("ML-Ayram — inicio de entrenamiento")
    logger.info(f"Pares:      {args.pairs}")
    logger.info(f"Timeframes: {args.timeframes}")
    logger.info(f"XGBoost:    {'sí' if train_xgb_flag  else 'no'}")
    logger.info(f"LSTM:       {'sí' if train_lstm_flag else 'no'}")
    logger.info(f"Optuna:     {'sí' if args.optimize   else 'no'}")
    logger.info("═" * 72)

    # ── Telegram: inicio de entrenamiento ─────────────────────────────────
    models_list = []
    if train_xgb_flag:  models_list.append("XGBoost")
    if train_lstm_flag: models_list.append("LSTM")
    n_combos = len(args.pairs) * len(args.timeframes) * len(models_list)
    _tg(
        f"🏋️ <b>Entrenamiento iniciado</b>\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"📊 Modelos: {', '.join(models_list)}\n"
        f"💱 Pares: {', '.join(args.pairs)}\n"
        f"⏱️ TFs: {', '.join(args.timeframes)}\n"
        f"🔢 Total: <b>{n_combos} entrenamientos</b>\n"
        f"{'🔍 Optuna: ' + str(args.trials) + ' trials' if args.optimize else ''}",
        silent=True,
    )
    completed = 0

    for tf in args.timeframes:
        for pair in args.pairs:
            rows = _labeled_row_count(engine, pair, tf)

            if rows < args.min_rows:
                logger.warning(f"⏭  {pair} {tf}: solo {rows} filas etiquetadas (mínimo {args.min_rows}) — omitiendo")
                if train_xgb_flag:
                    results.append(TrainResult(pair, tf, "xgb",  "skip", rows=rows, error_msg=f"{rows} filas < {args.min_rows}"))
                if train_lstm_flag:
                    results.append(TrainResult(pair, tf, "lstm", "skip", rows=rows, error_msg=f"{rows} filas < {args.min_rows}"))
                continue

            if train_xgb_flag:
                r = train_xgb(pair, tf, rows, optimize=args.optimize, n_trials=args.trials)
                results.append(r)
                completed += 1
                _tg_model_done(r, completed, n_combos)

            if train_lstm_flag:
                r = train_lstm(pair, tf, rows, epochs=args.epochs, patience=args.patience)
                results.append(r)
                completed += 1
                _tg_model_done(r, completed, n_combos)

    _print_summary(results)
    total_elapsed = time.time() - total_start
    logger.info(f"Tiempo total: {total_elapsed/60:.1f} min")

    # ── Telegram: resumen final ───────────────────────────────────────
    _tg_training_summary(results, total_elapsed)


if __name__ == "__main__":
    main()
