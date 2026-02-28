"""
src/analysis/monthly_summary.py
Genera resúmenes mensuales compactos del sistema ML-Ayram,
diseñados para análisis estratégico por IA (Claude/ChatGPT).

Output: JSON (~3-5 KB) con métricas clave + recomendaciones automáticas.
También genera un prompt listo para pegar en la IA.

Uso:
  python -m src.analysis.monthly_summary                    # mes anterior
  python -m src.analysis.monthly_summary --year 2026 --month 1
  python -m src.analysis.monthly_summary --last-n-days 30   # últimos 30 días
  python -m src.analysis.monthly_summary --prompt           # incluye prompt para IA
"""

import argparse
import json
import os
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from loguru import logger
from sqlalchemy import create_engine, text

load_dotenv()
DATABASE_URL = os.getenv("DATABASE_URL")

ALL_PAIRS      = ["EURUSD", "GBPUSD", "USDJPY", "EURJPY", "XAUUSD"]
ALL_TIMEFRAMES = ["H1", "H4"]
ACCOUNT_SIZE   = 10_000.0

PIP_SIZE = {
    "EURUSD": 0.0001, "GBPUSD": 0.0001,
    "USDJPY": 0.01,   "EURJPY": 0.01,   "XAUUSD": 0.10,
}


# ── Carga de datos ────────────────────────────────────────────────────────────

def _load_signals(engine, start_date, end_date) -> pd.DataFrame:
    return pd.read_sql(
        text("""
            SELECT pair, timeframe, direction, model_confidence, regime,
                   status, pnl_pips, pnl_usd, created_at, closed_at,
                   entry_price, sl_price, tp1_price, tp2_price, atr_at_signal
            FROM signals_log
            WHERE created_at >= :start AND created_at < :end
            ORDER BY created_at
        """),
        engine,
        params={"start": start_date, "end": end_date},
    )


def _load_market_stats(engine, start_date, end_date) -> dict:
    """Carga stats del mercado (ATR, ADX promedio) para el período."""
    stats = {}
    for pair in ALL_PAIRS:
        row = pd.read_sql(
            text("""
                SELECT
                    AVG(atr_14) as avg_atr,
                    AVG(adx) as avg_adx,
                    AVG(rsi_14) as avg_rsi,
                    COUNT(*) FILTER (WHERE trend_direction = 1) as bullish_bars,
                    COUNT(*) FILTER (WHERE trend_direction = -1) as bearish_bars,
                    COUNT(*) FILTER (WHERE trend_direction = 0) as lateral_bars,
                    COUNT(*) as total_bars
                FROM features_computed
                WHERE pair = :pair AND timeframe = 'H1'
                  AND timestamp >= :start AND timestamp < :end
                  AND atr_14 IS NOT NULL
            """),
            engine,
            params={"pair": pair, "start": start_date, "end": end_date},
        )
        if not row.empty:
            r = row.iloc[0]
            total = int(r["total_bars"]) or 1
            stats[pair] = {
                "avg_atr_h1":         round(float(r["avg_atr"] or 0), 6),
                "avg_adx_h1":         round(float(r["avg_adx"] or 0), 1),
                "avg_rsi_h1":         round(float(r["avg_rsi"] or 0), 1),
                "trending_pct":       round((int(r["bullish_bars"]) + int(r["bearish_bars"])) / total * 100, 1),
                "bullish_pct":        round(int(r["bullish_bars"]) / total * 100, 1),
                "bearish_pct":        round(int(r["bearish_bars"]) / total * 100, 1),
            }
    return stats


def _load_model_versions(engine) -> dict:
    """Carga información de los últimos modelos entrenados."""
    models = {}
    models_dir = Path("models/saved")
    if not models_dir.exists():
        return models

    for pair in ALL_PAIRS:
        for tf in ALL_TIMEFRAMES:
            key = f"{pair}_{tf}"
            # XGBoost
            metas = sorted(models_dir.glob(f"xgb_{pair}_{tf}_*_meta.json"))
            if metas:
                try:
                    meta = json.loads(metas[-1].read_text())
                    models[f"xgb_{key}"] = {
                        "trained_at": meta.get("trained_at", "unknown"),
                        "cv_f1":      meta.get("metrics", {}).get("cv_f1_mean", 0),
                        "age_days":   _age_days(meta.get("trained_at")),
                    }
                except (json.JSONDecodeError, KeyError):
                    pass

            # LSTM
            import torch
            pts = sorted(models_dir.glob(f"lstm_{pair}_{tf}_*.pt"))
            if pts:
                try:
                    ckpt = torch.load(str(pts[-1]), map_location="cpu", weights_only=False)
                    models[f"lstm_{key}"] = {
                        "trained_at": ckpt.get("trained_at", "unknown"),
                        "val_f1":     ckpt.get("metrics", {}).get("best_val_f1", 0),
                        "age_days":   _age_days(ckpt.get("trained_at")),
                    }
                except Exception:
                    pass

    return models


def _age_days(ts_str) -> int:
    if not ts_str or ts_str == "unknown":
        return -1
    try:
        t = datetime.fromisoformat(str(ts_str).replace("Z", "+00:00"))
        return (datetime.now(timezone.utc) - t).days
    except (ValueError, TypeError):
        return -1


# ── Cálculo del resumen ──────────────────────────────────────────────────────

def generate_summary(
    start_date: datetime,
    end_date:   datetime,
    period_label: str = "",
) -> dict:
    """Genera el resumen mensual completo."""

    engine = create_engine(DATABASE_URL)

    logger.info(f"Generando resumen: {start_date.date()} → {end_date.date()}")

    df = _load_signals(engine, start_date, end_date)
    market = _load_market_stats(engine, start_date, end_date)
    models = _load_model_versions(engine)

    df_closed = df[df["status"] != "open"].copy()
    df_open   = df[df["status"] == "open"].copy()

    # ── Por par ───────────────────────────────────────────────────────
    by_pair = {}
    for pair in ALL_PAIRS:
        p = df_closed[df_closed["pair"] == pair]
        total = len(p)
        wins = len(p[p["status"].isin(["tp1", "tp2"])])
        losses = len(p[p["status"] == "sl"])
        effective = wins + losses

        pnl_eur = float(p["pnl_usd"].sum()) if "pnl_usd" in p else 0
        pnl_pips = float(p["pnl_pips"].sum()) if "pnl_pips" in p else 0
        avg_conf = float(p["model_confidence"].mean()) if total > 0 else 0

        gross_p = float(p[p["pnl_usd"] > 0]["pnl_usd"].sum()) if "pnl_usd" in p else 0
        gross_l = abs(float(p[p["pnl_usd"] < 0]["pnl_usd"].sum())) if "pnl_usd" in p else 0
        pf = round(gross_p / gross_l, 2) if gross_l > 0 else 0

        by_pair[pair] = {
            "signals_total":    total,
            "signals_open":     len(df_open[df_open["pair"] == pair]),
            "wins":             wins,
            "losses":           losses,
            "win_rate":         round(wins / effective * 100, 1) if effective > 0 else 0,
            "profit_factor":    pf,
            "pnl_eur":          round(pnl_eur, 2),
            "pnl_pips":         round(pnl_pips, 1),
            "avg_confidence":   round(avg_conf, 3),
            "market":           market.get(pair, {}),
        }

    # ── Por timeframe ─────────────────────────────────────────────────
    by_tf = {}
    for tf in ALL_TIMEFRAMES:
        t = df_closed[df_closed["timeframe"] == tf]
        total = len(t)
        wins = len(t[t["status"].isin(["tp1", "tp2"])])
        losses = len(t[t["status"] == "sl"])
        effective = wins + losses
        pnl = float(t["pnl_usd"].sum()) if "pnl_usd" in t else 0

        by_tf[tf] = {
            "trades":     total,
            "wins":       wins,
            "losses":     losses,
            "win_rate":   round(wins / effective * 100, 1) if effective > 0 else 0,
            "pnl_eur":    round(pnl, 2),
        }

    # ── Por semana ────────────────────────────────────────────────────
    weekly = []
    if not df_closed.empty:
        df_closed["week"] = pd.to_datetime(df_closed["created_at"]).dt.isocalendar().week
        for week_num, wdf in df_closed.groupby("week"):
            wins_w = len(wdf[wdf["status"].isin(["tp1", "tp2"])])
            total_w = len(wdf[wdf["status"].isin(["tp1", "tp2", "sl"])])
            weekly.append({
                "week":     int(week_num),
                "trades":   len(wdf),
                "win_rate": round(wins_w / total_w * 100, 1) if total_w > 0 else 0,
                "pnl_eur":  round(float(wdf["pnl_usd"].sum()), 2),
            })

    # ── Globales ──────────────────────────────────────────────────────
    total_closed = len(df_closed)
    total_wins   = len(df_closed[df_closed["status"].isin(["tp1", "tp2"])])
    total_losses = len(df_closed[df_closed["status"] == "sl"])
    total_eff    = total_wins + total_losses
    total_pnl    = float(df_closed["pnl_usd"].sum()) if "pnl_usd" in df_closed else 0

    gross_p = float(df_closed[df_closed["pnl_usd"] > 0]["pnl_usd"].sum()) if "pnl_usd" in df_closed else 0
    gross_l = abs(float(df_closed[df_closed["pnl_usd"] < 0]["pnl_usd"].sum())) if "pnl_usd" in df_closed else 0

    # ── Detectar problemas automáticamente ────────────────────────────
    issues = []
    for pair, data in by_pair.items():
        if data["signals_total"] == 0:
            issues.append(f"{pair}: sin señales en todo el período")
        elif data["win_rate"] < 35 and data["signals_total"] >= 5:
            issues.append(f"{pair}: win rate {data['win_rate']}% muy bajo")
        if data.get("market", {}).get("avg_adx_h1", 30) < 18:
            issues.append(f"{pair}: ADX promedio {data['market'].get('avg_adx_h1', 0)} — mercado lateral")

    # ── Auto-recomendaciones ──────────────────────────────────────────
    recommendations = []
    for pair, data in by_pair.items():
        if data["signals_total"] == 0 and data.get("market", {}).get("avg_adx_h1", 30) < 20:
            recommendations.append(f"Reducir min_adx a 18 para {pair} (mercado lateral prolongado)")
        if data["win_rate"] > 60 and data["profit_factor"] > 2.0 and data["signals_total"] >= 10:
            recommendations.append(f"{pair} rendimiento excelente — considerar aumentar posición")
        if data["avg_confidence"] > 0 and data["avg_confidence"] < 0.56:
            recommendations.append(f"{pair}: confianza media baja ({data['avg_confidence']:.0%}) — reentrenar modelos")

    # Modelo más antiguo
    max_age = 0
    for k, v in models.items():
        age = v.get("age_days", 0)
        if age > max_age:
            max_age = age
    if max_age > 14:
        recommendations.append(f"Modelo más antiguo tiene {max_age}d — reentrenamiento necesario")

    # ── Construir JSON final ──────────────────────────────────────────
    summary = {
        "system":       "ML-Ayram",
        "version":      "1.0",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "period": {
            "label":      period_label or f"{start_date.date()} → {end_date.date()}",
            "start":      str(start_date.date()),
            "end":        str(end_date.date()),
            "days":       (end_date - start_date).days,
        },
        "global": {
            "total_signals":   len(df),
            "closed_signals":  total_closed,
            "open_signals":    len(df_open),
            "wins":            total_wins,
            "losses":          total_losses,
            "win_rate":        round(total_wins / total_eff * 100, 1) if total_eff > 0 else 0,
            "profit_factor":   round(gross_p / gross_l, 2) if gross_l > 0 else 0,
            "total_pnl_eur":   round(total_pnl, 2),
            "total_pnl_pips":  round(float(df_closed["pnl_pips"].sum()), 1) if "pnl_pips" in df_closed else 0,
            "avg_confidence":  round(float(df_closed["model_confidence"].mean()), 3) if total_closed > 0 else 0,
            "roi_pct":         round(total_pnl / ACCOUNT_SIZE * 100, 2),
        },
        "by_pair":       by_pair,
        "by_timeframe":  by_tf,
        "weekly":        weekly,
        "models":        models,
        "issues":        issues,
        "recommendations": recommendations,
    }

    return summary


# ── Generador de prompt para IA ───────────────────────────────────────────────

AI_PROMPT_TEMPLATE = """Eres un analista experto en trading algorítmico forex. Analiza el siguiente resumen mensual del bot ML-Ayram y proporciona ajustes estratégicos.

## Datos del Sistema
- Modelos: XGBoost (clasificación) + LSTM con Attention (dirección) en ensemble (55%/45%)
- Triple Barrier Labels: TP=2×ATR, SL=1×ATR, horizonte=20 velas
- Filtros: min_confidence=0.54, min_adx=20, sesión london/newyork/overlap, min_rr=1.5
- Pares: EURUSD, GBPUSD, USDJPY, EURJPY, XAUUSD
- Timeframes: H1 (primario), H4 (confirmación)

## Resumen del Período
```json
{summary_json}
```

## Preguntas de Análisis
1. **Rendimiento por par**: ¿Cuáles están underperforming y por qué? ¿Los datos de mercado (ATR, ADX, tendencia) explican el rendimiento?
2. **Filtros de señal**: ¿Son demasiado estrictos o laxos? ¿Qué ajustes concretos en signal_config.yaml?
3. **Patrones temporales**: ¿Hay patrones en las semanas/timeframes más rentables?
4. **Modelos**: ¿Necesitan reentrenamiento urgente? ¿La confianza promedio es adecuada?
5. **Gestión de riesgo**: ¿Los niveles de SL/TP son óptimos dado el ATR actual?

## Formato de Respuesta
- Diagnóstico (2-3 párrafos)
- Top 5 ajustes priorizados con valores concretos
- Código YAML actualizado para config/signal_config.yaml (solo secciones que cambian)
- Predicción: ¿qué esperar el próximo mes si se aplican los cambios?
"""


def generate_ai_prompt(summary: dict) -> str:
    """Genera un prompt optimizado para Claude/ChatGPT."""
    # Versión compacta del JSON (sin modelos detallados para ahorrar tokens)
    compact = {k: v for k, v in summary.items() if k != "models"}
    compact["models_summary"] = {
        k: {"age_days": v.get("age_days"), "metric": v.get("cv_f1", v.get("val_f1", 0))}
        for k, v in summary.get("models", {}).items()
    }

    summary_json = json.dumps(compact, indent=2, default=str)
    return AI_PROMPT_TEMPLATE.format(summary_json=summary_json)


# ── Motor principal ───────────────────────────────────────────────────────────

def run_monthly_summary(
    year:         int  = None,
    month:        int  = None,
    last_n_days:  int  = None,
    include_prompt: bool = True,
    notify:       bool = True,
) -> dict:

    now = datetime.now(timezone.utc)

    if last_n_days:
        end_date   = now
        start_date = now - timedelta(days=last_n_days)
        label = f"Últimos {last_n_days} días"
    elif year and month:
        start_date = datetime(year, month, 1, tzinfo=timezone.utc)
        if month == 12:
            end_date = datetime(year + 1, 1, 1, tzinfo=timezone.utc)
        else:
            end_date = datetime(year, month + 1, 1, tzinfo=timezone.utc)
        label = f"{year}-{month:02d}"
    else:
        # Mes anterior por defecto
        first_this_month = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        end_date   = first_this_month
        start_date = (first_this_month - timedelta(days=1)).replace(day=1)
        label = f"{start_date.year}-{start_date.month:02d}"

    logger.info("═" * 60)
    logger.info("ML-Ayram — Monthly Summary")
    logger.info(f"Período: {label}")
    logger.info("═" * 60)

    summary = generate_summary(start_date, end_date, label)

    # Guardar JSON
    output_dir = Path("results")
    output_dir.mkdir(parents=True, exist_ok=True)

    json_path = output_dir / f"summary_{label.replace(' ', '_').replace('/', '-')}.json"
    json_path.write_text(json.dumps(summary, indent=2, default=str))
    logger.info(f"Resumen JSON guardado: {json_path}")

    # Guardar prompt para IA
    if include_prompt:
        prompt = generate_ai_prompt(summary)
        prompt_path = output_dir / f"ai_prompt_{label.replace(' ', '_').replace('/', '-')}.md"
        prompt_path.write_text(prompt)
        logger.info(f"Prompt IA guardado: {prompt_path}")

    # Imprimir resumen rápido
    g = summary["global"]
    logger.info("")
    logger.info(f"📊 Señales: {g['total_signals']} total | {g['closed_signals']} cerradas | {g['open_signals']} abiertas")
    logger.info(f"🎯 Win rate: {g['win_rate']}%  |  PF: {g['profit_factor']}")
    logger.info(f"💰 PnL: {g['total_pnl_eur']:+.2f}€ ({g['total_pnl_pips']:+.1f} pips) | ROI: {g['roi_pct']:+.2f}%")

    if summary["issues"]:
        logger.warning("Problemas detectados:")
        for issue in summary["issues"]:
            logger.warning(f"  ⚠️ {issue}")

    if summary["recommendations"]:
        logger.info("Recomendaciones:")
        for rec in summary["recommendations"]:
            logger.info(f"  → {rec}")

    # Notificar Telegram
    if notify:
        try:
            from src.notifications.telegram import send_message
            lines = [
                f"📋 <b>Resumen Mensual — {label}</b>",
                f"━━━━━━━━━━━━━━━━━━━━",
                f"📊 Señales: {g['total_signals']} ({g['closed_signals']} cerradas)",
                f"🎯 Win rate: <b>{g['win_rate']}%</b>  |  PF: <b>{g['profit_factor']}</b>",
                f"💰 PnL: <b>{g['total_pnl_eur']:+.2f}€</b> ({g['roi_pct']:+.2f}%)",
                "",
            ]

            # Top/bottom pares
            pairs_sorted = sorted(summary["by_pair"].items(), key=lambda x: x[1]["pnl_eur"], reverse=True)
            for pair, data in pairs_sorted:
                emoji = "🟢" if data["pnl_eur"] > 0 else "🔴" if data["pnl_eur"] < 0 else "⚪"
                lines.append(
                    f"{emoji} {pair}: {data['pnl_eur']:+.0f}€ | WR {data['win_rate']}% | {data['signals_total']} trades"
                )

            if summary["issues"]:
                lines.append("")
                lines.append("<b>Problemas:</b>")
                for issue in summary["issues"][:3]:
                    lines.append(f"  ⚠️ {issue}")

            lines.append(f"\n<i>Prompt para IA disponible en: {prompt_path}</i>")
            send_message("\n".join(lines))
        except Exception as e:
            logger.error(f"Error Telegram: {e}")

    return summary


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="ML-Ayram Monthly Summary")
    parser.add_argument("--year",        type=int, default=None)
    parser.add_argument("--month",       type=int, default=None)
    parser.add_argument("--last-n-days", type=int, default=None, dest="last_n_days")
    parser.add_argument("--prompt",      action="store_true", default=True, help="Incluir prompt IA")
    parser.add_argument("--no-notify",   action="store_true", help="No enviar Telegram")
    args = parser.parse_args()

    run_monthly_summary(
        year          = args.year,
        month         = args.month,
        last_n_days   = args.last_n_days,
        include_prompt = args.prompt,
        notify        = not args.no_notify,
    )


if __name__ == "__main__":
    main()
