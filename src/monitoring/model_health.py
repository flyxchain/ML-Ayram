"""
src/monitoring/model_health.py
Monitoreo de salud de modelos: detecta degradación comparando
rendimiento reciente vs métricas OOS históricas.

Uso:
  python -m src.monitoring.model_health                 # chequeo completo
  python -m src.monitoring.model_health --days 14       # últimos 14 días
  python -m src.monitoring.model_health --threshold 0.3 # alerta si degrada >30%
  python -m src.monitoring.model_health --auto-retrain  # trigger reentrenamiento

Lógica:
  1. Carga señales cerradas de los últimos N días
  2. Calcula métricas actuales (win rate, PF, expectancy)
  3. Compara con métricas baseline (OOS walk-forward o config)
  4. Genera alertas si degradación supera umbrales
  5. Envía reporte por Telegram
  6. Guarda resultado en BD (model_performance)
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

# ── Umbrales por defecto ──────────────────────────────────────────────────────

BASELINE = {
    "min_win_rate":       45.0,    # % mínimo aceptable
    "min_profit_factor":  1.2,     # PF mínimo aceptable
    "max_drawdown_pct":   10.0,    # drawdown máximo % del capital
    "min_trades_period":  5,       # mínimo trades para evaluar
    "min_expectancy":     0.0,     # expectancy mínima (€/trade)
}

# Niveles de degradación
DEGRADE_WARN   = 0.20   # 20% peor que baseline → warning
DEGRADE_ALERT  = 0.35   # 35% peor → alerta crítica
DEGRADE_PAUSE  = 0.50   # 50% peor → recomendar pausar trading

ACCOUNT_SIZE = 10_000.0

# ── Dataclasses ───────────────────────────────────────────────────────────────

@dataclass
class HealthMetrics:
    pair:             str
    timeframe:        str
    period_days:      int
    total_signals:    int     = 0
    closed_signals:   int     = 0
    wins:             int     = 0
    losses:           int     = 0
    timeouts:         int     = 0
    win_rate:         float   = 0.0
    profit_factor:    float   = 0.0
    total_pnl_pips:   float   = 0.0
    total_pnl_eur:    float   = 0.0
    max_drawdown_eur: float   = 0.0
    max_drawdown_pct: float   = 0.0
    expectancy_eur:   float   = 0.0
    avg_confidence:   float   = 0.0
    # Diagnóstico
    status:           str     = "ok"     # ok | warning | alert | critical
    issues:           list    = field(default_factory=list)


@dataclass
class HealthReport:
    timestamp:        str
    period_days:      int
    total_pairs:      int     = 0
    healthy_pairs:    int     = 0
    warning_pairs:    int     = 0
    alert_pairs:      int     = 0
    critical_pairs:   int     = 0
    overall_status:   str     = "ok"
    global_pnl_eur:   float   = 0.0
    global_win_rate:  float   = 0.0
    global_pf:        float   = 0.0
    metrics:          list    = field(default_factory=list)
    recommendations:  list    = field(default_factory=list)


# ── Carga de datos ────────────────────────────────────────────────────────────

def _load_closed_signals(engine, days: int) -> pd.DataFrame:
    """Carga señales cerradas de los últimos N días."""
    cutoff = datetime.now(timezone.utc) - timedelta(days=days)
    df = pd.read_sql(
        text("""
            SELECT pair, timeframe, direction, model_confidence,
                   status, pnl_pips, pnl_usd, created_at, closed_at
            FROM signals_log
            WHERE created_at >= :cutoff
              AND status != 'open'
            ORDER BY created_at
        """),
        engine,
        params={"cutoff": cutoff},
    )
    return df


def _load_open_signals(engine) -> pd.DataFrame:
    """Carga señales abiertas actualmente."""
    df = pd.read_sql(
        text("""
            SELECT pair, timeframe, direction, model_confidence,
                   created_at, entry_price, sl_price, tp1_price
            FROM signals_log
            WHERE status = 'open'
            ORDER BY created_at
        """),
        engine,
    )
    return df


def _load_latest_walkforward(engine) -> dict:
    """
    Carga las métricas más recientes de walk-forward OOS
    para usar como baseline de comparación.
    """
    df = pd.read_sql(
        text("""
            SELECT pair, timeframe, hit_rate, profit_factor,
                   sharpe_ratio, max_drawdown, total_trades
            FROM model_performance
            WHERE is_oos = TRUE
            ORDER BY recorded_at DESC
            LIMIT 50
        """),
        engine,
    )
    if df.empty:
        return {}

    baseline = {}
    for _, row in df.iterrows():
        key = f"{row['pair']}_{row['timeframe']}"
        if key not in baseline:
            baseline[key] = {
                "win_rate":       float(row["hit_rate"] or 0) * 100,
                "profit_factor":  float(row["profit_factor"] or 0),
                "max_drawdown":   float(row["max_drawdown"] or 0),
            }
    return baseline


# ── Cálculo de métricas ──────────────────────────────────────────────────────

def _compute_health(
    df: pd.DataFrame,
    pair: str,
    tf: str,
    period_days: int,
    baseline_override: dict = None,
) -> HealthMetrics:
    """Calcula las métricas de salud para un par/tf."""

    h = HealthMetrics(pair=pair, timeframe=tf, period_days=period_days)

    subset = df[(df["pair"] == pair) & (df["timeframe"] == tf)]
    h.total_signals  = len(subset)
    h.closed_signals = len(subset)

    if h.closed_signals == 0:
        h.status = "warning"
        h.issues.append(f"Sin señales cerradas en {period_days} días")
        return h

    wins     = subset[subset["status"].isin(["tp1", "tp2"])]
    losses   = subset[subset["status"] == "sl"]
    timeouts = subset[subset["status"] == "expired"]

    h.wins     = len(wins)
    h.losses   = len(losses)
    h.timeouts = len(timeouts)

    effective = h.wins + h.losses
    h.win_rate = (h.wins / effective * 100) if effective > 0 else 0.0

    h.total_pnl_pips = float(subset["pnl_pips"].sum()) if "pnl_pips" in subset else 0.0
    h.total_pnl_eur  = float(subset["pnl_usd"].sum()) if "pnl_usd" in subset else 0.0

    gross_profit = float(subset[subset["pnl_usd"] > 0]["pnl_usd"].sum()) if "pnl_usd" in subset else 0.0
    gross_loss   = abs(float(subset[subset["pnl_usd"] < 0]["pnl_usd"].sum())) if "pnl_usd" in subset else 0.0
    h.profit_factor = round(gross_profit / gross_loss, 2) if gross_loss > 0 else float("inf")

    h.avg_confidence = float(subset["model_confidence"].mean()) if "model_confidence" in subset else 0.0

    if effective > 0:
        avg_win  = float(wins["pnl_usd"].mean())   if len(wins)   > 0 else 0.0
        avg_loss = float(losses["pnl_usd"].mean())  if len(losses) > 0 else 0.0
        wr = h.wins / effective
        h.expectancy_eur = round(wr * avg_win + (1 - wr) * avg_loss, 2)

    # Drawdown sobre equity curve
    equity = ACCOUNT_SIZE
    peak   = ACCOUNT_SIZE
    max_dd = 0.0
    for pnl in subset.sort_values("created_at")["pnl_usd"].fillna(0):
        equity += float(pnl)
        peak = max(peak, equity)
        max_dd = max(max_dd, peak - equity)
    h.max_drawdown_eur = round(max_dd, 2)
    h.max_drawdown_pct = round(max_dd / ACCOUNT_SIZE * 100, 2)

    # ── Diagnóstico ───────────────────────────────────────────────────
    bl = baseline_override or BASELINE

    if effective < BASELINE["min_trades_period"]:
        h.status = "warning"
        h.issues.append(f"Pocos trades ({effective}) para evaluar")
        return h

    # Verificar cada métrica
    if h.win_rate < bl.get("min_win_rate", BASELINE["min_win_rate"]):
        h.issues.append(f"Win rate {h.win_rate:.1f}% bajo mínimo {bl.get('min_win_rate', BASELINE['min_win_rate'])}%")

    pf_limit = bl.get("min_profit_factor", BASELINE["min_profit_factor"])
    if h.profit_factor != float("inf") and h.profit_factor < pf_limit:
        h.issues.append(f"Profit factor {h.profit_factor:.2f} bajo mínimo {pf_limit}")

    if h.max_drawdown_pct > BASELINE["max_drawdown_pct"]:
        h.issues.append(f"Drawdown {h.max_drawdown_pct:.1f}% supera máximo {BASELINE['max_drawdown_pct']}%")

    if h.expectancy_eur < BASELINE["min_expectancy"]:
        h.issues.append(f"Expectancy {h.expectancy_eur:.2f}€ negativa")

    # Determinar severidad
    n_issues = len(h.issues)
    if n_issues == 0:
        h.status = "ok"
    elif n_issues == 1:
        h.status = "warning"
    elif n_issues <= 2:
        h.status = "alert"
    else:
        h.status = "critical"

    # Degradación severa: drawdown > umbral de pausa
    if h.max_drawdown_pct > BASELINE["max_drawdown_pct"] * (1 + DEGRADE_PAUSE):
        h.status = "critical"
        h.issues.append("⚠️ RECOMENDAR PAUSAR TRADING")

    return h


# ── Generador de recomendaciones ──────────────────────────────────────────────

def _generate_recommendations(metrics: list[HealthMetrics]) -> list[str]:
    """Genera recomendaciones basadas en las métricas de todos los pares."""
    recs = []

    critical = [m for m in metrics if m.status == "critical"]
    alerts   = [m for m in metrics if m.status == "alert"]
    warnings = [m for m in metrics if m.status == "warning"]
    no_trades = [m for m in metrics if m.closed_signals == 0]

    if critical:
        pairs_str = ", ".join(f"{m.pair}/{m.timeframe}" for m in critical)
        recs.append(f"🔴 CRÍTICO: Pausar trading en {pairs_str} hasta revisión manual")

    if no_trades:
        pairs_str = ", ".join(f"{m.pair}/{m.timeframe}" for m in no_trades)
        recs.append(f"⚠️ Sin actividad: {pairs_str} — verificar filtros o datos")

    # Pares con drawdown alto
    high_dd = [m for m in metrics if m.max_drawdown_pct > 5.0 and m.closed_signals > 0]
    if high_dd:
        for m in high_dd:
            recs.append(f"📉 {m.pair}/{m.timeframe}: DD {m.max_drawdown_pct:.1f}% — considerar reducir posición")

    # Pares con win rate bajo pero PF alto (RR bueno, pocas operaciones ganadoras)
    low_wr_high_rr = [m for m in metrics if m.win_rate < 40 and m.profit_factor > 1.5 and m.closed_signals >= 5]
    if low_wr_high_rr:
        for m in low_wr_high_rr:
            recs.append(f"📊 {m.pair}/{m.timeframe}: WR bajo ({m.win_rate:.0f}%) pero PF alto ({m.profit_factor:.1f}) — buena gestión de RR")

    # Pares con confianza promedio baja
    low_conf = [m for m in metrics if 0 < m.avg_confidence < 0.58 and m.closed_signals >= 5]
    if low_conf:
        pairs_str = ", ".join(f"{m.pair}" for m in low_conf)
        recs.append(f"🤖 Confianza baja en {pairs_str} — considerar reentrenamiento o subir umbral")

    # Si todo está bien
    if not recs:
        recs.append("✅ Todos los pares dentro de parámetros aceptables")

    # Recomendación de reentrenamiento basada en degradación general
    total_closed = sum(m.closed_signals for m in metrics)
    total_wins   = sum(m.wins for m in metrics)
    if total_closed > 0:
        global_wr = total_wins / total_closed * 100
        if global_wr < 42:
            recs.append("🔄 Win rate global bajo — recomendado reentrenamiento inmediato")

    return recs


# ── Mensaje Telegram ──────────────────────────────────────────────────────────

def _build_telegram_message(report: HealthReport) -> str:
    """Construye mensaje HTML para Telegram."""
    status_emoji = {
        "ok":       "✅",
        "warning":  "⚠️",
        "alert":    "🟠",
        "critical": "🔴",
    }

    lines = [
        f"{status_emoji.get(report.overall_status, '❓')} <b>Model Health Check</b>",
        f"━━━━━━━━━━━━━━━━━━━━",
        f"📅 Período: últimos {report.period_days} días",
        f"💰 PnL global: <b>{report.global_pnl_eur:+.2f}€</b>",
        f"🎯 Win rate: {report.global_win_rate:.1f}%",
        f"📊 Profit factor: {report.global_pf:.2f}" if report.global_pf != float("inf") else "📊 Profit factor: ∞",
        "",
    ]

    # Estado por par
    for m in report.metrics:
        emoji = status_emoji.get(m["status"], "❓")
        pf = m["profit_factor"]
        pf_str = f"{pf:.1f}" if pf != float("inf") else "∞"
        lines.append(
            f"{emoji} <b>{m['pair']}/{m['timeframe']}</b>: "
            f"WR {m['win_rate']:.0f}% | PF {pf_str} | "
            f"PnL {m['total_pnl_eur']:+.0f}€ | {m['closed_signals']} trades"
        )
        if m["issues"]:
            for issue in m["issues"][:2]:
                lines.append(f"   ↳ {issue}")

    if report.recommendations:
        lines.append("")
        lines.append("<b>Recomendaciones:</b>")
        for rec in report.recommendations[:5]:
            lines.append(f"  • {rec}")

    lines.append("")
    lines.append(f"<i>🕐 {report.timestamp}</i>")
    return "\n".join(lines)


# ── Guardado en BD ────────────────────────────────────────────────────────────

def _save_to_db(engine, report: HealthReport) -> None:
    """Guarda métricas de health check en model_performance."""
    with engine.connect() as conn:
        for m in report.metrics:
            conn.execute(
                text("""
                    INSERT INTO model_performance
                        (model_version, pair, timeframe, period_start, period_end,
                         hit_rate, profit_factor, max_drawdown, total_trades, is_oos)
                    VALUES
                        (:ver, :pair, :tf, :ps, :pe, :hr, :pf, :dd, :tt, FALSE)
                """),
                {
                    "ver":  "health_check",
                    "pair": m["pair"],
                    "tf":   m["timeframe"],
                    "ps":   (datetime.now(timezone.utc) - timedelta(days=report.period_days)).date(),
                    "pe":   datetime.now(timezone.utc).date(),
                    "hr":   m["win_rate"] / 100,
                    "pf":   m["profit_factor"] if m["profit_factor"] != float("inf") else 99.99,
                    "dd":   m["max_drawdown_eur"],
                    "tt":   m["closed_signals"],
                },
            )
        conn.commit()
    logger.info("Métricas guardadas en model_performance")


# ── Motor principal ───────────────────────────────────────────────────────────

ALL_PAIRS      = ["EURUSD", "GBPUSD", "USDJPY", "EURJPY", "XAUUSD"]
ALL_TIMEFRAMES = ["H1", "H4"]


def run_health_check(
    days:           int   = 30,
    degrade_thresh: float = DEGRADE_ALERT,
    auto_retrain:   bool  = False,
    notify:         bool  = True,
    save_db:        bool  = True,
) -> HealthReport:

    engine = create_engine(DATABASE_URL)

    logger.info("═" * 60)
    logger.info("ML-Ayram — Model Health Check")
    logger.info(f"Período: últimos {days} días")
    logger.info("═" * 60)

    # Cargar datos
    df_closed = _load_closed_signals(engine, days)
    wf_baselines = _load_latest_walkforward(engine)

    logger.info(f"Señales cerradas cargadas: {len(df_closed)}")

    # Calcular métricas por par/tf
    all_metrics = []
    for tf in ALL_TIMEFRAMES:
        for pair in ALL_PAIRS:
            key = f"{pair}_{tf}"
            bl = wf_baselines.get(key, None)

            m = _compute_health(df_closed, pair, tf, days, bl)
            all_metrics.append(m)

            status_icon = {"ok": "✅", "warning": "⚠️", "alert": "🟠", "critical": "🔴"}.get(m.status, "?")
            logger.info(
                f"  {status_icon} {pair}/{tf}: {m.closed_signals} trades | "
                f"WR {m.win_rate:.1f}% | PF {m.profit_factor} | "
                f"PnL {m.total_pnl_eur:+.2f}€ | DD {m.max_drawdown_pct:.1f}%"
            )
            for issue in m.issues:
                logger.warning(f"     ↳ {issue}")

    # Construir reporte global
    recs = _generate_recommendations(all_metrics)

    total_closed = sum(m.closed_signals for m in all_metrics)
    total_wins   = sum(m.wins for m in all_metrics)
    total_pnl    = sum(m.total_pnl_eur for m in all_metrics)
    global_wr    = (total_wins / total_closed * 100) if total_closed > 0 else 0.0

    gross_p = sum(m.total_pnl_eur for m in all_metrics if m.total_pnl_eur > 0)
    gross_l = abs(sum(m.total_pnl_eur for m in all_metrics if m.total_pnl_eur < 0))
    global_pf = round(gross_p / gross_l, 2) if gross_l > 0 else float("inf")

    # Determinar estado global
    statuses = [m.status for m in all_metrics]
    if "critical" in statuses:
        overall = "critical"
    elif statuses.count("alert") >= 2:
        overall = "alert"
    elif "alert" in statuses or statuses.count("warning") >= 3:
        overall = "warning"
    else:
        overall = "ok"

    report = HealthReport(
        timestamp       = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        period_days     = days,
        total_pairs     = len(all_metrics),
        healthy_pairs   = sum(1 for m in all_metrics if m.status == "ok"),
        warning_pairs   = sum(1 for m in all_metrics if m.status == "warning"),
        alert_pairs     = sum(1 for m in all_metrics if m.status == "alert"),
        critical_pairs  = sum(1 for m in all_metrics if m.status == "critical"),
        overall_status  = overall,
        global_pnl_eur  = round(total_pnl, 2),
        global_win_rate = round(global_wr, 1),
        global_pf       = global_pf,
        metrics         = [asdict(m) for m in all_metrics],
        recommendations = recs,
    )

    # Imprimir resumen
    logger.info("")
    logger.info("═" * 60)
    logger.info(f"ESTADO GLOBAL: {overall.upper()}")
    logger.info(f"PnL total: {total_pnl:+.2f}€ | WR: {global_wr:.1f}% | PF: {global_pf}")
    logger.info(f"Pares: {report.healthy_pairs} OK / {report.warning_pairs} warn / "
                f"{report.alert_pairs} alert / {report.critical_pairs} critical")
    for rec in recs:
        logger.info(f"  → {rec}")
    logger.info("═" * 60)

    # Guardar en BD
    if save_db:
        try:
            _save_to_db(engine, report)
        except Exception as e:
            logger.error(f"Error guardando en BD: {e}")

    # Guardar JSON
    output_path = Path(f"results/health_{datetime.now().strftime('%Y%m%d_%H%M')}.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    data = asdict(report) if hasattr(report, '__dataclass_fields__') else report.__dict__
    # Manejar inf en JSON
    output_path.write_text(json.dumps(data, indent=2, default=str).replace(": Infinity", ": \"inf\""))
    logger.info(f"Reporte guardado: {output_path}")

    # Notificar Telegram
    if notify:
        try:
            from src.notifications.telegram import send_message
            msg = _build_telegram_message(report)
            send_message(msg)
            logger.info("Reporte enviado por Telegram")
        except Exception as e:
            logger.error(f"Error enviando Telegram: {e}")

    # Auto-retrain si hay degradación crítica
    if auto_retrain and overall in ("alert", "critical"):
        logger.warning("⚡ Degradación detectada — lanzando reentrenamiento automático")
        try:
            import subprocess
            subprocess.Popen(
                ["systemctl", "start", "ayram-train.service"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            logger.info("Reentrenamiento lanzado vía systemd")
        except Exception as e:
            logger.error(f"No se pudo lanzar reentrenamiento: {e}")

    return report


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="ML-Ayram Model Health Check")
    parser.add_argument("--days",        type=int,   default=30,           help="Período en días (default: 30)")
    parser.add_argument("--threshold",   type=float, default=DEGRADE_ALERT, help="Umbral degradación (default: 0.35)")
    parser.add_argument("--auto-retrain", action="store_true",             help="Lanzar reentrenamiento si degrada")
    parser.add_argument("--no-notify",   action="store_true",              help="No enviar Telegram")
    parser.add_argument("--no-save",     action="store_true",              help="No guardar en BD")
    args = parser.parse_args()

    run_health_check(
        days           = args.days,
        degrade_thresh = args.threshold,
        auto_retrain   = args.auto_retrain,
        notify         = not args.no_notify,
        save_db        = not args.no_save,
    )


if __name__ == "__main__":
    main()
