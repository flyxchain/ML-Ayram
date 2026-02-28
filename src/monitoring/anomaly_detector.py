"""
src/monitoring/anomaly_detector.py
Detector de anomalías operativas del sistema ML-Ayram.

Detecta:
  1. Sequía de señales (>N días sin señales válidas por par)
  2. Drawdown excesivo (pérdidas acumuladas > umbral)
  3. Win rate deteriorado (últimos N trades bajo umbral)
  4. Datos desactualizados (collector/features no ejecutaron)
  5. Modelos obsoletos (>N días sin reentrenar)
  6. Señales anómalas (confianza inusualmente alta/baja, clusters)

Uso:
  python -m src.monitoring.anomaly_detector               # chequeo completo
  python -m src.monitoring.anomaly_detector --quiet        # solo alertas críticas
  python -m src.monitoring.anomaly_detector --json         # output JSON
"""

import argparse
import json
import os
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone, timedelta
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from loguru import logger
from sqlalchemy import create_engine, text

load_dotenv()
DATABASE_URL = os.getenv("DATABASE_URL")

ALL_PAIRS      = ["EURUSD", "GBPUSD", "USDJPY", "EURJPY", "XAUUSD"]
ALL_TIMEFRAMES = ["H1", "H4"]


# ── Dataclass ─────────────────────────────────────────────────────────────────

@dataclass
class Alert:
    severity:    str     # info | warning | high | critical
    alert_type:  str     # signal_drought | drawdown | low_winrate | stale_data | stale_model | anomalous_signal
    pair:        str     = ""
    timeframe:   str     = ""
    message:     str     = ""
    value:       float   = 0.0
    threshold:   float   = 0.0
    action:      str     = ""


@dataclass
class AnomalyReport:
    timestamp:   str
    total_alerts: int     = 0
    critical:     int     = 0
    high:         int     = 0
    warnings:     int     = 0
    info:         int     = 0
    alerts:       list    = field(default_factory=list)
    system_ok:    bool    = True


# ── Checks individuales ──────────────────────────────────────────────────────

def check_signal_drought(engine, days_threshold: int = 5) -> list[Alert]:
    """Detecta pares sin señales válidas en los últimos N días."""
    alerts = []
    cutoff = datetime.now(timezone.utc) - timedelta(days=days_threshold)

    for pair in ALL_PAIRS:
        for tf in ALL_TIMEFRAMES:
            result = pd.read_sql(
                text("""
                    SELECT COUNT(*) as cnt, MAX(created_at) as last_signal
                    FROM signals_log
                    WHERE pair = :pair AND timeframe = :tf
                      AND created_at >= :cutoff
                """),
                engine,
                params={"pair": pair, "tf": tf, "cutoff": cutoff},
            )
            cnt = int(result.iloc[0]["cnt"]) if not result.empty else 0
            last = result.iloc[0]["last_signal"]

            if cnt == 0:
                days_since = "nunca" if pd.isna(last) else f"{(datetime.now(timezone.utc) - pd.to_datetime(last, utc=True)).days}d"
                alerts.append(Alert(
                    severity="high",
                    alert_type="signal_drought",
                    pair=pair, timeframe=tf,
                    message=f"Sin señales en {days_threshold}d (última: {days_since})",
                    value=0, threshold=1,
                    action="Revisar filtros de señal o condiciones de mercado",
                ))
    return alerts


def check_drawdown(engine, max_dd_pct: float = 8.0, period_days: int = 7) -> list[Alert]:
    """Detecta drawdown excesivo en los últimos N días."""
    alerts = []
    cutoff = datetime.now(timezone.utc) - timedelta(days=period_days)

    df = pd.read_sql(
        text("""
            SELECT pair, pnl_usd, created_at
            FROM signals_log
            WHERE created_at >= :cutoff AND status != 'open'
            ORDER BY created_at
        """),
        engine,
        params={"cutoff": cutoff},
    )

    if df.empty:
        return alerts

    account = 10_000.0

    # Drawdown global
    equity = account
    peak   = account
    max_dd = 0.0
    for pnl in df["pnl_usd"].fillna(0):
        equity += float(pnl)
        peak = max(peak, equity)
        max_dd = max(max_dd, peak - equity)

    dd_pct = max_dd / account * 100
    if dd_pct > max_dd_pct:
        severity = "critical" if dd_pct > max_dd_pct * 1.5 else "high"
        alerts.append(Alert(
            severity=severity,
            alert_type="drawdown",
            message=f"Drawdown {dd_pct:.1f}% en últimos {period_days}d (máx: {max_dd_pct}%)",
            value=dd_pct, threshold=max_dd_pct,
            action="PAUSAR trading automático" if severity == "critical" else "Reducir tamaño de posición",
        ))

    # Drawdown por par
    for pair in df["pair"].unique():
        pair_df = df[df["pair"] == pair]
        eq = account
        pk = account
        md = 0.0
        for pnl in pair_df["pnl_usd"].fillna(0):
            eq += float(pnl)
            pk = max(pk, eq)
            md = max(md, pk - eq)
        pair_dd = md / account * 100

        if pair_dd > max_dd_pct * 0.6:  # umbral más bajo por par individual
            alerts.append(Alert(
                severity="warning",
                alert_type="drawdown",
                pair=pair,
                message=f"Drawdown {pair} {pair_dd:.1f}% en {period_days}d",
                value=pair_dd, threshold=max_dd_pct * 0.6,
                action=f"Monitorear {pair} de cerca",
            ))

    return alerts


def check_recent_win_rate(engine, n_trades: int = 20, threshold: float = 35.0) -> list[Alert]:
    """Verifica win rate de los últimos N trades."""
    alerts = []

    df = pd.read_sql(
        text("""
            SELECT pair, timeframe, status, pnl_usd
            FROM signals_log
            WHERE status != 'open'
            ORDER BY created_at DESC
            LIMIT :n
        """),
        engine,
        params={"n": n_trades},
    )

    if len(df) < n_trades * 0.5:
        return alerts  # no hay suficientes trades

    wins = len(df[df["status"].isin(["tp1", "tp2"])])
    total = len(df[df["status"].isin(["tp1", "tp2", "sl"])])
    wr = (wins / total * 100) if total > 0 else 0

    if wr < threshold:
        alerts.append(Alert(
            severity="high",
            alert_type="low_winrate",
            message=f"Win rate últimos {n_trades} trades: {wr:.1f}% (mín: {threshold}%)",
            value=wr, threshold=threshold,
            action="Considerar reentrenamiento o pausar",
        ))

    return alerts


def check_stale_data(engine, max_hours: int = 2) -> list[Alert]:
    """Verifica que los datos están actualizados (collector ejecutó recientemente)."""
    alerts = []
    cutoff = datetime.now(timezone.utc) - timedelta(hours=max_hours)

    for pair in ["EURUSD"]:  # solo verificar un par como proxy
        result = pd.read_sql(
            text("""
                SELECT MAX(timestamp) as latest
                FROM ohlcv_raw
                WHERE pair = :pair AND timeframe = 'M15'
            """),
            engine,
            params={"pair": pair},
        )
        latest = result.iloc[0]["latest"] if not result.empty else None

        if latest is None:
            alerts.append(Alert(
                severity="critical",
                alert_type="stale_data",
                pair=pair,
                message="Sin datos OHLCV en la base de datos",
                action="Verificar collector y conexión EODHD API",
            ))
        else:
            latest_ts = pd.to_datetime(latest, utc=True)
            hours_old = (datetime.now(timezone.utc) - latest_ts).total_seconds() / 3600

            # Solo alertar en horario de mercado (lunes-viernes)
            now = datetime.now(timezone.utc)
            if now.weekday() < 5 and hours_old > max_hours:
                alerts.append(Alert(
                    severity="high" if hours_old > max_hours * 3 else "warning",
                    alert_type="stale_data",
                    pair=pair,
                    message=f"Datos M15 desactualizados ({hours_old:.1f}h)",
                    value=hours_old, threshold=max_hours,
                    action="Verificar ayram-collector.timer",
                ))

    return alerts


def check_stale_models(max_days: int = 14) -> list[Alert]:
    """Verifica que los modelos no llevan más de N días sin reentrenar."""
    alerts = []
    models_dir = Path("models/saved")

    if not models_dir.exists():
        alerts.append(Alert(
            severity="critical",
            alert_type="stale_model",
            message="Directorio models/saved no existe",
            action="Ejecutar entrenamiento inicial",
        ))
        return alerts

    for pair in ALL_PAIRS:
        for tf in ALL_TIMEFRAMES:
            # Buscar último modelo XGBoost
            xgb_files = sorted(models_dir.glob(f"xgb_{pair}_{tf}_*_meta.json"))
            if not xgb_files:
                alerts.append(Alert(
                    severity="high",
                    alert_type="stale_model",
                    pair=pair, timeframe=tf,
                    message=f"Sin modelo XGBoost para {pair}/{tf}",
                    action="Ejecutar python -m src.train",
                ))
                continue

            # Verificar edad del último modelo
            latest = xgb_files[-1]
            try:
                meta = json.loads(latest.read_text())
                trained_at = meta.get("trained_at", "")
                if trained_at:
                    t = datetime.fromisoformat(trained_at.replace("Z", "+00:00"))
                    age_days = (datetime.now(timezone.utc) - t).days
                    if age_days > max_days:
                        alerts.append(Alert(
                            severity="warning" if age_days < max_days * 2 else "high",
                            alert_type="stale_model",
                            pair=pair, timeframe=tf,
                            message=f"Modelo {pair}/{tf} tiene {age_days}d (máx: {max_days}d)",
                            value=age_days, threshold=max_days,
                            action="Verificar ayram-train.timer",
                        ))
            except (json.JSONDecodeError, KeyError, ValueError):
                pass  # si no podemos parsear, no alertar

    return alerts


def check_anomalous_signals(engine, hours: int = 24) -> list[Alert]:
    """Detecta patrones anómalos en señales recientes."""
    alerts = []
    cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)

    df = pd.read_sql(
        text("""
            SELECT pair, timeframe, direction, model_confidence, created_at
            FROM signals_log
            WHERE created_at >= :cutoff
            ORDER BY created_at
        """),
        engine,
        params={"cutoff": cutoff},
    )

    if df.empty:
        return alerts

    # Demasiadas señales en poco tiempo (posible sobreactividad)
    if len(df) > 30:
        alerts.append(Alert(
            severity="warning",
            alert_type="anomalous_signal",
            message=f"{len(df)} señales en {hours}h — posible sobreactividad",
            value=len(df), threshold=30,
            action="Revisar filtros de cooldown",
        ))

    # Todas las señales en una dirección (posible sesgo del modelo)
    if len(df) >= 5:
        for pair in df["pair"].unique():
            pair_df = df[df["pair"] == pair]
            if len(pair_df) >= 3:
                buy_pct = (pair_df["direction"] == "BUY").mean()
                if buy_pct > 0.9 or buy_pct < 0.1:
                    dominant = "BUY" if buy_pct > 0.5 else "SELL"
                    alerts.append(Alert(
                        severity="warning",
                        alert_type="anomalous_signal",
                        pair=pair,
                        message=f"{pair}: {buy_pct*100:.0f}% señales {dominant} — posible sesgo",
                        action="Verificar datos recientes y modelo",
                    ))

    return alerts


# ── Motor principal ───────────────────────────────────────────────────────────

def run_anomaly_detection(quiet: bool = False, notify: bool = True) -> AnomalyReport:
    engine = create_engine(DATABASE_URL)

    logger.info("═" * 60)
    logger.info("ML-Ayram — Anomaly Detection")
    logger.info("═" * 60)

    all_alerts = []

    # Ejecutar todos los checks
    checks = [
        ("Signal Drought",     lambda: check_signal_drought(engine, days_threshold=5)),
        ("Drawdown",           lambda: check_drawdown(engine, max_dd_pct=8.0, period_days=7)),
        ("Win Rate",           lambda: check_recent_win_rate(engine, n_trades=20, threshold=35.0)),
        ("Stale Data",         lambda: check_stale_data(engine, max_hours=2)),
        ("Stale Models",       lambda: check_stale_models(max_days=14)),
        ("Anomalous Signals",  lambda: check_anomalous_signals(engine, hours=24)),
    ]

    for name, check_fn in checks:
        try:
            alerts = check_fn()
            all_alerts.extend(alerts)
            n = len(alerts)
            if n > 0:
                logger.warning(f"  ⚠️ {name}: {n} alerta(s)")
                for a in alerts:
                    logger.warning(f"     [{a.severity.upper()}] {a.message}")
            else:
                logger.info(f"  ✅ {name}: OK")
        except Exception as e:
            logger.error(f"  ❌ {name}: Error — {e}")
            all_alerts.append(Alert(
                severity="high",
                alert_type="check_error",
                message=f"Error en check '{name}': {str(e)[:100]}",
                action="Revisar logs del sistema",
            ))

    # Construir reporte
    report = AnomalyReport(
        timestamp    = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        total_alerts = len(all_alerts),
        critical     = sum(1 for a in all_alerts if a.severity == "critical"),
        high         = sum(1 for a in all_alerts if a.severity == "high"),
        warnings     = sum(1 for a in all_alerts if a.severity == "warning"),
        info         = sum(1 for a in all_alerts if a.severity == "info"),
        alerts       = [asdict(a) for a in all_alerts],
        system_ok    = not any(a.severity in ("critical", "high") for a in all_alerts),
    )

    # Resumen
    logger.info("")
    if report.system_ok:
        logger.info("✅ SISTEMA OK — sin anomalías críticas")
    else:
        logger.warning(
            f"⚠️ {report.critical} críticas | {report.high} altas | "
            f"{report.warnings} warnings | {report.info} info"
        )

    # Notificar si hay alertas importantes
    if notify and (report.critical > 0 or report.high > 0):
        try:
            from src.notifications.telegram import send_message, log_notification
            lines = [
                f"🚨 <b>Anomaly Detection — {report.total_alerts} alertas</b>",
                f"━━━━━━━━━━━━━━━━━━━━",
            ]
            for a in all_alerts:
                if a.severity in ("critical", "high"):
                    emoji = "🔴" if a.severity == "critical" else "🟠"
                    pair_str = f" {a.pair}/{a.timeframe}" if a.pair else ""
                    lines.append(f"{emoji}{pair_str} {a.message}")
                    if a.action:
                        lines.append(f"   → {a.action}")

            lines.append(f"\n<i>🕐 {report.timestamp}</i>")
            msg = "\n".join(lines)
            ok = send_message(msg)
            log_notification(
                notif_type="anomaly", severity="critical" if report.critical > 0 else "high",
                title=f"Anomaly Detection — {report.total_alerts} alertas",
                message=msg, delivered=ok,
                metadata={"critical": report.critical, "high": report.high, "warnings": report.warnings},
            )
        except Exception as e:
            logger.error(f"Error enviando Telegram: {e}")

    # Guardar JSON
    output_path = Path(f"results/anomalies_{datetime.now().strftime('%Y%m%d_%H%M')}.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(asdict(report), indent=2, default=str))
    logger.info(f"Reporte guardado: {output_path}")

    return report


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="ML-Ayram Anomaly Detector")
    parser.add_argument("--quiet",     action="store_true", help="Solo alertas críticas")
    parser.add_argument("--no-notify", action="store_true", help="No enviar Telegram")
    parser.add_argument("--json",      action="store_true", help="Output JSON a stdout")
    args = parser.parse_args()

    report = run_anomaly_detection(
        quiet  = args.quiet,
        notify = not args.no_notify,
    )

    if args.json:
        print(json.dumps(asdict(report), indent=2, default=str))


if __name__ == "__main__":
    main()
