"""
src/notifications/telegram.py
Envía notificaciones de señales de trading al bot de Telegram.

Configuración necesaria en .env:
    TELEGRAM_BOT_TOKEN=123456:ABC-...
    TELEGRAM_CHAT_ID=-100123456789   # puede ser un chat personal o grupo

Tipos de mensaje:
  - send_signal()      → señal nueva (LONG / SHORT)
  - send_summary()     → resumen periódico de señales
  - send_error()       → alerta de error crítico del sistema
  - send_heartbeat()   → "sigo vivo" cada N horas
"""

import json
import os
import time
import requests
from datetime import datetime, timezone
from typing import Optional

from dotenv import load_dotenv
from loguru import logger
from sqlalchemy import create_engine, text

load_dotenv()

BOT_TOKEN     = os.getenv("TELEGRAM_BOT_TOKEN")
CHAT_ID       = os.getenv("TELEGRAM_CHAT_ID")
DATABASE_URL  = os.getenv("DATABASE_URL")

TELEGRAM_API = "https://api.telegram.org/bot{token}/{method}"

# Emojis por dirección
DIR_EMOJI = {1: "🟢", -1: "🔴", 0: "⚪"}
DIR_LABEL = {1: "LONG", -1: "SHORT", 0: "NEUTRAL"}

# Emojis por sesión
SESSION_EMOJI = {
    "london":    "🇬🇧",
    "newyork":   "🗽",
    "tokyo":     "🗼",
    "overlap":   "⚡",
    "offmarket": "🌙",
}

# Emojis por par
PAIR_EMOJI = {
    "EURUSD": "🇪🇺",
    "GBPUSD": "🇬🇧",
    "USDJPY": "🇯🇵",
    "EURJPY": "🌍",
    "XAUUSD": "🥇",
}


# ── Logging de notificaciones a BD ────────────────────────────────────────

_table_ensured = False

def _ensure_log_table():
    """Crea la tabla notification_log si no existe (una sola vez por ejecución)."""
    global _table_ensured
    if _table_ensured or not DATABASE_URL:
        return
    try:
        engine = create_engine(DATABASE_URL)
        with engine.connect() as conn:
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS notification_log (
                    id          BIGSERIAL       PRIMARY KEY,
                    created_at  TIMESTAMPTZ     DEFAULT NOW(),
                    notif_type  VARCHAR(30)     NOT NULL,
                    severity    VARCHAR(10)     DEFAULT 'info',
                    title       VARCHAR(200),
                    message     TEXT,
                    pair        VARCHAR(10),
                    timeframe   VARCHAR(5),
                    delivered   BOOLEAN         DEFAULT TRUE,
                    metadata    JSONB
                )
            """))
            conn.execute(text("CREATE INDEX IF NOT EXISTS idx_notif_created ON notification_log (created_at DESC)"))
            conn.commit()
        _table_ensured = True
    except Exception as e:
        logger.debug(f"notification_log table ensure skip: {e}")


def log_notification(
    notif_type: str,
    message: str,
    title: str = "",
    severity: str = "info",
    pair: str = None,
    timeframe: str = None,
    delivered: bool = True,
    metadata: dict = None,
) -> None:
    """Guarda una notificación enviada en la BD."""
    if not DATABASE_URL:
        return
    _ensure_log_table()
    try:
        engine = create_engine(DATABASE_URL)
        with engine.connect() as conn:
            conn.execute(
                text("""
                    INSERT INTO notification_log (notif_type, severity, title, message, pair, timeframe, delivered, metadata)
                    VALUES (:t, :s, :title, :msg, :pair, :tf, :delivered, :meta)
                """),
                {
                    "t": notif_type, "s": severity, "title": title,
                    "msg": message[:4000],  # limitar tamaño
                    "pair": pair, "tf": timeframe,
                    "delivered": delivered,
                    "meta": json.dumps(metadata) if metadata else None,
                },
            )
            conn.commit()
    except Exception as e:
        logger.debug(f"notification_log insert skip: {e}")


# ── Cliente HTTP básico ───────────────────────────────────────────────────────

def _post(method: str, payload: dict, retries: int = 3) -> bool:
    """Llama a la API de Telegram con reintentos."""
    if not BOT_TOKEN or not CHAT_ID:
        logger.warning("Telegram no configurado (BOT_TOKEN o CHAT_ID vacíos)")
        return False

    url = TELEGRAM_API.format(token=BOT_TOKEN, method=method)
    for attempt in range(1, retries + 1):
        try:
            resp = requests.post(url, json=payload, timeout=10)
            if resp.status_code == 200 and resp.json().get("ok"):
                return True
            logger.warning(f"Telegram error ({attempt}/{retries}): {resp.text}")
        except requests.RequestException as e:
            logger.warning(f"Telegram conexión fallida ({attempt}/{retries}): {e}")
        time.sleep(2 ** attempt)   # back-off exponencial: 2s, 4s, 8s

    logger.error("Telegram: no se pudo enviar el mensaje tras varios intentos")
    return False


def send_message(text: str, parse_mode: str = "HTML", silent: bool = False) -> bool:
    """Envía un mensaje de texto libre al chat configurado."""
    return _post("sendMessage", {
        "chat_id":              CHAT_ID,
        "text":                 text,
        "parse_mode":           parse_mode,
        "disable_notification": silent,
    })


# ── Mensajes específicos ──────────────────────────────────────────────────────

def send_signal(signal) -> bool:
    """
    Envía una notificación para una señal válida de trading.
    Acepta un objeto SignalResult de signals/generator.py
    """
    d     = signal.direction
    emoji = DIR_EMOJI.get(d, "⚪")
    label = DIR_LABEL.get(d, "?")
    pair_emoji    = PAIR_EMOJI.get(signal.pair, "💱")
    session_emoji = SESSION_EMOJI.get(signal.session, "🕐")

    # Barra de confianza visual  ████████░░  80%
    filled = round(signal.confidence * 10)
    bar    = "█" * filled + "░" * (10 - filled)

    ts_str = signal.timestamp.strftime("%d/%m/%Y %H:%M UTC") if hasattr(signal.timestamp, "strftime") else str(signal.timestamp)

    text = (
        f"{emoji} <b>{label} — {pair_emoji} {signal.pair} {signal.timeframe}</b>\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"📅 <b>Hora:</b> {ts_str}\n"
        f"{session_emoji} <b>Sesión:</b> {signal.session.capitalize()}\n"
        f"\n"
        f"💰 <b>Entrada:</b>  <code>{signal.entry_price:.5f}</code>\n"
        f"🎯 <b>Take Profit:</b> <code>{signal.tp_price:.5f}</code>  (+{signal.tp_pips:.1f} pips)\n"
        f"🛡 <b>Stop Loss:</b>  <code>{signal.sl_price:.5f}</code>  (-{signal.sl_pips:.1f} pips)\n"
        f"⚖️ <b>R:R:</b>  {signal.rr_ratio:.2f}\n"
        f"\n"
        f"🤖 <b>Confianza:</b>  {bar}  {signal.confidence:.0%}\n"
        f"   Long  {signal.prob_long:.0%}  |  Neutral {signal.prob_neutral:.0%}  |  Short {signal.prob_short:.0%}\n"
        f"   XGBoost: {DIR_LABEL.get(signal.xgb_direction,'?')}  |  LSTM: {DIR_LABEL.get(signal.lstm_direction,'?')}\n"
        f"\n"
        f"📊 <b>ADX:</b> {signal.adx:.1f}\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"<i>⚠️ No es asesoramiento financiero.</i>"
    )
    logger.info(f"Enviando señal Telegram: {signal.pair} {signal.timeframe} {label}")
    ok = send_message(text)
    log_notification(
        notif_type="signal", severity="info",
        title=f"{label} {signal.pair} {signal.timeframe}",
        message=text, pair=signal.pair, timeframe=signal.timeframe,
        delivered=ok,
        metadata={"confidence": signal.confidence, "rr": signal.rr_ratio, "direction": d},
    )
    return ok


def send_summary(signals: list, period: str = "última hora") -> bool:
    """
    Envía un resumen con todas las señales válidas de un período.
    signals: lista de SignalResult
    """
    if not signals:
        msg = f"📋 <b>Resumen señales — {period}</b>\n\nSin señales válidas en este período."
        ok = send_message(msg, silent=True)
        log_notification(notif_type="summary", title=f"Resumen {period}", message=msg, delivered=ok)
        return ok

    lines = [f"📋 <b>Resumen señales — {period}</b>\n"]
    for s in signals:
        emoji = DIR_EMOJI.get(s.direction, "⚪")
        lines.append(
            f"{emoji} {PAIR_EMOJI.get(s.pair,'')} <b>{s.pair}</b> {s.timeframe}  "
            f"{DIR_LABEL.get(s.direction,'')}  "
            f"Conf {s.confidence:.0%}  R:R {s.rr_ratio:.2f}"
        )

    lines.append(f"\n<i>Total señales: {len(signals)}</i>")
    msg = "\n".join(lines)
    ok = send_message(msg, silent=True)
    log_notification(notif_type="summary", title=f"Resumen {period} ({len(signals)} señales)", message=msg, delivered=ok)
    return ok


def send_error(error: str, context: Optional[str] = None) -> bool:
    """Alerta de error crítico del sistema."""
    ts = datetime.now(timezone.utc).strftime("%d/%m/%Y %H:%M UTC")
    msg = (
        f"🚨 <b>ERROR — ML-Ayram</b>\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"🕐 {ts}\n"
        f"📍 <b>Contexto:</b> {context or 'desconocido'}\n\n"
        f"<code>{error[:1000]}</code>"
    )
    ok = send_message(msg)
    log_notification(notif_type="error", severity="critical", title=f"Error: {context or 'sistema'}", message=msg, delivered=ok)
    return ok


def send_heartbeat(stats: Optional[dict] = None) -> bool:
    """
    Mensaje de 'sigo vivo' con estadísticas opcionales del sistema.
    stats puede tener: signals_today, pairs_active, last_signal_at
    """
    ts = datetime.now(timezone.utc).strftime("%d/%m/%Y %H:%M UTC")
    lines = [
        f"💓 <b>Heartbeat — ML-Ayram</b>",
        f"🕐 {ts}",
    ]
    if stats:
        lines.append("━━━━━━━━━━━━━━━━━━━━")
        if "signals_today" in stats:
            lines.append(f"📊 Señales hoy: <b>{stats['signals_today']}</b>")
        if "pairs_active" in stats:
            lines.append(f"💱 Pares activos: {stats['pairs_active']}")
        if "last_signal_at" in stats:
            lines.append(f"⏱ Última señal: {stats['last_signal_at']}")

    msg = "\n".join(lines)
    ok = send_message(msg, silent=True)
    log_notification(notif_type="heartbeat", title="Heartbeat", message=msg, delivered=ok, metadata=stats)
    return ok


# ── Test de conexión ──────────────────────────────────────────────────────────

def test_connection() -> bool:
    """Comprueba que el bot puede enviar mensajes."""
    ok = send_message("✅ <b>ML-Ayram conectado</b> — Telegram funcionando correctamente.")
    if ok:
        logger.success("Telegram: conexión OK")
    else:
        logger.error("Telegram: conexión FALLIDA")
    return ok


# ── Integración con el generador de señales ───────────────────────────────────

def notify_if_valid(signal) -> None:
    """
    Wrapper conveniente para usar desde generator.py:
    Si la señal es válida (direction != 0 y sin filter_reason) la notifica.
    """
    if signal and getattr(signal, 'direction', 0) != 0 and not getattr(signal, 'filter_reason', None):
        send_signal(signal)


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    test_connection()
