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

import os
import time
import requests
from datetime import datetime, timezone
from typing import Optional

from dotenv import load_dotenv
from loguru import logger

load_dotenv()

BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
CHAT_ID   = os.getenv("TELEGRAM_CHAT_ID")

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
    return send_message(text)


def send_summary(signals: list, period: str = "última hora") -> bool:
    """
    Envía un resumen con todas las señales válidas de un período.
    signals: lista de SignalResult
    """
    if not signals:
        return send_message(
            f"📋 <b>Resumen señales — {period}</b>\n\nSin señales válidas en este período.",
            silent=True,
        )

    lines = [f"📋 <b>Resumen señales — {period}</b>\n"]
    for s in signals:
        emoji = DIR_EMOJI.get(s.direction, "⚪")
        lines.append(
            f"{emoji} {PAIR_EMOJI.get(s.pair,'')} <b>{s.pair}</b> {s.timeframe}  "
            f"{DIR_LABEL.get(s.direction,'')}  "
            f"Conf {s.confidence:.0%}  R:R {s.rr_ratio:.2f}"
        )

    lines.append(f"\n<i>Total señales: {len(signals)}</i>")
    return send_message("\n".join(lines), silent=True)


def send_error(error: str, context: Optional[str] = None) -> bool:
    """Alerta de error crítico del sistema."""
    ts = datetime.now(timezone.utc).strftime("%d/%m/%Y %H:%M UTC")
    text = (
        f"🚨 <b>ERROR — ML-Ayram</b>\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"🕐 {ts}\n"
        f"📍 <b>Contexto:</b> {context or 'desconocido'}\n\n"
        f"<code>{error[:1000]}</code>"   # limitar longitud
    )
    return send_message(text)


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

    return send_message("\n".join(lines), silent=True)


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
