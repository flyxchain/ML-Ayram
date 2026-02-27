"""
ML-Ayram — Punto de entrada principal del bot
Uso:
  python main.py --mode paper-trading
  python main.py --mode live --account demo
  python main.py --mode live --account real
"""
import argparse
import asyncio
import logging
from pathlib import Path
from dotenv import load_dotenv
from loguru import logger

# Cargar variables de entorno
load_dotenv()

def parse_args():
    parser = argparse.ArgumentParser(description="ML-Ayram Trading Bot")
    parser.add_argument(
        "--mode",
        choices=["paper-trading", "live"],
        default="paper-trading",
        help="Modo de operación (default: paper-trading)"
    )
    parser.add_argument(
        "--account",
        choices=["demo", "real"],
        default="demo",
        help="Cuenta cTrader a usar (default: demo)"
    )
    parser.add_argument(
        "--max-lots",
        type=float,
        default=0.01,
        help="Tamaño máximo de posición en lotes (default: 0.01)"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Logging detallado"
    )
    return parser.parse_args()


async def main():
    args = parse_args()

    # Configurar logging
    log_level = "DEBUG" if args.verbose else "INFO"
    logger.add(
        "logs/bot.log",
        rotation="50 MB",
        retention="7 days",
        level=log_level,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}"
    )

    logger.info("=" * 60)
    logger.info(f"ML-Ayram Bot iniciando...")
    logger.info(f"Modo: {args.mode.upper()}")
    logger.info(f"Cuenta: {args.account.upper()}")
    logger.info(f"Max lotes: {args.max_lots}")
    logger.info("=" * 60)

    # TODO (se implementará en fases posteriores):
    # from src.data.collector import DataCollector
    # from src.signals.engine import SignalEngine
    # from src.notifications.telegram_bot import TelegramBot
    # from src.execution.ctrader_client import CTraderClient

    logger.info("Bot inicializado. Los módulos se añadirán progresivamente.")
    logger.info("Consulta docs/TUTORIAL_COMPLETO.md para el siguiente paso.")


if __name__ == "__main__":
    asyncio.run(main())
