"""
src/data/collector.py
Descarga datos OHLCV históricos y en tiempo real desde EODHD.
Pares: EURUSD, GBPUSD, USDJPY, EURJPY, XAUUSD
Timeframes: M15, H1, H4, D1
"""

import os
import time
import requests
import pandas as pd
from datetime import datetime, timedelta
from sqlalchemy import create_engine, text
from dotenv import load_dotenv
from loguru import logger

load_dotenv()

API_KEY = os.getenv("EODHD_API_KEY")
DATABASE_URL = os.getenv("DATABASE_URL")

# Mapeo par → símbolo EODHD
PAIR_MAP = {
    "EURUSD": "EURUSD.FOREX",
    "GBPUSD": "GBPUSD.FOREX",
    "USDJPY": "USDJPY.FOREX",
    "EURJPY": "EURJPY.FOREX",
    "XAUUSD": "XAUUSD.FOREX",
}

# Timeframes soportados por EODHD
TF_MAP = {
    "M15": "15m",
    "H1":  "1h",
    "H4":  "4h",  # EODHD no tiene H4 nativo, lo construimos desde H1
    "D1":  "1d",
}

BASE_URL = "https://eodhd.com/api/intraday/{symbol}"
EOD_URL  = "https://eodhd.com/api/eod/{symbol}"


def fetch_intraday(pair: str, interval: str, from_dt: datetime, to_dt: datetime) -> pd.DataFrame:
    """Descarga datos intradía desde EODHD."""
    symbol = PAIR_MAP[pair]
    from_ts = int(from_dt.timestamp())
    to_ts   = int(to_dt.timestamp())

    url = BASE_URL.format(symbol=symbol)
    params = {
        "interval":  interval,
        "api_token": API_KEY,
        "fmt":       "json",
        "from":      from_ts,
        "to":        to_ts,
    }

    resp = requests.get(url, params=params, timeout=30)
    resp.raise_for_status()
    data = resp.json()

    if not data:
        return pd.DataFrame()

    df = pd.DataFrame(data)
    df["timestamp"] = pd.to_datetime(df["datetime"])
    df = df.rename(columns={"open": "open", "high": "high", "low": "low", "close": "close"})
    df["pair"]      = pair
    df["timeframe"] = interval
    df["volume"]    = df.get("volume", None)
    return df[["pair", "timeframe", "timestamp", "open", "high", "low", "close", "volume"]]


def fetch_daily(pair: str, from_dt: datetime, to_dt: datetime) -> pd.DataFrame:
    """Descarga datos diarios (D1) desde EODHD."""
    symbol = PAIR_MAP[pair]
    url = EOD_URL.format(symbol=symbol)
    params = {
        "api_token": API_KEY,
        "fmt":       "json",
        "from":      from_dt.strftime("%Y-%m-%d"),
        "to":        to_dt.strftime("%Y-%m-%d"),
    }

    resp = requests.get(url, params=params, timeout=30)
    resp.raise_for_status()
    data = resp.json()

    if not data:
        return pd.DataFrame()

    df = pd.DataFrame(data)
    df["timestamp"] = pd.to_datetime(df["date"])
    df["pair"]      = pair
    df["timeframe"] = "D1"
    df["volume"]    = df.get("volume", None)
    return df[["pair", "timeframe", "timestamp", "open", "high", "low", "close", "volume"]]


def resample_h4(df_h1: pd.DataFrame) -> pd.DataFrame:
    """Construye H4 desde H1."""
    df = df_h1.copy().set_index("timestamp")
    df4 = df.resample("4h").agg({
        "open":   "first",
        "high":   "max",
        "low":    "min",
        "close":  "last",
        "volume": "sum",
    }).dropna(subset=["open"])
    df4 = df4.reset_index()
    df4["pair"]      = df_h1["pair"].iloc[0]
    df4["timeframe"] = "H4"
    return df4[["pair", "timeframe", "timestamp", "open", "high", "low", "close", "volume"]]


def save_to_db(df: pd.DataFrame, engine) -> int:
    """Guarda velas en la BD, ignorando duplicados."""
    if df.empty:
        return 0

    rows = df.to_dict("records")
    sql = text("""
        INSERT INTO ohlcv_raw (pair, timeframe, timestamp, open, high, low, close, volume)
        VALUES (:pair, :timeframe, :timestamp, :open, :high, :low, :close, :volume)
        ON CONFLICT (pair, timeframe, timestamp) DO NOTHING
    """)

    with engine.connect() as conn:
        result = conn.execute(sql, rows)
        conn.commit()
    return result.rowcount


def download_historical(years: int = 3):
    """
    Descarga 3 años de histórico para todos los pares y timeframes.
    Llama a esta función una sola vez para poblar la BD.
    """
    engine = create_engine(DATABASE_URL)
    to_dt   = datetime.utcnow()
    from_dt = to_dt - timedelta(days=365 * years)

    logger.info(f"Descargando histórico {years} años: {from_dt.date()} → {to_dt.date()}")

    total = 0
    for pair in PAIR_MAP:
        logger.info(f"Par: {pair}")

        # M15 y H1 — descargamos por trimestres para no superar límites
        for tf, interval in [("M15", "15m"), ("H1", "1h")]:
            chunk_start = from_dt
            while chunk_start < to_dt:
                chunk_end = min(chunk_start + timedelta(days=90), to_dt)
                try:
                    df = fetch_intraday(pair, interval, chunk_start, chunk_end)
                    df["timeframe"] = tf
                    saved = save_to_db(df, engine)
                    total += saved
                    logger.info(f"  {tf} {chunk_start.date()}→{chunk_end.date()}: {len(df)} velas, {saved} nuevas")
                except Exception as e:
                    logger.error(f"  Error {pair} {tf}: {e}")
                chunk_start = chunk_end
                time.sleep(0.5)  # respetar rate limits

        # H4 — construido desde H1
        try:
            df_h1 = pd.read_sql(
                f"SELECT * FROM ohlcv_raw WHERE pair='{pair}' AND timeframe='H1' ORDER BY timestamp",
                engine
            )
            if not df_h1.empty:
                df_h4 = resample_h4(df_h1)
                saved = save_to_db(df_h4, engine)
                total += saved
                logger.info(f"  H4 construido desde H1: {len(df_h4)} velas, {saved} nuevas")
        except Exception as e:
            logger.error(f"  Error H4 {pair}: {e}")

        # D1
        try:
            df_d1 = fetch_daily(pair, from_dt, to_dt)
            saved = save_to_db(df_d1, engine)
            total += saved
            logger.info(f"  D1: {len(df_d1)} velas, {saved} nuevas")
        except Exception as e:
            logger.error(f"  Error D1 {pair}: {e}")

        time.sleep(1)

    logger.success(f"✅ Descarga completada. Total velas nuevas: {total}")
    return total


def get_latest_candles(pair: str, timeframe: str, n: int = 500) -> pd.DataFrame:
    """
    Obtiene las últimas N velas para generación de señales en tiempo real.
    Primero intenta la BD, si no hay datos frescos llama a la API.
    """
    engine = create_engine(DATABASE_URL)
    df = pd.read_sql(
        f"""SELECT * FROM ohlcv_raw
            WHERE pair='{pair}' AND timeframe='{timeframe}'
            ORDER BY timestamp DESC LIMIT {n}""",
        engine
    )
    return df.sort_values("timestamp").reset_index(drop=True)


if __name__ == "__main__":
    logger.info("Iniciando descarga histórica completa...")
    download_historical(years=3)