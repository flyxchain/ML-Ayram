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
from datetime import datetime, timedelta, timezone
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


def _get_with_retry(url: str, params: dict, retries: int = 3) -> list:
    """GET con reintentos y back-off exponencial."""
    for attempt in range(1, retries + 1):
        try:
            resp = requests.get(url, params=params, timeout=30)
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            if attempt == retries:
                raise
            wait = 2 ** attempt
            logger.warning(f"  Intento {attempt}/{retries} fallido: {e} — reintentando en {wait}s")
            time.sleep(wait)
    return []


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

    data = _get_with_retry(url, params)

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

    data = _get_with_retry(url, params)

    if not data:
        return pd.DataFrame()

    df = pd.DataFrame(data)
    df["timestamp"] = pd.to_datetime(df["date"])
    df["pair"]      = pair
    df["timeframe"] = "D1"
    df["volume"]    = df.get("volume", None)
    return df[["pair", "timeframe", "timestamp", "open", "high", "low", "close", "volume"]]


# ── Resampleo inteligente con validación de completitud ───────────────────
#
# Regla: una vela resampleada solo se genera si:
#   - Histórica → tiene TODAS las sub-velas esperadas (4 M15 para H1, 4 H1 para H4)
#   - En formación (periodo actual) → se genera con lo que haya, es la vela "viva"
# Esto evita fabricar datos falsos pero permite mostrar la vela en curso.

def _is_current_period(period_start, freq_minutes: int) -> bool:
    """¿El periodo que empieza en period_start aún no ha cerrado?"""
    now = _utcnow()
    ts = pd.Timestamp(period_start)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    period_end = ts + pd.Timedelta(minutes=freq_minutes)
    return ts <= pd.Timestamp(now) < period_end


def resample_m15_to_h1(df_m15: pd.DataFrame, pair: str) -> pd.DataFrame:
    """
    Construye H1 desde M15.
    - Histórica: requiere 4 sub-velas (:00, :15, :30, :45)
    - En formación: genera con las que haya
    """
    df = df_m15.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.dropna(subset=["open", "high", "low", "close"])
    if df.empty:
        return pd.DataFrame()

    df = df.sort_values("timestamp")
    df["hour"] = df["timestamp"].dt.floor("h")

    rows = []
    for hour, g in df.groupby("hour"):
        is_current = _is_current_period(hour, 60)
        n = len(g)

        # Validar alineación: solo minutos :00, :15, :30, :45
        valid_minutes = g["timestamp"].dt.minute.isin([0, 15, 30, 45])
        g = g[valid_minutes]
        if g.empty:
            continue

        # Histórica incompleta → no generar
        if not is_current and len(g) < 4:
            continue

        g_sorted = g.sort_values("timestamp")
        rows.append({
            "pair": pair, "timeframe": "H1", "timestamp": hour,
            "open":   g_sorted.iloc[0]["open"],
            "high":   g["high"].max(),
            "low":    g["low"].min(),
            "close":  g_sorted.iloc[-1]["close"],
            "volume": g["volume"].sum() if g["volume"].notna().any() else None,
        })

    return pd.DataFrame(rows) if rows else pd.DataFrame()


def resample_h4(df_h1: pd.DataFrame, pair: str = None) -> pd.DataFrame:
    """
    Construye H4 desde H1.
    Bloques alineados: 00-03, 04-07, 08-11, 12-15, 16-19, 20-23.
    - Histórica: requiere 4 H1 por bloque
    - En formación: genera con las que haya
    """
    df = df_h1.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.dropna(subset=["open", "high", "low", "close"])
    if df.empty:
        return pd.DataFrame()

    _pair = pair or df["pair"].iloc[0]
    df = df.sort_values("timestamp")
    df["block"] = df["timestamp"].dt.floor("4h")

    rows = []
    for block, g in df.groupby("block"):
        is_current = _is_current_period(block, 240)

        # Validar: solo horas dentro del bloque correcto
        valid_hours = g["timestamp"].dt.hour.isin(
            [block.hour + i for i in range(4)]
        )
        g = g[valid_hours]
        if g.empty:
            continue

        if not is_current and len(g) < 4:
            continue

        g_sorted = g.sort_values("timestamp")
        rows.append({
            "pair": _pair, "timeframe": "H4", "timestamp": block,
            "open":   g_sorted.iloc[0]["open"],
            "high":   g["high"].max(),
            "low":    g["low"].min(),
            "close":  g_sorted.iloc[-1]["close"],
            "volume": g["volume"].sum() if g["volume"].notna().any() else None,
        })

    return pd.DataFrame(rows) if rows else pd.DataFrame()


def resample_h1_to_d1(df_h1: pd.DataFrame, pair: str) -> pd.DataFrame:
    """
    Construye D1 desde H1.
    Un día forex va de 00:00 a 23:00 UTC (24 velas H1).
    - Histórica: requiere al menos 20 H1 (permite huecos nocturnos)
    - En formación: genera con las que haya
    """
    df = df_h1.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.dropna(subset=["open", "high", "low", "close"])
    if df.empty:
        return pd.DataFrame()

    df = df.sort_values("timestamp")
    df["day"] = df["timestamp"].dt.floor("D")

    rows = []
    for day, g in df.groupby("day"):
        is_current = _is_current_period(day, 1440)

        # Forex tiene ~21-24 velas H1 por día (cierra viernes noche)
        if not is_current and len(g) < 20:
            continue

        g_sorted = g.sort_values("timestamp")
        rows.append({
            "pair": pair, "timeframe": "D1", "timestamp": day,
            "open":   g_sorted.iloc[0]["open"],
            "high":   g["high"].max(),
            "low":    g["low"].min(),
            "close":  g_sorted.iloc[-1]["close"],
            "volume": g["volume"].sum() if g["volume"].notna().any() else None,
        })

    return pd.DataFrame(rows) if rows else pd.DataFrame()


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
    to_dt   = _utcnow()
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

                    # FALLBACK: si EODHD no tiene H1 (ej: XAUUSD), resamplear desde M15
                    if df.empty and tf == "H1":
                        logger.info(f"  {pair} H1: EODHD vacío, resampleando desde M15...")
                        df_m15 = pd.read_sql(
                            text("SELECT * FROM ohlcv_raw WHERE pair = :pair AND timeframe = 'M15' AND timestamp >= :f AND timestamp < :t ORDER BY timestamp"),
                            engine,
                            params={"pair": pair, "f": str(chunk_start), "t": str(chunk_end)},
                        )
                        if not df_m15.empty:
                            df = resample_m15_to_h1(df_m15, pair)
                            logger.info(f"  {pair} H1: resampleadas {len(df)} velas desde M15")

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
                text("SELECT * FROM ohlcv_raw WHERE pair = :pair AND timeframe = 'H1' ORDER BY timestamp"),
                engine,
                params={"pair": pair},
            )
            if not df_h1.empty:
                df_h4 = resample_h4(df_h1, pair)
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
        text("""
            SELECT * FROM ohlcv_raw
            WHERE pair = :pair AND timeframe = :tf
            ORDER BY timestamp DESC
            LIMIT :n
        """),
        engine,
        params={"pair": pair, "tf": timeframe, "n": n},
    )
    return df.sort_values("timestamp").reset_index(drop=True)


def _utcnow() -> datetime:
    """Devuelve el momento actual en UTC con timezone info."""
    return datetime.now(timezone.utc)


def _ensure_aware(dt) -> datetime:
    """Convierte un datetime naive a UTC-aware si es necesario."""
    ts = pd.Timestamp(dt)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    return ts.to_pydatetime()


def download_incremental():
    """
    Descarga solo las velas nuevas desde el último timestamp guardado en la BD.
    Llamado por el timer systemd cada 15 minutos.
    """
    engine = create_engine(DATABASE_URL)
    to_dt  = _utcnow()
    total  = 0

    for pair in PAIR_MAP:
        # ── M15 y H1 ─────────────────────────────────────────────
        for tf, interval in [("M15", "15m"), ("H1", "1h")]:
            with engine.connect() as conn:
                row = conn.execute(
                    text("SELECT MAX(timestamp) FROM ohlcv_raw WHERE pair = :pair AND timeframe = :tf"),
                    {"pair": pair, "tf": tf},
                ).fetchone()

            last_ts = row[0] if row and row[0] else None
            if last_ts is None:
                logger.info(f"{pair} {tf}: sin datos previos, descargando histórico...")
                from_dt = to_dt - timedelta(days=365 * 3)
            else:
                from_dt = _ensure_aware(last_ts)
                if (to_dt - from_dt).total_seconds() < 60:
                    continue

            try:
                df = fetch_intraday(pair, interval, from_dt, to_dt)
                df["timeframe"] = tf

                # FALLBACK: si EODHD devuelve vacío para H1, resamplear desde M15
                if df.empty and tf == "H1":
                    logger.info(f"  {pair} H1: EODHD vacío, resampleando desde M15...")
                    m15_from = from_dt - timedelta(hours=1)  # margen extra
                    df_m15 = pd.read_sql(
                        text("SELECT * FROM ohlcv_raw WHERE pair = :pair AND timeframe = 'M15' AND timestamp >= :from_ts ORDER BY timestamp"),
                        engine,
                        params={"pair": pair, "from_ts": str(m15_from)},
                    )
                    if not df_m15.empty:
                        df = resample_m15_to_h1(df_m15, pair)
                        logger.info(f"  {pair} H1: resampleadas {len(df)} velas desde M15")

                saved = save_to_db(df, engine)
                total += saved
                if saved:
                    logger.info(f"  {pair} {tf}: +{saved} velas nuevas")
            except Exception as e:
                logger.error(f"  Error {pair} {tf}: {e}")
            time.sleep(0.3)

        # ── H4 (reconstruido desde H1, con validación de completitud) ─────
        try:
            with engine.connect() as conn:
                last_h4 = conn.execute(
                    text("SELECT MAX(timestamp) FROM ohlcv_raw WHERE pair = :pair AND timeframe = 'H4'"),
                    {"pair": pair},
                ).fetchone()[0]
            h1_from = (
                _ensure_aware(last_h4) - timedelta(hours=8)
                if last_h4
                else datetime(2000, 1, 1, tzinfo=timezone.utc)
            )
            df_h1_recent = pd.read_sql(
                text("SELECT * FROM ohlcv_raw WHERE pair = :pair AND timeframe = 'H1' AND timestamp >= :from_ts ORDER BY timestamp"),
                engine,
                params={"pair": pair, "from_ts": str(h1_from)},
            )
            if not df_h1_recent.empty:
                df_h4  = resample_h4(df_h1_recent, pair)
                saved  = save_to_db(df_h4, engine)
                total += saved
                if saved:
                    logger.info(f"  {pair} H4: +{saved} velas nuevas (desde H1)")
        except Exception as e:
            logger.error(f"  Error H4 {pair}: {e}")

        # ── D1 (diario, con fallback desde H1) ─────────────────────────
        try:
            with engine.connect() as conn:
                row = conn.execute(
                    text("SELECT MAX(timestamp) FROM ohlcv_raw WHERE pair = :pair AND timeframe = 'D1'"),
                    {"pair": pair},
                ).fetchone()
            last_d1 = row[0] if row and row[0] else None
            if last_d1 is None:
                d1_from = to_dt - timedelta(days=365 * 3)
            else:
                d1_from = _ensure_aware(last_d1)
                if (to_dt - d1_from).total_seconds() < 3600:
                    continue  # D1 no necesita actualizarse tan a menudo

            df_d1 = fetch_daily(pair, d1_from, to_dt)

            # FALLBACK: si EODHD devuelve vacío para D1, resamplear desde H1
            if df_d1.empty:
                logger.info(f"  {pair} D1: EODHD vacío, resampleando desde H1...")
                df_h1_for_d1 = pd.read_sql(
                    text("SELECT * FROM ohlcv_raw WHERE pair = :pair AND timeframe = 'H1' AND timestamp >= :from_ts ORDER BY timestamp"),
                    engine,
                    params={"pair": pair, "from_ts": str(d1_from)},
                )
                if not df_h1_for_d1.empty:
                    df_d1 = resample_h1_to_d1(df_h1_for_d1, pair)
                    logger.info(f"  {pair} D1: resampleadas {len(df_d1)} velas desde H1")

            saved = save_to_db(df_d1, engine)
            total += saved
            if saved:
                logger.info(f"  {pair} D1: +{saved} velas nuevas")
        except Exception as e:
            logger.error(f"  Error D1 {pair}: {e}")

        time.sleep(0.3)

    if total:
        logger.success(f"✅ Incremental: {total} velas nuevas")
    else:
        logger.debug("Incremental: sin datos nuevos")
    return total


if __name__ == "__main__":
    import sys
    if "--full" in sys.argv:
        logger.info("Iniciando descarga histórica completa...")
        download_historical(years=3)
    else:
        logger.info("Iniciando descarga incremental...")
        download_incremental()