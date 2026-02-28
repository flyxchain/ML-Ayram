"""
src/data/features.py
Cálculo de ~85 features técnicos, temporales y multi-timeframe.
Usa la librería `ta` (compatible Python 3.14).
"""

import pandas as pd
import numpy as np
from ta import trend, momentum, volatility, volume
from sqlalchemy import create_engine, text
from dotenv import load_dotenv
from loguru import logger
import os

load_dotenv()
DATABASE_URL = os.getenv("DATABASE_URL")

PAIRS = ["EURUSD", "GBPUSD", "USDJPY", "EURJPY", "XAUUSD"]
TIMEFRAMES = ["M15", "H1", "H4", "D1"]

# Sesiones forex (hora UTC)
SESSIONS = {
    "tokyo":   (0,  9),
    "london":  (7,  16),
    "newyork": (12, 21),
}


def get_session(hour: int) -> str:
    london  = SESSIONS["london"][0]  <= hour < SESSIONS["london"][1]
    newyork = SESSIONS["newyork"][0] <= hour < SESSIONS["newyork"][1]
    tokyo   = SESSIONS["tokyo"][0]   <= hour < SESSIONS["tokyo"][1]
    if london and newyork:
        return "overlap"
    if london:
        return "london"
    if newyork:
        return "newyork"
    if tokyo:
        return "tokyo"
    return "offmarket"


def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Recibe un DataFrame con columnas: timestamp, open, high, low, close, volume
    Devuelve el mismo DataFrame con ~85 features añadidos.
    Requiere mínimo 210 velas para calcular EMA200.
    """
    df = df.copy().sort_values("timestamp").reset_index(drop=True)

    c = df["close"]
    h = df["high"]
    l = df["low"]
    o = df["open"]

    # ── Tendencia ────────────────────────────────────────────────────────────
    df["ema_9"]    = trend.EMAIndicator(c, window=9).ema_indicator()
    df["ema_20"]   = trend.EMAIndicator(c, window=20).ema_indicator()
    df["ema_50"]   = trend.EMAIndicator(c, window=50).ema_indicator()
    df["ema_200"]  = trend.EMAIndicator(c, window=200).ema_indicator()
    df["sma_20"]   = trend.SMAIndicator(c, window=20).sma_indicator()
    df["sma_50"]   = trend.SMAIndicator(c, window=50).sma_indicator()

    macd = trend.MACD(c)
    df["macd_line"]   = macd.macd()
    df["macd_signal"] = macd.macd_signal()
    df["macd_hist"]   = macd.macd_diff()

    adx_ind = trend.ADXIndicator(h, l, c, window=14)
    df["adx"]     = adx_ind.adx()
    df["adx_pos"] = adx_ind.adx_pos()   # +DI
    df["adx_neg"] = adx_ind.adx_neg()   # -DI

    df["cci_20"]  = trend.CCIIndicator(h, l, c, window=20).cci()
    df["dpo_20"]  = trend.DPOIndicator(c, window=20).dpo()
    df["aroon_up"]   = trend.AroonIndicator(h, l, window=25).aroon_up()
    df["aroon_down"] = trend.AroonIndicator(h, l, window=25).aroon_down()
    df["aroon_ind"]  = trend.AroonIndicator(h, l, window=25).aroon_indicator()

    # Ichimoku
    ich = trend.IchimokuIndicator(h, l)
    df["ich_a"]    = ich.ichimoku_a()
    df["ich_b"]    = ich.ichimoku_b()
    df["ich_base"] = ich.ichimoku_base_line()
    df["ich_conv"] = ich.ichimoku_conversion_line()

    # ── Osciladores ──────────────────────────────────────────────────────────
    df["rsi_14"]  = momentum.RSIIndicator(c, window=14).rsi()
    df["rsi_7"]   = momentum.RSIIndicator(c, window=7).rsi()

    stoch = momentum.StochasticOscillator(h, l, c, window=14, smooth_window=3)
    df["stoch_k"] = stoch.stoch()
    df["stoch_d"] = stoch.stoch_signal()

    df["williams_r"] = momentum.WilliamsRIndicator(h, l, c, lbp=14).williams_r()
    df["roc_10"]     = momentum.ROCIndicator(c, window=10).roc()
    df["roc_20"]     = momentum.ROCIndicator(c, window=20).roc()

    ppo = momentum.PercentagePriceOscillator(c)
    df["ppo"]        = ppo.ppo()
    df["ppo_signal"] = ppo.ppo_signal()
    df["ppo_hist"]   = ppo.ppo_hist()

    df["tsi"] = momentum.TSIIndicator(c).tsi()

    # ── Volatilidad ──────────────────────────────────────────────────────────
    atr_ind = volatility.AverageTrueRange(h, l, c, window=14)
    df["atr_14"] = atr_ind.average_true_range()
    df["atr_7"]  = volatility.AverageTrueRange(h, l, c, window=7).average_true_range()

    bb = volatility.BollingerBands(c, window=20, window_dev=2)
    df["bb_upper"]  = bb.bollinger_hband()
    df["bb_middle"] = bb.bollinger_mavg()
    df["bb_lower"]  = bb.bollinger_lband()
    df["bb_width"]  = bb.bollinger_wband()
    df["bb_pct"]    = bb.bollinger_pband()   # posición dentro de las bandas

    kc = volatility.KeltnerChannel(h, l, c, window=20)
    df["kc_upper"]  = kc.keltner_channel_hband()
    df["kc_middle"] = kc.keltner_channel_mband()
    df["kc_lower"]  = kc.keltner_channel_lband()
    df["kc_width"]  = kc.keltner_channel_wband()
    df["kc_pct"]    = kc.keltner_channel_pband()

    df["dc_upper"] = volatility.DonchianChannel(h, l, c, window=20).donchian_channel_hband()
    df["dc_lower"] = volatility.DonchianChannel(h, l, c, window=20).donchian_channel_lband()
    df["dc_width"] = volatility.DonchianChannel(h, l, c, window=20).donchian_channel_wband()

    # ── Volumen (forex no tiene volumen real, usamos tick_volume si existe) ──
    vol_series = df.get("volume", pd.Series(np.ones(len(df))))
    vol_series = vol_series.fillna(1).infer_objects(copy=False)
    df["volume_sma_20"]   = vol_series.rolling(20).mean()
    df["volume_ratio_20"] = vol_series / df["volume_sma_20"].replace(0, np.nan)

    # ── Estructura de mercado ─────────────────────────────────────────────────
    window = 20
    df["swing_high_20"] = h.rolling(window).max()
    df["swing_low_20"]  = l.rolling(window).min()
    df["price_vs_sh"]   = (c - df["swing_high_20"]) / df["atr_14"]   # distancia al swing high en ATRs
    df["price_vs_sl"]   = (c - df["swing_low_20"])  / df["atr_14"]

    # Tendencia simple: EMA20 vs EMA50 vs EMA200
    df["trend_direction"] = np.where(
        (df["ema_20"] > df["ema_50"]) & (df["ema_50"] > df["ema_200"]),  1,   # alcista
        np.where(
            (df["ema_20"] < df["ema_50"]) & (df["ema_50"] < df["ema_200"]), -1,  # bajista
            0                                                                     # lateral
        )
    ).astype(int)

    # Vela actual
    df["body_size"]    = (c - o).abs() / df["atr_14"]
    df["upper_wick"]   = (h - df[["open", "close"]].max(axis=1)) / df["atr_14"]
    df["lower_wick"]   = (df[["open", "close"]].min(axis=1) - l) / df["atr_14"]
    df["is_bullish"]   = (c > o).astype(int)

    # Retorno logarítmico
    df["log_return_1"]  = np.log(c / c.shift(1))
    df["log_return_5"]  = np.log(c / c.shift(5))
    df["log_return_10"] = np.log(c / c.shift(10))

    # Normalización precio vs ATR
    df["close_vs_ema20"]  = (c - df["ema_20"])  / df["atr_14"]
    df["close_vs_ema50"]  = (c - df["ema_50"])  / df["atr_14"]
    df["close_vs_ema200"] = (c - df["ema_200"]) / df["atr_14"]

    # ── Temporales ────────────────────────────────────────────────────────────
    df["hour_of_day"]  = pd.to_datetime(df["timestamp"]).dt.hour
    df["day_of_week"]  = pd.to_datetime(df["timestamp"]).dt.dayofweek   # 0=lunes
    df["week_of_year"] = pd.to_datetime(df["timestamp"]).dt.isocalendar().week.astype(int)
    df["month"]        = pd.to_datetime(df["timestamp"]).dt.month

    # Sesión
    df["session_name"]  = df["hour_of_day"].apply(get_session)
    df["is_london"]     = (df["session_name"].isin(["london", "overlap"])).astype(int)
    df["is_newyork"]    = (df["session_name"].isin(["newyork", "overlap"])).astype(int)
    df["is_overlap"]    = (df["session_name"] == "overlap").astype(int)
    df["is_offmarket"]  = (df["session_name"] == "offmarket").astype(int)

    return df


def compute_htf_features(df_ltf: pd.DataFrame, df_htf: pd.DataFrame) -> pd.DataFrame:
    """
    Añade features del timeframe superior (HTF) al LTF.
    df_ltf: DataFrame del timeframe menor (ej H1)
    df_htf: DataFrame del timeframe mayor (ej H4), ya con features calculados
    """
    df_htf_simple = df_htf[["timestamp", "trend_direction", "rsi_14", "adx", "atr_14"]].copy()
    df_htf_simple.columns = ["timestamp", "htf_trend", "htf_rsi", "htf_adx", "htf_atr"]
    df_htf_simple = df_htf_simple.sort_values("timestamp")

    df_ltf = df_ltf.sort_values("timestamp")
    df_merged = pd.merge_asof(df_ltf, df_htf_simple, on="timestamp", direction="backward")
    return df_merged


def process_pair_timeframe(pair: str, timeframe: str, engine) -> int:
    """
    Calcula features para un par/timeframe y los guarda en features_computed.
    Modo incremental: solo procesa velas nuevas desde el último feature guardado.
    Devuelve el número de filas insertadas/actualizadas.
    """
    logger.info(f"Calculando features: {pair} {timeframe}")

    # Buscar el timestamp más reciente ya calculado
    with engine.connect() as conn:
        row = conn.execute(
            text("SELECT MAX(timestamp) FROM features_computed WHERE pair = :pair AND timeframe = :tf"),
            {"pair": pair, "tf": timeframe},
        ).fetchone()
    last_feature_ts = row[0] if row and row[0] else None

    # Para calcular indicadores de las velas nuevas necesitamos contexto histórico
    # (EMA200 requiere 200 velas, más margen = 250)
    CONTEXT_BARS = 250

    if last_feature_ts is None:
        # Primera vez: cargar todo
        df = pd.read_sql(
            text("SELECT * FROM ohlcv_raw WHERE pair = :pair AND timeframe = :tf ORDER BY timestamp"),
            engine,
            params={"pair": pair, "tf": timeframe},
        )
    else:
        # Incremental: cargar contexto + velas nuevas
        df = pd.read_sql(
            text("""
                SELECT * FROM ohlcv_raw
                WHERE pair = :pair AND timeframe = :tf
                  AND timestamp >= (
                      SELECT timestamp FROM ohlcv_raw
                      WHERE pair = :pair AND timeframe = :tf
                        AND timestamp <= :last_ts
                      ORDER BY timestamp DESC
                      LIMIT 1 OFFSET :ctx
                  )
                ORDER BY timestamp
            """),
            engine,
            params={"pair": pair, "tf": timeframe, "last_ts": str(last_feature_ts), "ctx": CONTEXT_BARS - 1},
        )
        if df.empty:
            logger.debug(f"  Sin velas nuevas para {pair} {timeframe}")
            return 0

    if len(df) < 210:
        logger.warning(f"  Insuficientes velas ({len(df)}) para {pair} {timeframe}, mínimo 210")
        return 0

    df = compute_features(df)

    # Añadir HTF features si es posible
    htf_map = {"M15": "H1", "H1": "H4", "H4": "D1"}
    if timeframe in htf_map:
        htf = htf_map[timeframe]
        df_htf_raw = pd.read_sql(
            text("""SELECT * FROM ohlcv_raw
                WHERE pair = :pair AND timeframe = :htf
                ORDER BY timestamp"""),
            engine,
            params={"pair": pair, "htf": htf},
        )
        if len(df_htf_raw) >= 210:
            df_htf = compute_features(df_htf_raw)
            df = compute_htf_features(df, df_htf)
        else:
            df["htf_trend"] = 0
            df["htf_rsi"]   = np.nan
            df["htf_adx"]   = np.nan
            df["htf_atr"]   = np.nan
    else:
        df["htf_trend"] = 0
        df["htf_rsi"]   = np.nan
        df["htf_adx"]   = np.nan
        df["htf_atr"]   = np.nan

    # Columnas a guardar en BD
    cols = [
        "pair", "timeframe", "timestamp",
        "ema_20", "ema_50", "ema_200",
        "macd_line", "macd_signal", "macd_hist",
        "adx", "adx_pos", "adx_neg",
        "rsi_14", "rsi_7",
        "stoch_k", "stoch_d",
        "williams_r", "roc_10", "cci_20",
        "atr_14", "atr_7",
        "bb_upper", "bb_middle", "bb_lower", "bb_width", "bb_pct",
        "kc_upper", "kc_lower", "kc_width",
        "dc_upper", "dc_lower", "dc_width",
        "volume_ratio_20",
        "swing_high_20", "swing_low_20",
        "price_vs_sh", "price_vs_sl",
        "trend_direction",
        "body_size", "upper_wick", "lower_wick", "is_bullish",
        "log_return_1", "log_return_5", "log_return_10",
        "close_vs_ema20", "close_vs_ema50", "close_vs_ema200",
        "hour_of_day", "day_of_week", "week_of_year", "month",
        "session_name", "is_london", "is_newyork", "is_overlap",
        "htf_trend", "htf_rsi", "htf_adx", "htf_atr",
    ]
    # Solo las columnas que existen
    cols = [c for c in cols if c in df.columns]
    df_save = df[cols].dropna(subset=["ema_200"])  # eliminar filas sin suficiente historia

    rows = df_save.to_dict("records")

    # Convertir tipos numpy a Python nativos
    for row in rows:
        for k, v in row.items():
            if isinstance(v, (np.integer,)):
                row[k] = int(v)
            elif isinstance(v, (np.floating,)):
                row[k] = float(v) if not np.isnan(v) else None
            elif pd.isna(v) if not isinstance(v, str) else False:
                row[k] = None

    insert_cols = ", ".join([c for c in cols])
    placeholders = ", ".join([f":{c}" for c in cols])
    conflict_cols = ", ".join([f"{c}=EXCLUDED.{c}" for c in cols if c not in ("pair","timeframe","timestamp")])

    sql = text(f"""
        INSERT INTO features_computed ({insert_cols})
        VALUES ({placeholders})
        ON CONFLICT (pair, timeframe, timestamp) DO UPDATE SET {conflict_cols}
    """)

    with engine.connect() as conn:
        conn.execute(sql, rows)
        conn.commit()

    logger.success(f"  ✅ {pair} {timeframe}: {len(df_save)} filas guardadas")
    return len(df_save)


def run_all():
    """Calcula features para todos los pares y timeframes."""
    engine = create_engine(DATABASE_URL)
    total = 0
    # Procesar en orden HTF→LTF para que los HTF features estén disponibles
    for tf in ["D1", "H4", "H1", "M15"]:
        for pair in PAIRS:
            try:
                total += process_pair_timeframe(pair, tf, engine)
            except Exception as e:
                logger.error(f"Error {pair} {tf}: {e}")
    logger.success(f"✅ Features completados. Total filas: {total}")


if __name__ == "__main__":
    run_all()
