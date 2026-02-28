"""
tests/test_features.py
Unit tests para src/data/features.py

Cubre:
  - get_session: mapeo hora UTC → sesión forex
  - compute_features: cálculo de ~85 indicadores técnicos
  - compute_htf_features: merge asof de features HTF → LTF
  - Integridad de columnas, tipos, NaN handling
"""

import pytest
import numpy as np
import pandas as pd

from src.data.features import compute_features, get_session, compute_htf_features


# ══════════════════════════════════════════════════════════════════════════════
# get_session
# ══════════════════════════════════════════════════════════════════════════════

class TestGetSession:
    def test_tokyo_early(self):
        assert get_session(2) == "tokyo"

    def test_tokyo_boundary(self):
        assert get_session(0) == "tokyo"
        assert get_session(6) == "tokyo"

    def test_london(self):
        assert get_session(9) == "london"
        assert get_session(10) == "london"

    def test_newyork(self):
        assert get_session(18) == "newyork"
        assert get_session(20) == "newyork"

    def test_overlap_london_newyork(self):
        """12-16 UTC es solapamiento Londres-Nueva York."""
        assert get_session(12) == "overlap"
        assert get_session(14) == "overlap"
        assert get_session(15) == "overlap"

    def test_offmarket(self):
        """Horas fuera de todas las sesiones → offmarket."""
        assert get_session(21) == "offmarket"
        assert get_session(22) == "offmarket"
        assert get_session(23) == "offmarket"

    def test_all_hours_return_valid_session(self):
        """Las 24 horas deben devolver una sesión válida."""
        valid = {"tokyo", "london", "newyork", "overlap", "offmarket"}
        for h in range(24):
            assert get_session(h) in valid


# ══════════════════════════════════════════════════════════════════════════════
# compute_features — Columnas esperadas
# ══════════════════════════════════════════════════════════════════════════════

class TestComputeFeaturesColumns:
    def test_returns_dataframe(self, ohlcv_eurusd_h1):
        result = compute_features(ohlcv_eurusd_h1)
        assert isinstance(result, pd.DataFrame)

    def test_same_length(self, ohlcv_eurusd_h1):
        result = compute_features(ohlcv_eurusd_h1)
        assert len(result) == len(ohlcv_eurusd_h1)

    def test_trend_columns(self, ohlcv_eurusd_h1):
        result = compute_features(ohlcv_eurusd_h1)
        for col in ["ema_9", "ema_20", "ema_50", "ema_200", "sma_20", "sma_50",
                     "macd_line", "macd_signal", "macd_hist",
                     "adx", "adx_pos", "adx_neg", "cci_20"]:
            assert col in result.columns, f"Falta columna de tendencia: {col}"

    def test_oscillator_columns(self, ohlcv_eurusd_h1):
        result = compute_features(ohlcv_eurusd_h1)
        for col in ["rsi_14", "rsi_7", "stoch_k", "stoch_d",
                     "williams_r", "roc_10", "roc_20", "ppo", "tsi"]:
            assert col in result.columns, f"Falta columna de oscilador: {col}"

    def test_volatility_columns(self, ohlcv_eurusd_h1):
        result = compute_features(ohlcv_eurusd_h1)
        for col in ["atr_14", "atr_7", "bb_upper", "bb_lower", "bb_width", "bb_pct",
                     "kc_upper", "kc_lower", "kc_width",
                     "dc_upper", "dc_lower", "dc_width"]:
            assert col in result.columns, f"Falta columna de volatilidad: {col}"

    def test_structure_columns(self, ohlcv_eurusd_h1):
        result = compute_features(ohlcv_eurusd_h1)
        for col in ["swing_high_20", "swing_low_20", "trend_direction",
                     "body_size", "upper_wick", "lower_wick", "is_bullish",
                     "log_return_1", "log_return_5", "log_return_10"]:
            assert col in result.columns, f"Falta columna de estructura: {col}"

    def test_temporal_columns(self, ohlcv_eurusd_h1):
        result = compute_features(ohlcv_eurusd_h1)
        for col in ["hour_of_day", "day_of_week", "week_of_year", "month",
                     "session_name", "is_london", "is_newyork", "is_overlap"]:
            assert col in result.columns, f"Falta columna temporal: {col}"


# ══════════════════════════════════════════════════════════════════════════════
# compute_features — Valores y rangos
# ══════════════════════════════════════════════════════════════════════════════

class TestComputeFeaturesValues:
    def test_rsi_range(self, ohlcv_eurusd_h1):
        result = compute_features(ohlcv_eurusd_h1)
        rsi = result["rsi_14"].dropna()
        assert rsi.min() >= 0, "RSI no puede ser negativo"
        assert rsi.max() <= 100, "RSI no puede superar 100"

    def test_stoch_range(self, ohlcv_eurusd_h1):
        result = compute_features(ohlcv_eurusd_h1)
        for col in ["stoch_k", "stoch_d"]:
            vals = result[col].dropna()
            assert vals.min() >= 0
            assert vals.max() <= 100

    def test_adx_non_negative(self, ohlcv_eurusd_h1):
        result = compute_features(ohlcv_eurusd_h1)
        assert result["adx"].dropna().min() >= 0

    def test_atr_positive(self, ohlcv_eurusd_h1):
        result = compute_features(ohlcv_eurusd_h1)
        atr = result["atr_14"].dropna()
        assert (atr >= 0).all(), "ATR debe ser no-negativo"

    def test_ema_order_last_row(self, ohlcv_eurusd_h1):
        """Las EMAs más cortas reaccionan más rápido, no testear orden absoluto."""
        result = compute_features(ohlcv_eurusd_h1)
        last = result.iloc[-1]
        # Solo verificamos que están calculadas
        assert not np.isnan(last["ema_9"])
        assert not np.isnan(last["ema_200"])

    def test_trend_direction_values(self, ohlcv_eurusd_h1):
        result = compute_features(ohlcv_eurusd_h1)
        valid = {-1, 0, 1}
        assert set(result["trend_direction"].dropna().unique()).issubset(valid)

    def test_is_bullish_binary(self, ohlcv_eurusd_h1):
        result = compute_features(ohlcv_eurusd_h1)
        assert set(result["is_bullish"].unique()).issubset({0, 1})

    def test_session_names_valid(self, ohlcv_eurusd_h1):
        result = compute_features(ohlcv_eurusd_h1)
        valid = {"tokyo", "london", "newyork", "overlap", "offmarket"}
        assert set(result["session_name"].unique()).issubset(valid)

    def test_bb_pct_centered(self, ohlcv_eurusd_h1):
        """Bollinger %B típicamente entre -0.5 y 1.5 (puede salirse)."""
        result = compute_features(ohlcv_eurusd_h1)
        bb = result["bb_pct"].dropna()
        assert bb.median() > -1 and bb.median() < 2

    def test_log_returns_not_all_zero(self, ohlcv_eurusd_h1):
        result = compute_features(ohlcv_eurusd_h1)
        assert result["log_return_1"].dropna().abs().sum() > 0

    def test_williams_r_range(self, ohlcv_eurusd_h1):
        result = compute_features(ohlcv_eurusd_h1)
        wr = result["williams_r"].dropna()
        assert wr.min() >= -100
        assert wr.max() <= 0


# ══════════════════════════════════════════════════════════════════════════════
# compute_features — NaN handling
# ══════════════════════════════════════════════════════════════════════════════

class TestComputeFeaturesNaN:
    def test_ema200_nan_first_199_rows(self, ohlcv_eurusd_h1):
        """Las primeras ~199 filas no tienen EMA200 calculable."""
        result = compute_features(ohlcv_eurusd_h1)
        first_valid = result["ema_200"].first_valid_index()
        assert first_valid is not None
        assert first_valid >= 190  # warmup ≈ 199

    def test_last_rows_no_nan_for_key_indicators(self, ohlcv_eurusd_h1):
        """Las últimas filas (fila 250+) deben tener indicadores completos."""
        result = compute_features(ohlcv_eurusd_h1)
        last = result.iloc[-1]
        key_cols = ["ema_20", "ema_200", "rsi_14", "atr_14", "adx", "macd_line",
                    "bb_upper", "stoch_k", "trend_direction"]
        for col in key_cols:
            assert not np.isnan(last[col]), f"{col} es NaN en la última fila"

    def test_insufficient_data_returns_all_nan_ema200(self, ohlcv_short):
        """Con solo 50 velas, EMA200 debe ser todo NaN."""
        result = compute_features(ohlcv_short)
        assert result["ema_200"].isna().all()


# ══════════════════════════════════════════════════════════════════════════════
# compute_features — Idempotencia
# ══════════════════════════════════════════════════════════════════════════════

class TestComputeFeaturesIdempotent:
    def test_does_not_modify_input(self, ohlcv_eurusd_h1):
        original_cols = set(ohlcv_eurusd_h1.columns)
        compute_features(ohlcv_eurusd_h1)
        assert set(ohlcv_eurusd_h1.columns) == original_cols

    def test_deterministic(self, ohlcv_eurusd_h1):
        r1 = compute_features(ohlcv_eurusd_h1)
        r2 = compute_features(ohlcv_eurusd_h1)
        pd.testing.assert_frame_equal(r1, r2)


# ══════════════════════════════════════════════════════════════════════════════
# compute_features — XAUUSD (diferente escala de precios)
# ══════════════════════════════════════════════════════════════════════════════

class TestComputeFeaturesXAUUSD:
    def test_atr_scale(self, ohlcv_xauusd_h1):
        """ATR de XAUUSD debe ser del orden de unidades (no fracciones como FX)."""
        result = compute_features(ohlcv_xauusd_h1)
        atr = result["atr_14"].dropna().mean()
        assert atr > 0.1, "ATR de XAUUSD debería ser mayor que 0.1"

    def test_feature_count_matches(self, ohlcv_eurusd_h1, ohlcv_xauusd_h1):
        """Misma cantidad de features independiente del par."""
        r1 = compute_features(ohlcv_eurusd_h1)
        r2 = compute_features(ohlcv_xauusd_h1)
        # Mismas columnas (salvo que el input ya tenía 'pair')
        shared_new = set(r1.columns) - set(ohlcv_eurusd_h1.columns)
        shared_xau = set(r2.columns) - set(ohlcv_xauusd_h1.columns)
        assert shared_new == shared_xau


# ══════════════════════════════════════════════════════════════════════════════
# compute_htf_features
# ══════════════════════════════════════════════════════════════════════════════

class TestComputeHtfFeatures:
    def test_adds_htf_columns(self, ohlcv_eurusd_h1, ohlcv_eurusd_h4):
        ltf = compute_features(ohlcv_eurusd_h1)
        htf = compute_features(ohlcv_eurusd_h4)
        merged = compute_htf_features(ltf, htf)
        for col in ["htf_trend", "htf_rsi", "htf_adx", "htf_atr"]:
            assert col in merged.columns

    def test_preserves_ltf_rows(self, ohlcv_eurusd_h1, ohlcv_eurusd_h4):
        ltf = compute_features(ohlcv_eurusd_h1)
        htf = compute_features(ohlcv_eurusd_h4)
        merged = compute_htf_features(ltf, htf)
        assert len(merged) == len(ltf)

    def test_htf_trend_valid_values(self, ohlcv_eurusd_h1, ohlcv_eurusd_h4):
        ltf = compute_features(ohlcv_eurusd_h1)
        htf = compute_features(ohlcv_eurusd_h4)
        merged = compute_htf_features(ltf, htf)
        valid = {-1, 0, 1}
        vals = set(merged["htf_trend"].dropna().unique())
        assert vals.issubset(valid)

    def test_htf_rsi_range(self, ohlcv_eurusd_h1, ohlcv_eurusd_h4):
        ltf = compute_features(ohlcv_eurusd_h1)
        htf = compute_features(ohlcv_eurusd_h4)
        merged = compute_htf_features(ltf, htf)
        rsi = merged["htf_rsi"].dropna()
        if len(rsi) > 0:
            assert rsi.min() >= 0
            assert rsi.max() <= 100
