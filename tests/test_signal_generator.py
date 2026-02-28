"""
tests/test_signal_generator.py
Unit tests para src/signals/generator.py

Cubre:
  - SignalResult: dataclass, propiedades, summary
  - _get_session: mapeo hora → sesión
  - _price_to_pips / _pip_size: conversiones de precio
  - Lógica de filtros: confianza, ADX, sesión, R:R, agreement
  - Cálculo TP/SL a partir de ATR
  - No requiere BD ni modelos — mock de predicciones
"""

import pytest
import numpy as np
from datetime import datetime, timezone

from src.signals.generator import (
    SignalResult,
    _get_session,
    _pip_size,
    _price_to_pips,
    FILTERS,
    PIP_SIZE,
    TP_MULTIPLIER,
    SL_MULTIPLIER,
)


# ══════════════════════════════════════════════════════════════════════════════
# _get_session
# ══════════════════════════════════════════════════════════════════════════════

class TestGetSession:
    def test_tokyo(self):
        assert _get_session(3) == "tokyo"

    def test_london(self):
        assert _get_session(9) == "london"

    def test_newyork(self):
        assert _get_session(18) == "newyork"

    def test_overlap(self):
        assert _get_session(13) == "overlap"

    def test_offmarket(self):
        assert _get_session(22) == "offmarket"

    def test_all_24_hours(self):
        valid = {"tokyo", "london", "newyork", "overlap", "offmarket"}
        for h in range(24):
            assert _get_session(h) in valid


# ══════════════════════════════════════════════════════════════════════════════
# _pip_size / _price_to_pips
# ══════════════════════════════════════════════════════════════════════════════

class TestPipConversions:
    def test_eurusd_pip_size(self):
        assert _pip_size("EURUSD") == 0.0001

    def test_usdjpy_pip_size(self):
        assert _pip_size("USDJPY") == 0.01

    def test_xauusd_pip_size(self):
        assert _pip_size("XAUUSD") == 0.1

    def test_unknown_pair_default(self):
        assert _pip_size("UNKNOWN") == 0.0001

    def test_price_to_pips_eurusd(self):
        """50 pips en EURUSD = 0.0050 de diferencia."""
        pips = _price_to_pips(0.0050, "EURUSD")
        assert pips == 50.0

    def test_price_to_pips_xauusd(self):
        """10 USD en XAUUSD = 100 pips (pip=0.1)."""
        pips = _price_to_pips(10.0, "XAUUSD")
        assert pips == 100.0

    def test_price_to_pips_negative_diff(self):
        """La función usa abs(), así que diferencias negativas dan pips positivos."""
        pips = _price_to_pips(-0.0025, "EURUSD")
        assert pips == 25.0

    def test_price_to_pips_zero(self):
        assert _price_to_pips(0.0, "EURUSD") == 0.0


# ══════════════════════════════════════════════════════════════════════════════
# SignalResult
# ══════════════════════════════════════════════════════════════════════════════

def _make_signal(**overrides) -> SignalResult:
    """Factory de SignalResult con valores por defecto válidos."""
    defaults = dict(
        pair="EURUSD", timeframe="H1",
        timestamp=datetime(2025, 6, 15, 12, 0, tzinfo=timezone.utc),
        direction=1, confidence=0.72,
        prob_long=0.72, prob_neutral=0.15, prob_short=0.13,
        xgb_direction=1, lstm_direction=1, agreement=True,
        entry_price=1.1000, tp_price=1.1050, sl_price=1.0975,
        tp_pips=50.0, sl_pips=25.0, rr_ratio=2.0,
        atr_14=0.0025, adx=28.5, session="london",
        filter_reason=None,
    )
    defaults.update(overrides)
    return SignalResult(**defaults)


class TestSignalResult:
    def test_is_valid_when_no_filter(self):
        s = _make_signal()
        assert s.is_valid is True

    def test_not_valid_with_filter_reason(self):
        s = _make_signal(filter_reason="ADX débil")
        assert s.is_valid is False

    def test_not_valid_when_neutral(self):
        s = _make_signal(direction=0)
        assert s.is_valid is False

    def test_direction_label_long(self):
        assert _make_signal(direction=1).direction_label == "LONG"

    def test_direction_label_short(self):
        assert _make_signal(direction=-1).direction_label == "SHORT"

    def test_direction_label_neutral(self):
        assert _make_signal(direction=0).direction_label == "NEUTRAL"

    def test_summary_valid(self):
        s = _make_signal()
        summary = s.summary()
        assert "LONG" in summary
        assert "EURUSD" in summary
        assert "1.1050" in summary  # TP
        assert "london" in summary

    def test_summary_filtered(self):
        s = _make_signal(filter_reason="confianza baja")
        summary = s.summary()
        assert "Sin señal" in summary
        assert "confianza baja" in summary


# ══════════════════════════════════════════════════════════════════════════════
# Lógica de filtros (sin BD ni modelos)
# ══════════════════════════════════════════════════════════════════════════════

class TestFilterLogic:
    """Verifica la lógica de filtros replicando el flujo de generate_signal."""

    def _apply_filters(self, direction, confidence, agreement, adx, session, rr_ratio, cfg=None):
        """Replica la cadena de filtros del generador."""
        cfg = cfg or FILTERS
        if direction == 0:
            return "neutral"
        if confidence < cfg["min_confidence"]:
            return f"confianza baja"
        if not agreement:
            return "XGBoost y LSTM no coinciden"
        if not np.isnan(adx) and adx < cfg["min_adx"]:
            return f"ADX débil"
        if not cfg["allow_offmarket"] and session == "offmarket":
            return "fuera de sesión"
        if rr_ratio < cfg["min_rr"]:
            return f"R:R insuficiente"
        return None  # pasó todos los filtros

    def test_neutral_filtered(self):
        assert self._apply_filters(0, 0.8, True, 30, "london", 2.0) == "neutral"

    def test_low_confidence_filtered(self):
        result = self._apply_filters(1, 0.50, True, 30, "london", 2.0)
        assert "confianza" in result

    def test_no_agreement_filtered(self):
        result = self._apply_filters(1, 0.70, False, 30, "london", 2.0)
        assert "no coinciden" in result

    def test_low_adx_filtered(self):
        result = self._apply_filters(1, 0.70, True, 15, "london", 2.0)
        assert "ADX" in result

    def test_offmarket_filtered(self):
        result = self._apply_filters(1, 0.70, True, 30, "offmarket", 2.0)
        assert "sesión" in result

    def test_low_rr_filtered(self):
        result = self._apply_filters(1, 0.70, True, 30, "london", 1.0)
        assert "R:R" in result

    def test_all_pass(self):
        """Señal que cumple todos los criterios → None (sin filtro)."""
        result = self._apply_filters(1, 0.70, True, 30, "london", 2.0)
        assert result is None

    def test_offmarket_allowed_if_configured(self):
        cfg = {**FILTERS, "allow_offmarket": True}
        result = self._apply_filters(1, 0.70, True, 30, "offmarket", 2.0, cfg)
        assert result is None

    def test_custom_min_confidence(self):
        cfg = {**FILTERS, "min_confidence": 0.80}
        result = self._apply_filters(1, 0.75, True, 30, "london", 2.0, cfg)
        assert "confianza" in result

    def test_filter_priority_order(self):
        """Los filtros se aplican en orden: neutral → confidence → agreement → ADX → session → RR."""
        # Neutral tiene prioridad sobre todo
        assert self._apply_filters(0, 0.50, False, 10, "offmarket", 0.5) == "neutral"
        # Confidence antes que agreement
        result = self._apply_filters(1, 0.50, False, 10, "offmarket", 0.5)
        assert "confianza" in result


# ══════════════════════════════════════════════════════════════════════════════
# Cálculo TP / SL
# ══════════════════════════════════════════════════════════════════════════════

class TestTpSlCalculation:
    """Verifica cálculos de TP/SL replicando la lógica del generador."""

    def test_long_tp_above_entry(self):
        entry, atr = 1.1000, 0.0025
        tp = entry + TP_MULTIPLIER * atr
        sl = entry - SL_MULTIPLIER * atr
        assert tp > entry
        assert sl < entry

    def test_short_tp_below_entry(self):
        entry, atr = 1.1000, 0.0025
        tp = entry - TP_MULTIPLIER * atr
        sl = entry + SL_MULTIPLIER * atr
        assert tp < entry
        assert sl > entry

    def test_rr_ratio_with_default_multipliers(self):
        """Con TP=2×ATR y SL=1×ATR → R:R = 2.0."""
        entry, atr = 1.1000, 0.0025
        tp = entry + TP_MULTIPLIER * atr
        sl = entry - SL_MULTIPLIER * atr
        tp_pips = _price_to_pips(tp - entry, "EURUSD")
        sl_pips = _price_to_pips(sl - entry, "EURUSD")
        rr = tp_pips / sl_pips if sl_pips > 0 else 0
        assert rr == pytest.approx(TP_MULTIPLIER / SL_MULTIPLIER, rel=1e-6)

    def test_xauusd_tp_sl_scale(self):
        entry, atr = 2650.0, 15.0
        tp = entry + TP_MULTIPLIER * atr  # 2650 + 30 = 2680
        sl = entry - SL_MULTIPLIER * atr  # 2650 - 15 = 2635
        tp_pips = _price_to_pips(tp - entry, "XAUUSD")  # 30/0.1 = 300
        sl_pips = _price_to_pips(sl - entry, "XAUUSD")  # 15/0.1 = 150
        assert tp_pips == 300.0
        assert sl_pips == 150.0

    def test_zero_atr_no_crash(self):
        """ATR=0 → TP=SL=entry."""
        entry, atr = 1.1000, 0.0
        tp = entry + TP_MULTIPLIER * atr
        sl = entry - SL_MULTIPLIER * atr
        assert tp == entry
        assert sl == entry


# ══════════════════════════════════════════════════════════════════════════════
# Constantes
# ══════════════════════════════════════════════════════════════════════════════

class TestConstants:
    def test_pip_sizes_defined(self):
        for pair in ["EURUSD", "GBPUSD", "USDJPY", "EURJPY", "XAUUSD"]:
            assert pair in PIP_SIZE

    def test_default_filters_sane(self):
        assert 0 < FILTERS["min_confidence"] < 1
        assert FILTERS["min_adx"] > 0
        assert FILTERS["min_rr"] > 0
        assert FILTERS["cooldown_bars"] >= 0
        assert isinstance(FILTERS["allow_offmarket"], bool)

    def test_multipliers_positive(self):
        assert TP_MULTIPLIER > 0
        assert SL_MULTIPLIER > 0
        assert TP_MULTIPLIER > SL_MULTIPLIER, "TP debe ser mayor que SL para R:R > 1"
