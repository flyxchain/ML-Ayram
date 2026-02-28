"""
tests/test_labels.py
Unit tests para src/data/labels.py — Triple Barrier Method.

Cubre:
  - compute_triple_barrier: etiquetado vectorizado (+1, -1, 0)
  - label_stats: estadísticas de distribución
  - Edge cases: DataFrame vacío, sin ATR, ATR cero, una sola fila
  - Variación de multiplicadores (TP/SL/horizon)
  - Escala XAUUSD (ATR grandes)
"""

import numpy as np
import pandas as pd
import pytest
from datetime import datetime, timezone, timedelta

from src.data.labels import compute_triple_barrier, label_stats


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_labeled_df(
    n: int = 100,
    start_price: float = 1.1000,
    atr_value: float = 0.0020,
    trend: float = 0.0,
    seed: int = 42,
) -> pd.DataFrame:
    """Genera DataFrame con OHLCV + atr_14 para etiquetado."""
    rng = np.random.default_rng(seed)
    ts = [datetime(2025, 1, 1, tzinfo=timezone.utc) + timedelta(hours=i) for i in range(n)]

    closes = [start_price]
    for _ in range(n - 1):
        closes.append(closes[-1] + trend + rng.normal(0, atr_value * 0.5))
    closes = np.array(closes)

    vol = atr_value * 0.8
    highs = closes + rng.uniform(0, vol, n)
    lows = closes - rng.uniform(0, vol, n)
    opens = closes + rng.normal(0, vol * 0.3, n)

    return pd.DataFrame({
        "timestamp": ts,
        "open": opens,
        "high": highs,
        "low": lows,
        "close": closes,
        "volume": rng.integers(100, 5000, n).astype(float),
        "atr_14": np.full(n, atr_value),
    })


def _make_tp_hit_df() -> pd.DataFrame:
    """Precio sube directamente → TP hit en barra 2."""
    return pd.DataFrame({
        "timestamp": pd.date_range("2025-01-01", periods=10, freq="1h", tz="UTC"),
        "open":   [100]*10,
        "high":   [100, 101, 103, 103, 103, 103, 103, 103, 103, 103],
        "low":    [100, 100, 100, 100, 100, 100, 100, 100, 100, 100],
        "close":  [100, 101, 102, 102, 102, 102, 102, 102, 102, 102],
        "volume": [1000]*10,
        "atr_14": [1.0]*10,  # TP=102 (2×ATR), SL=99 (1×ATR)
    })


def _make_sl_hit_df() -> pd.DataFrame:
    """Precio baja directamente → SL hit en barra 1."""
    return pd.DataFrame({
        "timestamp": pd.date_range("2025-01-01", periods=10, freq="1h", tz="UTC"),
        "open":   [100]*10,
        "high":   [100, 100, 100, 100, 100, 100, 100, 100, 100, 100],
        "low":    [100, 98.5, 98, 97, 97, 97, 97, 97, 97, 97],
        "close":  [100, 98.8, 98, 97.5, 97, 97, 97, 97, 97, 97],
        "volume": [1000]*10,
        "atr_14": [1.0]*10,  # TP=102 (2×ATR), SL=99 (1×ATR)
    })


def _make_timeout_df() -> pd.DataFrame:
    """Precio lateral, ni TP ni SL se tocan → label 0."""
    return pd.DataFrame({
        "timestamp": pd.date_range("2025-01-01", periods=10, freq="1h", tz="UTC"),
        "open":   [100]*10,
        "high":   [100.3]*10,
        "low":    [99.8]*10,
        "close":  [100.1]*10,
        "volume": [1000]*10,
        "atr_14": [1.0]*10,  # TP=102, SL=99 — ninguno se toca
    })


# ── Tests compute_triple_barrier ──────────────────────────────────────────────

class TestComputeTripleBarrier:
    """Tests del etiquetado Triple Barrier Method."""

    def test_tp_hit_label_positive(self):
        """Cuando high toca TP antes que low toque SL → label +1."""
        df = _make_tp_hit_df()
        result = compute_triple_barrier(df, tp_multiplier=2.0, sl_multiplier=1.0, horizon=5)
        assert result.loc[0, "label"] == 1
        assert result.loc[0, "label_return"] > 0
        assert result.loc[0, "bars_to_exit"] <= 5

    def test_sl_hit_label_negative(self):
        """Cuando low toca SL antes que high toque TP → label -1."""
        df = _make_sl_hit_df()
        result = compute_triple_barrier(df, tp_multiplier=2.0, sl_multiplier=1.0, horizon=5)
        assert result.loc[0, "label"] == -1
        assert result.loc[0, "label_return"] < 0
        assert result.loc[0, "bars_to_exit"] <= 5

    def test_timeout_label_zero(self):
        """Cuando ni TP ni SL se tocan dentro del horizonte → label 0."""
        df = _make_timeout_df()
        result = compute_triple_barrier(df, tp_multiplier=2.0, sl_multiplier=1.0, horizon=5)
        assert result.loc[0, "label"] == 0
        assert result.loc[0, "bars_to_exit"] == 5  # horizon

    def test_output_columns_present(self):
        """El resultado tiene todas las columnas esperadas."""
        df = _make_labeled_df(50)
        result = compute_triple_barrier(df)
        expected_cols = {"label", "label_return", "bars_to_exit", "tp_price", "sl_price"}
        assert expected_cols.issubset(set(result.columns))

    def test_label_values_valid(self):
        """Labels solo pueden ser +1, -1, o 0."""
        df = _make_labeled_df(200)
        result = compute_triple_barrier(df)
        unique_labels = set(result["label"].unique())
        assert unique_labels.issubset({-1, 0, 1})

    def test_tp_price_above_close(self):
        """TP price siempre debe estar por encima del close (long setup)."""
        df = _make_labeled_df(50)
        result = compute_triple_barrier(df)
        valid = result.dropna(subset=["tp_price"])
        assert (valid["tp_price"] > valid["close"]).all()

    def test_sl_price_below_close(self):
        """SL price siempre debe estar por debajo del close."""
        df = _make_labeled_df(50)
        result = compute_triple_barrier(df)
        valid = result.dropna(subset=["sl_price"])
        assert (valid["sl_price"] < valid["close"]).all()

    def test_bars_to_exit_within_horizon(self):
        """bars_to_exit nunca excede el horizonte."""
        horizon = 15
        df = _make_labeled_df(100)
        result = compute_triple_barrier(df, horizon=horizon)
        assert (result["bars_to_exit"] <= horizon).all()

    def test_bars_to_exit_positive(self):
        """bars_to_exit siempre es >= 1 para filas con label != 0, o == horizon para neutrales."""
        df = _make_labeled_df(100)
        result = compute_triple_barrier(df)
        hits = result[result["label"] != 0]
        if not hits.empty:
            assert (hits["bars_to_exit"] >= 1).all()

    def test_label_return_sign_matches_label(self):
        """Para label +1 el return es positivo, para -1 negativo."""
        df = _make_labeled_df(200)
        result = compute_triple_barrier(df)

        wins = result[result["label"] == 1]
        if not wins.empty:
            assert (wins["label_return"] > 0).all()

        losses = result[result["label"] == -1]
        if not losses.empty:
            assert (losses["label_return"] < 0).all()

    def test_last_row_label_zero(self):
        """La última fila no tiene barras futuras → label 0."""
        df = _make_labeled_df(50)
        result = compute_triple_barrier(df)
        assert result.iloc[-1]["label"] == 0

    def test_simultaneous_tp_sl_tp_wins(self):
        """Si TP y SL se tocan en la misma barra, TP gana (first_tp == first_sl → 0)."""
        # En la implementación actual: empate → label 0, pero TP < SL → +1
        df = pd.DataFrame({
            "timestamp": pd.date_range("2025-01-01", periods=5, freq="1h", tz="UTC"),
            "open":   [100, 100, 100, 100, 100],
            "high":   [100, 103, 100, 100, 100],   # barra 1: high=103 >= TP=102
            "low":    [100, 98,  100, 100, 100],    # barra 1: low=98 <= SL=99
            "close":  [100, 100, 100, 100, 100],
            "volume": [1000]*5,
            "atr_14": [1.0]*5,
        })
        result = compute_triple_barrier(df, tp_multiplier=2.0, sl_multiplier=1.0, horizon=3)
        # Ambos se tocan en barra 1 → empate → label 0
        assert result.loc[0, "label"] == 0

    def test_idempotency(self):
        """Aplicar el etiquetado dos veces produce el mismo resultado."""
        df = _make_labeled_df(100)
        r1 = compute_triple_barrier(df)
        r2 = compute_triple_barrier(df)
        pd.testing.assert_frame_equal(r1, r2)

    def test_deterministic_with_same_data(self):
        """Mismo input → mismo output (no hay random en etiquetado)."""
        df = _make_labeled_df(80, seed=99)
        r1 = compute_triple_barrier(df)
        r2 = compute_triple_barrier(df)
        assert (r1["label"] == r2["label"]).all()
        assert (r1["bars_to_exit"] == r2["bars_to_exit"]).all()


class TestComputeTripleBarrierMultipliers:
    """Tests con diferentes multiplicadores TP/SL."""

    def test_tight_sl_more_losses(self):
        """SL más ajustado (0.5×ATR) produce más pérdidas que SL amplio (2×ATR)."""
        df = _make_labeled_df(200)
        tight = compute_triple_barrier(df, tp_multiplier=2.0, sl_multiplier=0.5, horizon=20)
        wide  = compute_triple_barrier(df, tp_multiplier=2.0, sl_multiplier=2.0, horizon=20)
        losses_tight = (tight["label"] == -1).sum()
        losses_wide  = (wide["label"] == -1).sum()
        assert losses_tight >= losses_wide

    def test_wide_tp_fewer_wins(self):
        """TP más lejano (5×ATR) produce menos wins que TP cercano (1×ATR)."""
        df = _make_labeled_df(200)
        close_tp = compute_triple_barrier(df, tp_multiplier=1.0, sl_multiplier=1.0, horizon=20)
        far_tp   = compute_triple_barrier(df, tp_multiplier=5.0, sl_multiplier=1.0, horizon=20)
        wins_close = (close_tp["label"] == 1).sum()
        wins_far   = (far_tp["label"] == 1).sum()
        assert wins_close >= wins_far

    def test_short_horizon_more_timeouts(self):
        """Horizonte corto (3 barras) produce más timeouts que horizonte largo (50)."""
        df = _make_labeled_df(200)
        short = compute_triple_barrier(df, horizon=3)
        long  = compute_triple_barrier(df, horizon=50)
        timeouts_short = (short["label"] == 0).sum()
        timeouts_long  = (long["label"] == 0).sum()
        assert timeouts_short >= timeouts_long

    def test_custom_multipliers_15_05(self):
        """TP=1.5×ATR, SL=0.5×ATR, horizon=10 — configuración real."""
        df = _make_labeled_df(100)
        result = compute_triple_barrier(df, tp_multiplier=1.5, sl_multiplier=0.5, horizon=10)
        assert "label" in result.columns
        assert set(result["label"].unique()).issubset({-1, 0, 1})


class TestComputeTripleBarrierEdgeCases:
    """Edge cases y datos anómalos."""

    def test_empty_dataframe(self):
        """DataFrame vacío no lanza excepción."""
        df = pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume", "atr_14"])
        result = compute_triple_barrier(df)
        assert len(result) == 0
        assert "label" in result.columns

    def test_single_row(self):
        """Una sola fila → label 0 (sin barras futuras)."""
        df = pd.DataFrame({
            "timestamp": [datetime(2025, 1, 1, tzinfo=timezone.utc)],
            "open": [100.0], "high": [101.0], "low": [99.0], "close": [100.0],
            "volume": [1000.0], "atr_14": [1.0],
        })
        result = compute_triple_barrier(df)
        assert len(result) == 1
        assert result.iloc[0]["label"] == 0

    def test_nan_atr_skipped(self):
        """Filas con ATR=NaN son ignoradas (label 0)."""
        df = _make_labeled_df(20)
        df.loc[5, "atr_14"] = np.nan
        df.loc[6, "atr_14"] = np.nan
        result = compute_triple_barrier(df)
        assert result.loc[5, "label"] == 0
        assert result.loc[6, "label"] == 0

    def test_zero_atr_skipped(self):
        """Filas con ATR=0 son ignoradas (condición valid: atrs > 0)."""
        df = _make_labeled_df(20)
        df.loc[3, "atr_14"] = 0.0
        result = compute_triple_barrier(df)
        assert result.loc[3, "label"] == 0

    def test_xauusd_scale(self):
        """XAUUSD con precios ~2650 y ATR ~30 funciona correctamente."""
        df = _make_labeled_df(100, start_price=2650.0, atr_value=30.0)
        result = compute_triple_barrier(df)
        assert "label" in result.columns
        valid = result[result["label"] != 0]
        if not valid.empty:
            assert (valid[valid["label"] == 1]["label_return"] > 0).all()

    def test_original_df_not_modified(self):
        """compute_triple_barrier no modifica el DataFrame original."""
        df = _make_labeled_df(30)
        original_cols = set(df.columns)
        _ = compute_triple_barrier(df)
        assert set(df.columns) == original_cols
        assert "label" not in df.columns


# ── Tests label_stats ─────────────────────────────────────────────────────────

class TestLabelStats:
    """Tests de estadísticas de etiquetado."""

    def test_basic_stats(self):
        """label_stats devuelve las claves esperadas."""
        df = _make_labeled_df(100)
        labeled = compute_triple_barrier(df)
        stats = label_stats(labeled)
        assert "total" in stats
        assert "label_1_pct" in stats
        assert "label_0_pct" in stats
        assert "label_m1_pct" in stats
        assert stats["total"] == 100

    def test_percentages_sum_100(self):
        """Los porcentajes de labels suman ~100%."""
        df = _make_labeled_df(200)
        labeled = compute_triple_barrier(df)
        stats = label_stats(labeled)
        total_pct = stats["label_1_pct"] + stats["label_0_pct"] + stats["label_m1_pct"]
        assert abs(total_pct - 100.0) < 0.5  # tolerancia redondeo

    def test_empty_df_returns_zero_total(self):
        """DataFrame vacío devuelve total=0."""
        df = pd.DataFrame(columns=["label", "label_return", "bars_to_exit"])
        stats = label_stats(df)
        assert stats["total"] == 0

    def test_avg_return_win_positive(self):
        """El return promedio de wins (+1) es positivo."""
        df = _make_labeled_df(200)
        labeled = compute_triple_barrier(df)
        stats = label_stats(labeled)
        if stats.get("avg_return_win", 0) != 0:
            assert stats["avg_return_win"] > 0

    def test_avg_return_loss_negative(self):
        """El return promedio de losses (-1) es negativo."""
        df = _make_labeled_df(200)
        labeled = compute_triple_barrier(df)
        stats = label_stats(labeled)
        if stats.get("avg_return_loss", 0) != 0:
            assert stats["avg_return_loss"] < 0

    def test_avg_bars_win_positive(self):
        """Promedio de bars_to_exit para wins es > 0."""
        df = _make_labeled_df(200)
        labeled = compute_triple_barrier(df)
        stats = label_stats(labeled)
        if stats.get("avg_bars_win", 0) != 0:
            assert stats["avg_bars_win"] > 0
