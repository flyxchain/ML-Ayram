"""
tests/test_backtest_engine.py
Unit tests para src/backtest/engine.py

Cubre:
  - _calculate_lot_size: position sizing basado en riesgo
  - _simulate_trade: escenarios TP hit / SL hit / timeout, LONG y SHORT
  - _build_report: cálculos de métricas, equity curve, breakdowns
  - _tf_to_timedelta: conversión de timeframe a timedelta
  - TradeResult / BacktestReport: dataclasses
"""

import pytest
import numpy as np
import pandas as pd
from datetime import timedelta, datetime, timezone
from dataclasses import asdict

from src.backtest.engine import (
    _calculate_lot_size,
    _simulate_trade,
    _build_report,
    _tf_to_timedelta,
    TradeResult,
    BacktestReport,
    PIP_SIZE,
    PIP_VALUE_PER_LOT,
    SPREAD_PIPS,
    SLIPPAGE_PIPS,
    ACCOUNT_SIZE,
    RISK_PCT,
    MIN_LOT,
    MAX_LOT,
)


# ══════════════════════════════════════════════════════════════════════════════
# _tf_to_timedelta
# ══════════════════════════════════════════════════════════════════════════════

class TestTfToTimedelta:
    def test_m15(self):
        assert _tf_to_timedelta("M15") == timedelta(minutes=15)

    def test_h1(self):
        assert _tf_to_timedelta("H1") == timedelta(hours=1)

    def test_h4(self):
        assert _tf_to_timedelta("H4") == timedelta(hours=4)

    def test_d1(self):
        assert _tf_to_timedelta("D1") == timedelta(days=1)

    def test_unknown_defaults_to_h1(self):
        assert _tf_to_timedelta("W1") == timedelta(hours=1)


# ══════════════════════════════════════════════════════════════════════════════
# _calculate_lot_size
# ══════════════════════════════════════════════════════════════════════════════

class TestCalculateLotSize:
    def test_eurusd_normal(self):
        """Con 25 pips SL y 1% riesgo de 10k → lote ~4.0."""
        lot = _calculate_lot_size(25.0, "EURUSD")
        # risk=100€, pip_val=10€, SL=25 pips → 100/(25×10) = 0.4
        assert lot == 0.4

    def test_xauusd_wide_sl(self):
        """XAUUSD con SL amplio → lote pequeño."""
        lot = _calculate_lot_size(100.0, "XAUUSD")
        # 100€ / (100×10) = 0.10
        assert lot == 0.1

    def test_zero_sl_returns_min_lot(self):
        lot = _calculate_lot_size(0.0, "EURUSD")
        assert lot == MIN_LOT

    def test_negative_sl_returns_min_lot(self):
        lot = _calculate_lot_size(-10.0, "EURUSD")
        assert lot == MIN_LOT

    def test_tiny_sl_capped_at_max_lot(self):
        """SL de 0.1 pip → lote enorme → capped a MAX_LOT."""
        lot = _calculate_lot_size(0.1, "EURUSD")
        assert lot == MAX_LOT

    def test_respects_min_lot(self):
        """SL extremadamente grande → lote mínimo."""
        lot = _calculate_lot_size(50000.0, "EURUSD")
        assert lot == MIN_LOT

    def test_unknown_pair_uses_default_pip_value(self):
        """Par no configurado usa pip_val=10."""
        lot = _calculate_lot_size(25.0, "AUDCAD")
        assert lot == 0.4

    def test_round_to_2_decimals(self):
        lot = _calculate_lot_size(33.3, "EURUSD")
        # 100/(33.3×10) = 0.3003... → round to 0.30
        assert lot == round(lot, 2)


# ══════════════════════════════════════════════════════════════════════════════
# _simulate_trade — LONG
# ══════════════════════════════════════════════════════════════════════════════

class TestSimulateTradeLong:
    def test_tp_hit(self, sample_signal, forward_bars_tp_hit):
        """LONG que alcanza TP → result='tp_hit', PnL positivo."""
        trade = _simulate_trade(sample_signal, forward_bars_tp_hit)
        assert trade.result == "tp_hit"
        assert trade.pnl_pips > 0
        assert trade.pnl_eur > 0
        assert trade.exit_price == sample_signal["tp_price"]
        assert trade.bars_to_exit <= len(forward_bars_tp_hit)

    def test_sl_hit(self, sample_signal, forward_bars_sl_hit):
        """LONG que toca SL → result='sl_hit', PnL negativo."""
        trade = _simulate_trade(sample_signal, forward_bars_sl_hit)
        assert trade.result == "sl_hit"
        assert trade.pnl_pips < 0
        assert trade.pnl_eur < 0
        assert trade.exit_price == sample_signal["sl_price"]

    def test_timeout(self, sample_signal, forward_bars_timeout):
        """LONG sin tocar niveles → result='timeout'."""
        trade = _simulate_trade(sample_signal, forward_bars_timeout)
        assert trade.result == "timeout"
        assert trade.bars_to_exit == len(forward_bars_timeout)

    def test_spread_slippage_applied(self, sample_signal, forward_bars_tp_hit):
        """El actual_entry incluye spread + slippage sobre entry_price."""
        trade = _simulate_trade(sample_signal, forward_bars_tp_hit)
        pip_size = PIP_SIZE["EURUSD"]
        expected_spread = SPREAD_PIPS["EURUSD"] * pip_size
        expected_slip = SLIPPAGE_PIPS * pip_size
        expected_entry = sample_signal["entry_price"] + expected_spread + expected_slip
        assert abs(trade.actual_entry - expected_entry) < 1e-9

    def test_lot_size_positive(self, sample_signal, forward_bars_tp_hit):
        trade = _simulate_trade(sample_signal, forward_bars_tp_hit)
        assert trade.lot_size >= MIN_LOT
        assert trade.lot_size <= MAX_LOT

    def test_metadata_preserved(self, sample_signal, forward_bars_tp_hit):
        trade = _simulate_trade(sample_signal, forward_bars_tp_hit)
        assert trade.pair == "EURUSD"
        assert trade.timeframe == "H1"
        assert trade.direction == 1
        assert trade.confidence == 0.72


# ══════════════════════════════════════════════════════════════════════════════
# _simulate_trade — SHORT
# ══════════════════════════════════════════════════════════════════════════════

class TestSimulateTradeShort:
    def test_tp_hit_short(self, sample_signal_short):
        """SHORT XAUUSD: TP=2640 → barras que bajan alcanzan TP."""
        ohlcv = pd.DataFrame({
            "timestamp": pd.date_range("2025-06-15 20:00", periods=5, freq="4h", tz="UTC"),
            "open":  [2648, 2645, 2642, 2638, 2635],
            "high":  [2652, 2648, 2645, 2642, 2638],
            "low":   [2644, 2641, 2638, 2634, 2630],  # barra 1: low=2641>2640; barra 2: low=2638<=2640 ✓
            "close": [2645, 2642, 2639, 2635, 2632],
        })
        trade = _simulate_trade(sample_signal_short, ohlcv)
        assert trade.result == "tp_hit"
        assert trade.pnl_pips > 0

    def test_sl_hit_short(self, sample_signal_short):
        """SHORT XAUUSD: SL=2655 → barras que suben tocan SL."""
        ohlcv = pd.DataFrame({
            "timestamp": pd.date_range("2025-06-15 20:00", periods=5, freq="4h", tz="UTC"),
            "open":  [2651, 2653, 2656, 2658, 2660],
            "high":  [2654, 2656, 2659, 2661, 2663],  # barra 0: high=2654<2655; barra 1: high=2656>=2655 ✓
            "low":   [2649, 2651, 2654, 2656, 2658],
            "close": [2653, 2655, 2657, 2660, 2662],
        })
        trade = _simulate_trade(sample_signal_short, ohlcv)
        assert trade.result == "sl_hit"
        assert trade.pnl_pips < 0

    def test_spread_slippage_short(self, sample_signal_short):
        """Para SHORT el spread/slippage se resta del entry."""
        ohlcv = pd.DataFrame({
            "timestamp": pd.date_range("2025-06-15 20:00", periods=3, freq="4h", tz="UTC"),
            "open": [2649, 2648, 2647], "high": [2651, 2650, 2649],
            "low": [2647, 2646, 2645], "close": [2648, 2647, 2646],
        })
        trade = _simulate_trade(sample_signal_short, ohlcv)
        pip_size = PIP_SIZE["XAUUSD"]
        expected = sample_signal_short["entry_price"] - SPREAD_PIPS["XAUUSD"] * pip_size - SLIPPAGE_PIPS * pip_size
        assert abs(trade.actual_entry - expected) < 1e-6


# ══════════════════════════════════════════════════════════════════════════════
# _simulate_trade — Edge cases
# ══════════════════════════════════════════════════════════════════════════════

class TestSimulateTradeEdge:
    def test_empty_ohlcv(self, sample_signal):
        """Sin barras forward: timeout inmediato con entry como exit."""
        ohlcv = pd.DataFrame(columns=["timestamp", "open", "high", "low", "close"])
        trade = _simulate_trade(sample_signal, ohlcv)
        assert trade.result == "timeout"
        assert trade.bars_to_exit == 0

    def test_single_bar_tp(self, sample_signal):
        """Una sola barra que toca TP."""
        ohlcv = pd.DataFrame({
            "timestamp": [pd.Timestamp("2025-06-15 13:00", tz="UTC")],
            "open": [1.1010], "high": [1.1060], "low": [1.0990], "close": [1.1040],
        })
        trade = _simulate_trade(sample_signal, ohlcv)
        # TP=1.1050, high=1.1060 → TP hit
        # Pero SL=1.0975, low=1.0990 → SL no hit
        assert trade.result == "tp_hit"

    def test_bar_hits_both_tp_and_sl_tp_wins(self, sample_signal):
        """Si una barra toca AMBOS niveles, gana TP (por orden del loop: high check primero para LONG)."""
        ohlcv = pd.DataFrame({
            "timestamp": [pd.Timestamp("2025-06-15 13:00", tz="UTC")],
            "open": [1.1000], "high": [1.1060], "low": [1.0960], "close": [1.1000],
        })
        trade = _simulate_trade(sample_signal, ohlcv)
        # high >= TP se chequea antes que low <= SL en el loop
        assert trade.result == "tp_hit"


# ══════════════════════════════════════════════════════════════════════════════
# _build_report
# ══════════════════════════════════════════════════════════════════════════════

class TestBuildReport:
    def _make_trades(self):
        """Genera lista de TradeResult variados para probar métricas."""
        base = dict(
            entry_price=1.1, tp_price=1.105, sl_price=1.097,
            tp_pips=50, sl_pips=30, confidence=0.7,
            adx=25, session="london", actual_entry=1.1001,
        )
        return [
            TradeResult(signal_id=1, pair="EURUSD", timeframe="H1", direction=1, rr_ratio=2.0,
                        exit_price=1.105, result="tp_hit", pnl_pips=49.0, pnl_eur=196.0,
                        lot_size=0.4, bars_to_exit=5, opened_at="2025-06-01 10:00",
                        closed_at="2025-06-01 15:00", **base),
            TradeResult(signal_id=2, pair="EURUSD", timeframe="H1", direction=-1, rr_ratio=2.0,
                        exit_price=1.097, result="sl_hit", pnl_pips=-30.0, pnl_eur=-120.0,
                        lot_size=0.4, bars_to_exit=3, opened_at="2025-06-02 14:00",
                        closed_at="2025-06-02 17:00", **base),
            TradeResult(signal_id=3, pair="XAUUSD", timeframe="H4", direction=1, rr_ratio=1.8,
                        exit_price=2660.0, result="tp_hit", pnl_pips=100.0, pnl_eur=400.0,
                        lot_size=0.4, bars_to_exit=8, opened_at="2025-07-10 08:00",
                        closed_at="2025-07-10 16:00", **{**base, "entry_price": 2650.0}),
            TradeResult(signal_id=4, pair="XAUUSD", timeframe="H4", direction=-1, rr_ratio=2.0,
                        exit_price=2650.0, result="timeout", pnl_pips=-5.0, pnl_eur=-20.0,
                        lot_size=0.4, bars_to_exit=40, opened_at="2025-07-15 12:00",
                        closed_at="2025-07-16 12:00", **{**base, "entry_price": 2650.0}),
        ]

    def test_counts(self):
        trades = self._make_trades()
        signals_df = pd.DataFrame({"id": [1, 2, 3, 4]})
        report = _build_report(
            trades, signals_df,
            ["EURUSD", "XAUUSD"], ["H1", "H4"],
            datetime(2025, 6, 1), datetime(2025, 8, 1), 0.54,
        )
        assert report.total_trades == 4
        assert report.wins == 2
        assert report.losses == 1
        assert report.timeouts == 1

    def test_pnl_sums(self):
        trades = self._make_trades()
        signals_df = pd.DataFrame({"id": range(4)})
        report = _build_report(
            trades, signals_df,
            ["EURUSD", "XAUUSD"], ["H1", "H4"],
            datetime(2025, 6, 1), datetime(2025, 8, 1), 0.54,
        )
        assert report.total_pnl_eur == round(196.0 - 120.0 + 400.0 - 20.0, 2)
        assert report.gross_profit == round(196.0 + 400.0, 2)
        assert report.gross_loss == round(120.0 + 20.0, 2)

    def test_win_rate(self):
        trades = self._make_trades()
        signals_df = pd.DataFrame({"id": range(4)})
        report = _build_report(
            trades, signals_df,
            ["EURUSD", "XAUUSD"], ["H1", "H4"],
            datetime(2025, 6, 1), datetime(2025, 8, 1), 0.54,
        )
        assert report.win_rate == 50.0

    def test_profit_factor(self):
        trades = self._make_trades()
        signals_df = pd.DataFrame({"id": range(4)})
        report = _build_report(
            trades, signals_df,
            ["EURUSD", "XAUUSD"], ["H1", "H4"],
            datetime(2025, 6, 1), datetime(2025, 8, 1), 0.54,
        )
        expected_pf = round(596.0 / 140.0, 2)
        assert report.profit_factor == expected_pf

    def test_equity_curve_length(self):
        trades = self._make_trades()
        signals_df = pd.DataFrame({"id": range(4)})
        report = _build_report(
            trades, signals_df,
            ["EURUSD", "XAUUSD"], ["H1", "H4"],
            datetime(2025, 6, 1), datetime(2025, 8, 1), 0.54,
        )
        assert len(report.equity_curve) == 4

    def test_by_pair_breakdown(self):
        trades = self._make_trades()
        signals_df = pd.DataFrame({"id": range(4)})
        report = _build_report(
            trades, signals_df,
            ["EURUSD", "XAUUSD"], ["H1", "H4"],
            datetime(2025, 6, 1), datetime(2025, 8, 1), 0.54,
        )
        assert "EURUSD" in report.by_pair
        assert "XAUUSD" in report.by_pair
        assert report.by_pair["EURUSD"]["trades"] == 2
        assert report.by_pair["XAUUSD"]["trades"] == 2

    def test_by_month_breakdown(self):
        trades = self._make_trades()
        signals_df = pd.DataFrame({"id": range(4)})
        report = _build_report(
            trades, signals_df,
            ["EURUSD", "XAUUSD"], ["H1", "H4"],
            datetime(2025, 6, 1), datetime(2025, 8, 1), 0.54,
        )
        assert "2025-06" in report.by_month
        assert "2025-07" in report.by_month

    def test_max_drawdown(self):
        trades = self._make_trades()
        signals_df = pd.DataFrame({"id": range(4)})
        report = _build_report(
            trades, signals_df,
            ["EURUSD", "XAUUSD"], ["H1", "H4"],
            datetime(2025, 6, 1), datetime(2025, 8, 1), 0.54,
        )
        assert report.max_drawdown >= 0
        assert report.max_drawdown_pct >= 0

    def test_empty_trades(self):
        signals_df = pd.DataFrame({"id": [1]})
        report = _build_report(
            [], signals_df,
            ["EURUSD"], ["H1"],
            datetime(2025, 6, 1), datetime(2025, 8, 1), 0.54,
        )
        assert report.total_trades == 0
        assert report.win_rate == 0
        assert report.total_pnl_eur == 0

    def test_all_wins(self):
        """100% win rate → profit_factor = inf (o muy alto)."""
        t = TradeResult(
            signal_id=1, pair="EURUSD", timeframe="H1", direction=1,
            confidence=0.8, entry_price=1.1, tp_price=1.105, sl_price=1.097,
            tp_pips=50, sl_pips=30, rr_ratio=2.0, adx=25, session="london",
            actual_entry=1.1001, exit_price=1.105, result="tp_hit",
            pnl_pips=49, pnl_eur=196, lot_size=0.4, bars_to_exit=5,
            opened_at="2025-06-01 10:00", closed_at="2025-06-01 15:00",
        )
        signals_df = pd.DataFrame({"id": [1]})
        report = _build_report(
            [t], signals_df,
            ["EURUSD"], ["H1"],
            datetime(2025, 6, 1), datetime(2025, 8, 1), 0.54,
        )
        assert report.win_rate == 100.0
        assert report.profit_factor == float("inf")


# ══════════════════════════════════════════════════════════════════════════════
# Constants sanity checks
# ══════════════════════════════════════════════════════════════════════════════

class TestConstants:
    def test_all_pairs_have_pip_size(self):
        for pair in ["EURUSD", "GBPUSD", "USDJPY", "EURJPY", "XAUUSD"]:
            assert pair in PIP_SIZE
            assert PIP_SIZE[pair] > 0

    def test_all_pairs_have_pip_value(self):
        for pair in ["EURUSD", "GBPUSD", "USDJPY", "EURJPY", "XAUUSD"]:
            assert pair in PIP_VALUE_PER_LOT
            assert PIP_VALUE_PER_LOT[pair] > 0

    def test_all_pairs_have_spread(self):
        for pair in ["EURUSD", "GBPUSD", "USDJPY", "EURJPY", "XAUUSD"]:
            assert pair in SPREAD_PIPS
            assert SPREAD_PIPS[pair] >= 0

    def test_risk_params_sane(self):
        assert ACCOUNT_SIZE > 0
        assert 0 < RISK_PCT < 1
        assert MIN_LOT > 0
        assert MAX_LOT > MIN_LOT
