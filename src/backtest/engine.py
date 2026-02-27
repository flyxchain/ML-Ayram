"""
src/backtest/engine.py
Backtesting realista sobre señales históricas almacenadas en la BD.

Flujo:
  1. Carga señales válidas de la tabla `signals`
  2. Para cada señal busca en `ohlcv_raw` la primera barra que toca TP o SL
  3. Calcula PnL con lot sizing basado en riesgo, spread y slippage
  4. Genera un informe completo de métricas

Uso:
  python -m src.backtest.engine
  python -m src.backtest.engine --pair EURUSD --tf H1 --days 90
  python -m src.backtest.engine --pair EURUSD GBPUSD --min-confidence 0.60
  python -m src.backtest.engine --output results/backtest_20260227.json
"""

import argparse
import json
import os
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from loguru import logger
from sqlalchemy import create_engine, text

load_dotenv()
DATABASE_URL = os.getenv("DATABASE_URL")

# ── Constantes ────────────────────────────────────────────────────────────────

PIP_SIZE = {
    "EURUSD": 0.0001,
    "GBPUSD": 0.0001,
    "USDJPY": 0.01,
    "EURJPY": 0.01,
    "XAUUSD": 0.10,
}

# Valor monetario de 1 pip con 1 lote estándar (en EUR aproximado)
PIP_VALUE_PER_LOT = {
    "EURUSD": 10.0,
    "GBPUSD": 10.0,
    "USDJPY": 6.5,    # ~90 JPY/EUR aproximado
    "EURJPY": 10.0,
    "XAUUSD": 10.0,   # 1 pip = 0.10 USD ≈ 0.09 EUR; por simplicidad 10€/lote
}

# Spread típico por par (en pips) — afecta al entry
SPREAD_PIPS = {
    "EURUSD": 0.5,
    "GBPUSD": 0.8,
    "USDJPY": 0.5,
    "EURJPY": 0.8,
    "XAUUSD": 3.0,
}

# Slippage adicional en pips (ejecución de mercado)
SLIPPAGE_PIPS = 0.5

# Horizonte máximo de búsqueda TP/SL en barras
MAX_HORIZON = 40

# Parámetros de position sizing
ACCOUNT_SIZE   = 10_000.0   # € de cuenta base para cálculos
RISK_PCT       = 0.01       # 1% de riesgo por trade
MIN_LOT        = 0.01
MAX_LOT        = 5.0


# ── Dataclasses ───────────────────────────────────────────────────────────────

@dataclass
class TradeResult:
    signal_id:     int
    pair:          str
    timeframe:     str
    direction:     int
    confidence:    float
    entry_price:   float
    tp_price:      float
    sl_price:      float
    tp_pips:       float
    sl_pips:       float
    rr_ratio:      float
    adx:           float
    session:       str

    # Ejecución simulada
    actual_entry:  float = 0.0   # entry + spread/slippage
    exit_price:    float = 0.0
    result:        str   = ""    # "tp_hit" | "sl_hit" | "timeout"
    pnl_pips:      float = 0.0
    pnl_eur:       float = 0.0
    lot_size:      float = 0.0
    bars_to_exit:  int   = 0
    opened_at:     str   = ""
    closed_at:     str   = ""


@dataclass
class BacktestReport:
    # Configuración
    pairs:          list
    timeframes:     list
    date_from:      str
    date_to:        str
    min_confidence: float
    account_size:   float
    risk_pct:       float

    # Resumen
    total_signals:  int   = 0   # señales encontradas en el período
    total_trades:   int   = 0   # señales válidas simuladas (direction != 0, no filtradas)
    wins:           int   = 0
    losses:         int   = 0
    timeouts:       int   = 0

    # PnL
    total_pnl_eur:  float = 0.0
    total_pnl_pips: float = 0.0
    gross_profit:   float = 0.0
    gross_loss:     float = 0.0
    profit_factor:  float = 0.0

    # Ratios
    win_rate:       float = 0.0
    avg_win_eur:    float = 0.0
    avg_loss_eur:   float = 0.0
    avg_rr:         float = 0.0
    expectancy:     float = 0.0   # E[PnL] por trade = WR×AvgWin - LR×AvgLoss

    # Drawdown
    max_drawdown:   float = 0.0
    max_drawdown_pct: float = 0.0

    # Por par
    by_pair:        dict  = field(default_factory=dict)

    # Por timeframe
    by_timeframe:   dict  = field(default_factory=dict)

    # Por mes
    by_month:       dict  = field(default_factory=dict)

    # Curva de equity
    equity_curve:   list  = field(default_factory=list)   # [{date, equity, drawdown}]

    # Trades individuales
    trades:         list  = field(default_factory=list)


# ── Carga de datos ────────────────────────────────────────────────────────────

def _load_signals(
    engine,
    pairs:          list,
    timeframes:     list,
    date_from:      datetime,
    date_to:        datetime,
    min_confidence: float,
) -> pd.DataFrame:
    """Carga señales válidas (direction != 0, sin filter_reason) de la BD."""
    pair_list = ", ".join(f"'{p}'" for p in pairs)
    tf_list   = ", ".join(f"'{t}'" for t in timeframes)

    df = pd.read_sql(
        text(f"""
            SELECT *
            FROM signals
            WHERE pair        IN ({pair_list})
              AND timeframe   IN ({tf_list})
              AND timestamp   >= :date_from
              AND timestamp   <= :date_to
              AND direction   != 0
              AND filter_reason IS NULL
              AND confidence  >= :min_conf
            ORDER BY timestamp
        """),
        engine,
        params={
            "date_from": date_from,
            "date_to":   date_to,
            "min_conf":  min_confidence,
        },
    )
    logger.info(f"Señales cargadas: {len(df)}")
    return df


def _load_ohlcv(engine, pair: str, timeframe: str, from_ts, to_ts) -> pd.DataFrame:
    """Carga velas OHLCV en un rango para simular la ejecución."""
    df = pd.read_sql(
        text("""
            SELECT timestamp, open, high, low, close
            FROM ohlcv_raw
            WHERE pair      = :pair
              AND timeframe = :tf
              AND timestamp >= :from_ts
              AND timestamp <= :to_ts
            ORDER BY timestamp
        """),
        engine,
        params={"pair": pair, "tf": timeframe, "from_ts": from_ts, "to_ts": to_ts},
    )
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df.reset_index(drop=True)


# ── Simulación de un trade ────────────────────────────────────────────────────

def _calculate_lot_size(sl_pips: float, pair: str) -> float:
    """Calcula el tamaño de lote basado en riesgo fijo."""
    pip_val = PIP_VALUE_PER_LOT.get(pair, 10.0)
    risk_eur = ACCOUNT_SIZE * RISK_PCT
    if sl_pips <= 0 or pip_val <= 0:
        return MIN_LOT
    lot = risk_eur / (sl_pips * pip_val)
    return round(max(MIN_LOT, min(MAX_LOT, lot)), 2)


def _simulate_trade(signal: pd.Series, ohlcv: pd.DataFrame) -> TradeResult:
    """
    Simula la ejecución de una señal buscando el primer bar que toca TP o SL.
    Aplica spread y slippage al entry.
    """
    pair      = signal["pair"]
    pip_size  = PIP_SIZE.get(pair, 0.0001)
    pip_val   = PIP_VALUE_PER_LOT.get(pair, 10.0)
    spread    = SPREAD_PIPS.get(pair, 1.0) * pip_size
    slippage  = SLIPPAGE_PIPS * pip_size
    direction = int(signal["direction"])

    # Entry real con spread + slippage (siempre nos penaliza)
    if direction == 1:    # LONG: compramos al ask = entry + spread + slippage
        actual_entry = signal["entry_price"] + spread + slippage
    else:                 # SHORT: vendemos al bid = entry - spread - slippage
        actual_entry = signal["entry_price"] - spread - slippage

    tp = float(signal["tp_price"])
    sl = float(signal["sl_price"])

    # Buscar primera barra que toca TP o SL
    result_str    = "timeout"
    exit_price    = float(ohlcv.iloc[-1]["close"]) if len(ohlcv) else actual_entry
    bars_to_exit  = len(ohlcv)
    closed_at     = str(ohlcv.iloc[-1]["timestamp"]) if len(ohlcv) else str(signal["timestamp"])

    for i, row in ohlcv.iterrows():
        h, l = float(row["high"]), float(row["low"])

        if direction == 1:
            if h >= tp:
                result_str   = "tp_hit"
                exit_price   = tp
                bars_to_exit = i + 1
                closed_at    = str(row["timestamp"])
                break
            if l <= sl:
                result_str   = "sl_hit"
                exit_price   = sl
                bars_to_exit = i + 1
                closed_at    = str(row["timestamp"])
                break
        else:  # SHORT
            if l <= tp:
                result_str   = "tp_hit"
                exit_price   = tp
                bars_to_exit = i + 1
                closed_at    = str(row["timestamp"])
                break
            if h >= sl:
                result_str   = "sl_hit"
                exit_price   = sl
                bars_to_exit = i + 1
                closed_at    = str(row["timestamp"])
                break

    # PnL en pips y en euros
    if direction == 1:
        pnl_pips = (exit_price - actual_entry) / pip_size
    else:
        pnl_pips = (actual_entry - exit_price) / pip_size

    sl_pips  = abs(signal["sl_pips"]) if signal["sl_pips"] else abs(actual_entry - sl) / pip_size
    lot_size = _calculate_lot_size(sl_pips, pair)
    pnl_eur  = pnl_pips * pip_val * lot_size

    return TradeResult(
        signal_id    = int(signal.get("id", 0)),
        pair         = pair,
        timeframe    = signal["timeframe"],
        direction    = direction,
        confidence   = float(signal["confidence"]),
        entry_price  = float(signal["entry_price"]),
        tp_price     = tp,
        sl_price     = sl,
        tp_pips      = float(signal.get("tp_pips", 0)),
        sl_pips      = float(signal.get("sl_pips", 0)),
        rr_ratio     = float(signal.get("rr_ratio", 0)),
        adx          = float(signal.get("adx", 0)),
        session      = str(signal.get("session", "")),
        actual_entry = round(actual_entry, 6),
        exit_price   = round(exit_price, 6),
        result       = result_str,
        pnl_pips     = round(pnl_pips, 2),
        pnl_eur      = round(pnl_eur, 2),
        lot_size     = lot_size,
        bars_to_exit = bars_to_exit,
        opened_at    = str(signal["timestamp"]),
        closed_at    = closed_at,
    )


# ── Motor principal ───────────────────────────────────────────────────────────

def run_backtest(
    pairs:          list        = None,
    timeframes:     list        = None,
    days:           int         = 90,
    date_from:      datetime    = None,
    date_to:        datetime    = None,
    min_confidence: float       = 0.54,
    output_path:    Path        = None,
) -> BacktestReport:
    """
    Ejecuta el backtest completo y devuelve un BacktestReport.
    """
    pairs      = pairs      or ["EURUSD", "GBPUSD", "USDJPY", "EURJPY", "XAUUSD"]
    timeframes = timeframes or ["M15", "H1", "H4"]

    if date_to is None:
        date_to = datetime.now(timezone.utc)
    if date_from is None:
        date_from = date_to - timedelta(days=days)

    logger.info("═" * 60)
    logger.info("ML-Ayram — Backtest")
    logger.info(f"Período:    {date_from.date()} → {date_to.date()}")
    logger.info(f"Pares:      {pairs}")
    logger.info(f"Timeframes: {timeframes}")
    logger.info(f"Confianza:  ≥ {min_confidence:.0%}")
    logger.info("═" * 60)

    engine = create_engine(DATABASE_URL)

    signals_df = _load_signals(engine, pairs, timeframes, date_from, date_to, min_confidence)

    if signals_df.empty:
        logger.warning("Sin señales en el período. Verifica que el generador haya corrido.")
        return BacktestReport(
            pairs=pairs, timeframes=timeframes,
            date_from=str(date_from.date()), date_to=str(date_to.date()),
            min_confidence=min_confidence, account_size=ACCOUNT_SIZE, risk_pct=RISK_PCT,
            total_signals=0,
        )

    trades: list[TradeResult] = []

    # Agrupar por par+timeframe para cargar OHLCV en bloque
    for (pair, tf), group in signals_df.groupby(["pair", "timeframe"]):
        logger.info(f"Simulando {pair} {tf}: {len(group)} señales")

        for _, signal in group.iterrows():
            sig_ts = pd.to_datetime(signal["timestamp"], utc=True)

            # Cargar las MAX_HORIZON barras siguientes a la señal
            ohlcv = _load_ohlcv(
                engine, pair, tf,
                from_ts = sig_ts,
                to_ts   = sig_ts + _tf_to_timedelta(tf) * (MAX_HORIZON + 1),
            )
            # Excluir la barra de la señal misma (la primera es la de entrada)
            ohlcv = ohlcv[ohlcv["timestamp"] > sig_ts].reset_index(drop=True)

            if ohlcv.empty:
                logger.debug(f"  Sin barras forward para {pair} {tf} @ {sig_ts} — omitiendo")
                continue

            trade = _simulate_trade(signal, ohlcv)
            trades.append(trade)

    if not trades:
        logger.warning("Ningún trade simulado (sin barras OHLCV forward).")
        return BacktestReport(
            pairs=pairs, timeframes=timeframes,
            date_from=str(date_from.date()), date_to=str(date_to.date()),
            min_confidence=min_confidence, account_size=ACCOUNT_SIZE, risk_pct=RISK_PCT,
            total_signals=len(signals_df),
        )

    report = _build_report(
        trades, signals_df, pairs, timeframes,
        date_from, date_to, min_confidence,
    )

    _print_report(report)

    if output_path:
        _save_report(report, output_path)

    return report


# ── Construcción del informe ──────────────────────────────────────────────────

def _build_report(
    trades:         list[TradeResult],
    signals_df:     pd.DataFrame,
    pairs:          list,
    timeframes:     list,
    date_from:      datetime,
    date_to:        datetime,
    min_confidence: float,
) -> BacktestReport:

    report = BacktestReport(
        pairs           = pairs,
        timeframes      = timeframes,
        date_from       = str(date_from.date()),
        date_to         = str(date_to.date()),
        min_confidence  = min_confidence,
        account_size    = ACCOUNT_SIZE,
        risk_pct        = RISK_PCT,
        total_signals   = len(signals_df),
        total_trades    = len(trades),
    )

    wins     = [t for t in trades if t.result == "tp_hit"]
    losses   = [t for t in trades if t.result == "sl_hit"]
    timeouts = [t for t in trades if t.result == "timeout"]

    report.wins     = len(wins)
    report.losses   = len(losses)
    report.timeouts = len(timeouts)

    pnls = [t.pnl_eur for t in trades]
    report.total_pnl_eur  = round(sum(pnls), 2)
    report.total_pnl_pips = round(sum(t.pnl_pips for t in trades), 2)
    report.gross_profit   = round(sum(t.pnl_eur for t in trades if t.pnl_eur > 0), 2)
    report.gross_loss     = round(abs(sum(t.pnl_eur for t in trades if t.pnl_eur < 0)), 2)
    report.profit_factor  = round(
        report.gross_profit / report.gross_loss if report.gross_loss > 0 else float("inf"), 2
    )

    n = len(trades)
    report.win_rate    = round(len(wins) / n * 100, 1) if n else 0.0
    report.avg_win_eur = round(sum(t.pnl_eur for t in wins)   / len(wins)   if wins   else 0.0, 2)
    report.avg_loss_eur= round(sum(t.pnl_eur for t in losses) / len(losses) if losses else 0.0, 2)
    report.avg_rr      = round(sum(t.rr_ratio for t in trades) / n, 2) if n else 0.0

    wr = report.win_rate / 100
    lr = 1 - wr
    report.expectancy  = round(wr * report.avg_win_eur + lr * report.avg_loss_eur, 2)

    # ── Curva de equity y drawdown ────────────────────────────────────────
    equity = ACCOUNT_SIZE
    peak   = ACCOUNT_SIZE
    max_dd = 0.0
    equity_curve = []

    for t in sorted(trades, key=lambda x: x.opened_at):
        equity += t.pnl_eur
        peak    = max(peak, equity)
        dd      = peak - equity
        dd_pct  = dd / peak * 100 if peak > 0 else 0
        max_dd  = max(max_dd, dd)
        equity_curve.append({
            "date":       t.opened_at[:10],
            "equity":     round(equity, 2),
            "drawdown":   round(dd, 2),
            "drawdown_pct": round(dd_pct, 2),
        })

    report.equity_curve       = equity_curve
    report.max_drawdown       = round(max_dd, 2)
    report.max_drawdown_pct   = round(max_dd / ACCOUNT_SIZE * 100, 2) if ACCOUNT_SIZE else 0

    # ── Breakdown por par ─────────────────────────────────────────────────
    for pair in pairs:
        pt = [t for t in trades if t.pair == pair]
        if not pt:
            continue
        pw = [t for t in pt if t.result == "tp_hit"]
        report.by_pair[pair] = {
            "trades":   len(pt),
            "wins":     len(pw),
            "win_rate": round(len(pw) / len(pt) * 100, 1),
            "pnl_eur":  round(sum(t.pnl_eur for t in pt), 2),
            "pnl_pips": round(sum(t.pnl_pips for t in pt), 2),
        }

    # ── Breakdown por timeframe ───────────────────────────────────────────
    for tf in timeframes:
        tt = [t for t in trades if t.timeframe == tf]
        if not tt:
            continue
        tw = [t for t in tt if t.result == "tp_hit"]
        report.by_timeframe[tf] = {
            "trades":   len(tt),
            "wins":     len(tw),
            "win_rate": round(len(tw) / len(tt) * 100, 1),
            "pnl_eur":  round(sum(t.pnl_eur for t in tt), 2),
        }

    # ── Breakdown por mes ─────────────────────────────────────────────────
    for t in trades:
        month = t.opened_at[:7]   # "2026-01"
        if month not in report.by_month:
            report.by_month[month] = {"trades": 0, "wins": 0, "pnl_eur": 0.0}
        report.by_month[month]["trades"]  += 1
        report.by_month[month]["wins"]    += 1 if t.result == "tp_hit" else 0
        report.by_month[month]["pnl_eur"] = round(
            report.by_month[month]["pnl_eur"] + t.pnl_eur, 2
        )

    report.trades = [asdict(t) for t in trades]
    return report


# ── Helpers ───────────────────────────────────────────────────────────────────

def _tf_to_timedelta(tf: str) -> timedelta:
    return {"M15": timedelta(minutes=15), "H1": timedelta(hours=1),
            "H4": timedelta(hours=4),     "D1": timedelta(days=1)}.get(tf, timedelta(hours=1))


def _print_report(r: BacktestReport) -> None:
    """Imprime el informe en consola."""
    pf_str = str(r.profit_factor) if r.profit_factor != float("inf") else "∞"
    logger.info("")
    logger.info("═" * 60)
    logger.info("RESULTADOS DEL BACKTEST")
    logger.info("═" * 60)
    logger.info(f"Período:        {r.date_from} → {r.date_to}")
    logger.info(f"Señales totales:{r.total_signals:>8}")
    logger.info(f"Trades simulados:{r.total_trades:>7}")
    logger.info("─" * 60)
    logger.info(f"Wins / Losses:  {r.wins} / {r.losses}  (timeout: {r.timeouts})")
    logger.info(f"Win rate:       {r.win_rate}%")
    logger.info(f"PnL total:      {r.total_pnl_eur:+.2f}€  ({r.total_pnl_pips:+.1f} pips)")
    logger.info(f"Profit factor:  {pf_str}")
    logger.info(f"Expectancy:     {r.expectancy:+.2f}€ / trade")
    logger.info(f"Avg win:        {r.avg_win_eur:+.2f}€   |  Avg loss: {r.avg_loss_eur:+.2f}€")
    logger.info(f"Max drawdown:   {r.max_drawdown:.2f}€  ({r.max_drawdown_pct:.1f}%)")
    logger.info("─" * 60)
    logger.info("Por par:")
    for pair, d in r.by_pair.items():
        logger.info(
            f"  {pair:<8} {d['trades']:>4} trades  "
            f"WR {d['win_rate']:>5.1f}%  "
            f"PnL {d['pnl_eur']:>+8.2f}€"
        )
    logger.info("Por timeframe:")
    for tf, d in r.by_timeframe.items():
        logger.info(
            f"  {tf:<5} {d['trades']:>4} trades  "
            f"WR {d['win_rate']:>5.1f}%  "
            f"PnL {d['pnl_eur']:>+8.2f}€"
        )
    logger.info("Por mes:")
    for month, d in sorted(r.by_month.items()):
        wr = round(d["wins"] / d["trades"] * 100, 1) if d["trades"] else 0
        logger.info(
            f"  {month}  {d['trades']:>4} trades  "
            f"WR {wr:>5.1f}%  "
            f"PnL {d['pnl_eur']:>+8.2f}€"
        )
    logger.info("═" * 60)


def _save_report(report: BacktestReport, path: Path) -> None:
    """Guarda el informe completo en JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Serializar — convertir inf a string para JSON válido
    data = asdict(report)
    if data.get("profit_factor") == float("inf"):
        data["profit_factor"] = "inf"

    path.write_text(json.dumps(data, indent=2, default=str))
    logger.success(f"Informe guardado: {path}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="ML-Ayram Backtesting")
    parser.add_argument("--pairs",          nargs="+",  default=None)
    parser.add_argument("--tf",             nargs="+",  default=None,  dest="timeframes")
    parser.add_argument("--days",           type=int,   default=90)
    parser.add_argument("--min-confidence", type=float, default=0.54,  dest="min_confidence")
    parser.add_argument("--output",         type=str,   default=None)
    args = parser.parse_args()

    output = Path(args.output) if args.output else Path(
        f"results/backtest_{datetime.now().strftime('%Y%m%d_%H%M')}.json"
    )

    run_backtest(
        pairs           = args.pairs,
        timeframes      = args.timeframes,
        days            = args.days,
        min_confidence  = args.min_confidence,
        output_path     = output,
    )


if __name__ == "__main__":
    main()
