"""
src/utils/pipeline_logger.py
Registro de ejecuciones del pipeline para tracking visual en dashboard.

Uso:
    from src.utils.pipeline_logger import pipeline_run, is_forex_market_open, log_skip

    # Como context manager
    with pipeline_run("collector") as run:
        rows = do_collection()
        run["rows_processed"] = rows

    # Saltar si mercado cerrado
    if not is_forex_market_open():
        log_skip("collector", "Mercado cerrado (fin de semana)")
        sys.exit(0)
"""

import json
import os
import time
from contextlib import contextmanager
from datetime import datetime, timezone

from dotenv import load_dotenv
from loguru import logger
from sqlalchemy import create_engine, text

load_dotenv()

_engine = None


def _get_engine():
    global _engine
    if _engine is None:
        _engine = create_engine(os.getenv("DATABASE_URL"), pool_pre_ping=True)
    return _engine


def ensure_table():
    """Crea la tabla pipeline_runs si no existe."""
    ddl = """
    CREATE TABLE IF NOT EXISTS pipeline_runs (
        id              SERIAL PRIMARY KEY,
        task            VARCHAR(50)  NOT NULL,
        started_at      TIMESTAMPTZ  NOT NULL DEFAULT NOW(),
        finished_at     TIMESTAMPTZ,
        status          VARCHAR(20)  DEFAULT 'running',
        error_message   TEXT,
        rows_processed  INT,
        duration_seconds FLOAT,
        details         JSONB
    )
    """
    idx = """
    CREATE INDEX IF NOT EXISTS idx_pipeline_task_started
        ON pipeline_runs(task, started_at DESC)
    """
    with _get_engine().begin() as conn:
        conn.execute(text(ddl))
        conn.execute(text(idx))


@contextmanager
def pipeline_run(task: str):
    """
    Context manager para registrar ejecución de un paso del pipeline.

    Ejemplo:
        with pipeline_run("collector") as run:
            n = download_data()
            run["rows_processed"] = n
            run["details"] = {"pairs": 5}
    """
    ensure_table()
    eng = _get_engine()
    start = time.time()
    run_id = None
    now = datetime.now(timezone.utc)

    with eng.begin() as conn:
        result = conn.execute(
            text(
                "INSERT INTO pipeline_runs (task, started_at, status) "
                "VALUES (:task, :started_at, 'running') RETURNING id"
            ),
            {"task": task, "started_at": now},
        )
        run_id = result.scalar()

    info = {"rows_processed": None, "details": None}

    try:
        yield info

        duration = time.time() - start
        details_json = json.dumps(info["details"]) if info.get("details") else None
        with eng.begin() as conn:
            conn.execute(
                text(
                    "UPDATE pipeline_runs "
                    "SET finished_at = :finished, status = 'success', "
                    "    duration_seconds = :dur, rows_processed = :rows, "
                    "    details = :details "
                    "WHERE id = :id"
                ),
                {
                    "finished": datetime.now(timezone.utc),
                    "dur": round(duration, 2),
                    "rows": info.get("rows_processed"),
                    "details": details_json,
                    "id": run_id,
                },
            )
        logger.info(f"[pipeline] {task} completado en {duration:.1f}s")

    except Exception as e:
        duration = time.time() - start
        with eng.begin() as conn:
            conn.execute(
                text(
                    "UPDATE pipeline_runs "
                    "SET finished_at = :finished, status = 'error', "
                    "    error_message = :err, duration_seconds = :dur "
                    "WHERE id = :id"
                ),
                {
                    "finished": datetime.now(timezone.utc),
                    "err": str(e)[:2000],
                    "dur": round(duration, 2),
                    "id": run_id,
                },
            )
        logger.error(f"[pipeline] {task} falló en {duration:.1f}s: {e}")
        raise


def log_skip(task: str, reason: str):
    """Registra una ejecución saltada (ej: mercado cerrado)."""
    ensure_table()
    now = datetime.now(timezone.utc)
    with _get_engine().begin() as conn:
        conn.execute(
            text(
                "INSERT INTO pipeline_runs "
                "(task, started_at, finished_at, status, error_message, duration_seconds) "
                "VALUES (:task, :ts, :ts, 'skipped', :reason, 0)"
            ),
            {"task": task, "ts": now, "reason": reason},
        )
    logger.info(f"[pipeline] {task} saltado: {reason}")


def is_forex_market_open() -> bool:
    """
    Comprueba si el mercado forex está abierto.

    Horario forex:
      Abre:   Domingo  22:00 UTC
      Cierra: Viernes  22:00 UTC

    Cerrado: Viernes 22:00 → Domingo 22:00
    """
    now = datetime.now(timezone.utc)
    wd = now.weekday()   # 0=Lun … 4=Vie, 5=Sáb, 6=Dom
    h = now.hour

    if wd == 5:                    # Sábado completo
        return False
    if wd == 4 and h >= 22:        # Viernes ≥ 22:00
        return False
    if wd == 6 and h < 22:         # Domingo < 22:00
        return False
    return True
