#!/bin/bash
# train_watchdog.sh
# Lanza el training y lo relanza automáticamente con --resume si falla.
# Uso: bash scripts/train_watchdog.sh

cd /home/ayram/ml-ayram
source venv/bin/activate

MAX_RETRIES=5
RETRY=0
SLEEP_BETWEEN=30  # segundos antes de reintentar

echo "$(date) — Watchdog iniciado"

while [ $RETRY -lt $MAX_RETRIES ]; do
    LOG="logs/train_$(date +%Y%m%d_%H%M).log"

    if [ $RETRY -eq 0 ]; then
        echo "$(date) — Intento $((RETRY+1)): lanzando training limpio"
        python -m src.train 2>&1 | tee "$LOG"
    else
        echo "$(date) — Intento $((RETRY+1)): relanzando con --resume"
        python -m src.train --resume 2>&1 | tee "$LOG"
    fi

    EXIT_CODE=$?

    if [ $EXIT_CODE -eq 0 ]; then
        echo "$(date) — ✅ Training completado con éxito"
        exit 0
    else
        RETRY=$((RETRY+1))
        if [ $RETRY -lt $MAX_RETRIES ]; then
            echo "$(date) — ⚠️  Training terminó con error (código $EXIT_CODE). Reintento $RETRY/$MAX_RETRIES en ${SLEEP_BETWEEN}s..."
            sleep $SLEEP_BETWEEN
        fi
    fi
done

echo "$(date) — ❌ Training falló después de $MAX_RETRIES intentos"
exit 1
