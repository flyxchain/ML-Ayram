#!/bin/bash
# deploy/systemd/install.sh
# Instala y activa los servicios systemd de ML-Ayram.
# Ejecutar como root en el servidor: sudo bash deploy/systemd/install.sh
set -euo pipefail

PROJECT_DIR="/home/ayram/ml-ayram"
SYSTEMD_DIR="/etc/systemd/system"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=== ML-Ayram: instalando servicios systemd ==="

# 1. Verificar que el proyecto existe
if [ ! -d "$PROJECT_DIR" ]; then
    echo "❌ No se encontró $PROJECT_DIR"
    exit 1
fi

# 2. Verificar que el venv existe
if [ ! -f "$PROJECT_DIR/venv/bin/uvicorn" ]; then
    echo "❌ No se encontró el venv en $PROJECT_DIR/venv"
    echo "   Ejecuta: cd $PROJECT_DIR && python3 -m venv venv && venv/bin/pip install -r requirements.txt"
    exit 1
fi

# 3. Verificar que el .env existe
if [ ! -f "$PROJECT_DIR/.env" ]; then
    echo "❌ No se encontró $PROJECT_DIR/.env"
    exit 1
fi

# 4. Copiar unit files (servicios + timers)
echo "→ Copiando unit files..."
for f in ayram-dashboard.service ayram-signals.service \
         ayram-collector.service ayram-collector.timer \
         ayram-features.service  ayram-features.timer \
         ayram-train.service     ayram-train.timer \
         ayram-positions.service  ayram-positions.timer \
         ayram-walkforward.service ayram-walkforward.timer \
         ayram-anomaly.service    ayram-anomaly.timer; do
    cp "$SCRIPT_DIR/$f" "$SYSTEMD_DIR/"
    chmod 644 "$SYSTEMD_DIR/$f"
done

# 5. Recargar systemd
echo "→ Recargando systemd..."
systemctl daemon-reload

# 6. Activar para arranque automático
echo "→ Activando servicios y timers en el arranque..."
systemctl enable ayram-dashboard.service
systemctl enable ayram-signals.service
systemctl enable ayram-collector.timer
systemctl enable ayram-features.timer
systemctl enable ayram-train.timer
systemctl enable ayram-positions.timer
systemctl enable ayram-walkforward.timer
systemctl enable ayram-anomaly.timer

# 7. Detener los procesos nohup actuales si existen
echo "→ Deteniendo procesos nohup existentes (si los hay)..."
pkill -f "uvicorn src.dashboard.app" 2>/dev/null && echo "   Dashboard nohup detenido" || echo "   (ninguno activo)"
pkill -f "src.signals.generator"    2>/dev/null && echo "   Signals nohup detenido"  || echo "   (ninguno activo)"

# 8. Iniciar servicios y timers
echo "→ Iniciando servicios y timers..."
systemctl start ayram-dashboard.service
sleep 3
systemctl start ayram-signals.service
systemctl start ayram-collector.timer
systemctl start ayram-features.timer
systemctl start ayram-train.timer
systemctl start ayram-positions.timer
systemctl start ayram-walkforward.timer
systemctl start ayram-anomaly.timer

# 9. Estado final
echo ""
echo "=== Estado ==="
for unit in ayram-dashboard.service ayram-signals.service \
            ayram-collector.timer ayram-features.timer \
            ayram-train.timer ayram-positions.timer \
            ayram-walkforward.timer ayram-anomaly.timer; do
    printf "%-35s " "$unit:"
    systemctl is-active "$unit" 2>/dev/null || echo "inactive"
done

echo ""
echo "✅ Instalación completada."
echo ""
echo "Comandos útiles:"
echo "  journalctl -u ayram-dashboard -f     # logs dashboard en vivo"
echo "  journalctl -u ayram-signals -f       # logs señales en vivo"
echo "  journalctl -u ayram-collector -f     # logs último collector"
echo "  journalctl -u ayram-features -f      # logs último features"
echo "  systemctl list-timers                # ver próximas ejecuciones"
echo "  systemctl restart ayram-dashboard    # reiniciar dashboard"
echo "  systemctl stop ayram-signals         # parar señales"
echo "  journalctl -u ayram-train -f          # logs último entrenamiento"
echo "  journalctl -u ayram-walkforward -f    # logs walk-forward"
echo "  systemctl start ayram-train.service   # forzar entrenamiento ahora"
echo "  systemctl list-timers ayram-*         # ver próximas ejecuciones"
