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

# 4. Copiar unit files
echo "→ Copiando unit files..."
cp "$SCRIPT_DIR/ayram-dashboard.service" "$SYSTEMD_DIR/"
cp "$SCRIPT_DIR/ayram-signals.service"   "$SYSTEMD_DIR/"
chmod 644 "$SYSTEMD_DIR/ayram-dashboard.service"
chmod 644 "$SYSTEMD_DIR/ayram-signals.service"

# 5. Recargar systemd
echo "→ Recargando systemd..."
systemctl daemon-reload

# 6. Activar para arranque automático
echo "→ Activando servicios en el arranque..."
systemctl enable ayram-dashboard.service
systemctl enable ayram-signals.service

# 7. Detener los procesos nohup actuales si existen
echo "→ Deteniendo procesos nohup existentes (si los hay)..."
pkill -f "uvicorn src.dashboard.app" 2>/dev/null && echo "   Dashboard nohup detenido" || echo "   (ninguno activo)"
pkill -f "src.signals.generator"    2>/dev/null && echo "   Signals nohup detenido"  || echo "   (ninguno activo)"

# 8. Iniciar servicios
echo "→ Iniciando servicios..."
systemctl start ayram-dashboard.service
sleep 3
systemctl start ayram-signals.service

# 9. Estado final
echo ""
echo "=== Estado ==="
systemctl status ayram-dashboard.service --no-pager -l
echo ""
systemctl status ayram-signals.service --no-pager -l

echo ""
echo "✅ Instalación completada."
echo ""
echo "Comandos útiles:"
echo "  journalctl -u ayram-dashboard -f     # logs dashboard en vivo"
echo "  journalctl -u ayram-signals -f       # logs señales en vivo"
echo "  systemctl restart ayram-dashboard    # reiniciar dashboard"
echo "  systemctl stop ayram-signals         # parar señales"
echo "  systemctl disable ayram-signals      # no arrancar en el boot"
