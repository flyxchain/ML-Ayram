#!/bin/bash
# deploy/deploy.sh
# Sincroniza el código local con el servidor y reinicia los servicios.
#
# Uso:
#   ./deploy/deploy.sh              # deploy completo
#   ./deploy/deploy.sh --no-restart # solo sincronizar código, sin reiniciar
#   ./deploy/deploy.sh --services   # solo reiniciar servicios (sin rsync)
#   ./deploy/deploy.sh --install    # primera instalación (instala systemd units)
#
# Requisitos:
#   - Clave SSH configurada para ayram@206.81.31.156
#   - rsync instalado en local
#   - .env en el servidor ya configurado

set -euo pipefail

# ── Configuración ─────────────────────────────────────────────────────────────

REMOTE_USER="ayram"
REMOTE_HOST="206.81.31.156"
REMOTE_DIR="/home/ayram/ml-ayram"
VENV="$REMOTE_DIR/venv/bin"

SSH="ssh -o StrictHostKeyChecking=no $REMOTE_USER@$REMOTE_HOST"

# Servicios en orden de dependencia
SERVICES_ONESHOT=("ayram-collector" "ayram-features")   # oneshot — gestionados por timers
SERVICES_DAEMON=("ayram-dashboard" "ayram-signals")     # daemons continuos
TIMERS=("ayram-collector" "ayram-features")

# Archivos/dirs a excluir del rsync
RSYNC_EXCLUDES=(
    ".git"
    ".gitignore"
    "venv/"
    "__pycache__/"
    "*.pyc"
    "*.pyo"
    ".env"           # NUNCA sobreescribir el .env del servidor
    "models/saved/"  # modelos entrenados viven en el servidor
    "results/"
    "logs/"
    "*.log"
    ".DS_Store"
    "node_modules/"
)

# ── Flags ─────────────────────────────────────────────────────────────────────

DO_SYNC=true
DO_RESTART=true
DO_INSTALL=false

for arg in "$@"; do
    case $arg in
        --no-restart) DO_RESTART=false ;;
        --services)   DO_SYNC=false    ;;
        --install)    DO_INSTALL=true  ;;
    esac
done

# ── Helpers ───────────────────────────────────────────────────────────────────

log()  { echo -e "\033[1;34m[deploy]\033[0m $*"; }
ok()   { echo -e "\033[1;32m[  ok  ]\033[0m $*"; }
warn() { echo -e "\033[1;33m[ warn ]\033[0m $*"; }
fail() { echo -e "\033[1;31m[ fail ]\033[0m $*"; exit 1; }

check_ssh() {
    log "Verificando conexión SSH a $REMOTE_USER@$REMOTE_HOST..."
    $SSH "echo ok" > /dev/null 2>&1 || fail "No se puede conectar por SSH. Verifica tu clave y VPN."
    ok "SSH OK"
}

# ── 1. Sincronizar código ─────────────────────────────────────────────────────

sync_code() {
    log "Sincronizando código → $REMOTE_HOST:$REMOTE_DIR"

    EXCLUDE_FLAGS=""
    for ex in "${RSYNC_EXCLUDES[@]}"; do
        EXCLUDE_FLAGS="$EXCLUDE_FLAGS --exclude=$ex"
    done

    rsync -avz --delete \
        $EXCLUDE_FLAGS \
        --checksum \
        -e "ssh -o StrictHostKeyChecking=no" \
        ./ "$REMOTE_USER@$REMOTE_HOST:$REMOTE_DIR/"

    ok "Código sincronizado"
}

# ── 2. Instalar dependencias ──────────────────────────────────────────────────

install_deps() {
    log "Instalando dependencias Python..."
    $SSH "cd $REMOTE_DIR && $VENV/pip install -q --upgrade pip && $VENV/pip install -q -r requirements.server.txt"
    ok "Dependencias instaladas"
}

# ── 3. Instalar/actualizar units systemd ─────────────────────────────────────

install_systemd() {
    log "Instalando units systemd..."

    # Copiar todos los .service y .timer al directorio de systemd
    $SSH "sudo cp $REMOTE_DIR/deploy/systemd/*.service /etc/systemd/system/ && \
          sudo cp $REMOTE_DIR/deploy/systemd/*.timer   /etc/systemd/system/ && \
          sudo systemctl daemon-reload"

    # Habilitar timers
    for timer in "${TIMERS[@]}"; do
        $SSH "sudo systemctl enable --now ${timer}.timer" && ok "Timer ${timer} habilitado" || warn "Timer ${timer}: ya estaba habilitado"
    done

    # Habilitar daemons
    for svc in "${SERVICES_DAEMON[@]}"; do
        $SSH "sudo systemctl enable ${svc}.service" && ok "Servicio ${svc} habilitado" || warn "Servicio ${svc}: ya estaba habilitado"
    done

    ok "Units systemd instaladas"
}

# ── 4. Reiniciar servicios ────────────────────────────────────────────────────

restart_services() {
    log "Recargando systemd y reiniciando servicios..."

    $SSH "sudo systemctl daemon-reload"

    for svc in "${SERVICES_DAEMON[@]}"; do
        log "  Reiniciando $svc..."
        $SSH "sudo systemctl restart ${svc}.service" && ok "  $svc reiniciado" || warn "  $svc: error al reiniciar (ver: journalctl -u $svc -n 20)"
    done

    # Los oneshot no se reinician manualmente — los manejan los timers
    # Pero sí recargamos los timers por si hubo cambios en OnCalendar
    for timer in "${TIMERS[@]}"; do
        $SSH "sudo systemctl restart ${timer}.timer" && ok "  Timer ${timer} recargado" || warn "  Timer ${timer}: error"
    done
}

# ── 5. Health check ───────────────────────────────────────────────────────────

health_check() {
    log "Verificando estado de servicios..."
    sleep 3   # dar tiempo a que arranquen

    all_ok=true
    for svc in "${SERVICES_DAEMON[@]}"; do
        status=$($SSH "systemctl is-active ${svc}.service" 2>/dev/null || echo "unknown")
        if [ "$status" = "active" ]; then
            ok "  $svc: ACTIVO"
        else
            warn "  $svc: $status"
            all_ok=false
        fi
    done

    for timer in "${TIMERS[@]}"; do
        status=$($SSH "systemctl is-active ${timer}.timer" 2>/dev/null || echo "unknown")
        if [ "$status" = "active" ]; then
            ok "  ${timer}.timer: ACTIVO"
        else
            warn "  ${timer}.timer: $status"
            all_ok=false
        fi
    done

    if $all_ok; then
        ok "Dashboard: http://$REMOTE_HOST:8000"
    else
        warn "Algunos servicios tienen problemas. Revisar con:"
        warn "  ssh $REMOTE_USER@$REMOTE_HOST 'journalctl -u ayram-signals -n 30'"
    fi
}

# ── Main ──────────────────────────────────────────────────────────────────────

echo ""
echo "╔══════════════════════════════════════╗"
echo "║     ML-Ayram Deploy Script           ║"
echo "╚══════════════════════════════════════╝"
echo ""

check_ssh

if $DO_SYNC; then
    sync_code
    install_deps
fi

if $DO_INSTALL; then
    install_systemd
fi

if $DO_RESTART; then
    restart_services
    health_check
fi

echo ""
ok "Deploy completado ✓"
echo ""
