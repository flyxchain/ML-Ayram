# Deploy — Servicios systemd

## Arquitectura de servicios

### Daemons (siempre activos)

| Servicio | Qué hace | Puerto |
|----------|----------|--------|
| `ayram-dashboard` | FastAPI + Uvicorn | 8000 |
| `ayram-signals`   | Generador de señales (bucle cada 900s) | — |

### Timers (ejecución programada)

| Timer | Servicio | Cuándo | Qué hace |
|-------|----------|--------|----------|
| `ayram-collector` | oneshot | Cada 15 min | Descarga velas OHLCV |
| `ayram-features`  | oneshot | Cada 3h (:10) | Calcula features + labels |
| `ayram-train`     | oneshot | Domingos 02:00 UTC | Reentrenamiento XGB+LSTM+Optuna |
| `ayram-positions` | oneshot | Cada 5 min | Abre/cierra posiciones (TP/SL) |
| `ayram-walkforward` | oneshot | 1er domingo del mes 04:00 | Walk-forward validation |
| `ayram-anomaly`   | oneshot | Cada 6h | Detección de anomalías |

### Flujo de datos

```
collector (15m) → features+labels (3h) → train (semanal)
                                              ↓
signals (daemon 900s) → positions (5m) → trades_history
                                              ↓
                              anomaly (6h) + walkforward (mensual)
```

---

## Instalación (una sola vez)

```bash
# Desde el servidor
cd /home/ayram/ml-ayram
sudo bash deploy/systemd/install.sh
```

El script instala todos los services/timers, los habilita en el boot y mata los procesos
`nohup` que hubiera activos.

---

## Comandos del día a día

```bash
# Ver logs en vivo
journalctl -u ayram-dashboard -f
journalctl -u ayram-signals -f
journalctl -u ayram-collector -f
journalctl -u ayram-features -f
journalctl -u ayram-positions -f

# Estado rápido de todo
systemctl status ayram-dashboard ayram-signals
systemctl list-timers ayram-*

# Reiniciar (ej. tras hacer git pull)
systemctl restart ayram-dashboard
systemctl restart ayram-signals

# Forzar ejecución manual de un timer
systemctl start ayram-collector.service
systemctl start ayram-features.service
systemctl start ayram-train.service
systemctl start ayram-positions.service

# Parar/arrancar manualmente
systemctl stop  ayram-signals
systemctl start ayram-signals
```

---

## Actualizar código en producción

```bash
# Opción 1: deploy completo desde local
./deploy/deploy.sh

# Opción 2: manual en el servidor
cd /home/ayram/ml-ayram
git pull
sudo systemctl daemon-reload
systemctl restart ayram-dashboard ayram-signals
```

---

## Estructura de archivos

```
deploy/systemd/
├── ayram-dashboard.service     # Daemon: dashboard FastAPI
├── ayram-signals.service       # Daemon: generador de señales
├── ayram-collector.service     # Oneshot: descarga OHLCV
├── ayram-collector.timer       # Timer: cada 15 min
├── ayram-features.service      # Oneshot: features + labels
├── ayram-features.timer        # Timer: cada 3h
├── ayram-train.service         # Oneshot: entrenamiento semanal
├── ayram-train.timer           # Timer: domingos 02:00
├── ayram-positions.service     # Oneshot: gestión de posiciones
├── ayram-positions.timer       # Timer: cada 5 min
├── ayram-walkforward.service   # Oneshot: walk-forward validation
├── ayram-walkforward.timer     # Timer: 1er domingo del mes
├── ayram-anomaly.service       # Oneshot: detección anomalías
├── ayram-anomaly.timer         # Timer: cada 6h
├── install.sh                  # Script de instalación automática
└── README.md                   # Este archivo
```
