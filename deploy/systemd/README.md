# Deploy — Servicios systemd

Dos servicios gestionados por systemd:

| Servicio | Qué hace | Puerto |
|----------|----------|--------|
| `ayram-dashboard` | FastAPI + Uvicorn | 8000 |
| `ayram-signals`   | Generador de señales (bucle cada 60s) | — |

---

## Instalación (una sola vez)

```bash
# 1. Subir los unit files al servidor
scp deploy/systemd/*.service ayram@206.81.31.156:/tmp/
scp deploy/systemd/install.sh ayram@206.81.31.156:/tmp/

# 2. Conectarse y ejecutar el instalador
ssh ayram@206.81.31.156
sudo bash /tmp/install.sh
```

El script instala los services, los habilita en el boot y mata los procesos
`nohup` que hubiera activos.

---

## Comandos del día a día

```bash
# Ver logs en vivo
journalctl -u ayram-dashboard -f
journalctl -u ayram-signals -f

# Estado rápido de ambos
systemctl status ayram-dashboard ayram-signals

# Reiniciar (ej. tras hacer git pull)
systemctl restart ayram-dashboard
systemctl restart ayram-signals

# Parar/arrancar manualmente
systemctl stop  ayram-signals
systemctl start ayram-signals

# Desactivar arranque automático (sin borrar el servicio)
systemctl disable ayram-signals
```

---

## Actualizar el código en producción

```bash
cd /home/ayram/ml-ayram
git pull
systemctl restart ayram-dashboard; systemctl restart ayram-signals
```

---

## Añadir el generador de señales más adelante

El servicio `ayram-signals` se instala desactivado hasta que haya modelos entrenados.
Una vez entrenados, activarlo con:

```bash
systemctl enable --now ayram-signals
```

---

## Estructura de archivos

```
deploy/systemd/
├── ayram-dashboard.service   # Unit file del dashboard
├── ayram-signals.service     # Unit file del generador de señales
├── install.sh                # Script de instalación automática
└── README.md                 # Este archivo
```
