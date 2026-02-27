# ML-Ayram — Tutorial Completo de Implementación
# Bot de Trading Algorítmico con ML para Forex

> **Proyecto:** ML-Ayram  
> **Objetivo:** Sistema de trading con ML que analiza pares de divisas, genera señales por Telegram y ejecuta trades automáticamente en cTrader  
> **Última actualización del tutorial:** Feb 2026  
> **Estado actual:** FASE 0 — Configuración de infraestructura

---

## ÍNDICE

1. [Resumen de la Arquitectura](#1-resumen-de-la-arquitectura)
2. [Fase 0 — Preparar tu máquina local](#2-fase-0--preparar-tu-máquina-local)
3. [Fase 1 — Crear el servidor (DigitalOcean O Google Cloud)](#3-fase-1--crear-el-servidor)
4. [Fase 2 — Configurar la Base de Datos](#4-fase-2--configurar-la-base-de-datos)
5. [Fase 3 — Conectar cTrader Open API](#5-fase-3--conectar-ctrader-open-api)
6. [Fase 4 — Feature Engineering y Dataset](#6-fase-4--feature-engineering-y-dataset)
7. [Fase 5 — Entrenar el Modelo ML](#7-fase-5--entrenar-el-modelo-ml)
8. [Fase 6 — Motor de Señales](#8-fase-6--motor-de-señales)
9. [Fase 7 — Bot de Telegram](#9-fase-7--bot-de-telegram)
10. [Fase 8 — Paper Trading (validación real)](#10-fase-8--paper-trading)
11. [Fase 9 — Auto-Trader (ejecución automática)](#11-fase-9--auto-trader)
12. [Fase 10 — Dashboard en Netlify](#12-fase-10--dashboard-en-netlify)
13. [Mantenimiento y Monitorización](#13-mantenimiento)
14. [Checklist de Estado del Proyecto](#14-checklist)

---

## 1. Resumen de la Arquitectura

```
cTrader API ──► Data Collector ──► PostgreSQL (TimescaleDB)
                                        │
                                   Feature Engine
                                        │
                              XGBoost (régimen) + LSTM (dirección)
                                        │
                                  Signal Engine (confluencia multi-TF)
                                        │
                            ┌───────────┴────────────┐
                        Telegram Bot            Auto-Trader
                      (notificaciones)       (cTrader orders)
```

**Stack tecnológico:**
- Lenguaje: Python 3.11
- Servidor: DigitalOcean Droplet $12/mes (2vCPU/2GB) → Opción GCP e2-medium
- Base de datos: PostgreSQL 15 + TimescaleDB (Supabase free o DO Managed)
- ML: XGBoost + TensorFlow/Keras (LSTM)
- Notificaciones: python-telegram-bot
- Ejecución: cTrader Open API (Protobuf/TCP)
- Dashboard: Netlify (HTML estático)
- Gestión de secretos: archivo .env local en el servidor
- Scheduler: APScheduler (dentro del proceso Python)

**Pares objetivo:** EURUSD, GBPUSD, USDJPY, EURJPY, XAUUSD  
**Timeframes:** M15, H1, H4, D1  
**Señales por día esperadas:** 2-5 (con filtros estrictos de calidad)

---

## 2. Fase 0 — Preparar tu Máquina Local

### 2.1 Instalar Git

Descarga Git desde https://git-scm.com/download/win  
Durante la instalación, elige "Git Bash" y deja las opciones por defecto.

Verificar:
```bash
git --version
# Debe mostrar: git version 2.x.x
```

### 2.2 Python en local

Ya instalado: **Python 3.14.3** ✔️

Verificar en terminal:
```bash
python --version
# Python 3.14.3
```

> **Nota importante sobre Python 3.14.3:** Es la última versión estable pero muy reciente.
> TensorFlow no tiene wheel para 3.14, por lo que el proyecto usa **PyTorch** para la LSTM.
> El servidor de producción (Linux) usará Python 3.11 donde todas las librerías ML tienen soporte completo.
> El código está escrito para ser compatible con ambas versiones.

### 2.3 Instalar Visual Studio Code

Descarga desde https://code.visualstudio.com/  
Extensiones recomendadas (instalar desde VSCode):
- Python (Microsoft)
- Pylance
- GitLens
- SQLTools + PostgreSQL driver
- Remote SSH (para editar archivos directamente en el servidor)

### 2.4 Instalar un cliente SSH

- Windows 10/11: OpenSSH ya está incluido. Abre "Terminal" o "PowerShell" y usa `ssh`.
- Alternativa gráfica: https://www.putty.org/ (más visual)
- Recomendado: **MobaXterm** (https://mobaxterm.mobatek.net/) — combina SSH, editor y transferencia de archivos.

### 2.5 Inicializar el proyecto local

El proyecto ya está creado en:
```
C:\Users\Usuario\Documents\Webs\ML-Ayram\
```

Abre una terminal en esa carpeta y ejecuta:
```bash
git init
git remote add origin https://github.com/TU_USUARIO/ml-ayram.git
```

(Crea primero el repositorio en GitHub — privado)

### 2.6 Crear entorno virtual local

```bash
cd C:\Users\Usuario\Documents\Webs\ML-Ayram
python -m venv venv
venv\Scripts\activate       # Windows
# o: source venv/bin/activate   # si usas Git Bash
pip install -r requirements.txt
```

---

## 3. Fase 1 — Crear el Servidor

### OPCIÓN A: DigitalOcean (más fácil, recomendada para empezar)

#### Paso 1: Crear cuenta
Ve a https://www.digitalocean.com  
Usa código de referido para obtener $200 de crédito: busca en Google "DigitalOcean $200 credit 2025"  
Regístrate con tarjeta de crédito (no te cobran hasta acabar los créditos).

#### Paso 2: Crear el Droplet

1. Click en "Create" → "Droplets"
2. Configuración:
   - **Imagen:** Ubuntu 24.04 LTS (64-bit)
   - **Plan:** Basic → Regular → $12/mo (2 vCPU, 2GB RAM, 60GB SSD)
   - **Región:** Frankfurt (FRA1) o Amsterdam (AMS3) — más cercano, menor latencia a brokers europeos
   - **Authentication:** SSH Keys → "New SSH Key"
3. Crear SSH Key en tu Windows:
   ```bash
   # Abre PowerShell y ejecuta:
   ssh-keygen -t ed25519 -C "ml-ayram"
   # Guarda en C:\Users\Usuario\.ssh\id_ed25519
   # Pulsa Enter para no poner contraseña (más cómodo para scripts)
   
   # Ver la clave pública para copiarla en DigitalOcean:
   cat C:\Users\Usuario\.ssh\id_ed25519.pub
   ```
4. Copia el contenido de la clave pública en el campo de DigitalOcean
5. **Nombre del Droplet:** ml-ayram-prod
6. Click "Create Droplet" — tardará ~1 minuto

#### Paso 3: Conectar al servidor

```bash
# En tu terminal local (sustituye IP por la IP del Droplet):
ssh root@TU_IP_DROPLET

# Primera vez pedirá confirmar, escribe: yes
```

#### Paso 4: Configuración inicial del servidor

```bash
# Actualizar sistema
apt update && apt upgrade -y

# Instalar herramientas básicas
apt install -y python3.11 python3.11-venv python3-pip git curl wget unzip htop nano ufw

# Configurar firewall
ufw allow OpenSSH
ufw allow 8443/tcp   # Puerto para dashboard (opcional)
ufw enable

# Crear usuario no-root para el bot (más seguro)
adduser ayram
usermod -aG sudo ayram

# Copiar SSH key al nuevo usuario
mkdir -p /home/ayram/.ssh
cp /root/.ssh/authorized_keys /home/ayram/.ssh/
chown -R ayram:ayram /home/ayram/.ssh
chmod 700 /home/ayram/.ssh

# Desde ahora siempre conectar con:
# ssh ayram@TU_IP_DROPLET
```

#### Paso 5: Clonar el proyecto en el servidor

```bash
# Conectar como usuario ayram
ssh ayram@TU_IP_DROPLET

# Clonar el repositorio
git clone https://github.com/TU_USUARIO/ml-ayram.git ~/ml-ayram
cd ~/ml-ayram

# Crear entorno virtual Python
python3.11 -m venv venv
source venv/bin/activate

# Instalar dependencias
pip install -r requirements.txt
```

---

### OPCIÓN B: Google Cloud Platform (más potente, con $300 de crédito)

#### Paso 1: Crear cuenta GCP
Ve a https://cloud.google.com  
Activa el período de prueba gratuito ($300 por 90 días).  
Crea un proyecto nuevo llamado "ml-ayram".

#### Paso 2: Habilitar APIs necesarias

Ve a "APIs & Services" → "Enable APIs":
- Compute Engine API
- Cloud SQL Admin API
- Secret Manager API
- Cloud Storage API
- Cloud Logging API

#### Paso 3: Crear la VM (Compute Engine)

1. Ve a "Compute Engine" → "VM instances" → "Create Instance"
2. Configuración:
   - **Nombre:** ml-ayram-prod
   - **Región:** europe-west1 (Bélgica) o europe-west4 (Países Bajos)
   - **Tipo de máquina:** e2-medium (2 vCPU, 4GB RAM) — ~$27/mes
   - **SO:** Ubuntu 22.04 LTS
   - **Disco:** 30 GB SSD (suficiente)
   - **Acceso a red:** Deja los defaults
3. En "Firewall": marca "Allow HTTP" y "Allow HTTPS" (para dashboard)
4. Click "Create"

#### Paso 4: Conectar vía SSH a GCP

Opción fácil: directamente desde la consola web, click en "SSH" en la lista de VMs.

Opción avanzada (desde tu terminal local):
```bash
# Instalar Google Cloud CLI: https://cloud.google.com/sdk/docs/install
gcloud auth login
gcloud config set project ml-ayram
gcloud compute ssh ml-ayram-prod --zone=europe-west1-b
```

#### Paso 5: Configurar la VM (mismo proceso que DigitalOcean)

```bash
sudo apt update && sudo apt upgrade -y
sudo apt install -y python3.11 python3.11-venv python3-pip git curl wget unzip htop
```

#### Paso 6: Configurar Secret Manager para las credenciales

```bash
# Instalar cliente de GCP en el servidor
curl -fsSL https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key add -
# Sigue la guía oficial para instalar gcloud CLI en Ubuntu

# Guardar secretos (desde tu local o la consola web):
gcloud secrets create CTRADER_CLIENT_ID --data-file=-
gcloud secrets create CTRADER_CLIENT_SECRET --data-file=-
gcloud secrets create TELEGRAM_BOT_TOKEN --data-file=-
gcloud secrets create DATABASE_URL --data-file=-
```

---

## 4. Fase 2 — Configurar la Base de Datos

### OPCIÓN A: Supabase (gratis para empezar — RECOMENDADO)

#### Paso 1: Crear proyecto en Supabase
1. Ve a https://supabase.com y crea una cuenta
2. "New Project" → nombre: "ml-ayram" → genera una contraseña fuerte (guárdala)
3. Región: Europe (Frankfurt)
4. Espera ~2 minutos a que se cree

#### Paso 2: Obtener la cadena de conexión
1. En el dashboard de Supabase: Settings → Database
2. Copia la "Connection string" de tipo "URI" (modo Transaction Pooling)
   Tendrá este formato: `postgresql://postgres:TU_PASS@db.xxxx.supabase.co:5432/postgres`
3. Guárdala, la usarás en el archivo `.env` del servidor

#### Paso 3: Ejecutar el schema SQL
1. En Supabase: ve a "SQL Editor"
2. Copia y ejecuta el contenido de `config/schema.sql` (archivo en este proyecto)

### OPCIÓN B: DigitalOcean Managed PostgreSQL ($15/mes)

1. En DigitalOcean: "Databases" → "Create Database Cluster"
2. Motor: PostgreSQL 15
3. Plan: Basic ($15/mes, 1GB RAM, 10GB)
4. Misma región que tu Droplet
5. Una vez creado, copia la "Connection String" del panel

### OPCIÓN C: Cloud SQL en GCP (~$10/mes)

```bash
# Desde la consola GCP o con gcloud CLI:
gcloud sql instances create ml-ayram-db \
  --database-version=POSTGRES_15 \
  --tier=db-f1-micro \
  --region=europe-west1 \
  --storage-size=20GB

# Crear base de datos y usuario:
gcloud sql databases create mlayram --instance=ml-ayram-db
gcloud sql users create ayramuser --instance=ml-ayram-db --password=TU_CONTRASEÑA
```

### Instalar TimescaleDB (solo si usas servidor propio)

Si usas Supabase, TimescaleDB ya está incluido. Si usas Cloud SQL o tu propia instancia:
```bash
# En el servidor PostgreSQL:
sudo apt install -y timescaledb-2-postgresql-15
# Editar postgresql.conf: shared_preload_libraries = 'timescaledb'
sudo systemctl restart postgresql

# En psql:
CREATE EXTENSION IF NOT EXISTS timescaledb;
```

---

## 5. Fase 3 — Conectar cTrader Open API

### Paso 1: Obtener credenciales de cTrader

1. Ve a https://connect.ctrader.com
2. "Create Application" → nombre: "ML-Ayram-Bot"
3. Anota: **Client ID** y **Client Secret**
4. En tu cuenta de cTrader (o cTrader demo):
   - Configuración → Open API → Activa el acceso
   - Anota tu **Account ID** (número de cuenta)

### Paso 2: Instalar la librería

```bash
# En el servidor, con el venv activado:
pip install ctrader-open-api
```

### Paso 3: Test de conexión

```bash
# En el servidor:
cd ~/ml-ayram
source venv/bin/activate
python scripts/test_ctrader_connection.py
```

Si ves "Connected successfully" y los datos de tu cuenta, estás listo para la siguiente fase.

### Notas importantes sobre cTrader Open API:
- Usa protocolo Protobuf/TCP (no REST)
- Servidor demo: demo.ctraderapi.com:5035
- Servidor live: live.ctraderapi.com:5035
- **Siempre desarrolla y testea en DEMO primero**
- Rate limits: máximo 100 requests/segundo (más que suficiente)

---

## 6. Fase 4 — Feature Engineering y Dataset

### Paso 1: Descargar datos históricos

```bash
python src/data/collector.py --mode historical --pairs EURUSD,GBPUSD,USDJPY,EURJPY,XAUUSD --timeframes M15,H1,H4,D1 --years 3
```

Esto descargará ~15-20 millones de filas y tardará 30-60 minutos.  
Los datos se guardan en la tabla `ohlcv_raw` de PostgreSQL.

### Paso 2: Calcular features

```bash
python src/data/features.py --compute-all
```

Calcula los ~85 features por vela y los guarda en `features_computed`.  
Duración estimada: 20-40 minutos la primera vez.

### Paso 3: Generar etiquetas (Triple-Barrier Method)

```bash
python src/data/labeler.py --barrier-up 1.5 --barrier-down 1.0 --max-bars 20
```

Parámetros del método Triple-Barrier:
- `barrier-up`: múltiplo de ATR para barrera de beneficio (TP)
- `barrier-down`: múltiplo de ATR para barrera de pérdida (SL)  
- `max-bars`: máximo de velas antes de expirar la señal

### Paso 4: Análisis exploratorio

Abre el Jupyter notebook:
```bash
pip install jupyter
jupyter notebook docs/exploratory_analysis.ipynb
```

Métricas que debes revisar antes de entrenar:
- Distribución de labels: idealmente ~40% BUY / ~40% SELL / ~20% NEUTRAL
- Correlación entre features (eliminar las muy correlacionadas > 0.95)
- Importancia de features con un XGBoost rápido (Random Forest feature importance)

---

## 7. Fase 5 — Entrenar el Modelo ML

### Entrenamiento XGBoost (clasificador de régimen)

```bash
python src/models/train_regime_classifier.py \
  --pairs EURUSD,GBPUSD \
  --timeframe H1 \
  --walk-forward-periods 8 \
  --optuna-trials 100
```

Duración estimada: 2-4 horas en servidor básico.

Métricas mínimas para continuar:
- Accuracy OOS > 58%
- F1-score (clase BUY/SELL) > 0.55
- No más de 15% diferencia entre train y test accuracy (sign of overfitting)

### Entrenamiento LSTM (predictor de dirección)

```bash
python src/models/train_lstm.py \
  --sequence-length 60 \
  --epochs 100 \
  --batch-size 256 \
  --dropout 0.3
```

Duración estimada: 4-8 horas (mucho más en servidor básico sin GPU).  
**Nota:** Para el reentrenamiento mensual, considera usar Google Colab (gratis, con GPU) y luego subir el modelo al servidor.

### Tracking con MLflow

```bash
# Iniciar el servidor MLflow (corre en background en el servidor):
mlflow server --host 0.0.0.0 --port 5000 &

# Ver experimentos:
# Abre en el navegador: http://TU_IP_SERVIDOR:5000
# (recuerda abrir el puerto 5000 en el firewall)
```

### Guardar el modelo final

Los modelos se guardan automáticamente en `~/ml-ayram/models/saved/`  
- `regime_classifier_vX.pkl` — modelo XGBoost
- `lstm_predictor_vX.h5` — modelo LSTM
- `scaler_vX.pkl` — normalizador de features

---

## 8. Fase 6 — Motor de Señales

### Configurar parámetros del Signal Engine

Edita `config/signal_config.yaml`:
```yaml
signal_engine:
  min_confidence: 0.72        # Confianza mínima del modelo (72%)
  min_timeframe_confluence: 2  # Mínimo 2 timeframes alineados
  max_simultaneous_signals: 3  # Máximo 3 señales abiertas a la vez
  cooldown_hours: 4            # Horas de espera entre señales del mismo par
  
risk_manager:
  max_risk_per_trade: 0.015   # 1.5% del capital por operación
  max_daily_drawdown: 0.04    # Stop del día si pérdida > 4%
  kelly_fraction: 0.25        # Kelly Criterion al 25% (conservador)
  
sl_tp:
  sl_atr_multiplier: 1.5      # SL = 1.5 × ATR(14)
  tp1_rr: 1.5                 # TP1 = RR 1:1.5 (cierre 50%)
  tp2_rr: 2.5                 # TP2 = RR 1:2.5 (cierre 50% restante)
  trailing_stop: true          # Activar trailing stop en TP1
```

### Probar el motor en modo dry-run

```bash
python src/signals/engine.py --mode dry-run --verbose
```

Verás en consola las señales que hubiera generado en los últimos 7 días sin ejecutar nada.

---

## 9. Fase 7 — Bot de Telegram

### Paso 1: Crear el bot

1. En Telegram, busca "@BotFather"
2. Escribe `/newbot`
3. Nombre: "ML Ayram Bot"
4. Username: "@ml_ayram_bot" (debe ser único, pon algo personal)
5. Copia el **token** que te da BotFather

### Paso 2: Obtener tu Chat ID

1. Busca "@userinfobot" en Telegram
2. Escríbele cualquier cosa y te responderá con tu **Chat ID** (un número)
3. Este número es tu ID personal para recibir mensajes

### Paso 3: Configurar en .env

```bash
# En el servidor, edita el archivo .env:
nano ~/ml-ayram/.env

# Añade:
TELEGRAM_BOT_TOKEN=tu_token_aqui
TELEGRAM_CHAT_ID=tu_chat_id_aqui
```

### Paso 4: Probar el bot

```bash
python src/notifications/telegram_bot.py --test
```

Deberías recibir un mensaje de prueba en Telegram.

### Comandos disponibles del bot:
- `/start` — Inicia el bot y muestra estado
- `/status` — Estado del sistema y posiciones abiertas
- `/positions` — Lista de posiciones activas
- `/pause` — Pausa las señales (sin cerrar posiciones)
- `/resume` — Reanuda las señales
- `/close_all` — Cierra todas las posiciones (¡emergencia!)
- `/performance` — Resumen de rendimiento de los últimos 30 días
- `/settings` — Ver configuración actual

---

## 10. Fase 8 — Paper Trading (validación real — MUY IMPORTANTE)

Esta fase dura **mínimo 4 semanas** y es obligatoria antes de poner dinero real.

### Activar paper trading mode

```bash
python main.py --mode paper-trading
```

En este modo el sistema:
- Analiza el mercado en tiempo real con el modelo entrenado
- Envía señales reales por Telegram como si fueran en vivo
- Registra todas las señales y sus resultados en la BD
- NO ejecuta ninguna orden real en cTrader

### Métricas que debes alcanzar antes de pasar a live:

| Métrica | Mínimo requerido |
|---------|-----------------|
| Hit Rate (% operaciones ganadoras) | > 52% |
| Profit Factor | > 1.25 |
| Ratio Sharpe (anualizado) | > 1.0 |
| Máximo Drawdown | < 12% |
| Nº operaciones en 4 semanas | > 25 (suficiente muestra estadística) |

Si no alcanzas estas métricas, ajusta los parámetros del signal engine o revisa el entrenamiento del modelo.

### Ver métricas de paper trading

```bash
python scripts/performance_report.py --period 30days
```

---

## 11. Fase 9 — Auto-Trader (ejecución automática)

**Solo llegas aquí si el paper trading fue positivo durante 4+ semanas.**

### Activar con cuenta DEMO primero

```bash
# Primero en demo:
python main.py --mode live --account demo --max-lots 0.01
```

Corre en demo durante 2 semanas adicionales. Verifica que las órdenes se ejecutan correctamente con los SL/TP correctos.

### Activar con cuenta REAL (con capital mínimo)

```bash
# En real, pero con tamaño mínimo:
python main.py --mode live --account real --max-lots 0.01
```

Empieza con 0.01 lots (micro lotes) independientemente del capital. Escala progresivamente solo si los resultados son consistentes.

### Configurar como servicio del sistema (para que arranque automáticamente)

```bash
# En el servidor, crear el archivo de servicio:
sudo nano /etc/systemd/system/ml-ayram.service

# Contenido:
[Unit]
Description=ML Ayram Trading Bot
After=network.target

[Service]
Type=simple
User=ayram
WorkingDirectory=/home/ayram/ml-ayram
Environment=PATH=/home/ayram/ml-ayram/venv/bin
ExecStart=/home/ayram/ml-ayram/venv/bin/python main.py --mode live
Restart=always
RestartSec=30

[Install]
WantedBy=multi-user.target

# Activar el servicio:
sudo systemctl daemon-reload
sudo systemctl enable ml-ayram
sudo systemctl start ml-ayram

# Ver logs en tiempo real:
sudo journalctl -u ml-ayram -f
```

---

## 12. Fase 10 — Dashboard en Netlify

El dashboard muestra el historial de señales y métricas de rendimiento.

### Configurar y desplegar

```bash
# En tu máquina local:
cd C:\Users\Usuario\Documents\Webs\ML-Ayram\src\dashboard

# Instalar Netlify CLI:
npm install -g netlify-cli

# Login en Netlify:
netlify login

# Desplegar:
netlify deploy --prod --dir .
```

El dashboard se actualiza consultando un endpoint del servidor que expone las métricas en JSON.

---

## 13. Mantenimiento

### Reentrenamiento mensual del modelo

Se ejecuta automáticamente el primer domingo de cada mes gracias a APScheduler.  
También puedes forzarlo:
```bash
python scripts/retrain_model.py --force
```

### Actualizar el código

```bash
cd ~/ml-ayram
git pull origin main
sudo systemctl restart ml-ayram
```

### Ver logs del bot

```bash
# Logs del servicio systemd:
sudo journalctl -u ml-ayram -n 100 -f

# Logs propios del bot (archivo):
tail -f ~/ml-ayram/logs/bot.log
```

### Backups de la base de datos

```bash
# Si usas Supabase: backups automáticos incluidos en el dashboard
# Si usas PostgreSQL propio, agregar al crontab:
0 2 * * * pg_dump $DATABASE_URL > ~/backups/db_$(date +%Y%m%d).sql
```

---

## 14. Checklist de Estado del Proyecto

Usa este checklist para saber en qué punto estás:

### FASE 0 — Preparación local
- [X] Git instalado y configurado
- [X] Python 3.14.3 instalado (nota: servidor usará Python 3.11)
- [X] VSCode instalado con extensiones
- [ ] Repositorio GitHub privado creado
- [ ] venv local creado y dependencias instaladas (verificar compatibilidad con 3.14)

### FASE 1 — Servidor
- [ ] Cuenta DigitalOcean / GCP creada
- [ ] Créditos gratuitos activados
- [ ] Servidor/VM creado y arrancando
- [ ] SSH configurado y funcionando
- [ ] Usuario `ayram` creado en el servidor
- [ ] Dependencias del sistema instaladas

### FASE 2 — Base de Datos
- [ ] PostgreSQL accesible desde el servidor
- [ ] TimescaleDB instalado/habilitado
- [ ] Schema SQL ejecutado (todas las tablas creadas)
- [ ] Conexión verificada desde el código Python

### FASE 3 — cTrader
- [ ] Credenciales Open API obtenidas
- [ ] Conexión de prueba exitosa en DEMO
- [ ] Script de descarga de histórico probado

### FASE 4 — Dataset
- [ ] Datos históricos descargados (3 años, 5 pares, 4 TFs)
- [ ] Features calculados y guardados en BD
- [ ] Labels generados con Triple-Barrier
- [ ] Análisis exploratorio completado

### FASE 5 — Modelo ML
- [ ] XGBoost entrenado (Accuracy OOS > 58%)
- [ ] LSTM entrenado
- [ ] Walk-forward validation completado
- [ ] Modelos guardados en /models/saved/

### FASE 6 — Signal Engine
- [ ] Motor de señales configurado
- [ ] Dry-run probado y revisado
- [ ] Risk manager configurado

### FASE 7 — Telegram
- [ ] Bot creado en BotFather
- [ ] Chat ID obtenido
- [ ] Mensajes de prueba recibidos
- [ ] Comandos /status y /pause funcionando

### FASE 8 — Paper Trading
- [ ] Paper trading activo
- [ ] Hit Rate > 52% después de 4 semanas
- [ ] Profit Factor > 1.25
- [ ] Drawdown < 12%

### FASE 9 — Live Trading
- [ ] Probado en cuenta DEMO live
- [ ] Activado en cuenta REAL con 0.01 lots
- [ ] Servicio systemd configurado (arranque automático)
- [ ] Alertas de monitoring configuradas

### FASE 10 — Dashboard
- [ ] Dashboard desplegado en Netlify
- [ ] Métricas en tiempo real visibles

---

*Proyecto ML-Ayram | Uso personal | No compartir credenciales*
