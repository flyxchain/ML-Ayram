# ML-Ayram — Tutorial Completo de Implementación
# Bot de Trading Algorítmico con ML para Forex

> **Proyecto:** ML-Ayram
> **Objetivo:** Sistema de trading con ML que analiza pares de divisas, genera señales por Telegram y ejecuta trades automáticamente (modo simulado hasta tener cuenta demo)
> **Última actualización:** Feb 2026
> **Estado actual:** FASE 4 — Dataset (descarga en progreso)

---

## ÍNDICE

1. [Resumen de la Arquitectura](#1-resumen-de-la-arquitectura)
2. [Fase 0 — Preparar la máquina local](#2-fase-0)
3. [Fase 1 — Servidor DigitalOcean](#3-fase-1)
4. [Fase 2 — Base de Datos Supabase](#4-fase-2)
5. [Fase 3 — Fuente de datos EODHD](#5-fase-3)
6. [Fase 4 — Dataset: descarga, features y labels](#6-fase-4)
7. [Fase 5 — Entrenar los modelos ML](#7-fase-5)
8. [Fase 6 — Generador de señales y ensemble](#8-fase-6)
9. [Fase 7 — Bot de Telegram](#9-fase-7)
10. [Fase 8 — Paper Trading](#10-fase-8)
11. [Fase 9 — Live Trading (cuando haya demo)](#11-fase-9)
12. [Fase 10 — Dashboard Netlify](#12-fase-10)
13. [Mantenimiento](#13-mantenimiento)

---

## 1. Resumen de la Arquitectura

```
EODHD API ──► collector.py ──► ohlcv_raw (PostgreSQL/Supabase)
                                        │
                                   features.py (~85 features)
                                        │
                                    labels.py (Triple-Barrier)
                                        │
                              features_computed (PostgreSQL)
                                        │
                      ┌─────────────────┴─────────────────┐
               xgboost_model.py                   lstm_model.py
               (XGBoost + Optuna)            (LSTM + Attention, PyTorch)
                      └─────────────────┬─────────────────┘
                                   ensemble.py
                               (votación 55/45)
                                        │
                              signal_generator.py
                          (TP/SL/tamaño posición)
                                        │
                        ┌───────────────┴───────────────┐
                   signals_log                    positions_active
                   (PostgreSQL)                   (PostgreSQL)
                        │
                   telegram_bot.py          [futuro: ctrader_client.py]
                  (notificaciones)          (ejecución real cuando demo)
```

**Stack tecnológico:**
- Lenguaje: Python 3.14.3 (local) / Python 3.12.3 (servidor)
- Servidor: DigitalOcean Droplet $12/mes (2vCPU/2GB, Frankfurt)
- Base de datos: Supabase PostgreSQL 17.6 (sin TimescaleDB)
- Fuente de datos: EODHD API (forex intradía M15/H1/D1)
- ML: XGBoost + PyTorch LSTM con Attention
- Optimización: Optuna (hiperparámetros XGBoost)
- Tracking: MLflow (solo servidor)
- Notificaciones: python-telegram-bot
- Ejecución: Simulada (paper trading); cTrader cuando haya demo
- Dashboard: Netlify (HTML estático)
- Scheduler: APScheduler

**Pares:** EURUSD, GBPUSD, USDJPY, EURJPY, XAUUSD
**Timeframes:** M15, H1, H4 (resampleado desde H1), D1
**Señales esperadas:** 2-5 por día con filtros estrictos

---

## 2. Fase 0 — Preparar la Máquina Local ✅

### Completado:
- Git instalado
- Python 3.14.3 instalado en Windows
- VSCode instalado con extensiones Python, Pylance, GitLens, Remote SSH
- Repositorio GitHub privado: https://github.com/flyxchain/ML-Ayram
- venv local creado, dependencias instaladas (pandas 2.3.3, numpy 2.4.2, xgboost 3.2.0, torch 2.10.0)

### Estructura del venv local

```bash
cd C:\Users\Usuario\Documents\Webs\ML-Ayram
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

### Notas Python 3.14 vs servidor

| Librería | Local 3.14.3 | Servidor 3.12.3 |
|---|---|---|
| PyTorch | ✅ | ✅ |
| XGBoost | ✅ | ✅ |
| librería `ta` | ✅ | ✅ |
| mlflow | ❌ (cmake/MSVC) | ✅ |
| pandas-ta | ❌ (numba) | ❌ (usar `ta`) |
| TensorFlow | ❌ | ⚠️ (usar PyTorch) |

---

## 3. Fase 1 — Servidor DigitalOcean ✅

### Datos del servidor

- **IP:** 206.81.31.156
- **Región:** Frankfurt (FRA1)
- **SO:** Ubuntu 24.04 LTS
- **Plan:** $12/mes (2vCPU, 2GB RAM, 50GB SSD)
- **Usuario de trabajo:** ayram (no root)
- **Python:** 3.12.3 con venv en ~/ml-ayram/venv
- **CUDA:** 12.8 (soporte GPU disponible aunque no se use inicialmente)

### Conectar al servidor

```bash
ssh root@206.81.31.156
# o directamente como ayram:
ssh ayram@206.81.31.156

# En el servidor, activar entorno:
cd ~/ml-ayram && source venv/bin/activate
```

### Firewall activo

```
UFW: SSH (22) + HTTPS (443)
```

### Pull de cambios del repo

```bash
cd ~/ml-ayram && git pull
```

---

## 4. Fase 2 — Base de Datos Supabase ✅

### Datos de conexión

- **Proyecto:** ML-Ayram
- **Región:** West EU (Ireland)
- **Motor:** PostgreSQL 17.6
- **Host:** solo IPv6 → habilitado IPv6 en Droplet DigitalOcean para resolverlo

### Tablas creadas (schema.sql)

| Tabla | Descripción |
|---|---|
| `ohlcv_raw` | Velas OHLCV brutas por par/timeframe |
| `features_computed` | ~85 features + labels por vela |
| `signals_log` | Señales generadas con resultados |
| `model_performance` | Métricas de cada modelo por período |
| `positions_active` | Posiciones abiertas del paper trader |
| `performance_summary_30d` | Vista de rendimiento 30 días |

### Ejecutar/actualizar schema

Abre el SQL Editor en Supabase y ejecuta `config/schema.sql`.

### Nota importante

TimescaleDB **no está disponible** en Supabase free tier. El schema usa PostgreSQL estándar con índices temporales. El rendimiento es adecuado para este proyecto.

---

## 5. Fase 3 — Fuente de Datos EODHD ✅

### Por qué EODHD (no cTrader)

cTrader Open API requiere OAuth2. El endpoint de autorización de Spotware devolvía 404. Además, crear cuentas demo en cTrader desde España tiene coste. Se decidió:

1. **EODHD** como fuente de datos históricos y en tiempo real
2. **Modo simulado** para la ejecución (paper trading con PnL matemático)
3. Las credenciales de cTrader están guardadas para activación futura cuando se disponga de cuenta demo

### API Key EODHD

- Acceso a forex intradía verificado: M15 (`15m`), H1 (`1h`), D1 (endpoint EOD)
- H4 se construye por resample desde H1 con `resample_h4()`
- Histórico: hasta 3+ años disponibles

### Módulo collector.py

```python
# Descarga histórica completa (una sola vez, ~20-30 min):
python -m src.data.collector

# Obtener últimas N velas para señales en tiempo real:
from src.data.collector import get_latest_candles
df = get_latest_candles("EURUSD", "H1", n=300)
```

### Variables de entorno necesarias

```
EODHD_API_KEY=tu_api_key
DATABASE_URL=postgresql://...
```

---

## 6. Fase 4 — Dataset: Descarga, Features y Labels

### 6.1 Descarga histórica

```bash
# En el servidor (proceso largo, usar nohup):
mkdir -p ~/ml-ayram/logs
nohup python -m src.data.collector > logs/download_historical.log 2>&1 &

# Ver progreso:
tail -f ~/ml-ayram/logs/download_historical.log
```

Descarga 3 años de M15, H1, H4, D1 para 5 pares. Dura ~20-30 minutos.

### 6.2 Calcular features (~85 indicadores)

```bash
# Cuando termine la descarga:
python -m src.data.features
```

El módulo `features.py` calcula:
- **Tendencia:** EMA 9/20/50/200, MACD, ADX/DI, CCI, DPO, Aroon, Ichimoku
- **Osciladores:** RSI 7/14, Stoch K/D, Williams %R, ROC, PPO, TSI
- **Volatilidad:** ATR 7/14, Bollinger Bands, Keltner Channel, Donchian Channel
- **Estructura de mercado:** Swing high/low, distancia en ATRs, tendencia EMA cruzada
- **Vela:** body size, wicks, log returns 1/5/10, precio vs EMAs
- **Temporales:** hora, día, semana, mes, sesión (London/NY/Tokio/overlap)
- **Multi-timeframe (HTF):** tendencia, RSI, ADX, ATR del TF superior

Procesa en orden D1 → H4 → H1 → M15 para que los HTF features estén disponibles.

### 6.3 Generar etiquetas (Triple-Barrier Method)

```bash
python -m src.data.labels
```

**Parámetros por defecto:**
- TP = 2x ATR (barrera superior)
- SL = 1x ATR (barrera inferior)
- Horizon = 20 velas máximo

**Distribución esperada de labels:** ~35% (+1) / ~35% (-1) / ~30% (0)

**Etiquetas:**
- `+1` → precio tocó TP antes que SL (long ganador)
- `-1` → precio tocó SL antes que TP (long perdedor)
- `0` → venció el horizonte sin tocar barrera

---

## 7. Fase 5 — Entrenar los Modelos ML

### 7.1 XGBoost (xgboost_model.py)

```bash
# Sin optimización (rápido, ~5 min por par):
python -m src.models.xgboost_model

# Con optimización Optuna (lento, mejor resultado):
python -c "
from src.models.xgboost_model import train_and_save
train_and_save('EURUSD', 'H1', optimize=True, n_trials=50)
"
```

**Validación:** TimeSeriesSplit con 5 folds (nunca mezcla futuro con pasado)
**Objetivo:** CV F1 weighted > 0.55
**Tracking:** MLflow en http://localhost:5000 del servidor

### 7.2 LSTM con Attention (lstm_model.py)

```bash
python -m src.models.lstm_model
```

**Arquitectura:**
- LSTM 2 capas con Attention
- Secuencias de 60 velas como contexto
- Early stopping (paciencia 10 épocas)
- Pesos de clase para compensar desbalanceo
- Gradient clipping para estabilidad

**Duración estimada:** 1-3 horas por par en CPU. El servidor tiene CUDA disponible — si se necesita acelerar, instalar drivers NVIDIA y relanzar.

**Objetivo:** val F1 > 0.53

### 7.3 Ensemble (ensemble.py)

Combina XGBoost (55%) y LSTM (45%). Solo emite señal cuando:
1. Ambos modelos coinciden en dirección
2. Probabilidad de la clase predicha ≥ 52%
3. Diferencia entre mejor y segunda clase ≥ 0.15

```python
from src.models.ensemble import get_latest_signal
signal = get_latest_signal(df, "EURUSD", "H1")
# signal.direction: +1, -1, 0
# signal.confidence: float
# signal.agreement: bool
```

---

## 8. Fase 6 — Generador de Señales

```bash
# Scan completo de todos los pares:
python -m src.trading.signal_generator
```

### Gestión de riesgo

| Parámetro | Valor |
|---|---|
| Riesgo por operación | 1% del capital |
| Capital simulado inicial | 10.000€ |
| TP | 2x ATR |
| SL | 1x ATR |
| Ratio riesgo/beneficio | 1:2 |
| Máximo por operación | 10 lotes |
| Mínimo por operación | 0.01 lotes |

### Flujo de cada ciclo

1. `update_positions()` — revisa si alguna posición abierta tocó TP o SL
2. Para cada par activo, llama al ensemble y genera señal
3. Si hay señal, calcula tamaño de posición y guarda en BD
4. Registra en `signals_log` y `positions_active`

---

## 9. Fase 7 — Bot de Telegram

### Crear el bot

1. Busca `@BotFather` en Telegram
2. `/newbot` → nombre: ML Ayram Bot
3. Copia el token

### Obtener Chat ID

Busca `@userinfobot` y escríbele cualquier cosa — te da tu ID.

### Configurar .env

```
TELEGRAM_BOT_TOKEN=tu_token
TELEGRAM_CHAT_ID=tu_chat_id
```

### Comandos disponibles (por implementar)

- `/status` — Estado del sistema y posiciones abiertas
- `/positions` — Lista de posiciones activas
- `/pause` / `/resume` — Pausar/reanudar señales
- `/close_all` — Cerrar todas las posiciones (emergencia)
- `/performance` — Resumen de los últimos 30 días

---

## 10. Fase 8 — Paper Trading

Esta fase dura **mínimo 4 semanas** antes de activar dinero real.

En modo simulado el sistema:
- Analiza el mercado en tiempo real con precios reales de EODHD
- Genera señales con el ensemble
- Calcula PnL matemáticamente (sin ejecutar órdenes)
- Registra todo en la BD para análisis

### Métricas mínimas para continuar a live

| Métrica | Mínimo |
|---|---|
| Hit Rate (% operaciones ganadoras) | > 52% |
| Profit Factor | > 1.25 |
| Sharpe Ratio (anualizado) | > 1.0 |
| Máximo Drawdown | < 12% |
| Número de operaciones en 4 semanas | > 25 |

---

## 11. Fase 9 — Live Trading

**Solo accesible cuando haya cuenta demo de cTrader disponible.**

Credenciales cTrader ya guardadas en .env:
- Client ID: 21838_G2BlJoy7B8vs4AkVWXaWdcojrLIQGyf83GmJ60cmgigH5uUkug
- Account ID: 2016020 (cuenta real Pepperstone — usar SOLO en demo)

Para activar: cambiar `CTRADER_ENV=demo` en .env y crear `src/trading/ctrader_client.py`.

### Servicio systemd (cuando esté en producción)

```bash
sudo nano /etc/systemd/system/ml-ayram.service

[Unit]
Description=ML Ayram Trading Bot
After=network.target

[Service]
Type=simple
User=ayram
WorkingDirectory=/home/ayram/ml-ayram
Environment=PATH=/home/ayram/ml-ayram/venv/bin
ExecStart=/home/ayram/ml-ayram/venv/bin/python main.py
Restart=always
RestartSec=30

[Install]
WantedBy=multi-user.target

sudo systemctl daemon-reload
sudo systemctl enable ml-ayram
sudo systemctl start ml-ayram
```

---

## 12. Fase 10 — Dashboard Netlify

Dashboard HTML estático que lee métricas de la BD vía API.

```bash
# Instalar Netlify CLI:
npm install -g netlify-cli
netlify login

# Desplegar:
cd src/dashboard
netlify deploy --prod --dir .
```

---

## 13. Mantenimiento

### Ver logs de descarga

```bash
tail -f ~/ml-ayram/logs/download_historical.log
```

### Reentrenamiento mensual

El bot reentrenará automáticamente el primer domingo de cada mes con APScheduler. También se puede forzar:

```bash
python -m src.training.train_pipeline --force
```

### Actualizar código

```bash
cd ~/ml-ayram && git pull
sudo systemctl restart ml-ayram
```

### Backup de modelos

Los modelos `.pt` y `.ubj` están en `.gitignore`. Hacer copia manual periódicamente:

```bash
cp -r ~/ml-ayram/models/saved/ ~/backups/models_$(date +%Y%m%d)/
```

---

*Proyecto ML-Ayram | Uso personal | No compartir credenciales*
