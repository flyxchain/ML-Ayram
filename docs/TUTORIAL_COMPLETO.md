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
8. [Fase 5b — Backtesting y Walk-Forward](#8-fase-5b)
9. [Fase 6 — Generador de señales y ensemble](#9-fase-6)
10. [Fase 7 — Bot de Telegram](#10-fase-7)
11. [Fase 8 — Paper Trading](#11-fase-8)
12. [Fase 9 — Live Trading (cuando haya demo)](#12-fase-9)
13. [Fase 10 — Dashboard FastAPI](#13-fase-10)
14. [Deploy y Servicios systemd](#14-deploy)
15. [Mantenimiento](#15-mantenimiento)

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
                          ┌─────────────┴─────────────┐
                   signals/generator.py        backtest/engine.py
                  (filtros + confianza)        (validación histórica)
                          │                    backtest/walk_forward.py
                          │                    (validación OOS)
                   execution/position_manager.py
                   (sizing + tracking)
                          │
           ┌──────────────┼──────────────┐
    signals (BD)    positions_active    trades_history
           │              (BD)              (BD)
    notifications/telegram.py    [futuro: ctrader_client.py]
    (señales/errores/heartbeat)  (ejecución real cuando demo)
           │
    dashboard/app.py (FastAPI)
    (API + SPA en puerto 8000)
```

**Stack tecnológico:**
- Lenguaje: Python 3.14.3 (local) / Python 3.12.3 (servidor)
- Servidor: DigitalOcean Droplet $12/mes (2vCPU/2GB, Frankfurt)
- Base de datos: Supabase PostgreSQL 17.6 (sin TimescaleDB)
- Fuente de datos: EODHD API (forex intradía M15/H1/D1)
- ML: XGBoost + PyTorch LSTM con Attention
- Optimización: Optuna (hiperparámetros XGBoost)
- Tracking: MLflow (solo servidor)
- Notificaciones: requests (API HTTP de Telegram)
- Ejecución: Simulada (paper trading); cTrader cuando haya demo
- Dashboard: FastAPI + Uvicorn (puerto 8000, servido desde servidor)
- Backtesting: Motor propio con Walk-Forward Validation
- Configuración: signal_config.yaml centralizado
- Deploy: rsync + systemd (deploy/deploy.sh)
- Servicios: systemd (dashboard, signals, collector, features)

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
| FastAPI + Uvicorn | ✅ | ✅ |
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

### Servicios systemd instalados

Ver sección [14. Deploy y Servicios systemd](#14-deploy) para detalles completos.

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
| `signals_log` | Señales generadas con TP1/TP2, régimen, modo (paper/live) |
| `model_performance` | Métricas de cada modelo por período (IS/OOS) |
| `positions_active` | Posiciones abiertas del paper trader (con trailing SL) |
| `performance_summary_30d` | Vista de rendimiento 30 días |

### Ejecutar/actualizar schema

Abre el SQL Editor en Supabase y ejecuta `config/schema.sql`.

También disponible el script:
```bash
python scripts/init_db.py
```

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
- **Tendencia:** EMA 20/50/200, MACD, ADX/DI, CCI, DPO, Aroon, Ichimoku
- **Osciladores:** RSI 14, Stoch K/D, Williams %R, ROC, PPO, TSI
- **Volatilidad:** ATR 14, Bollinger Bands (20, 2σ), Keltner Channel, Donchian Channel
- **Estructura de mercado:** Swing high/low (20), distancia en ATRs, tendencia EMA cruzada
- **Vela:** body size, wicks, log returns 1/5/10, precio vs EMAs
- **Temporales:** hora, día, semana, mes, sesión (London/NY/Tokio/overlap)
- **Multi-timeframe (HTF):** tendencia, RSI, ADX, ATR del TF superior

Procesa en orden D1 → H4 → H1 → M15 para que los HTF features estén disponibles.

Los períodos de cada indicador están centralizados en `config/signal_config.yaml` → sección `features`.

### 6.3 Generar etiquetas (Triple-Barrier Method)

```bash
python -m src.data.labels
```

**Parámetros (signal_config.yaml → labeling):**
- TP = 1.5× ATR (barrera superior)
- SL = 1.0× ATR (barrera inferior)
- Horizon = 20 velas máximo
- Min return threshold = 0.0003 (filtrar movimientos < 3 pips)

**Distribución esperada de labels:** ~35% (+1) / ~35% (-1) / ~30% (0)

**Etiquetas:**
- `+1` → precio tocó TP antes que SL (long ganador)
- `-1` → precio tocó SL antes que TP (long perdedor)
- `0` → venció el horizonte sin tocar barrera

---

## 7. Fase 5 — Entrenar los Modelos ML

### 7.1 Orquestador de entrenamiento (src/train.py)

```bash
# Entrenar todo (todos los pares y TF):
python -m src.train

# Solo ciertos pares:
python -m src.train --pairs EURUSD GBPUSD

# Solo ciertos timeframes:
python -m src.train --timeframes H1 H4

# Solo XGBoost o solo LSTM:
python -m src.train --xgb-only
python -m src.train --lstm-only

# Con optimización Optuna:
python -m src.train --optimize --trials 30
```

### 7.2 XGBoost (src/models/xgboost_model.py)

**Hiperparámetros por defecto (signal_config.yaml):**
- n_estimators: 500
- max_depth: 6
- learning_rate: 0.05
- subsample: 0.8
- colsample_bytree: 0.8
- min_child_weight: 10

**Validación:** TimeSeriesSplit con 5 folds (nunca mezcla futuro con pasado)
**Objetivo:** CV F1 weighted > 0.55
**Tracking:** MLflow en http://localhost:5000 del servidor
**Optimización:** Optuna con hasta 100 trials (configurable)

### 7.3 LSTM con Attention (src/models/lstm_model.py)

**Hiperparámetros por defecto (signal_config.yaml):**
- sequence_length: 60 velas
- layers: 2
- units: 128
- dropout: 0.3
- batch_size: 256
- epochs: 100
- Early stopping: paciencia 10 épocas
- Pesos de clase para compensar desbalanceo
- Gradient clipping para estabilidad

**Duración estimada:** 1-3 horas por par en CPU. El servidor tiene CUDA disponible — si se necesita acelerar, instalar drivers NVIDIA y relanzar.

**Objetivo:** val F1 > 0.53

### 7.4 Ensemble (src/models/ensemble.py)

Combina XGBoost (55%) y LSTM (45%). Solo emite señal cuando:
1. Ambos modelos coinciden en dirección
2. Confianza del ensemble ≥ 72% (configurable en signal_config.yaml)
3. Mínimo 2 timeframes alineados

```python
from src.models.ensemble import get_latest_signal
signal = get_latest_signal(df, "EURUSD", "H1")
# signal.direction: +1, -1, 0
# signal.confidence: float
# signal.agreement: bool
```

---

## 8. Fase 5b — Backtesting y Walk-Forward

### 8.1 Backtesting (src/backtest/engine.py)

Motor de backtesting realista sobre señales almacenadas en BD:

```bash
# Backtest completo:
python -m src.backtest.engine

# Filtrado por par, timeframe, período:
python -m src.backtest.engine --pair EURUSD --tf H1 --days 90

# Múltiples pares + filtro de confianza:
python -m src.backtest.engine --pair EURUSD GBPUSD --min-confidence 0.60

# Guardar resultados:
python -m src.backtest.engine --output results/backtest_20260228.json
```

Características:
- Simulación de spread y slippage realistas
- PnL calculado con pip sizes correctos por par
- Lot sizing basado en riesgo configurable
- Informe con métricas completas (hit rate, PF, Sharpe, drawdown, etc.)

### 8.2 Walk-Forward Validation (src/backtest/walk_forward.py)

Validación OOS sin lookahead bias:

```bash
# Walk-forward completo:
python -m src.backtest.walk_forward

# Personalizado:
python -m src.backtest.walk_forward --pairs EURUSD GBPUSD --tf H1
python -m src.backtest.walk_forward --folds 6 --is-months 6 --oos-months 1
python -m src.backtest.walk_forward --expanding   # ventana expansiva (default: rolling)

# Guardar resultados:
python -m src.backtest.walk_forward --output results/wf_20260228.json
```

**Configuración Walk-Forward (signal_config.yaml → model.training):**
- 8 períodos OOS de 3 meses cada uno
- Mínimo accuracy OOS: 0.55
- Mínimo profit factor OOS: 1.25
- Soporta ventana rolling y expanding

---

## 9. Fase 6 — Generador de Señales

### 9.1 Generador principal (src/signals/generator.py)

Genera señales accionables combinando el ensemble con filtros técnicos:

1. Carga últimas N velas con features calculados
2. Pide predicción al ensemble (XGBoost + LSTM)
3. Aplica filtros: ADX, sesión activa, confianza mínima 72%, cooldown 4h
4. Calcula TP/SL en pips y en precio
5. Guarda señal en BD (tabla `signals`)
6. Devuelve objeto SignalResult para notificaciones/ejecución

### 9.2 Gestión de posiciones (src/execution/position_manager.py)

Consume señales de la tabla `signals` y gestiona posiciones en `positions_active` y `trades_history`.

### 9.3 Gestión de riesgo (signal_config.yaml)

| Parámetro | Valor |
|---|---|
| Riesgo por operación | 1.5% del capital |
| Capital simulado inicial | 10.000€ |
| Max drawdown diario | 4% |
| Max drawdown semanal | 8% |
| Kelly fraction | 25% |
| SL | 1.5 × ATR(14) |
| TP1 | 1.5:1 RR (cierre 50% posición) |
| TP2 | 2.5:1 RR (trailing stop) |
| Trailing stop | 1.0 × ATR desde máximo |
| Máximo por operación | 1.0 lotes |
| Mínimo por operación | 0.01 lotes |
| Máximo señales simultáneas | 3 |
| Cooldown entre señales | 4 horas |
| Sesiones activas | London, New York, Overlap |

### 9.4 Signal generator legacy (src/trading/signal_generator.py)

Versión anterior del generador de señales. Mantenido por compatibilidad, pero la versión principal es `src/signals/generator.py`.

---

## 10. Fase 7 — Bot de Telegram

### Módulo: src/notifications/telegram.py

Funciones implementadas:
- `send_signal()` → señal nueva (LONG / SHORT) con detalles
- `send_summary()` → resumen periódico de señales
- `send_error()` → alerta de error crítico del sistema
- `send_heartbeat()` → "sigo vivo" cada N horas

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

---

## 11. Fase 8 — Paper Trading

Esta fase dura **mínimo 4 semanas** antes de activar dinero real.

En modo simulado el sistema:
- Analiza el mercado en tiempo real con precios reales de EODHD
- Genera señales con el ensemble (filtros de signal_config.yaml)
- Calcula PnL matemáticamente (sin ejecutar órdenes)
- Gestiona posiciones con trailing stop (position_manager.py)
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

## 12. Fase 9 — Live Trading

**Solo accesible cuando haya cuenta demo de cTrader disponible.**

Credenciales cTrader ya guardadas en .env:
- Client ID: 21838_G2BlJoy7B8vs4AkVWXaWdcojrLIQGyf83GmJ60cmgigH5uUkug
- Account ID: 2016020 (cuenta real Pepperstone — usar SOLO en demo)

Para activar: cambiar `CTRADER_ENV=demo` en .env y crear `src/trading/ctrader_client.py`.

Se puede testear la conexión con:
```bash
python scripts/test_ctrader_connection.py
```

---

## 13. Fase 10 — Dashboard FastAPI

### Backend: src/dashboard/app.py

El dashboard es una aplicación FastAPI que expone una API REST y sirve un frontend SPA.

**Endpoints disponibles:**
- `GET /` → SPA (index.html)
- `GET /api/status` → estado del sistema
- `GET /api/signals/latest` → señales recientes
- `GET /api/signals/history` → historial paginado
- `GET /api/chart/{pair}/{tf}` → velas OHLCV + señales superpuestas
- `GET /api/metrics` → distribución y stats de señales
- `GET /api/performance` → rendimiento real de trades cerrados
- `GET /api/positions` → posiciones abiertas actualmente
- `GET /api/config` → configuración actual de filtros
- `POST /api/config` → actualizar filtros del generador

### Frontend: src/dashboard/static/index.html

SPA que consume la API del backend.

### Arranque

```bash
# Local:
uvicorn src.dashboard.app:app --host 0.0.0.0 --port 8000 --workers 1

# En servidor (gestionado por systemd):
systemctl start ayram-dashboard
```

---

## 14. Deploy y Servicios systemd

### Script de deploy (deploy/deploy.sh)

```bash
# Deploy completo (rsync + reinicio):
./deploy/deploy.sh

# Solo sincronizar código sin reiniciar:
./deploy/deploy.sh --no-restart

# Solo reiniciar servicios:
./deploy/deploy.sh --services

# Primera instalación de systemd units:
./deploy/deploy.sh --install
```

### Servicios systemd

| Servicio | Función | Puerto |
|---|---|---|
| `ayram-dashboard` | FastAPI + Uvicorn | 8000 |
| `ayram-signals` | Generador de señales (bucle 60s) | — |
| `ayram-collector` | Descarga de datos (con timer) | — |
| `ayram-features` | Cálculo de features (con timer) | — |

### Instalación manual (una sola vez)

```bash
# Subir unit files al servidor
scp deploy/systemd/*.service ayram@206.81.31.156:/tmp/
scp deploy/systemd/install.sh ayram@206.81.31.156:/tmp/

# Conectarse y ejecutar el instalador
ssh ayram@206.81.31.156
sudo bash /tmp/install.sh
```

### Comandos del día a día

```bash
# Ver logs en vivo
journalctl -u ayram-dashboard -f
journalctl -u ayram-signals -f

# Estado rápido
systemctl status ayram-dashboard ayram-signals

# Reiniciar tras git pull
systemctl restart ayram-dashboard
systemctl restart ayram-signals

# Activar señales cuando haya modelos entrenados
systemctl enable --now ayram-signals
```

### Actualizar código en producción

```bash
cd /home/ayram/ml-ayram
git pull
systemctl restart ayram-dashboard; systemctl restart ayram-signals
```

---

## 15. Mantenimiento

### Ver logs de descarga

```bash
tail -f ~/ml-ayram/logs/download_historical.log
```

### Reentrenamiento

Automático: domingos a las 2am UTC (configurable en signal_config.yaml).

Manual:
```bash
python -m src.train --optimize --trials 50
```

### Actualizar código

```bash
# Opción 1: Script de deploy (desde local)
./deploy/deploy.sh

# Opción 2: Manual (desde servidor)
cd ~/ml-ayram && git pull
systemctl restart ayram-dashboard; systemctl restart ayram-signals
```

### Backup de modelos

Los modelos `.pt` y `.ubj` están en `.gitignore`. Hacer copia manual periódicamente:

```bash
cp -r ~/ml-ayram/models/saved/ ~/backups/models_$(date +%Y%m%d)/
```

### Ejecutar backtests periódicamente

```bash
# Backtest rápido:
python -m src.backtest.engine --days 30 --output results/backtest_$(date +%Y%m%d).json

# Walk-forward completo:
python -m src.backtest.walk_forward --output results/wf_$(date +%Y%m%d).json
```

---

*Proyecto ML-Ayram | Uso personal | No compartir credenciales*
