# ML-Ayram — Tutorial Completo de Implementación
# Bot de Trading Algorítmico con ML para Forex

> **Proyecto:** ML-Ayram
> **Objetivo:** Sistema de trading con ML que analiza pares de divisas, genera señales por Telegram y ejecuta trades automáticamente (modo simulado hasta tener cuenta demo)
> **Última actualización:** Feb 2026
> **Estado actual:** FASE 4 — Dataset (descarga en progreso) | Infraestructura completa | Dashboard 15 secciones

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
14. [Fase 11 — Monitoreo y Alertas](#14-fase-11)
15. [Fase 12 — Análisis IA Mensual](#15-fase-12)
16. [Fase 13 — Comparador de Modelos](#16-fase-13)
17. [Deploy y Servicios systemd](#17-deploy)
18. [Mantenimiento](#18-mantenimiento)
19. [Roadmap de Mejoras](#19-roadmap)

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
           │
    ┌──────┼──────────────────────────────────┐
    │      │                                  │
 Telegram  │                        monitoring/
 (alertas) │                    ┌── model_health.py
           │                    │   (degradación modelos)
    dashboard/app.py            ├── anomaly_detector.py
    (FastAPI :8000)             │   (6 checks operativos)
    15 secciones SPA            └── analysis/monthly_summary.py
    30+ endpoints API               (resumen IA mensual)
    responsive (móvil+desktop)
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
- Dashboard: FastAPI + Uvicorn (puerto 8000), 15 secciones SPA, responsive
- Monitoreo: model_health + anomaly_detector (automáticos con systemd)
- Análisis: monthly_summary (genera prompts para Claude/ChatGPT)
- Backtesting: Motor propio con Walk-Forward Validation
- Configuración: signal_config.yaml centralizado
- Deploy: rsync + systemd (deploy/deploy.sh)
- Servicios: 2 daemons + 5 timers en systemd

**Pares:** EURUSD, GBPUSD, USDJPY, EURJPY, XAUUSD
**Timeframes:** M15, H1, H4 (resampleado desde H1), D1
**Señales esperadas:** 2-5 por día con filtros estrictos

---

## 2. Fase 0 — Preparar la Máquina Local ✅

### Completado:
- Git, Python 3.14.3, VSCode instalados
- Repositorio GitHub privado: https://github.com/flyxchain/ML-Ayram
- venv local creado, dependencias instaladas

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
- **CUDA:** 12.8 (soporte GPU disponible)

### Conectar al servidor

```bash
ssh root@206.81.31.156
su - ayram
cd ~/ml-ayram && source venv/bin/activate
```

### Firewall activo

```
UFW: SSH (22) + HTTPS (443) + Puerto 8000 (dashboard)
```

---

## 4. Fase 2 — Base de Datos Supabase ✅

### Datos de conexión

- **Proyecto:** ML-Ayram
- **Región:** West EU (Ireland)
- **Motor:** PostgreSQL 17.6
- **Host:** solo IPv6 → habilitado IPv6 en Droplet

### Tablas principales

| Tabla | Descripción |
|---|---|
| `ohlcv_raw` | Velas OHLCV brutas por par/timeframe |
| `features_computed` | ~85 features + labels por vela |
| `signals` | Señales generadas con filtros aplicados |
| `model_performance` | Métricas de cada modelo por período (IS/OOS) |
| `positions_active` | Posiciones abiertas del paper trader |
| `trades_history` | Trades cerrados con PnL |

### Ejecutar/actualizar schema

```bash
python scripts/init_db.py
```

---

## 5. Fase 3 — Fuente de Datos EODHD ✅

### Por qué EODHD (no cTrader)

cTrader Open API requiere OAuth2 y la autorización no funcionó. EODHD provee forex intradía sin problemas. cTrader queda reservado para ejecución cuando haya cuenta demo.

### Módulo collector.py

```bash
# Descarga histórica completa (~20-30 min):
python -m src.data.collector

# Timer automático: cada 15 min en servidor
```

---

## 6. Fase 4 — Dataset: Descarga, Features y Labels

### 6.1 Descarga histórica

```bash
# En el servidor:
nohup python -m src.data.collector > logs/download_historical.log 2>&1 &
tail -f ~/ml-ayram/logs/download_historical.log
```

Descarga 3 años de M15, H1, H4, D1 para 5 pares.

### 6.2 Calcular features (~85 indicadores)

```bash
python -m src.data.features
```

El módulo calcula: tendencia (EMA, MACD, ADX, Ichimoku), osciladores (RSI, Stoch, Williams %R), volatilidad (ATR, Bollinger, Keltner), estructura de mercado, patrones de vela, temporales, multi-timeframe.

Procesa en orden D1 → H4 → H1 → M15 para que los HTF features estén disponibles.

### 6.3 Generar etiquetas (Triple-Barrier Method)

```bash
python -m src.data.labels
```

TP=1.5×ATR, SL=1.0×ATR, Horizonte=20 velas. Distribución esperada ~35/35/30.

---

## 7. Fase 5 — Entrenar los Modelos ML

### 7.1 Orquestador (src/train.py)

```bash
python -m src.train                               # entrenar todo
python -m src.train --pairs EURUSD GBPUSD          # solo ciertos pares
python -m src.train --optimize --trials 30         # con Optuna
python -m src.train --xgb-only                     # solo XGBoost
python -m src.train --lstm-only                    # solo LSTM
```

### 7.2 XGBoost
- Validación TimeSeriesSplit (5 folds), objetivo CV F1 > 0.55
- Tracking MLflow, optimización Optuna
- Output: `models/saved/xgb_{par}_{tf}_{timestamp}.ubj` + `_meta.json`
- Meta JSON contiene: `cv_f1_mean`, `cv_f1_std`, `cv_f1_folds`, `features`, `label_map`

### 7.3 LSTM con Attention
- Secuencia de 60 velas, 2 capas × 128 unidades
- Early stopping (paciencia 10), gradient clipping
- Objetivo val F1 > 0.53
- Output: `models/saved/lstm_{par}_{tf}_{timestamp}.pt`
- Checkpoint contiene: `model_state`, `model_config` (hidden_size, num_layers), `scaler_*`, `feature_cols`, `metrics` (best_val_f1), `pair`, `timeframe`

### 7.4 Ensemble
- XGBoost 55% + LSTM 45% (hardcoded en `ensemble.py`)
- Solo emite señal si ambos coinciden + confianza ≥ 72%

### 7.5 Reentrenamiento automático

El timer `ayram-train.timer` ejecuta reentrenamiento cada domingo a las 02:00 UTC con:
```bash
python -m src.train --optimize --trials 30
```

---

## 8. Fase 5b — Backtesting y Walk-Forward

### 8.1 Backtesting (src/backtest/engine.py)

```bash
python -m src.backtest.engine --pair EURUSD --tf H1 --days 90
python -m src.backtest.engine --output results/backtest_$(date +%Y%m%d).json
```

Simulación realista con spread, slippage, lot sizing basado en riesgo.

### 8.2 Walk-Forward (src/backtest/walk_forward.py)

```bash
python -m src.backtest.walk_forward --folds 8 --is-months 6 --oos-months 1 --expanding
```

Validación OOS sin lookahead bias. Mínimos: accuracy 0.55, PF 1.25.

El timer `ayram-walkforward.timer` ejecuta walk-forward el 1er domingo de cada mes a las 04:00 UTC, encadenado con model_health y monthly_summary.

---

## 9. Fase 6 — Generador de Señales

### 9.1 Generador principal (src/signals/generator.py)

1. Carga últimas N velas con features
2. Pide predicción al ensemble
3. Aplica filtros: ADX, sesión, confianza 72%, cooldown 4h
4. Calcula TP/SL en pips y precio
5. Guarda señal en BD + notificación Telegram

### 9.2 Gestión de posiciones (src/execution/position_manager.py)

Gestiona posiciones en `positions_active` y cierra en `trades_history`.

### 9.3 Gestión de riesgo

| Parámetro | Valor |
|---|---|
| Riesgo por operación | 1.5% del capital |
| Capital simulado | 10.000€ |
| Max drawdown diario/semanal | 4% / 8% |
| SL | 1.5 × ATR(14) |
| TP1 | 1.5:1 RR (cierre 50%) |
| TP2 | 2.5:1 RR (trailing stop) |
| Trailing stop | 1.0 × ATR desde máximo |
| Max simultáneas | 3 señales |
| Cooldown | 4 horas |

---

## 10. Fase 7 — Bot de Telegram

### Módulo: src/notifications/telegram.py

Funciones: `send_signal()`, `send_summary()`, `send_error()`, `send_heartbeat()`.

### Crear el bot

1. `@BotFather` → `/newbot` → copiar token
2. `@userinfobot` → obtener chat ID
3. Configurar en `.env`: `TELEGRAM_BOT_TOKEN` y `TELEGRAM_CHAT_ID`

---

## 11. Fase 8 — Paper Trading

Mínimo 4 semanas antes de activar dinero real.

### Métricas mínimas

| Métrica | Mínimo |
|---|---|
| Hit Rate | > 52% |
| Profit Factor | > 1.25 |
| Sharpe Ratio | > 1.0 |
| Máximo Drawdown | < 12% |
| Trades en 4 semanas | > 25 |

---

## 12. Fase 9 — Live Trading

Solo cuando haya cuenta demo de cTrader. Credenciales guardadas en .env.

---

## 13. Fase 10 — Dashboard FastAPI ✅

### Backend: src/dashboard/app.py (~1630 líneas, 30+ endpoints)

**Endpoints completos:**

| Endpoint | Método | Descripción |
|---|---|---|
| `/` | GET | SPA (index.html) |
| `/api/status` | GET | Estado general: señales 24h/7d, último modelo, última señal |
| `/api/signals/latest` | GET | Señales más recientes (últimas 20) |
| `/api/signals/history` | GET | Historial paginado con filtros (par, TF, dirección, fecha) |
| `/api/chart/{pair}/{tf}` | GET | Velas OHLCV + señales superpuestas para lightweight-charts |
| `/api/metrics` | GET | Distribución de señales, confianza media, tasas long/short |
| `/api/performance` | GET | Rendimiento trades cerrados: PnL, win rate, profit factor |
| `/api/positions` | GET | Posiciones abiertas con PnL flotante |
| `/api/monitor` | GET | Frescura de datos OHLCV, features, señales por par/TF |
| `/api/health` | GET | Salud de modelos: F1 scores, antigüedad, estado |
| `/api/anomalies` | GET | Alertas de anomalías (6 checks) |
| `/api/summary` | GET | Resumen mensual IA + prompt generado |
| `/api/pipeline` | GET | Logs del pipeline de ejecución |
| `/api/services` | GET | Estado de servicios systemd + timers |
| `/api/correlations` | GET | Correlaciones entre pares de divisas |
| `/api/bot` | GET/POST | Configuración del bot (mode, risk, pairs) |
| `/api/train/status` | GET | Estado del entrenamiento en curso |
| `/api/backtest/run` | POST | Ejecutar backtest con parámetros custom |
| `/api/backtest/quick-stats` | GET | Stats rápidos del último backtest |
| `/api/models/compare` | GET | Comparador XGBoost vs LSTM side-by-side |
| `/api/docs-list` | GET | Lista documentación (.md en docs/) |
| `/api/docs-content/{file}` | GET | Contenido raw de un archivo .md |
| `/api/notifications` | GET | Historial de notificaciones Telegram |
| `/api/alert-rules` | GET/POST/PUT/DELETE | CRUD de reglas de alerta personalizadas |
| `/api/alert-rules/test/{id}` | POST | Test de una regla de alerta |
| `/api/config` | GET/POST | Filtros del generador de señales en caliente |

### Frontend: src/dashboard/static/index.html (~3000 líneas, 15 secciones)

SPA responsive con 15 secciones:

| # | Página | Descripción |
|---|---|---|
| 1 | **Dashboard** | 4 KPIs, posiciones abiertas con PnL flotante, señales recientes |
| 2 | **Pipeline** | Estado y logs de cada etapa: collector, features, labels, train |
| 3 | **Gráfico** | Velas OHLCV interactivas (lightweight-charts), señales overlay, TP/SL |
| 4 | **Historial** | Tabla paginada de señales con filtros par/TF/dirección/fecha |
| 5 | **Métricas** | Charts Chart.js: distribución long/short, por par, por día, por TF |
| 6 | **Rendimiento** | PnL, win rate, PF, drawdown, equity curve, desglose por par |
| 7 | **Monitor** | Frescura datos OHLCV/features por par/TF, auto-refresh 30s |
| 8 | **Mercado** | Sesiones activas, correlaciones entre pares |
| 9 | **Train** | Progreso entrenamiento en vivo, modelos completados, F1s, log |
| 10 | **Bot** | Config bot: mode, risk, pares activos |
| 11 | **Señales** | Editor filtros en caliente (confianza, ADX, R:R, cooldown) |
| 12 | **🎯 Backtest** | Motor backtesting interactivo con KPIs |
| 13 | **📚 Docs** | Visor documentación Markdown del proyecto |
| 14 | **🔔 Alertas** | Historial notificaciones + CRUD reglas de alerta |
| 15 | **🧠 Modelos** | Comparador XGB vs LSTM: F1 side-by-side, wins, barras progreso |

**Responsive:** hamburger menu en móvil (<768px), grids adaptativos, touch targets 44px, safe-area-inset para notch, scroll-snap en tablas.

### Arranque

```bash
# Local:
uvicorn src.dashboard.app:app --host 0.0.0.0 --port 8000 --workers 1

# Servidor (gestionado por systemd, auto-reinicio):
systemctl start ayram-dashboard
```

---

## 14. Fase 11 — Monitoreo y Alertas ✅

### 14.1 Model Health (src/monitoring/model_health.py)

Diagnóstico de degradación de modelos ML. Compara rendimiento actual con baselines del walk-forward.

```bash
python -m src.monitoring.model_health --days 30 --threshold 0.35 --auto-retrain
```

**Funcionamiento:**
- Carga señales cerradas de los últimos N días
- Calcula métricas por par/TF: win rate, profit factor, PnL, drawdown, expectancy
- Compara con baselines OOS almacenados en BD
- Detecta degradación con 3 niveles:
  - **Warning** (20%): rendimiento ligeramente inferior
  - **Alert** (35%): rendimiento significativamente inferior
  - **Critical** (50%): modelo probablemente inútil
- Genera recomendaciones automáticas
- Guarda métricas en tabla `model_performance`
- Envía reporte por Telegram
- Output JSON: `results/health_YYYYMMDD_HHMM.json`
- Opción `--auto-retrain`: dispara reentrenamiento automático si degradación crítica

**Umbrales configurables:**
- min_win_rate: 45%, min_profit_factor: 1.2, max_drawdown: 10%, min_trades: 5

### 14.2 Anomaly Detector (src/monitoring/anomaly_detector.py)

6 checks automáticos que se ejecutan cada 6 horas:

```bash
python -m src.monitoring.anomaly_detector --quiet --no-notify
```

| Check | Qué detecta | Umbral |
|---|---|---|
| Signal Drought | Sin señales válidas por par/TF | >5 días |
| Drawdown | Drawdown excesivo | >8% en 7 días |
| Recent Win Rate | Racha perdedora | <35% últimos 20 trades |
| Stale Data | OHLCV desactualizado | >2h (solo horario mercado) |
| Stale Models | Modelos sin reentrenar | >14 días |
| Anomalous Signals | Exceso o sesgo de señales | >30/24h o sesgo >90% |

**Severidades:** info, warning, high, critical
**Telegram:** solo alertas high/critical
**Output:** `results/anomalies_YYYYMMDD_HHMM.json`

### 14.3 Sistema de alertas personalizadas (dashboard)

La pestaña 🔔 Alertas del dashboard permite:
- Ver historial de todas las notificaciones Telegram enviadas
- Crear reglas de alerta personalizadas (nombre, condición, severidad, canal)
- Editar y eliminar reglas existentes
- Probar una regla de alerta antes de activarla

**Endpoints:** `GET/POST/PUT/DELETE /api/alert-rules`, `POST /api/alert-rules/test/{id}`

### 14.4 Timer automático

```
ayram-anomaly.timer → cada 6h (00:30, 06:30, 12:30, 18:30)
```

El walk-forward mensual encadena: walk_forward → model_health → monthly_summary.

---

## 15. Fase 12 — Análisis IA Mensual ✅

### monthly_summary.py (src/analysis/monthly_summary.py)

Genera un resumen completo de rendimiento y un prompt optimizado para análisis con Claude o ChatGPT.

```bash
python -m src.analysis.monthly_summary                    # mes anterior
python -m src.analysis.monthly_summary --year 2026 --month 1
python -m src.analysis.monthly_summary --last-n-days 30
python -m src.analysis.monthly_summary --prompt            # muestra prompt en terminal
```

**Contenido del JSON resumen:**
- **global:** total señales, win rate, PF, PnL, ROI
- **by_pair:** métricas + market stats (ATR, ADX, tendencia)
- **by_timeframe:** rendimiento H1 vs H4
- **weekly:** desglose semanal
- **models:** edad y métricas últimos modelos
- **issues:** problemas detectados automáticamente
- **recommendations:** ajustes sugeridos

**Prompt IA generado:**
- Contexto del sistema (modelos, filtros, pares, TF)
- JSON resumen completo
- 5 preguntas de análisis: rendimiento por par, ajuste filtros, patrones temporales, estado modelos, gestión riesgo
- Formato respuesta esperado: diagnóstico + top 5 ajustes + YAML actualizado + predicción

**Output:**
- `results/summary_LABEL.json` — datos crudos
- `results/ai_prompt_LABEL.md` — prompt listo para pegar en Claude/ChatGPT
- Notificación Telegram con resumen ejecutivo

### Timer automático

Ejecutado por `ayram-walkforward.service` el 1er domingo de cada mes:
```
walk_forward → model_health → monthly_summary
```

---

## 16. Fase 13 — Comparador de Modelos ✅

### Qué es

Herramienta visual en el dashboard para comparar el rendimiento de XGBoost vs LSTM side-by-side por cada combinación par/timeframe. Sirve para evaluar si los pesos del ensemble (actualmente 55/45) son óptimos o deberían ajustarse.

### Backend: `/api/models/compare`

- Escanea el directorio `models/saved/` buscando archivos de ambos modelos
- **XGBoost:** lee `xgb_{par}_{tf}_{timestamp}_meta.json` → extrae `cv_f1_mean`, `cv_f1_std`, `cv_f1_folds`, features, fecha
- **LSTM:** carga `lstm_{par}_{tf}_{timestamp}.pt` con `torch.load()` → extrae `best_val_f1`, history, hidden_size, num_layers, features
- Determina el ganador por F1 más alto
- Devuelve resumen global: total pares, wins de cada modelo, ties, F1 medios

### Frontend: pestaña 🧠 Modelos

- **Filtros:** dropdown de par y timeframe, actualizan la vista dinámicamente
- **6 tarjetas resumen:** total par/TF, wins XGB, wins LSTM, ties, avg F1 XGB, avg F1 LSTM
- **Tarjetas de comparación:** por cada par/TF:
  - Header con nombre del par y badge del timeframe
  - Dos columnas: XGBoost 🌲 | LSTM 🧠
  - F1 score con ⭐ para el ganador
  - Barras de progreso del F1 (azul para XGB, morado para LSTM)
  - Métricas específicas: std/folds (XGB), hidden_size/layers (LSTM)
  - Número de features, tamaño de archivo, fecha de entrenamiento
- Maneja el caso de modelos ausentes ("No entrenado")
- Datos cacheados en `_modelsData` para filtros sin recarga

### Cómo usarlo

1. Entrenar modelos: `python -m src.train`
2. Abrir dashboard → pestaña 🧠 Modelos
3. Revisar qué modelo domina por par/TF
4. Si hay diferencias claras, considerar pesos dinámicos del ensemble por par/TF

---

## 17. Deploy y Servicios systemd

### Script de deploy (deploy/deploy.sh)

```bash
./deploy/deploy.sh              # deploy completo (rsync + reinicio todos los servicios)
./deploy/deploy.sh --no-restart # solo sincronizar código
./deploy/deploy.sh --services   # solo reiniciar servicios
./deploy/deploy.sh --install    # primera instalación systemd
```

### Instalación de servicios (una sola vez)

```bash
# En el servidor:
cd ~/ml-ayram
sudo bash deploy/systemd/install.sh
```

Esto copia 12 unit files (6 services + 5 timers + install.sh), habilita todo para arranque automático e inicia los servicios.

### Servicios daemon (siempre activos)

| Servicio | Función | Restart | Puerto |
|---|---|---|---|
| `ayram-dashboard` | FastAPI + Uvicorn | always (10s) | 8000 |
| `ayram-signals` | Generador señales (bucle 60s) | always (30s) | — |

**Importante:** ambos tienen `Restart=always` + `WantedBy=multi-user.target`, por lo que sobreviven reinicios del servidor automáticamente.

### Timers (tareas programadas)

| Timer | Frecuencia | Servicio que ejecuta |
|---|---|---|
| `ayram-collector.timer` | Cada 15 min | Descarga OHLCV |
| `ayram-features.timer` | Cada 3 horas | Recalcula features + labels |
| `ayram-anomaly.timer` | Cada 6 horas | 6 checks operativos |
| `ayram-train.timer` | Domingos 02:00 UTC | Reentrenamiento Optuna |
| `ayram-walkforward.timer` | 1er dom/mes 04:00 UTC | WF + health + resumen IA |

**Persistencia:** todos los timers tienen `Persistent=true`, lo que significa que si el servidor estuvo apagado cuando tocaba ejecutar, lo ejecutará al arrancar.

### Orquestación completa

```
┌─── Cada 15 minutos ────────────────────────────┐
│  ayram-collector   → descarga velas EODHD       │
│  ayram-signals     → genera señales (daemon)     │
└──────────────────────────────────────────────────┘

┌─── Cada 3 horas ───────────────────────────────┐
│  ayram-features    → recalcula features + labels │
└──────────────────────────────────────────────────┘

┌─── Cada 6 horas ───────────────────────────────┐
│  ayram-anomaly     → 6 checks operativos         │
└──────────────────────────────────────────────────┘

┌─── Cada domingo 02:00 UTC ─────────────────────┐
│  ayram-train       → reentrenamiento Optuna      │
└──────────────────────────────────────────────────┘

┌─── 1er domingo/mes 04:00 UTC ──────────────────┐
│  ayram-walkforward → walk-forward validation     │
│                    → model_health check          │
│                    → monthly AI summary           │
└──────────────────────────────────────────────────┘

┌─── Siempre activo ─────────────────────────────┐
│  ayram-dashboard   → FastAPI :8000 (auto-restart)│
└──────────────────────────────────────────────────┘
```

### Comandos del día a día

```bash
# Logs en vivo
journalctl -u ayram-dashboard -f
journalctl -u ayram-signals -f
journalctl -u ayram-train -f
journalctl -u ayram-walkforward -f
journalctl -u ayram-anomaly -f

# Estado
systemctl status ayram-dashboard ayram-signals
systemctl list-timers ayram-*

# Reiniciar tras deploy
systemctl restart ayram-dashboard ayram-signals

# Forzar ejecuciones manuales
systemctl start ayram-train.service          # entrenar ahora
systemctl start ayram-anomaly.service        # check anomalías ahora
python -m src.monitoring.model_health --days 30
python -m src.analysis.monthly_summary --last-n-days 30 --prompt

# Actualizar código
cd ~/ml-ayram && git pull
systemctl restart ayram-dashboard ayram-signals
```

---

## 18. Mantenimiento

### Reentrenamiento

Automático: domingos 2am UTC (timer). Manual:
```bash
python -m src.train --optimize --trials 50
```

### Monitoreo

Automático: cada 6h (anomaly_detector). Manual:
```bash
python -m src.monitoring.anomaly_detector
python -m src.monitoring.model_health --days 30
```

### Análisis mensual

Automático: 1er domingo/mes (walkforward encadenado). Manual:
```bash
python -m src.analysis.monthly_summary --last-n-days 30 --prompt
```

El archivo `results/ai_prompt_*.md` se copia y pega en Claude o ChatGPT para obtener análisis estratégico.

### Comparar modelos

Acceder al dashboard → pestaña 🧠 Modelos. Los datos se actualizan automáticamente cuando hay modelos nuevos en `models/saved/`.

### Backup de modelos

```bash
cp -r ~/ml-ayram/models/saved/ ~/backups/models_$(date +%Y%m%d)/
```

### Backtests periódicos

```bash
python -m src.backtest.engine --days 30 --output results/backtest_$(date +%Y%m%d).json
python -m src.backtest.walk_forward --output results/wf_$(date +%Y%m%d).json
```

### Verificar que todo sobrevive un reinicio

```bash
sudo reboot
# Esperar 1-2 min, reconectar:
ssh ayram@206.81.31.156
systemctl status ayram-dashboard ayram-signals
systemctl list-timers ayram-*
# Todos deben aparecer como active
```

---

## 19. Roadmap de Mejoras

Priorizado por impacto en la rentabilidad del sistema. Referencia detallada en `PROMPT_CONTINUIDAD.md`.

### 🔴 Alto impacto — Rentabilidad

| # | Mejora | Descripción |
|---|---|---|
| 1 | **Pesos dinámicos del ensemble** | Ajustar 55/45 automáticamente por par/TF según F1 del backtest |
| 2 | **Confluencia multi-timeframe** | Scoring: señal que coincide en M15+H1+H4 puntúa más alto |
| 3 | **Circuit breaker por drawdown** | Pausar trading si DD acumulado supera umbral |
| 4 | **Walk-forward en pipeline semanal** | Integrar WF en reentrenamiento para validación OOS real |

### 🟡 Impacto medio — Operativa

| # | Mejora | Descripción |
|---|---|---|
| 5 | Equity curve en dashboard | Gráfica de PnL acumulado en el tiempo |
| 6 | Feature importance tracking | Guardar y visualizar importancias XGB por reentrenamiento |
| 7 | Detección de régimen de mercado | Clasificador trending/ranging/volátil |
| 8 | Análisis de slippage | Precio señal vs precio ejecución |
| 9 | Autenticación del dashboard | Login JWT o HTTP Basic |

### 🟢 Nice to have

| # | Mejora | Descripción |
|---|---|---|
| 10 | Model registry con versionado | Cada señal registra versión del modelo, rollback automático |
| 11 | Paper trading mode explícito | Flag sin ejecución real |
| 12 | Correlación entre pares | Check pre-apertura para limitar exposición |
| 13 | Test coverage | Cubrir ensemble, position_manager, anomaly_detector |

---

*Proyecto ML-Ayram | Uso personal | No compartir credenciales*
