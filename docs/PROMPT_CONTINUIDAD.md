# PROMPT DE CONTINUIDAD — ML-AYRAM
# Copia este prompt completo al inicio de cada chat nuevo con Claude

---

Estoy desarrollando un proyecto llamado **ML-Ayram**, un sistema de trading algorítmico con Machine Learning para Forex.

El proyecto está en: `C:\Users\Usuario\Documents\Webs\ML-Ayram\`
Repositorio GitHub: https://github.com/flyxchain/ML-Ayram

## Arquitectura del sistema

- **Python local:** 3.14.3 | **Python servidor (Linux):** 3.12.3
- **Datos:** EODHD API (forex intradía M15/H1/H4/D1) → PostgreSQL (Supabase)
- **Ejecución:** Modo SIMULADO (paper trading local, sin broker demo)
- **ML:** Ensemble XGBoost + LSTM (PyTorch) con votación ponderada 55/45
- **Features:** ~85 features técnicos + temporales + multi-timeframe por vela
- **Etiquetado:** Triple-Barrier Method (López de Prado), TP=1.5xATR, SL=1xATR
- **Señales:** Ensemble con umbral de confianza 72% y acuerdo entre modelos
- **SL/TP:** ATR dinámico (SL=1.5×ATR, TP1=1.5:1 RR, TP2=2.5:1 RR, trailing stop)
- **Notificaciones:** Bot de Telegram
- **Servidor:** DigitalOcean Droplet 206.81.31.156 (Ubuntu 24.04, $12/mes, 2vCPU/2GB)
- **BD:** Supabase PostgreSQL 17.6 (West EU, sin TimescaleDB)
- **Dashboard:** FastAPI + Uvicorn (puerto 8000) — SPA con gráficos, monitor, rendimiento
- **Monitoreo:** model_health.py + anomaly_detector.py (detección degradación continua)
- **Análisis IA:** monthly_summary.py (genera prompts con métricas para análisis con Claude/ChatGPT)
- **Tracking ML:** MLflow
- **Pares:** EURUSD, GBPUSD, USDJPY, EURJPY, XAUUSD
- **Timeframes:** M15, H1, H4, D1 (H4 construido desde H1 por resample)
- **Scheduler:** systemd timers (collector, features, train, walkforward, anomaly)
- **Servicios daemon:** systemd (ayram-dashboard, ayram-signals) — Restart=always
- **Deploy:** Script rsync + systemd (deploy/deploy.sh)
- **Backtesting:** Motor propio + Walk-Forward Validation
- **Configuración:** signal_config.yaml centralizado

## Orquestación completa de servicios

```
Cada 15 min    → ayram-collector.timer   (descarga velas EODHD)
Cada 15 min    → ayram-signals.service   (genera señales, bucle continuo, daemon)
Cada 3 horas   → ayram-features.timer    (recalcula features + labels)
Cada 6 horas   → ayram-anomaly.timer     (6 checks operativos)
Cada domingo   → ayram-train.timer       (reentrenamiento Optuna, 02:00 UTC)
1er dom/mes    → ayram-walkforward.timer (WF + health + resumen IA, 04:00 UTC)
Siempre activo → ayram-dashboard.service (FastAPI en :8000, daemon)
```

Todos los servicios se reinician automáticamente si el servidor se reinicia (systemd enable + Restart=always para daemons).

## ⚠️ Decisiones técnicas importantes

| Tema | Decisión | Motivo |
|---|---|---|
| Fuente de datos | **EODHD API** (no cTrader) | OAuth de cTrader no funcionó; EODHD tiene intradía forex |
| Ejecución de órdenes | **Simulada** (paper trading) | Sin cuenta demo disponible en España sin coste |
| cTrader | Solo credenciales guardadas | Se activa cuando haya demo disponible |
| TensorFlow | ❌ → **PyTorch** | Sin wheel para Python 3.14 |
| pandas-ta | ❌ → librería **`ta`** | pandas-ta usa numba, sin soporte Python 3.14 |
| TimescaleDB | ❌ → PostgreSQL estándar | No disponible en Supabase free tier |
| H4 | Construido desde H1 con resample | EODHD no tiene H4 nativo |
| mlflow | Solo en servidor | pyarrow requiere cmake+MSVC en Windows |
| Dashboard | **FastAPI** (no Netlify estático) | Backend necesario para API de métricas |
| Backtesting | Motor propio + Walk-Forward | Validación OOS sin lookahead bias |
| Monitoreo | model_health + anomaly_detector | Degradación de modelos detectada automáticamente |
| Análisis IA | monthly_summary genera prompts | Análisis mensual asistido por Claude/ChatGPT |

## Estructura de carpetas del proyecto

```
ML-Ayram/
├── config/
│   ├── schema.sql                  ← Schema PostgreSQL ✅
│   └── signal_config.yaml          ← Configuración centralizada ✅
├── deploy/
│   ├── deploy.sh                   ← Script de deploy rsync + reinicio servicios ✅
│   └── systemd/
│       ├── ayram-collector.service  ✅
│       ├── ayram-collector.timer    ✅
│       ├── ayram-dashboard.service  ✅  (daemon, Restart=always)
│       ├── ayram-features.service   ✅
│       ├── ayram-features.timer     ✅
│       ├── ayram-signals.service    ✅  (daemon, Restart=always)
│       ├── ayram-train.service      ✅  (reentrenamiento semanal)
│       ├── ayram-train.timer        ✅  (domingos 02:00 UTC)
│       ├── ayram-walkforward.service ✅ (WF + health + IA mensual)
│       ├── ayram-walkforward.timer  ✅  (1er domingo/mes 04:00 UTC)
│       ├── ayram-anomaly.service    ✅  (6 checks operativos)
│       ├── ayram-anomaly.timer      ✅  (cada 6 horas)
│       ├── install.sh               ✅
│       └── README.md                ✅
├── docs/
│   ├── PROMPT_CONTINUIDAD.md       ← Este archivo
│   ├── TUTORIAL_COMPLETO.md        ← Tutorial paso a paso
│   └── COMO_FUNCIONA.md            ← Explicación detallada del sistema
├── logs/                            ← Logs del bot en producción
├── models/
│   └── saved/                       ← Modelos entrenados (.pt, .ubj)
├── results/                         ← Resultados de backtests, health, anomalías, IA
├── scripts/
│   ├── init_db.py                   ← Inicialización de la BD ✅
│   └── test_ctrader_connection.py   ← Test de conexión cTrader ✅
├── src/
│   ├── __init__.py                  ✅
│   ├── train.py                     ← Orquestador entrenamiento XGB+LSTM ✅
│   ├── analysis/                    ← MÓDULO ANÁLISIS IA
│   │   ├── __init__.py              ✅
│   │   └── monthly_summary.py       ← Resumen mensual + prompt para Claude/ChatGPT ✅
│   ├── backtest/
│   │   ├── __init__.py              ✅
│   │   ├── engine.py                ← Backtesting sobre señales históricas ✅
│   │   └── walk_forward.py          ← Walk-Forward Validation ✅
│   ├── dashboard/
│   │   ├── __init__.py              ✅
│   │   ├── app.py                   ← FastAPI backend (API + SPA) ✅
│   │   └── static/
│   │       └── index.html           ← Frontend SPA del dashboard ✅
│   ├── data/
│   │   ├── __init__.py              ✅
│   │   ├── collector.py             ← Descarga OHLCV desde EODHD ✅
│   │   ├── features.py              ← ~85 features técnicos/temporales ✅
│   │   └── labels.py                ← Triple-Barrier Method ✅
│   ├── execution/
│   │   ├── __init__.py              ✅
│   │   └── position_manager.py      ← Gestión posiciones simuladas ✅
│   ├── models/
│   │   ├── __init__.py              ✅
│   │   ├── xgboost_model.py         ← XGBoost + Optuna + MLflow ✅
│   │   ├── lstm_model.py            ← LSTM + Attention (PyTorch) ✅
│   │   └── ensemble.py              ← Votación ponderada XGB+LSTM ✅
│   ├── monitoring/                  ← MÓDULO MONITOREO
│   │   ├── __init__.py              ✅
│   │   ├── model_health.py          ← Diagnóstico degradación modelos ✅
│   │   └── anomaly_detector.py      ← 6 checks operativos cada 6h ✅
│   ├── notifications/
│   │   ├── __init__.py              ✅
│   │   └── telegram.py              ← Bot Telegram ✅
│   ├── signals/
│   │   ├── __init__.py              ✅
│   │   └── generator.py             ← Generador de señales con filtros ✅
│   └── trading/
│       ├── __init__.py              ✅
│       └── signal_generator.py      ← Señales + gestión riesgo (legacy) ✅
├── main.py                          ← Punto de entrada principal ✅
├── requirements.txt                 ← Dependencias local ✅
├── requirements.server.txt          ← Dependencias servidor ✅
├── .env / .env.example              ✅
├── .gitignore                       ✅
└── README.md                        ✅
```

## Variables de entorno (.env en servidor ~/ml-ayram/.env)

```
EODHD_API_KEY=694d385412e069.56149556
DATABASE_URL=postgresql://postgres:ff6P*Pe*QK_9kaJ@[host_supabase]:5432/postgres
DATABASE_POOL_SIZE=5
DATABASE_MAX_OVERFLOW=10
TELEGRAM_BOT_TOKEN=
TELEGRAM_CHAT_ID=
MLFLOW_TRACKING_URI=http://localhost:5000
MLFLOW_EXPERIMENT_NAME=ml-ayram-forex
BOT_MODE=paper-trading
LOG_LEVEL=INFO
LOG_FILE=logs/bot.log
```

⚠️ Las credenciales de cTrader están guardadas en .env del servidor pero no se usan hasta tener demo.

## Configuración de señales (config/signal_config.yaml)

Parámetros clave centralizados:
- **Confianza mínima:** 72%
- **Confluencia mínima TF:** 2 timeframes alineados
- **Máximo señales simultáneas:** 3
- **Cooldown:** 4 horas entre señales del mismo par
- **Sesiones activas:** London, New York, Overlap
- **Riesgo por trade:** 1.5% del capital
- **Max drawdown diario:** 4% | Semanal: 8%
- **SL:** 1.5 × ATR(14) | **TP1:** 1.5:1 RR | **TP2:** 2.5:1 RR
- **Trailing stop:** activado al alcanzar TP1 (1.0 × ATR)
- **Walk-forward:** 8 períodos OOS de 3 meses, mín accuracy 0.55, mín PF 1.25
- **Reentrenamiento automático:** domingos 2am UTC

## Conexión al servidor

```bash
ssh root@206.81.31.156
su - ayram
cd ~/ml-ayram && source venv/bin/activate
```

## Deploy al servidor

```bash
./deploy/deploy.sh              # deploy completo (rsync + reinicio todos los servicios)
./deploy/deploy.sh --no-restart # solo código
./deploy/deploy.sh --install    # primera instalación systemd
```

## Servicios systemd

### Servicios daemon (siempre activos, auto-reinicio)

| Servicio | Función | Restart | Puerto |
|---|---|---|---|
| `ayram-dashboard` | FastAPI + Uvicorn | always (10s) | 8000 |
| `ayram-signals` | Generador señales (bucle 60s) | always (30s) | — |

### Timers (tareas programadas)

| Timer | Servicio | Frecuencia |
|---|---|---|
| `ayram-collector.timer` | ayram-collector.service | Cada 15 min |
| `ayram-features.timer` | ayram-features.service | Cada 3 horas |
| `ayram-anomaly.timer` | ayram-anomaly.service | Cada 6 horas |
| `ayram-train.timer` | ayram-train.service | Domingos 02:00 UTC |
| `ayram-walkforward.timer` | ayram-walkforward.service | 1er domingo/mes 04:00 UTC |

### Comandos del día a día

```bash
# Logs en vivo
journalctl -u ayram-dashboard -f
journalctl -u ayram-signals -f
journalctl -u ayram-train -f
journalctl -u ayram-anomaly -f

# Estado
systemctl status ayram-dashboard ayram-signals
systemctl list-timers ayram-*

# Reiniciar tras git pull
systemctl restart ayram-dashboard ayram-signals

# Forzar ejecuciones manuales
systemctl start ayram-train.service
python -m src.monitoring.model_health --days 30
python -m src.monitoring.anomaly_detector
python -m src.analysis.monthly_summary --last-n-days 30 --prompt
```

## Dashboard API (FastAPI)

Endpoints en `src/dashboard/app.py`:

| Endpoint | Descripción |
|---|---|
| `GET /` | SPA (index.html) |
| `GET /api/status` | Estado del sistema |
| `GET /api/signals/latest` | Señales recientes |
| `GET /api/signals/history` | Historial paginado con filtros |
| `GET /api/chart/{pair}/{tf}` | Velas OHLCV + señales superpuestas |
| `GET /api/metrics` | Distribución y stats de señales |
| `GET /api/performance` | Rendimiento trades cerrados + equity curve |
| `GET /api/positions` | Posiciones abiertas con PnL flotante |
| `GET /api/monitor` | Frescura de datos OHLCV y features |
| `GET /api/config` | Configuración filtros actual |
| `POST /api/config` | Actualizar filtros en caliente |

Frontend SPA con 7 secciones: Dashboard, Gráfico (lightweight-charts), Historial, Métricas (Chart.js), Rendimiento, Monitor de datos, Configuración.

## Sistema de monitoreo

### model_health.py
- Compara rendimiento actual vs baselines OOS del walk-forward
- Detecta degradación: warning (20%), alert (35%), critical (50%)
- Auto-retrain opcional si degradación crítica
- Genera reporte Telegram + JSON

### anomaly_detector.py (cada 6h)
6 checks automáticos:
1. Signal Drought — sin señales >5 días
2. Drawdown — DD >8% últimos 7 días
3. Recent Win Rate — WR <35% últimos 20 trades
4. Stale Data — OHLCV >2h sin actualizar
5. Stale Models — modelos >14 días sin reentrenar
6. Anomalous Signals — >30 señales/24h o sesgo >90%

### monthly_summary.py (1er domingo/mes)
- Genera JSON con métricas globales + por par + por TF + semanal
- Crea prompt optimizado para Claude/ChatGPT con análisis estratégico
- Output: `results/summary_LABEL.json` + `results/ai_prompt_LABEL.md`

## Estado actual del proyecto

**Fecha de última actualización:** Feb 2026

### Checklist

```
FASE 0 — Preparación local           [COMPLETADA]
FASE 1 — Servidor DigitalOcean       [COMPLETADA]
FASE 2 — Base de Datos Supabase      [COMPLETADA]
FASE 3 — Fuente de datos EODHD       [COMPLETADA]
FASE 4 — Dataset                      [EN PROGRESO — descarga histórica]
FASE 5 — Modelos ML                   [CÓDIGO LISTO — pendiente datos reales]
FASE 5b — Backtesting + Walk-Forward  [CÓDIGO LISTO — pendiente datos reales]
FASE 6 — Signal Engine                [CÓDIGO LISTO — pendiente modelos]
FASE 7 — Telegram                     [CÓDIGO LISTO — pendiente configurar bot]
FASE 8 — Paper Trading                [PENDIENTE]
FASE 9 — Live Trading                 [PENDIENTE]
FASE 10 — Dashboard                   [COMPLETADA]
FASE 11 — Monitoreo + Alertas         [COMPLETADA]
FASE 12 — Análisis IA mensual         [COMPLETADA]
```

## Notas técnicas

- **Nunca commitear .env** con credenciales
- **Modelos grandes (.pt, .ubj)** en .gitignore
- **H4** se construye con `resample_h4()` desde H1
- **Paper trading:** PnL calculado matemáticamente
- **Servidor Python 3.12.3 / Local Python 3.14.3** — compatible ambas versiones
- **CUDA 12.8** disponible en servidor (torch con +cu128)
- **Todos los servicios sobreviven reinicios** del servidor (systemd enable + Restart=always)
- **deploy.sh** sincroniza código y reinicia todos los servicios y timers automáticamente

---

**Al iniciar un nuevo chat, pega este prompt y añade:**

"Continuamos desde [FASE X]. El último paso completado fue [DESCRIPCIÓN]. Necesito ayuda con [TAREA]."

---
*ML-Ayram | Proyecto de uso personal | No compartir públicamente*
