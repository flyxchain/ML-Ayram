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
- **Dashboard:** FastAPI + Uvicorn (servido desde el servidor, puerto 8000)
- **Tracking ML:** MLflow
- **Pares:** EURUSD, GBPUSD, USDJPY, EURJPY, XAUUSD
- **Timeframes:** M15, H1, H4, D1 (H4 construido desde H1 por resample)
- **Scheduler:** APScheduler / systemd timers
- **Servicios:** systemd en el servidor Linux (ayram-dashboard, ayram-signals, ayram-collector, ayram-features)
- **Deploy:** Script rsync + systemd (deploy/deploy.sh)
- **Backtesting:** Motor propio + Walk-Forward Validation
- **Configuración:** signal_config.yaml centralizado

## ⚠️ Decisiones técnicas importantes

| Tema | Decisión | Motivo |
|---|---|---|
| Fuente de datos | **EODHD API** (no cTrader) | OAuth de cTrader no funcionó; EODHD tiene intradía forex M15/H1/D1 |
| Ejecución de órdenes | **Simulada** (paper trading) | Sin cuenta demo disponible en España sin coste |
| cTrader | Solo credenciales guardadas | Se activa cuando haya demo disponible |
| TensorFlow | ❌ → **PyTorch** | Sin wheel para Python 3.14 |
| pandas-ta | ❌ → librería **`ta`** | pandas-ta usa numba, sin soporte Python 3.14 |
| TimescaleDB | ❌ → PostgreSQL estándar | No disponible en Supabase free tier |
| H4 | Construido desde H1 con resample | EODHD no tiene H4 nativo |
| mlflow | Solo en servidor | pyarrow requiere cmake+MSVC en Windows |
| Dashboard | **FastAPI** (no Netlify estático) | Backend necesario para API de métricas y configuración dinámica |
| Backtesting | Motor propio + Walk-Forward | Validación OOS sin lookahead bias |

## Estructura de carpetas del proyecto

```
ML-Ayram/
├── config/
│   ├── schema.sql                  ← Schema PostgreSQL (sin TimescaleDB) ✅
│   └── signal_config.yaml          ← Configuración centralizada de señales/riesgo/modelos ✅
├── deploy/
│   ├── deploy.sh                   ← Script de deploy rsync + reinicio servicios ✅
│   └── systemd/
│       ├── ayram-collector.service  ✅
│       ├── ayram-collector.timer    ✅
│       ├── ayram-dashboard.service  ✅
│       ├── ayram-features.service   ✅
│       ├── ayram-features.timer     ✅
│       ├── ayram-signals.service    ✅
│       ├── install.sh               ✅
│       └── README.md                ✅
├── docs/
│   ├── PROMPT_CONTINUIDAD.md       ← Este archivo
│   └── TUTORIAL_COMPLETO.md        ← Tutorial paso a paso
├── logs/                            ← Logs del bot en producción
├── models/
│   └── saved/                       ← Modelos entrenados (.pt, .ubj)
├── results/                         ← Resultados de backtests y walk-forward
├── scripts/
│   ├── init_db.py                   ← Inicialización de la BD ✅
│   └── test_ctrader_connection.py   ← Test de conexión cTrader ✅
├── src/
│   ├── __init__.py                  ✅
│   ├── train.py                     ← Orquestador entrenamiento XGB+LSTM ✅
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
│   ├── notifications/
│   │   ├── __init__.py              ✅
│   │   └── telegram.py              ← Bot Telegram (señales/errores/heartbeat) ✅
│   ├── signals/
│   │   ├── __init__.py              ✅
│   │   └── generator.py             ← Generador de señales con filtros ✅
│   └── trading/
│       ├── __init__.py              ✅
│       └── signal_generator.py      ← Señales + gestión riesgo + BD (legacy) ✅
├── main.py                          ← Punto de entrada principal ✅
├── requirements.txt                 ← Dependencias local (sin mlflow) ✅
├── requirements.server.txt          ← Dependencias servidor (con mlflow) ✅
├── .env / .env.example              ✅
├── .gitignore                       ✅
└── README.md                        ✅
```

## Variables de entorno (.env en servidor ~/ml-ayram/.env)

```
# ── cTrader Open API ─────────────────────────────────────
CTRADER_CLIENT_ID=21838_G2BlJoy7B8vs4AkVWXaWdcojrLIQGyf83GmJ60cmgigH5uUkug
CTRADER_CLIENT_SECRET=53hsPolaU5L1QlKaY1FROVcLeXcnjReQFIc1iIspnQ7My7jE4O
CTRADER_ACCOUNT_ID=2016020
CTRADER_ENV=live
CTRADER_HOST_DEMO=demo.ctraderapi.com
CTRADER_HOST_LIVE=live.ctraderapi.com
CTRADER_PORT=5035

# ── EODHD ────────────────────────────────────────────────
EODHD_API_KEY=694d385412e069.56149556

# ── Base de datos ────────────────────────────────────────
DATABASE_URL=postgresql://postgres:ff6P*Pe*QK_9kaJ@[host_supabase]:5432/postgres
DATABASE_POOL_SIZE=5
DATABASE_MAX_OVERFLOW=10

# ── Telegram Bot ─────────────────────────────────────────
TELEGRAM_BOT_TOKEN=
TELEGRAM_CHAT_ID=

# ── MLflow ───────────────────────────────────────────────
MLFLOW_TRACKING_URI=http://localhost:5000
MLFLOW_EXPERIMENT_NAME=ml-ayram-forex

# ── Configuración del bot ────────────────────────────────
BOT_MODE=paper-trading
LOG_LEVEL=INFO
LOG_FILE=logs/bot.log

# ── APIs externas ────────────────────────────────────────
FRED_API_KEY=
```

⚠️ Cambiar contraseña de BD después de este chat.

## Configuración de señales (config/signal_config.yaml)

Parámetros clave centralizados:
- **Confianza mínima:** 72%
- **Confluencia mínima TF:** 2 timeframes alineados
- **Máximo señales simultáneas:** 3
- **Cooldown:** 4 horas entre señales del mismo par
- **Sesiones activas:** London, New York, Overlap
- **Riesgo por trade:** 1.5% del capital
- **Max drawdown diario:** 4% | Semanal: 8%
- **Kelly fraction:** 25%
- **SL:** 1.5 × ATR(14)
- **TP1:** 1.5:1 RR (cierre 50% posición) | **TP2:** 2.5:1 RR
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
# Desde local (sincroniza + reinicia servicios):
./deploy/deploy.sh

# Solo código sin reiniciar:
./deploy/deploy.sh --no-restart

# Primera instalación de systemd:
./deploy/deploy.sh --install
```

## Servicios systemd

| Servicio | Función | Estado |
|---|---|---|
| `ayram-dashboard` | FastAPI + Uvicorn (puerto 8000) | Activo |
| `ayram-signals` | Generador de señales (bucle 60s) | Desactivado (espera modelos) |
| `ayram-collector` | Descarga de datos + timer | Activo |
| `ayram-features` | Cálculo de features + timer | Activo |

```bash
# Ver logs
journalctl -u ayram-dashboard -f
journalctl -u ayram-signals -f

# Estado
systemctl status ayram-dashboard ayram-signals

# Reiniciar tras git pull
systemctl restart ayram-dashboard ayram-signals
```

## Estado actual del proyecto

**Fecha de última actualización:** Feb 2026
**Fase actual:** FASE 4 — Dataset (descarga histórica en progreso)

### Checklist

```
FASE 0 — Preparación local
[X] Git, Python 3.14.3, VSCode instalados
[X] venv local creado, dependencias instaladas
[X] Repositorio GitHub privado creado y primer commit

FASE 1 — Servidor
[X] DigitalOcean Droplet (206.81.31.156, Frankfurt, Ubuntu 24.04)
[X] SSH configurado, usuario ayram creado
[X] Firewall UFW (SSH + 443)
[X] Python 3.12.3 + venv + requirements.server.txt instalado
[X] Repo clonado en ~/ml-ayram
[X] Servicios systemd configurados (deploy/systemd/)
[X] Script de deploy creado (deploy/deploy.sh)

FASE 2 — Base de Datos
[X] Supabase PostgreSQL 17.6 (West EU)
[X] IPv6 habilitado en Droplet (Supabase usa IPv6)
[X] Schema ejecutado (sin TimescaleDB)
[X] .env configurado en servidor
[X] script init_db.py creado

FASE 3 — Fuente de datos
[X] EODHD API key verificada (intradía forex OK)
[X] collector.py creado y probado (200 velas H1 EURUSD OK)
[ ] Descarga histórica completa (en progreso)

FASE 4 — Dataset
[ ] Descarga histórica completada (3 años, 5 pares)
[ ] features.py ejecutado en servidor
[ ] labels.py ejecutado en servidor

FASE 5 — Modelo
[X] xgboost_model.py creado (XGBoost + Optuna + MLflow)
[X] lstm_model.py creado (LSTM + Attention, PyTorch)
[X] ensemble.py creado (votación ponderada 55/45)
[X] src/train.py creado (orquestador de entrenamiento)
[ ] XGBoost entrenado con datos reales (CV F1 > 0.55)
[ ] LSTM entrenado con datos reales
[ ] Ensemble validado con datos reales

FASE 5b — Backtesting
[X] backtest/engine.py creado (backtesting sobre señales BD)
[X] backtest/walk_forward.py creado (Walk-Forward Validation)
[ ] Walk-forward ejecutado con resultados satisfactorios

FASE 6 — Signal Engine
[X] src/signals/generator.py creado (señales con filtros)
[X] src/execution/position_manager.py creado (gestión posiciones)
[X] config/signal_config.yaml creado (configuración centralizada)
[ ] signal engine probado end-to-end con modelos reales

FASE 7 — Telegram
[X] src/notifications/telegram.py creado (señales/errores/heartbeat)
[ ] Bot de Telegram configurado y probado

FASE 8 — Paper Trading
[ ] 4 semanas con métricas OK

FASE 9 — Live Trading
[ ] Activado (cuando haya cuenta demo disponible)

FASE 10 — Dashboard
[X] src/dashboard/app.py creado (FastAPI backend con API completa)
[X] src/dashboard/static/index.html creado (SPA frontend)
[X] ayram-dashboard.service configurado en systemd
[ ] Desplegado y accesible públicamente
```

## Dependencias principales

**requirements.txt (local — sin mlflow):**
```
pandas>=2.1.0
numpy>=2.0.0
ta>=0.11.0
scikit-learn>=1.4.0
xgboost>=2.0.0
torch>=2.2.0
optuna>=3.4.0
joblib>=1.3.0
SQLAlchemy>=2.0.0
psycopg2-binary>=2.9.0
fastapi>=0.110.0
uvicorn>=0.27.0
pydantic>=2.0.0
requests>=2.31.0
python-dotenv>=1.0.0
loguru>=0.7.0
# Dev/análisis (solo local):
jupyter>=1.0.0
matplotlib>=3.8.0
seaborn>=0.13.0
plotly>=5.18.0
```

**requirements.server.txt (servidor — con mlflow, sin dev tools):**
```
(todo lo anterior excepto jupyter/matplotlib/seaborn/plotly) + mlflow>=2.9.0
```

## Dashboard API (FastAPI)

Endpoints disponibles en `src/dashboard/app.py`:
- `GET /` → SPA (index.html)
- `GET /api/status` → estado del sistema
- `GET /api/signals/latest` → señales recientes
- `GET /api/signals/history` → historial paginado
- `GET /api/chart/{pair}/{tf}` → velas OHLCV + señales
- `GET /api/metrics` → distribución y stats
- `GET /api/performance` → rendimiento trades cerrados
- `GET /api/positions` → posiciones abiertas
- `GET /api/config` → configuración actual
- `POST /api/config` → actualizar filtros

```bash
# Arrancar localmente:
uvicorn src.dashboard.app:app --host 0.0.0.0 --port 8000 --workers 1
```

## Notas técnicas

- **Nunca commitear .env** con credenciales
- **Modelos grandes (.pt, .ubj)** en .gitignore
- **H4** se construye con `resample_h4()` desde H1 en collector.py
- **Paper trading:** PnL calculado matemáticamente en position_manager.py
- **Switch live:** cuando haya demo, cambiar `CTRADER_ENV=demo` y activar ctrader_client.py
- **Servidor Python 3.12.3 / Local Python 3.14.3** — código compatible con ambas versiones
- El servidor tiene soporte CUDA 12.8 (torch instalado con +cu128)
- **Backtesting** incluye simulación de spread y slippage
- **Walk-Forward** soporta ventana rolling y expanding
- **signal_config.yaml** centraliza todos los parámetros ajustables del bot

---

**Al iniciar un nuevo chat, pega este prompt y añade:**

"Continuamos desde [FASE X]. El último paso completado fue [DESCRIPCIÓN]. Necesito ayuda con [TAREA]."

---
*ML-Ayram | Proyecto de uso personal | No compartir públicamente*
