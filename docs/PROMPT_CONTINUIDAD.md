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
- **Dashboard:** FastAPI + Uvicorn (puerto 8000) — SPA con 15 secciones, responsive
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
│   ├── schema.sql                  ← Schema PostgreSQL
│   └── signal_config.yaml          ← Configuración centralizada
├── deploy/
│   ├── deploy.sh                   ← Script de deploy rsync + reinicio servicios
│   └── systemd/
│       ├── ayram-collector.service + .timer   (cada 15 min)
│       ├── ayram-dashboard.service            (daemon, Restart=always, :8000)
│       ├── ayram-features.service + .timer    (cada 3h)
│       ├── ayram-signals.service              (daemon, Restart=always)
│       ├── ayram-train.service + .timer       (domingos 02:00 UTC)
│       ├── ayram-walkforward.service + .timer (1er dom/mes 04:00 UTC)
│       ├── ayram-anomaly.service + .timer     (cada 6h)
│       ├── install.sh
│       └── README.md
├── docs/
│   ├── PROMPT_CONTINUIDAD.md       ← Este archivo
│   ├── TUTORIAL_COMPLETO.md        ← Tutorial paso a paso
│   └── COMO_FUNCIONA.md            ← Explicación detallada del sistema
├── logs/                            ← Logs del bot en producción
├── models/
│   └── saved/                       ← Modelos entrenados (.pt, .ubj, _meta.json)
├── results/                         ← Resultados de backtests, health, anomalías, IA
├── scripts/
│   ├── init_db.py                   ← Inicialización de la BD
│   └── test_ctrader_connection.py   ← Test de conexión cTrader
├── src/
│   ├── __init__.py
│   ├── train.py                     ← Orquestador entrenamiento XGB+LSTM
│   ├── analysis/
│   │   └── monthly_summary.py       ← Resumen mensual + prompt para Claude/ChatGPT
│   ├── backtest/
│   │   ├── engine.py                ← Backtesting sobre señales históricas
│   │   └── walk_forward.py          ← Walk-Forward Validation
│   ├── dashboard/
│   │   ├── app.py                   ← FastAPI backend (~1630 líneas, 30+ endpoints)
│   │   └── static/
│   │       └── index.html           ← Frontend SPA (~3000 líneas, 15 secciones)
│   ├── data/
│   │   ├── collector.py             ← Descarga OHLCV desde EODHD
│   │   ├── features.py              ← ~85 features técnicos/temporales
│   │   └── labels.py                ← Triple-Barrier Method
│   ├── execution/
│   │   └── position_manager.py      ← Gestión posiciones simuladas
│   ├── models/
│   │   ├── xgboost_model.py         ← XGBoost + Optuna + MLflow
│   │   ├── lstm_model.py            ← LSTM + Attention (PyTorch)
│   │   └── ensemble.py              ← Votación ponderada XGB+LSTM (55/45 hardcoded)
│   ├── monitoring/
│   │   ├── model_health.py          ← Diagnóstico degradación modelos
│   │   └── anomaly_detector.py      ← 6 checks operativos cada 6h
│   ├── notifications/
│   │   └── telegram.py              ← Bot Telegram
│   ├── signals/
│   │   └── generator.py             ← Generador señales + filtros + persistencia BD
│   ├── trading/
│   │   └── signal_generator.py      ← Señales + gestión riesgo (legacy)
│   └── utils/
├── tests/
│   ├── conftest.py
│   ├── test_backtest_engine.py
│   ├── test_dashboard_api.py
│   ├── test_features.py
│   ├── test_labels.py
│   └── test_signal_generator.py
├── main.py                          ← Punto de entrada principal
├── requirements.txt                 ← Dependencias local
├── requirements.server.txt          ← Dependencias servidor
├── .env / .env.example
├── .gitignore
└── README.md
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

## Dashboard — API completa (src/dashboard/app.py)

### Endpoints principales

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
| `/api/train/status` | GET | Estado del entrenamiento en curso (progreso, modelos, F1s) |
| `/api/backtest/run` | POST | Ejecutar backtest con parámetros custom |
| `/api/backtest/quick-stats` | GET | Stats rápidos del último backtest |
| `/api/models/compare` | GET | **Comparador XGBoost vs LSTM** side-by-side por par/TF |
| `/api/docs-list` | GET | Lista documentación (.md en docs/) |
| `/api/docs-content/{file}` | GET | Contenido raw de un archivo .md |
| `/api/notifications` | GET | Historial de notificaciones Telegram |
| `/api/alert-rules` | GET/POST/PUT/DELETE | CRUD de reglas de alerta personalizadas |
| `/api/alert-rules/test/{id}` | POST | Test de una regla de alerta |
| `/api/config` | GET/POST | Filtros del generador de señales en caliente |

### Frontend — 15 secciones

| Página | Descripción |
|---|---|
| Dashboard | Resumen: 4 KPIs, señales recientes, estado rápido |
| Pipeline | Logs del pipeline de datos/señales |
| Gráfico | Velas OHLCV con lightweight-charts + señales overlay |
| Historial | Tabla paginada de señales con filtros avanzados |
| Métricas | Charts de distribución (Chart.js): confianza, dirección, pares |
| Rendimiento | PnL por trade, equity curve, stats globales |
| Monitor | Frescura de datos: OHLCV, features, señales, modelos |
| Mercado | Correlaciones entre pares y data de mercado |
| Train | Progreso del entrenamiento en vivo, modelos completados, F1s |
| Bot | Configuración del bot: mode, risk, pairs activos |
| Señales | Filtros del generador en caliente (confianza, ADX, sesión) |
| 🎯 Backtest | Motor de backtesting interactivo con KPIs |
| 📚 Docs | Visor de documentación Markdown del proyecto |
| 🔔 Alertas | Historial de notificaciones + CRUD de reglas de alerta |
| 🧠 Modelos | **Comparador XGBoost vs LSTM**: F1 side-by-side, wins, barras |

Responsive: hamburger menu en móvil, grids adaptativos, touch targets 44px, safe-area-inset.

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

## Modelos ML — Detalles

### XGBoost (src/models/xgboost_model.py)
- 5-fold CV estratificado, métrica: F1 weighted
- Optimización Optuna opcional (n_trials configurable)
- Guarda: `models/saved/xgb_{pair}_{tf}_{timestamp}.ubj` + `_meta.json`
- Meta JSON contiene: `cv_f1_mean`, `cv_f1_std`, `cv_f1_folds`, `features`, `label_map`

### LSTM (src/models/lstm_model.py)
- ForexLSTM con Attention, early stopping por val_f1
- Guarda: `models/saved/lstm_{pair}_{tf}_{timestamp}.pt` (torch checkpoint)
- Checkpoint contiene: `model_state`, `model_config` (hidden_size, num_layers), `scaler_*`, `feature_cols`, `metrics` (best_val_f1), `pair`, `timeframe`

### Ensemble (src/models/ensemble.py)
- Votación ponderada: XGB_WEIGHT=0.55, LSTM_WEIGHT=0.45 (actualmente hardcoded)
- Combina probabilidades: `prob = xgb_weight * xgb_prob + lstm_weight * lstm_prob`
- Requiere acuerdo entre ambos modelos para generar señal

## Estado actual del proyecto

**Fecha de última actualización:** 28 Feb 2026

### Checklist de fases

```
FASE 0  — Preparación local           [COMPLETADA]
FASE 1  — Servidor DigitalOcean       [COMPLETADA]
FASE 2  — Base de Datos Supabase      [COMPLETADA]
FASE 3  — Fuente de datos EODHD       [COMPLETADA]
FASE 4  — Dataset                      [EN PROGRESO — descarga histórica]
FASE 5  — Modelos ML                   [CÓDIGO LISTO — pendiente datos reales]
FASE 5b — Backtesting + Walk-Forward   [CÓDIGO LISTO — pendiente datos reales]
FASE 6  — Signal Engine                [CÓDIGO LISTO — pendiente modelos]
FASE 7  — Telegram                     [CÓDIGO LISTO — pendiente configurar bot]
FASE 8  — Paper Trading                [PENDIENTE]
FASE 9  — Live Trading                 [PENDIENTE]
FASE 10 — Dashboard                    [COMPLETADA — 15 secciones, 30+ endpoints]
FASE 11 — Monitoreo + Alertas          [COMPLETADA — health, anomalías, alertas]
FASE 12 — Análisis IA mensual          [COMPLETADA]
FASE 13 — Comparador de Modelos        [COMPLETADA — pestaña 🧠 Modelos]
```

## Roadmap de mejoras (priorizado por impacto)

### 🔴 Alto impacto — Rentabilidad

| # | Mejora | Descripción | Archivos afectados |
|---|---|---|---|
| 1 | **Pesos dinámicos del ensemble** | Ajustar 55/45 automáticamente por par/TF según F1 del backtest tras cada reentrenamiento | `ensemble.py`, `train.py` |
| 2 | **Confluencia multi-timeframe** | Scoring de confluencia: señal que coincide en M15+H1+H4 puntúa más alto | `generator.py`, nuevo `confluence.py` |
| 3 | **Circuit breaker por drawdown** | Pausar trading automáticamente si DD acumulado supera umbral diario/semanal | `position_manager.py` |
| 4 | **Walk-forward en pipeline semanal** | Integrar walk_forward.py en el ciclo de reentrenamiento para validación OOS real | `train.py`, `walk_forward.py` |

### 🟡 Impacto medio — Operativa y confianza

| # | Mejora | Descripción | Archivos afectados |
|---|---|---|---|
| 5 | **Equity curve en dashboard** | Gráfica de PnL acumulado en el tiempo | `app.py`, `index.html` |
| 6 | **Feature importance tracking** | Guardar importancias XGB con cada reentrenamiento, visualizar tendencias | `xgboost_model.py`, `app.py`, `index.html` |
| 7 | **Detección de régimen de mercado** | Clasificador trending/ranging/volátil para ajustar filtros dinámicamente | nuevo `regime.py`, `generator.py` |
| 8 | **Análisis de slippage** | Comparar precio de señal vs precio de ejecución | `position_manager.py`, `app.py` |
| 9 | **Autenticación del dashboard** | Login JWT o HTTP Basic para proteger la API en producción | `app.py` |

### 🟢 Nice to have — Profesionalización

| # | Mejora | Descripción | Archivos afectados |
|---|---|---|---|
| 10 | **Model registry con versionado** | Cada señal registra qué versión del modelo la generó, rollback automático | `ensemble.py`, `generator.py` |
| 11 | **Paper trading mode explícito** | Flag que registra todo sin ejecutar en broker real | `position_manager.py` |
| 12 | **Correlación entre pares** | Check pre-apertura para limitar exposición duplicada a una divisa | `position_manager.py` |
| 13 | **Test coverage** | Cubrir ensemble, position_manager, anomaly_detector | `tests/` |

## Servicios systemd — Referencia rápida

```bash
# Logs en vivo
journalctl -u ayram-dashboard -f
journalctl -u ayram-signals -f
journalctl -u ayram-train -f

# Estado
systemctl status ayram-dashboard ayram-signals
systemctl list-timers ayram-*

# Reiniciar tras deploy
systemctl restart ayram-dashboard ayram-signals

# Forzar ejecuciones manuales
systemctl start ayram-train.service
python -m src.monitoring.model_health --days 30
python -m src.monitoring.anomaly_detector
python -m src.analysis.monthly_summary --last-n-days 30 --prompt
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
- **Dashboard endpoints docs:** `/api/docs-list` (listado) + `/api/docs-content/{filename}` (contenido)
- **Dashboard comparador:** `/api/models/compare` (escanea models/saved/, lee meta.json y .pt)

---

**Al iniciar un nuevo chat, pega este prompt y añade:**

"Continuamos con ML-Ayram. El último paso completado fue [DESCRIPCIÓN]. Necesito ayuda con [TAREA]. El roadmap de mejoras está en el punto [#N]."

---
*ML-Ayram | Proyecto de uso personal | No compartir públicamente*
