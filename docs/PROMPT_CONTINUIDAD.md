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
- **Etiquetado:** Triple-Barrier Method (López de Prado), TP=2xATR, SL=1xATR
- **Señales:** Ensemble con umbral de confianza y acuerdo entre modelos
- **SL/TP:** ATR dinámico (SL=1×ATR, TP=2×ATR, ratio 1:2)
- **Notificaciones:** Bot de Telegram
- **Servidor:** DigitalOcean Droplet 206.81.31.156 (Ubuntu 24.04, $12/mes, 2vCPU/2GB)
- **BD:** Supabase PostgreSQL 17.6 (West EU, sin TimescaleDB)
- **Dashboard:** Netlify (HTML estático)
- **Tracking ML:** MLflow
- **Pares:** EURUSD, GBPUSD, USDJPY, EURJPY, XAUUSD
- **Timeframes:** M15, H1, H4, D1 (H4 construido desde H1 por resample)
- **Scheduler:** APScheduler
- **Servicio:** systemd en el servidor Linux

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

## Estructura de carpetas del proyecto

```
ML-Ayram/
├── docs/
│   ├── TUTORIAL_COMPLETO.md
│   └── PROMPT_CONTINUIDAD.md       ← Este archivo
├── src/
│   ├── data/
│   │   ├── collector.py            ← Descarga OHLCV desde EODHD ✅
│   │   ├── features.py             ← ~85 features técnicos/temporales ✅
│   │   └── labels.py               ← Triple-Barrier Method ✅
│   ├── models/
│   │   ├── xgboost_model.py        ← XGBoost + Optuna + MLflow ✅
│   │   ├── lstm_model.py           ← LSTM + Attention (PyTorch) ✅
│   │   └── ensemble.py             ← Votación ponderada XGB+LSTM ✅
│   └── trading/
│       ├── __init__.py             ✅
│       └── signal_generator.py     ← Señales + gestión riesgo + BD ✅
├── config/
│   └── schema.sql                  ← Schema PostgreSQL (sin TimescaleDB) ✅
├── models/
│   └── saved/                      ← Modelos entrenados (.pt, .ubj)
├── logs/                           ← Logs del bot en producción
├── requirements.txt                ← Dependencias local (sin mlflow) ✅
├── requirements.server.txt         ← Dependencias servidor (con mlflow) ✅
└── .env / .env.example
```

**Módulos pendientes de crear:**
- `src/bot.py` — Scheduler principal (APScheduler)
- `src/notifications/telegram_bot.py` — Bot Telegram
- `src/training/train_pipeline.py` — Pipeline entrenamiento completo
- `main.py` — Punto de entrada

## Variables de entorno (.env en servidor ~/ml-ayram/.env)

```
EODHD_API_KEY=694d385412e069.56149556

CTRADER_CLIENT_ID=21838_G2BlJoy7B8vs4AkVWXaWdcojrLIQGyf83GmJ60cmgigH5uUkug
CTRADER_CLIENT_SECRET=53hsPolaU5L1QlKaY1FROVcLeXcnjReQFIc1iIspnQ7My7jE4O
CTRADER_ACCOUNT_ID=2016020
CTRADER_ENV=live

DATABASE_URL=postgresql://postgres:ff6P*Pe*QK_9kaJ@[host_supabase]:5432/postgres

TELEGRAM_BOT_TOKEN=
TELEGRAM_CHAT_ID=

MLFLOW_TRACKING_URI=http://localhost:5000
```

⚠️ Cambiar contraseña de BD después de este chat.

## Conexión al servidor

```bash
ssh root@206.81.31.156
su - ayram
cd ~/ml-ayram && source venv/bin/activate
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

FASE 2 — Base de Datos
[X] Supabase PostgreSQL 17.6 (West EU)
[X] IPv6 habilitado en Droplet (Supabase usa IPv6)
[X] Schema ejecutado (sin TimescaleDB)
[X] .env configurado en servidor

FASE 3 — Fuente de datos
[X] EODHD API key verificada (intradía forex OK)
[X] collector.py creado y probado (200 velas H1 EURUSD OK)
[ ] Descarga histórica completa (en progreso, PID 3405)

FASE 4 — Dataset
[ ] Descarga histórica completada (3 años, 5 pares)
[ ] features.py ejecutado en servidor
[ ] labels.py ejecutado en servidor

FASE 5 — Modelo
[ ] XGBoost entrenado (CV F1 > 0.55)
[ ] LSTM entrenado
[ ] Ensemble validado

FASE 6 — Signal Engine
[ ] signal_generator.py probado end-to-end

FASE 7 — Telegram
[ ] Bot creado y recibiendo mensajes

FASE 8 — Paper Trading
[ ] 4 semanas con métricas OK

FASE 9 — Live Trading
[ ] Activado (cuando haya cuenta demo disponible)

FASE 10 — Dashboard
[ ] Desplegado en Netlify
```

## Dependencias principales

**requirements.txt (local — sin mlflow):**
```
pandas>=2.1.0
ta>=0.11.0
numpy>=2.0.0
scikit-learn>=1.4.0
xgboost>=2.0.0
torch>=2.2.0
optuna>=3.4.0
SQLAlchemy>=2.0.0
psycopg2-binary>=2.9.0
APScheduler>=3.10.0
python-dotenv>=1.0.0
loguru>=0.7.0
requests>=2.31.0
python-telegram-bot>=21.0.0
# mlflow     ← solo en servidor
```

**requirements.server.txt (servidor — con mlflow):**
```
(todo lo anterior) + mlflow>=3.2.0
```

## Notas técnicas

- **Nunca commitear .env** con credenciales
- **Modelos grandes (.pt, .ubj)** en .gitignore
- **H4** se construye con `resample_h4()` desde H1 en collector.py
- **Paper trading:** PnL calculado matemáticamente en signal_generator.py
- **Switch live:** cuando haya demo, cambiar `CTRADER_ENV=demo` y activar ctrader_client.py
- **Servidor Python 3.12.3 / Local Python 3.14.3** — código compatible con ambas versiones
- El servidor tiene soporte CUDA 12.8 (torch instalado con +cu128)

---

**Al iniciar un nuevo chat, pega este prompt y añade:**

"Continuamos desde [FASE X]. El último paso completado fue [DESCRIPCIÓN]. Necesito ayuda con [TAREA]."

---
*ML-Ayram | Proyecto de uso personal | No compartir públicamente*
