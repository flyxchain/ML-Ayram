# PROMPT DE CONTINUIDAD — ML-AYRAM
# Copia este prompt completo al inicio de cada chat nuevo con Claude

---

Estoy desarrollando un proyecto llamado **ML-Ayram**, un sistema de trading algorítmico con Machine Learning para Forex.

El proyecto está en: `C:\Users\Usuario\Documents\Webs\ML-Ayram\`

## Arquitectura del sistema

- **Python local:** 3.14.3 | **Python servidor (Linux):** 3.11 (más estable para librerías ML)
- **Datos:** cTrader Open API (Protobuf/TCP) → PostgreSQL + TimescaleDB
- **ML:** Ensemble XGBoost (clasificador de régimen) + LSTM bidireccional (predictor de dirección)
- **Framework LSTM:** PyTorch (en lugar de TensorFlow — sin wheel oficial para Python 3.14)
- **Features:** ~85 features técnicos + temporales + multi-timeframe por vela
- **Etiquetado:** Triple-Barrier Method (López de Prado)
- **Señales:** Motor de confluencia multi-TF (M15/H1/H4/D1), umbral confianza >72%
- **SL/TP:** ATR dinámico (SL=1.5×ATR, TP1=1.5RR, TP2=2.5RR, trailing en TP1)
- **Notificaciones:** Bot de Telegram (python-telegram-bot)
- **Ejecución:** Auto-trader vía cTrader Open API
- **Servidor:** DigitalOcean Droplet o GCP e2-medium
- **BD:** Supabase free (PostgreSQL) o DO Managed PostgreSQL
- **Dashboard:** Netlify (HTML estático, lectura de métricas)
- **Tracking ML:** MLflow
- **Pares:** EURUSD, GBPUSD, USDJPY, EURJPY, XAUUSD
- **Scheduler:** APScheduler (dentro del proceso Python)
- **Servicio:** systemd en el servidor Linux

## ⚠️ Nota de compatibilidad — Python 3.14.3

Python 3.14 es muy reciente. Impacto en librerías:

| Librería | Estado en Python 3.14 | Solución |
|---|---|---|
| TensorFlow | ❌ Sin wheel oficial | Usar **PyTorch** para LSTM |
| PyTorch | ✅ Soportado (wheel disponible) | Usar esto para LSTM |
| XGBoost | ✅ Soportado | Sin cambios |
| scikit-learn | ✅ Soportado | Sin cambios |
| pandas / numpy | ✅ Soportados | Sin cambios |
| pandas-ta | ❌ Usa numba, sin soporte 3.14 | Sustituida por librería **`ta`** |
| ctrader-open-api | ⚠️ Por verificar | Probar al instalar |

**Estrategia dual:** Desarrollo local en Python 3.14.3. Servidor de producción en Python 3.11 (estable con todas las librerías ML). El código está escrito para ser compatible con ambas versiones.

## Estructura de carpetas del proyecto

```
ML-Ayram/
├── docs/
│   ├── TUTORIAL_COMPLETO.md       ← Tutorial paso a paso completo
│   ├── PROMPT_CONTINUIDAD.md      ← Este archivo
│   └── ARCHITECTURE_DIAGRAM.html ← Diagrama visual de la arquitectura
├── src/
│   ├── data/
│   │   ├── collector.py           ← Descarga datos de cTrader
│   │   ├── features.py            ← Cálculo de ~85 features
│   │   └── labeler.py             ← Triple-Barrier labeling
│   ├── models/
│   │   ├── regime_classifier.py   ← XGBoost para régimen de mercado
│   │   ├── lstm_predictor.py      ← LSTM con PyTorch para dirección
│   │   ├── ensemble.py            ← Combinación de modelos
│   │   └── trainer.py             ← Pipeline de entrenamiento + walk-forward
│   ├── signals/
│   │   ├── engine.py              ← Motor de señales con confluencia multi-TF
│   │   └── risk_manager.py        ← Gestión de riesgo y tamaño de posición
│   ├── execution/
│   │   └── ctrader_client.py      ← Conexión y órdenes en cTrader
│   ├── notifications/
│   │   └── telegram_bot.py        ← Bot de Telegram
│   └── dashboard/
│       └── index.html             ← Dashboard estático para Netlify
├── scripts/
│   ├── test_ctrader_connection.py ← Test de conexión a cTrader
│   ├── retrain_model.py           ← Forzar reentrenamiento
│   └── performance_report.py     ← Informe de rendimiento
├── config/
│   ├── schema.sql                 ← Schema de la base de datos
│   ├── signal_config.yaml         ← Parámetros del motor de señales
│   └── settings.py                ← Configuración general del proyecto
├── models/
│   └── saved/                     ← Modelos entrenados (.pt, .pkl)
├── logs/                          ← Logs del bot en producción
├── tests/                         ← Tests unitarios
├── main.py                        ← Punto de entrada principal
├── requirements.txt               ← Dependencias Python
├── .env.example                   ← Plantilla de variables de entorno
├── .gitignore                     ← Ignora .env y modelos grandes
└── README.md                      ← Descripción general del proyecto
```

## Variables de entorno necesarias (.env)

```
CTRADER_CLIENT_ID=
CTRADER_CLIENT_SECRET=
CTRADER_ACCOUNT_ID=
CTRADER_ENV=demo   # o 'live'

DATABASE_URL=postgresql://user:pass@host:5432/dbname

TELEGRAM_BOT_TOKEN=
TELEGRAM_CHAT_ID=

MLFLOW_TRACKING_URI=http://localhost:5000
```

## Estado actual del proyecto

<!-- ACTUALIZA ESTA SECCIÓN EN CADA SESIÓN -->

**Fecha de última actualización:** Feb 2026  
**Fase actual:** FASE 0 — Preparación local (en progreso)  
**Último paso completado:** Git, Python 3.14.3 y VSCode instalados en Windows  
**Próximo paso a hacer:** Crear repo GitHub privado + venv local + verificar instalación de dependencias

### Checklist rápido (marca con X lo que está hecho):

```
FASE 0 — Preparación local
[X] Git instalado
[X] Python 3.14.3 instalado en Windows
[X] VSCode instalado con extensiones
[ ] Repositorio GitHub privado creado
[X] venv local creado y dependencias instaladas (Python 3.14.3 OK) (verificar compatibilidad 3.14)

FASE 1 — Servidor
[ ] Cuenta DO/GCP creada con créditos
[ ] Servidor creado y accesible por SSH
[ ] Python 3.11 + venv en el servidor (nota: servidor usa 3.11, no 3.14)

FASE 2 — Base de Datos
[ ] PostgreSQL accesible
[ ] Schema ejecutado

FASE 3 — cTrader
[ ] Credenciales API obtenidas
[ ] Test de conexión OK

FASE 4 — Dataset
[ ] Datos históricos descargados
[ ] Features calculados
[ ] Labels generados

FASE 5 — Modelo
[ ] XGBoost entrenado (OOS >58%)
[ ] LSTM (PyTorch) entrenado
[ ] Walk-forward validation OK

FASE 6 — Signal Engine
[ ] Motor configurado y testeado

FASE 7 — Telegram
[ ] Bot creado y recibiendo mensajes

FASE 8 — Paper Trading
[ ] 4 semanas completadas con métricas OK

FASE 9 — Live Trading
[ ] Auto-trader en DEMO funcionando
[ ] Servicio systemd activo

FASE 10 — Dashboard
[ ] Desplegado en Netlify
```

## Notas técnicas importantes

- **Python 3.14.3 local / Python 3.11 servidor** — estrategia dual por compatibilidad
- **PyTorch en lugar de TensorFlow** para la LSTM (TF no tiene wheel para Python 3.14)
- **Siempre desarrollar en cTrader DEMO** antes de tocar cuenta real
- El modelo se **reentrena automáticamente** el primer domingo de cada mes con APScheduler
- El walk-forward validation usa **expanding window** (no rolling) para evitar overfitting
- El etiquetado **Triple-Barrier** está implementado en `src/data/labeler.py`
- El bot de Telegram tiene **kill-switch** con el comando `/close_all`
- Los modelos guardados usan extensión **.pt** (PyTorch) en lugar de .h5 (TensorFlow)
- Los modelos grandes (.pt, .pkl) están en **.gitignore** — se regeneran
- **Nunca commitear el archivo .env** con credenciales reales
- El servidor usa **systemd** para arrancar el bot automáticamente tras reinicios
- Los logs del bot están en `~/ml-ayram/logs/bot.log` en el servidor

## Dependencias principales (requirements.txt)

```
# Nota: instaladas en Python 3.14.3 local y Python 3.11 en servidor

ctrader-open-api==0.9.2       # Última versión estable (la librería nunca llegó a 2.x)
pandas>=2.1.0
ta>=0.11.0                    # Reemplaza pandas-ta (pandas-ta usa numba, sin soporte Python 3.14)
numpy>=2.0.0                  # NumPy 2.x compatible con Python 3.14
scikit-learn>=1.4.0
xgboost>=2.0.0
torch>=2.2.0                  # PyTorch (en lugar de TensorFlow)
optuna>=3.4.0
mlflow>=2.9.0
python-telegram-bot>=21.0.0
SQLAlchemy>=2.0.0
psycopg2-binary>=2.9.0
APScheduler>=3.10.0
python-dotenv>=1.0.0
pyyaml>=6.0.1
scipy>=1.11.0
joblib>=1.3.0
loguru>=0.7.0
rich>=13.0.0
httpx>=0.26.0
```

---

**Cuando empieces un nuevo chat, pega este prompt y añade al final:**

"Continuamos desde [FASE X — NOMBRE]. El último paso completado fue [DESCRIPCIÓN]. Necesito ayuda con [TAREA ESPECÍFICA]."

**Ejemplo:**
"Continuamos desde FASE 1 — Servidor. El último paso completado fue la creación del Droplet en DigitalOcean. Necesito ayuda con la configuración inicial del servidor."

---

*ML-Ayram | Proyecto de uso personal | No compartir públicamente*
