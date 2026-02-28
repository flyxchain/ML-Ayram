# ML-Ayram — Cómo Funciona Todo (Explicación Detallada)

> Este documento explica con detalle cómo funciona el sistema ML-Ayram, paso a paso,
> en un lenguaje que pueda entender cualquier persona. No hace falta saber programar
> para seguirlo — es una guía de "qué hace cada pieza y por qué".

---

## ¿Qué es ML-Ayram?

ML-Ayram es un bot de trading para el mercado de divisas (Forex). En lugar de que una persona esté mirando gráficos todo el día, el sistema hace lo siguiente de forma automática:

1. **Descarga datos** de precios del mercado cada 15 minutos
2. **Calcula indicadores** técnicos (como los que usan los traders profesionales)
3. **Usa inteligencia artificial** para predecir si un par de divisas va a subir o bajar
4. **Genera señales** de compra (LONG) o venta (SHORT) con niveles de entrada, beneficio y pérdida
5. **Monitorea** que todo funcione bien y avisa si algo falla
6. **Analiza** su propio rendimiento cada mes y propone mejoras

Todo esto funciona en un servidor en la nube (DigitalOcean, en Frankfurt) las 24 horas del día, los 7 días de la semana, sin intervención humana.

---

## Los 5 pares de divisas que vigila

| Par | Qué es | Ejemplo |
|---|---|---|
| EURUSD | Euro vs Dólar americano | "El euro sube frente al dólar" |
| GBPUSD | Libra vs Dólar | "La libra baja frente al dólar" |
| USDJPY | Dólar vs Yen japonés | "El dólar sube frente al yen" |
| EURJPY | Euro vs Yen | "El euro baja frente al yen" |
| XAUUSD | Oro vs Dólar | "El oro sube" |

Cada par se analiza en 4 escalas de tiempo diferentes:

| Timeframe | Significa | Cada vela representa |
|---|---|---|
| M15 | 15 minutos | Un período de 15 minutos |
| H1 | 1 hora | Un período de 1 hora |
| H4 | 4 horas | Un período de 4 horas |
| D1 | 1 día | Un día entero |

---

## El Pipeline Completo: De Datos Brutos a Señales

Imagina una fábrica con varias estaciones de trabajo. Los datos entran por un lado y las señales de trading salen por el otro.

### Estación 1: Descarga de datos (collector.py)

**Qué hace:** Cada 15 minutos, el sistema llama a un servicio externo (EODHD API) y le pide los últimos precios de cada par de divisas. Recibe 4 datos por cada vela: precio de apertura (Open), máximo (High), mínimo (Low) y cierre (Close), más el volumen.

**Analogía:** Es como un termómetro que mide la temperatura cada 15 minutos y apunta el resultado en un cuaderno.

**Dónde se guarda:** En la tabla `ohlcv_raw` de la base de datos PostgreSQL (alojada en Supabase).

**Cuándo se ejecuta:** Automáticamente cada 15 minutos gracias a `ayram-collector.timer`.

---

### Estación 2: Cálculo de indicadores (features.py)

**Qué hace:** Toma los precios brutos y calcula ~85 indicadores técnicos. Estos indicadores son cálculos matemáticos que los traders llevan usando décadas para entender el mercado.

**Algunos ejemplos de lo que calcula:**

| Indicador | Para qué sirve | Analogía |
|---|---|---|
| EMA (Media Exponencial) | Muestra la tendencia del precio | Si la temperatura media de la semana sube, hace más calor |
| RSI | Mide si el precio ha subido "demasiado rápido" | Si llevas corriendo 10 km sin parar, seguramente pares pronto |
| MACD | Detecta cambios de tendencia | Como cuando el viento cambia de dirección |
| ADX | Mide la fuerza de la tendencia | No es lo mismo una brisa que un huracán |
| ATR (Rango Medio Real) | Mide la volatilidad (cuánto se mueve el precio) | Algunos días el precio se mueve 20 pips, otros 100 |
| Bollinger Bands | Banda de "precio normal" | Si la temperatura sale de la media ±2σ, es anómalo |

Además calcula:
- **Features temporales:** hora del día, día de la semana, sesión de mercado (Londres, Nueva York, Tokio)
- **Features multi-timeframe:** qué dice el timeframe superior (si en H4 la tendencia es alcista pero en M15 baja, es información útil)

**Analogía general:** Es como un médico que toma la presión, el pulso, la temperatura, hace analíticas... todo para tener un "perfil completo" del paciente (en este caso, del mercado).

**Dónde se guarda:** En la tabla `features_computed`.

**Cuándo se ejecuta:** Cada 3 horas con `ayram-features.timer`.

---

### Estación 3: Etiquetado histórico (labels.py)

**Qué hace:** Solo se usa durante el entrenamiento. Mira los datos históricos y etiqueta cada vela:
- **+1 (LONG ganador):** el precio subió lo suficiente para tocar el beneficio antes que la pérdida
- **-1 (SHORT ganador):** el precio bajó lo suficiente
- **0 (Neutro):** no pasó nada significativo en las siguientes 20 velas

**Método:** Triple-Barrier Method de Marcos López de Prado (un investigador financiero reconocido). Coloca 3 "barreras" alrededor del precio:
- Barrera superior: Take Profit = 1.5× ATR (beneficio)
- Barrera inferior: Stop Loss = 1.0× ATR (pérdida)
- Barrera temporal: 20 velas máximo (tiempo límite)

La primera barrera que el precio toque determina la etiqueta.

**Analogía:** Es como ver partidos de fútbol ya jugados y anotar "ganó local", "ganó visitante" o "empate" para que la IA aprenda patrones.

---

### Estación 4: Los modelos de IA (xgboost_model.py + lstm_model.py)

Aquí es donde entra la inteligencia artificial. Hay **dos modelos** diferentes que trabajan en equipo:

#### Modelo 1: XGBoost (el analista rápido)

**Qué es:** Un modelo de "árboles de decisión potenciados". Imagina un árbol de preguntas:
- ¿El RSI es mayor que 70? → Sí → ¿El ADX es mayor que 25? → Sí → Probablemente baje (SHORT)
- XGBoost construye cientos de estos árboles y los combina

**Puntos fuertes:** Muy rápido, entiende bien relaciones entre indicadores, no necesita GPU
**Se entrena con:** Optuna (prueba miles de combinaciones de parámetros para encontrar la mejor)

#### Modelo 2: LSTM (la red neuronal con memoria)

**Qué es:** Una red neuronal que recuerda secuencias. No solo mira la vela actual, sino las **últimas 60 velas** en orden, detectando patrones temporales.

**Puntos fuertes:** Entiende patrones de secuencia (por ejemplo: "después de 3 velas rojas seguidas con volumen creciente, suele haber rebote")
**Incluye:** Mecanismo de atención (como cuando lees un texto y te fijas más en ciertas palabras clave)

#### Cómo trabajan juntos: El Ensemble

Los dos modelos votan:
- **XGBoost tiene peso 55%** (es más estable)
- **LSTM tiene peso 45%** (aporta visión temporal)
- Solo se emite señal si **ambos están de acuerdo** en la dirección
- La confianza combinada debe ser **≥ 72%**

**Analogía:** Es como tener dos médicos especialistas. Uno es generalista (XGBoost) y otro es neurólogo (LSTM). Solo actúas si los dos coinciden en el diagnóstico y están bastante seguros.

---

### Estación 5: El generador de señales (generator.py)

**Qué hace:** Toma la predicción del ensemble y la filtra con criterios prácticos:

| Filtro | Qué comprueba | Por qué |
|---|---|---|
| Confianza ≥ 72% | El ensemble está bastante seguro | Evitar señales dudosas |
| ADX ≥ 20 | Hay tendencia real (no lateral) | No operar cuando el mercado no se mueve |
| Sesión activa | Solo Londres, Nueva York u Overlap | El mercado tiene más liquidez en estas horas |
| Cooldown 4h | No repetir señal del mismo par | Evitar sobreoperar |
| R:R ≥ 1.5 | El beneficio potencial es 1.5× la pérdida | Solo trades que merecen la pena |

Si pasa todos los filtros, genera una **señal accionable** con:
- **Par y dirección:** EURUSD LONG
- **Precio de entrada:** 1.08523
- **Take Profit:** 1.08892 (+36.9 pips)
- **Stop Loss:** 1.08277 (-24.6 pips)
- **Confianza:** 78%
- **R:R:** 1.50

**Cuándo se ejecuta:** El servicio `ayram-signals` corre en bucle continuo (cada 60 segundos comprueba si hay nuevas velas con features para analizar).

---

### Estación 6: Gestión de posiciones (position_manager.py)

**Qué hace:** Cuando llega una señal válida:

1. **Calcula el tamaño de la posición** — ¿Cuántos lotes operar? Se basa en arriesgar máximo 1.5% del capital por operación
2. **Abre la posición** en la tabla `positions_active`
3. **Monitorea** si el precio toca el TP o el SL
4. **Trailing stop:** cuando el precio alcanza TP1 (primer objetivo), sube el SL a breakeven y deja correr hasta TP2
5. **Cierra y registra** el resultado en `trades_history`

**Actualmente:** Todo es simulado (paper trading). El PnL se calcula matemáticamente con precios reales. No se ejecuta ninguna orden real hasta tener cuenta demo.

---

## El Sistema de Monitoreo

El sistema no solo opera — también se vigila a sí mismo constantemente.

### Monitor 1: Anomaly Detector (cada 6 horas)

Es como un sistema de alarmas que comprueba 6 cosas:

| Alarma | Qué vigila | Cuándo salta |
|---|---|---|
| 🔇 Signal Drought | ¿Se están generando señales? | Si un par lleva >5 días sin señal |
| 📉 Drawdown | ¿Estamos perdiendo demasiado? | Si las pérdidas superan el 8% en 7 días |
| ❌ Win Rate | ¿Estamos acertando? | Si ganamos menos del 35% de los últimos 20 trades |
| 📊 Stale Data | ¿Los datos son frescos? | Si los precios llevan >2h sin actualizarse |
| 🧠 Stale Models | ¿Los modelos están actualizados? | Si llevan >14 días sin reentrenarse |
| ⚡ Anomalous Signals | ¿El sistema se ha vuelto loco? | Si genera >30 señales en 24h o el 90% son en la misma dirección |

Las alertas graves llegan por **Telegram** automáticamente.

### Monitor 2: Model Health (mensual)

Es como una revisión médica del rendimiento:

1. Mira cómo han funcionado los modelos los últimos 30 días
2. Compara el rendimiento actual con lo que dieron en las pruebas históricas (walk-forward)
3. Clasifica el estado:
   - 🟢 **OK:** rendimiento normal
   - 🟡 **Warning (−20%):** un poco peor de lo esperado
   - 🟠 **Alert (−35%):** significativamente peor
   - 🔴 **Critical (−50%):** el modelo probablemente ya no sirve → puede disparar reentrenamiento automático

### Monitor 3: Análisis IA Mensual

Cada mes genera automáticamente:
1. Un **JSON con todas las métricas** (rendimiento global, por par, por timeframe, por semana)
2. Un **prompt optimizado** para pegar en Claude o ChatGPT

El prompt le pide a la IA que analice: rendimiento por par, filtros que ajustar, patrones temporales, estado de los modelos y gestión de riesgo. Devuelve un diagnóstico con los 5 ajustes más importantes a hacer.

---

## El Dashboard (lo que ves en el navegador)

Accesible en `http://206.81.31.156:8000`, tiene 7 pestañas:

### 1. Dashboard (página principal)
Lo primero que ves: cuántas señales ha dado hoy, cuál fue la última, posiciones abiertas con PnL en tiempo real, y las señales más recientes.

### 2. Gráfico
Un gráfico de velas interactivo (como el de TradingView) donde puedes seleccionar cualquier par y timeframe. Las señales LONG aparecen como flechas verdes y las SHORT como flechas rojas. Líneas punteadas marcan TP y SL.

### 3. Historial
Todas las señales históricas en una tabla con filtros (par, TF, dirección, período). Se ve la confianza de cada modelo, el ADX, y si la señal pasó los filtros.

### 4. Métricas
Gráficos de distribución: cuántas señales LONG vs SHORT, señales por par, por día, por timeframe. Confianza media y acuerdo entre modelos.

### 5. Rendimiento
Lo más importante: PnL total, win rate, profit factor, max drawdown. Una curva de equity que muestra cómo evoluciona el capital. Desglose por par y los últimos 10 trades.

### 6. Monitor
Estado de salud de los datos: ¿cuándo fue la última vela descargada para cada par/TF? ¿Cuándo se calcularon los últimos features? Si algo está retrasado, aparece en amarillo o rojo.

### 7. Configuración
Editor de filtros en tiempo real. Puedes cambiar la confianza mínima, el ADX, el R:R, etc. sin tocar código. Los cambios se aplican inmediatamente a las nuevas señales.

---

## Timing: Cuándo Ocurre Cada Cosa

### Cada 15 minutos
- **Collector:** descarga las últimas velas de precios de EODHD
- **Signals:** comprueba si hay algo nuevo que analizar

### Cada 3 horas
- **Features:** recalcula los 85 indicadores técnicos con los nuevos datos

### Cada 6 horas
- **Anomaly Detector:** 6 comprobaciones de salud del sistema

### Cada domingo a las 02:00 UTC
- **Entrenamiento:** los modelos se reentrenan con los datos más recientes
- Se usa Optuna para encontrar los mejores parámetros (prueba 30 combinaciones)
- Dura entre 1-6 horas dependiendo de los datos

### El primer domingo de cada mes a las 04:00 UTC
Se ejecutan 3 cosas encadenadas:
1. **Walk-Forward Validation:** comprueba que los modelos funcionan bien en datos que no han visto
2. **Model Health:** diagnóstico de degradación
3. **Monthly Summary:** genera el resumen mensual y el prompt para análisis con IA

### Siempre activo (24/7)
- **Dashboard:** FastAPI sirviendo la web en el puerto 8000
- **Signals:** el generador de señales corriendo en bucle continuo

---

## ¿Qué pasa si el servidor se reinicia?

Nada. Todo se recupera automáticamente:

- Los **servicios daemon** (dashboard, signals) tienen `Restart=always` en systemd → si se caen por cualquier motivo, se reinician en 10-30 segundos
- Los **timers** están habilitados con `systemctl enable` → arrancan con el servidor
- Los timers tienen `Persistent=true` → si perdieron una ejecución durante el apagado, la ejecutan nada más arrancar
- Todo está registrado en el journal de systemd → se puede ver qué pasó con `journalctl`

---

## El Flujo Completo: Ejemplo de un Trade

Pongamos un ejemplo de cómo funciona todo junto:

1. **14:15 UTC** — El collector descarga las últimas velas M15 y H1 de EURUSD
2. **15:00 UTC** — El features timer calcula los 85 indicadores para las nuevas velas
3. **15:01 UTC** — El signals service detecta features nuevos y pide predicción al ensemble:
   - XGBoost dice: LONG con 82% confianza
   - LSTM dice: LONG con 76% confianza
   - Ensemble combinado: LONG con 79% confianza (pasa el umbral de 72%)
4. **15:01 UTC** — El generador aplica filtros:
   - ✅ ADX = 28 (> 20, hay tendencia)
   - ✅ Sesión = London-NY Overlap (la mejor)
   - ✅ No hay cooldown activo para EURUSD
   - ✅ R:R = 1.8 (> 1.5)
5. **15:01 UTC** — Se genera la señal:
   - EURUSD LONG | Entrada: 1.08523 | TP: 1.08892 | SL: 1.08277
6. **15:01 UTC** — Position manager calcula: con 10.000€ y riesgo 1.5%, arriesga 150€ → abre 0.61 lotes
7. **15:01 UTC** — Telegram envía notificación al móvil
8. **15:01 UTC** — La señal aparece en el dashboard

**Horas después:**
- El precio sube hasta 1.08750 → trailing stop se activa
- El precio sigue hasta 1.08892 → TP tocado → posición cerrada
- PnL: +36.9 pips × 0.61 lotes × 10€/pip = +225.09€
- El trade aparece en `trades_history` y en el dashboard en la pestaña "Rendimiento"

**Fin de mes:**
- El monthly_summary calcula que EURUSD tuvo 58% win rate y PF 1.6
- Genera un prompt para Claude que dice: "EURUSD fue tu par más rentable, considera aumentar el tamaño de posición un 10%"

---

## Resumen Visual del Sistema

```
 MERCADO (EODHD)
      │
      ▼ cada 15 min
 ┌────────────┐
 │ COLLECTOR  │ ──► ohlcv_raw (precios en BD)
 └────────────┘
      │
      ▼ cada 3h
 ┌────────────┐
 │ FEATURES   │ ──► features_computed (85 indicadores en BD)
 └────────────┘
      │
      ▼ continuo
 ┌────────────┐     ┌──────────┐
 │ ENSEMBLE   │ ◄───│ XGBoost  │ (55%)
 │ (predicción│ ◄───│ LSTM     │ (45%)
 └────────────┘     └──────────┘
      │
      ▼ si pasa filtros
 ┌────────────┐
 │ SEÑAL      │ ──► signals (BD) + Telegram
 └────────────┘
      │
      ▼
 ┌────────────┐
 │ POSITION   │ ──► positions_active / trades_history (BD)
 │ MANAGER    │
 └────────────┘
      │
      ▼
 ┌────────────────────────────────────────────┐
 │            MONITOREO CONTINUO              │
 │                                            │
 │  cada 6h → anomaly_detector (6 checks)    │
 │  mensual → model_health (degradación)     │
 │  mensual → monthly_summary (análisis IA)  │
 └────────────────────────────────────────────┘
      │
      ▼
 ┌────────────┐
 │ DASHBOARD  │ ──► Web en :8000 (7 pestañas)
 │ (FastAPI)  │     Gráficos, métricas, monitor
 └────────────┘

 ┌────────────────────────────────────────────┐
 │         REENTRENAMIENTO AUTOMÁTICO         │
 │                                            │
 │  cada domingo → train (Optuna, 30 trials)  │
 │  1er dom/mes → walk-forward validation     │
 └────────────────────────────────────────────┘
```

---

## Tecnologías Usadas (para curiosos)

| Qué | Tecnología | Para qué |
|---|---|---|
| Servidor | DigitalOcean (Ubuntu 24.04) | Ejecutar todo 24/7 |
| Base de datos | Supabase PostgreSQL | Guardar precios, features, señales, trades |
| Datos de mercado | EODHD API | Precios forex en tiempo real e históricos |
| ML modelo 1 | XGBoost | Predicción rápida con árboles de decisión |
| ML modelo 2 | PyTorch LSTM | Red neuronal con memoria para secuencias |
| Optimización | Optuna | Encontrar los mejores parámetros automáticamente |
| API web | FastAPI + Uvicorn | Servir el dashboard y la API |
| Frontend | HTML/CSS/JS + Chart.js + lightweight-charts | Interfaz visual del dashboard |
| Notificaciones | Telegram Bot API | Alertas al móvil |
| Tareas programadas | systemd timers | Ejecutar cosas a horas fijas |
| Servicios | systemd services | Mantener procesos vivos 24/7 |
| Deploy | rsync + bash | Subir código al servidor |
| Tracking ML | MLflow | Registrar experimentos y métricas de modelos |

---

## Estado Actual y Próximos Pasos

### ✅ Lo que ya funciona
- Toda la infraestructura de servidor, BD y deploy
- Pipeline completo de datos (collector → features → labels)
- Los dos modelos ML (XGBoost + LSTM) — código listo
- Generador de señales con filtros
- Backtesting + Walk-Forward validation
- Dashboard web con 7 secciones
- Sistema de monitoreo (anomalías + health + análisis IA)
- Reentrenamiento automático semanal
- Todos los servicios systemd con auto-reinicio

### ⏳ Lo que falta
- Completar la descarga histórica (3 años de datos)
- Entrenar los modelos con datos reales
- Configurar el bot de Telegram
- 4 semanas de paper trading con resultados satisfactorios
- Cuenta demo de cTrader para ejecución real

---

*ML-Ayram | Proyecto de uso personal | Feb 2026*
