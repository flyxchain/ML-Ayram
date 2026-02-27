# DEPRECADO — ver src/execution/position_manager.py
#
# Este archivo fue reemplazado porque:
#   1. Duplicaba la función generate_signal() de src/signals/generator.py
#   2. Usaba get_latest_candles() (API en vivo) en vez de features_computed
#   3. El cálculo de pips para XAUUSD era incorrecto
#   4. Las tablas signals_log/positions_active no estaban creadas
#
# La lógica de lot sizing y gestión de posiciones vive ahora en:
#   → src/execution/position_manager.py
raise ImportError("Este módulo está deprecado. Usa src.execution.position_manager")
