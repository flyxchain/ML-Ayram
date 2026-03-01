"""
src/utils/db.py
Motor SQLAlchemy compartido con pool limitado.

Supabase Transaction mode (puerto 6543) soporta muchas conexiones concurrentes.
Limitamos el pool para no agotar los límites del plan.

Todos los módulos deben importar el engine desde aquí:
    from src.utils.db import engine
"""

import os
from sqlalchemy import create_engine
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL", "")

# Pool conservador: máximo 3 conexiones activas por proceso
# Con Transaction mode (puerto 6543) en Supabase esto es seguro
engine = create_engine(
    DATABASE_URL,
    pool_size=2,       # conexiones persistentes en el pool
    max_overflow=1,    # conexiones extra permitidas bajo carga
    pool_timeout=30,   # segundos antes de lanzar error si no hay conexión libre
    pool_pre_ping=True # verifica que la conexión sigue viva antes de usarla
)
