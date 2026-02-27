"""
Script de prueba de conexión a cTrader Open API
Ejecutar con: python scripts/test_ctrader_connection.py
"""
import os
import sys
from dotenv import load_dotenv

load_dotenv()

def test_connection():
    client_id = os.getenv("CTRADER_CLIENT_ID")
    client_secret = os.getenv("CTRADER_CLIENT_SECRET")
    account_id = os.getenv("CTRADER_ACCOUNT_ID")
    env = os.getenv("CTRADER_ENV", "demo")

    if not all([client_id, client_secret, account_id]):
        print("❌ Faltan credenciales en el archivo .env")
        print("   Verifica: CTRADER_CLIENT_ID, CTRADER_CLIENT_SECRET, CTRADER_ACCOUNT_ID")
        sys.exit(1)

    print(f"✓ Credenciales cargadas")
    print(f"  Client ID: {client_id[:8]}...")
    print(f"  Account ID: {account_id}")
    print(f"  Entorno: {env.upper()}")
    print()
    print("Para testear la conexión real, implementa aquí la llamada")
    print("a ctrader-open-api una vez instalada la librería.")
    print()
    print("Documentación: https://spotware.github.io/open-api-docs/")

if __name__ == "__main__":
    test_connection()
