
import os
from pathlib import Path
from dotenv import load_dotenv
from google import genai

load_dotenv(Path(__file__).resolve().parent / ".env")

api_key = os.getenv("GOOGLE_API_KEY")

if not api_key:
    print("ERROR: Falta GOOGLE_API_KEY en .env")
    raise SystemExit(1)

client = genai.Client(api_key=api_key)

print("\nMODELOS DISPONIBLES PARA TU API:\n")

try:
    for modelo in client.models.list():
        nombre = modelo.name
        acciones = getattr(modelo, "supported_actions", []) or []

        if "gemini" in nombre.lower() and (
            "generateContent" in acciones
            or not acciones
        ):
            print(nombre)

except Exception as error:
    print("Error consultando modelos:", error)

finally:
    client.close()
