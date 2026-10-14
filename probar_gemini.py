
import os
from pathlib import Path
from dotenv import load_dotenv
from google import genai

load_dotenv(Path(__file__).resolve().parent / ".env")

client = genai.Client(
    api_key=os.getenv("GOOGLE_API_KEY")
)

modelos = [
    "gemini-3.1-flash-lite",
    "gemini-3.5-flash-lite",
    "gemini-flash-lite-latest"
]

for modelo in modelos:
    print(f"\nProbando: {modelo}", flush=True)

    try:
        respuesta = client.models.generate_content(
            model=modelo,
            contents="Responde únicamente: Hola CUTonalá"
        )

        print("FUNCIONA:", respuesta.text)
        print("MODELO RECOMENDADO:", modelo)
        break

    except Exception as error:
        print("FALLÓ:", str(error)[:350])

else:
    print("\nNingún modelo respondió correctamente.")

client.close()
