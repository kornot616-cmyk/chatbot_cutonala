
import os
import time
import random
from pathlib import Path

from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

from langchain_google_genai import (
    GoogleGenerativeAIEmbeddings,
    ChatGoogleGenerativeAI
)

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


# ======================================================
# 1. CONFIGURACIÓN
# ======================================================

BASE_DIR = Path(__file__).resolve().parent

load_dotenv(BASE_DIR / ".env")

PDF_PATH = BASE_DIR / "conocimientos"
DB_PATH = BASE_DIR / "faiss_index"

# Mantener el mismo modelo de embeddings del índice.
EMBEDDING_MODEL = "gemini-embedding-2"

# Modelo que ya probaste con tu API.
CHAT_MODEL = "gemini-3.1-flash-lite"

if not os.getenv("GOOGLE_API_KEY"):
    raise RuntimeError(
        "No se encontró GOOGLE_API_KEY en .env"
    )

print("=" * 55)
print(" TUTOR DE ÉTICA Y NORMATIVA - CUTONALÁ")
print("=" * 55)


# ======================================================
# 2. EMBEDDINGS
# ======================================================

embeddings = GoogleGenerativeAIEmbeddings(
    model=EMBEDDING_MODEL
)


# ======================================================
# 3. CARGAR O CREAR FAISS
# ======================================================

def cargar_conocimiento():

    if DB_PATH.exists():

        print("\nCargando conocimiento local...")

        db = FAISS.load_local(
            str(DB_PATH),
            embeddings,
            allow_dangerous_deserialization=True
        )

        print("FAISS cargado correctamente.")

        return db

    print("\nPrimera ejecución: procesando PDF...")

    if not PDF_PATH.exists():
        PDF_PATH.mkdir(parents=True, exist_ok=True)

        raise FileNotFoundError(
            "Coloca tus PDF en conocimientos/ "
            "y vuelve a ejecutar."
        )

    documentos = PyPDFDirectoryLoader(
        str(PDF_PATH)
    ).load()

    if not documentos:
        raise RuntimeError(
            "No se encontraron PDF con texto."
        )

    print(f"Páginas cargadas: {len(documentos)}")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200,
        chunk_overlap=250,
        separators=["\n\n", "\n", ". ", " ", ""]
    )

    fragmentos = splitter.split_documents(
        documentos
    )

    print(f"Fragmentos generados: {len(fragmentos)}")

    db = None
    tamano_lote = 10

    for inicio in range(0, len(fragmentos), tamano_lote):

        lote = fragmentos[
            inicio:inicio + tamano_lote
        ]

        for intento in range(3):

            try:
                print(
                    f"Indexando fragmentos "
                    f"{inicio + 1} a "
                    f"{min(inicio + tamano_lote, len(fragmentos))}..."
                )

                if db is None:
                    db = FAISS.from_documents(
                        lote, embeddings
                    )
                else:
                    db.add_documents(lote)

                break

            except Exception as error:

                if intento == 2:
                    raise

                espera = 20 * (intento + 1)

                print(
                    f"Error temporal: {error}\n"
                    f"Reintentando en {espera} segundos..."
                )

                time.sleep(espera)

    db.save_local(str(DB_PATH))

    print("Conocimiento guardado en faiss_index.")

    return db


vector_store = cargar_conocimiento()

retriever = vector_store.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 4}
)


# ======================================================
# 4. GEMINI
# ======================================================

llm = ChatGoogleGenerativeAI(
    model=CHAT_MODEL,
    temperature=0.2,
    max_retries=0,
    timeout=90,
    max_output_tokens=300
)


# ======================================================
# 5. INSTRUCCIONES DEL TUTOR
# ======================================================

CONSEJERO_SYSTEM_PROMPT = """
Eres el Tutor de Ética y Normativa Institucional
del Centro Universitario de Tonalá (CUTonalá).

Tu función es orientar a estudiantes con información
de los documentos institucionales recuperados.

Ofreces orientación educativa, no asesoría legal.

REGLAS:

1. RESPONDE LA PREGUNTA DEL ESTUDIANTE.

Si pregunta para qué sirve una instancia:
- Explica qué hace.
- Explica qué problemas atiende.
- Explica cuándo conviene acudir.

Si cuenta un problema:
- Recomienda la instancia adecuada.
- Explica cómo puede ayudarle.
- Pregunta únicamente lo necesario.

Si pide un contacto:
- Proporciona únicamente datos presentes
  en los fragmentos documentales.
- No mezcles contactos de diferentes carreras.

2. RESPETA EL CONTEXTO DE LA CONVERSACIÓN.

Usa el historial para interpretar:
- "Sí, por favor".
- "Ya fui y no me ayudaron".
- "¿Y después?".
- "Dame su correo".

No vuelvas a preguntar información conocida.

3. RESPETA LAS RUTAS DE ATENCIÓN.

PASO 0:
Profesor de confianza.

NIVEL 1:
Coordinación de Licenciatura.
Tutorías y Orientación Educativa.
Módulo de Primer Contacto.
Direcciones de División.

NIVEL 2:
Secretarías del CUTonalá.
Consejo de Centro.

NIVEL 3:
Defensoría de los Derechos Universitarios.
Contraloría General de la UdeG.
Rectoría General.

NIVEL 4:
Instancias externas, según corresponda.

La gravedad del caso no equivale al nivel institucional.

Si Coordinación no atiende un problema académico,
puede corresponder acudir a Dirección de División,
que también pertenece al nivel 1.

No todos los casos deben recorrer esos niveles
en orden. Considera la naturaleza de los hechos.

4. UTILIDAD DE LAS INSTANCIAS.

Siempre que recomiendes una dependencia,
explica para qué sirve en el caso concreto.

No te limites a mencionar su nombre.

5. INFORMACIÓN VERIFICABLE.

Usa únicamente la información documental recuperada.

No inventes correos, teléfonos, ubicaciones,
personas, horarios ni procedimientos.

Si falta información, reconócelo.

Nunca afirmes como comprobada una acusación
hecha por un estudiante.

6. SEGURIDAD.

En casos de violencia, amenazas o posibles delitos,
no obligues al estudiante a agotar todas
las instancias universitarias.

7. ESTILO.

Responde en español.
Sé cercano, respetuoso y profesional.
Evita repetir información innecesariamente.
Da respuestas breves pero útiles.
No prometas resultados.

HISTORIAL:
{history}

CONTEXTO RECUPERADO DE LOS PDF:
{context}

PREGUNTA:
{input}

RESPUESTA:
"""

prompt_template = ChatPromptTemplate.from_template(
    CONSEJERO_SYSTEM_PROMPT
)

cadena_respuesta = (
    prompt_template
    | llm
    | StrOutputParser()
)


# ======================================================
# 6. FORMATEAR LOS FRAGMENTOS PDF
# ======================================================

def format_docs(documentos):

    bloques = []

    for doc in documentos:

        nombre = Path(
            str(doc.metadata.get("source", "PDF"))
        ).name

        pagina = doc.metadata.get("page")

        if isinstance(pagina, int):
            pagina += 1

        bloques.append(
            f"PDF: {nombre}\n"
            f"Página: {pagina}\n"
            f"Contenido:\n{doc.page_content}"
        )

    return "\n\n---\n\n".join(bloques)


# ======================================================
# 7. MEMORIA DE CONVERSACIÓN
# ======================================================

historial = []


def formatear_historial():

    return "\n".join(
        f"{rol}: {texto}"
        for rol, texto in historial[-8:]
    )


def construir_busqueda(pregunta):

    anteriores = [
        texto
        for rol, texto in historial
        if rol == "Estudiante"
    ]

    contexto_anterior = " ".join(
        anteriores[-3:]
    )

    return (
        f"Conversación previa: {contexto_anterior}\n"
        f"Pregunta actual: {pregunta}\n"
        "CUTonalá, orientación institucional, "
        "funciones, contactos, quejas, procedimientos."
    )[-1800:]


# ======================================================
# 8. BUSCAR EN FAISS
# ======================================================

def consultar_faiss(pregunta):

    print(
        "\n[1] Buscando en FAISS...",
        flush=True
    )

    inicio = time.perf_counter()

    documentos = retriever.invoke(
        construir_busqueda(pregunta)
    )

    duracion = time.perf_counter() - inicio

    print(
        f"[2] FAISS respondió en "
        f"{duracion:.2f} segundos.",
        flush=True
    )

    print(
        f"Fragmentos recuperados: {len(documentos)}",
        flush=True
    )

    return documentos


# ======================================================
# 9. CONSULTAR GEMINI CON REINTENTOS
# ======================================================

def consultar_gemini(pregunta, documentos):

    contexto = format_docs(documentos)

    max_intentos = 3

    for intento in range(1, max_intentos + 1):

        print(
            f"\n[3] Consultando {CHAT_MODEL} "
            f"({intento}/{max_intentos})...",
            flush=True
        )

        inicio = time.perf_counter()

        try:

            respuesta = cadena_respuesta.invoke({
                "input": pregunta,
                "history": formatear_historial(),
                "context": contexto
            })

            duracion = time.perf_counter() - inicio

            print(
                f"[4] Respuesta recibida en "
                f"{duracion:.2f} segundos.",
                flush=True
            )

            return respuesta

        except Exception as error:

            detalle = str(error).upper()

            temporal = any(
                codigo in detalle
                for codigo in (
                    "503",
                    "UNAVAILABLE",
                    "504",
                    "DEADLINE_EXCEEDED",
                    "429",
                    "RESOURCE_EXHAUSTED"
                )
            )

            if not temporal or intento == max_intentos:
                raise

            espera = min(
                20,
                (2 ** intento) + random.uniform(0, 2)
            )

            print(
                f"Gemini no respondió. "
                f"Reintentando en {espera:.1f} segundos...",
                flush=True
            )

            time.sleep(espera)


# ======================================================
# 10. CHAT EN TERMINAL
# ======================================================

print("\n" + "=" * 55)
print("CHATBOT CUTONALÁ LISTO")
print("Modelo:", CHAT_MODEL)
print("FAISS: cargado")
print("=" * 55)

print("\nComandos: salir | reiniciar")
print("-" * 55)


while True:

    try:
        pregunta = input(
            "\nEstudiante: "
        ).strip()

    except (KeyboardInterrupt, EOFError):
        print("\nTutor: Hasta pronto.")
        break

    if not pregunta:
        continue

    if pregunta.lower() == "salir":
        print("\nTutor: Hasta pronto.")
        break

    if pregunta.lower() == "reiniciar":

        historial.clear()

        print(
            "\nTutor: Conversación reiniciada. "
            "¿En qué puedo ayudarte?"
        )

        continue

    if pregunta.lower() in {
        "hola",
        "buenas",
        "buenos dias",
        "buenas tardes",
        "buenas noches"
    }:

        respuesta = (
            "¡Hola! Soy el Tutor de Ética y "
            "Normativa de CUTonalá. "
            "¿En qué puedo ayudarte?"
        )

        print(f"\nTutor: {respuesta}")

        historial.extend([
            ("Estudiante", pregunta),
            ("Tutor", respuesta)
        ])

        continue

    print("\nConsultando documentos...")

    try:
        documentos = consultar_faiss(pregunta)

    except Exception as error:
        print("\nERROR EN FAISS")
        print("Detalle:", error)
        continue

    try:
        respuesta = consultar_gemini(
            pregunta, documentos
        )

    except Exception as error:
        print("\nERROR EN GEMINI")
        print("Detalle:", error)
        print(
            "La recuperación documental funcionó, "
            "pero Gemini no pudo responder."
        )
        continue

    print(f"\nTutor: {respuesta}")

    historial.extend([
        ("Estudiante", pregunta),
        ("Tutor", respuesta)
    ])

    if len(historial) > 20:
        historial = historial[-20:]
