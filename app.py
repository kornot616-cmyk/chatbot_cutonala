# app.py
import os
from pathlib import Path
from flask import Flask, render_template_string, request, jsonify
from dotenv import load_dotenv

from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from automata_favoritismo import AutomataFavoritismo

app = Flask(__name__)

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

DB_PATH = BASE_DIR / "faiss_index"
EMBEDDING_MODEL = "gemini-embedding-2"
CHAT_MODEL = "gemini-3.1-flash-lite"

if not os.getenv("GOOGLE_API_KEY"):
    raise RuntimeError("No se encontró GOOGLE_API_KEY en .env")

embeddings = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL)
vector_store = FAISS.load_local(
    str(DB_PATH),
    embeddings,
    allow_dangerous_deserialization=True
)
retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 4})

llm = ChatGoogleGenerativeAI(
    model=CHAT_MODEL,
    temperature=0.2,
    max_retries=0,
    timeout=90,
    max_output_tokens=350
)

SYSTEM_PROMPT = """
Eres el Tutor de Ética y Normativa Institucional del Centro Universitario de Tonalá (CUTonalá).
Ofreces orientación educativa y normativa clara, empática y profesional, sin asesoría legal vinculante.

REGLAS:
1. Usa únicamente información verificable de los documentos institucionales recuperados. No inventes correos, nombres ni ubicaciones.
2. Sé conciso, directo y respetuoso. No repitas información que ya se mencionó en el historial.
3. Si el AUTÓMATA FINITO DETERMINISTA (AFD) está activo, guía la conversación según la directiva del estado actual sin mencionar códigos internos (como q0, q1, e1, R1, etc.) al estudiante; háblale en lenguaje natural.

ESTADO DEL PROTOCOLO:
{estado_automata}

HISTORIAL DE CONVERSACIÓN:
{history}

FRAGMENTOS NORMATIVOS (FAISS):
{context}

MENSAJE DEL ESTUDIANTE:
{input}

RESPUESTA DEL TUTOR:
"""

cadena_respuesta = ChatPromptTemplate.from_template(SYSTEM_PROMPT) | llm | StrOutputParser()

historial = []
automata = AutomataFavoritismo()

PALABRAS_TRIGGER = (
    "favoritismo", "preferencia", "alumnas", "puntos extra",
    "segundas oportunidades", "excluyen", "comentarios degradantes",
    "trato desigual", "daño moral", "injusto"
)

HTML_INTERFAZ = """
<!DOCTYPE html>
<html lang="es">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Tutor de Ética y Normativa - CUTonalá</title>
    <style>
        :root { --azul-udeg: #002855; --azul-claro: #005596; --dorado: #C59B27; --fondo: #F4F6F9; --blanco: #FFFFFF; --texto: #1E293B; }
        * { box-sizing: border-box; margin: 0; padding: 0; font-family: 'Segoe UI', system-ui, sans-serif; }
        body { background-color: var(--fondo); color: var(--texto); display: flex; justify-content: center; align-items: center; height: 100vh; }
        .chat-container { width: 100%; max-width: 780px; height: 90vh; background: var(--blanco); border-radius: 12px; box-shadow: 0 8px 24px rgba(0,0,0,0.08); display: flex; flex-direction: column; overflow: hidden; border: 1px solid #E2E8F0; }
        .chat-header { background: var(--azul-udeg); color: var(--blanco); padding: 18px 24px; display: flex; justify-content: space-between; align-items: center; border-bottom: 4px solid var(--dorado); }
        .chat-header h1 { font-size: 1.15rem; font-weight: 600; }
        .chat-header p { font-size: 0.8rem; opacity: 0.85; margin-top: 2px; }
        .btn-reset { background: transparent; border: 1px solid rgba(255,255,255,0.4); color: var(--blanco); padding: 6px 12px; border-radius: 6px; cursor: pointer; font-size: 0.8rem; transition: 0.2s; }
        .btn-reset:hover { background: rgba(255,255,255,0.15); }
        .chat-box { flex: 1; padding: 24px; overflow-y: auto; display: flex; flex-direction: column; gap: 14px; }
        .msg { max-width: 80%; padding: 12px 16px; border-radius: 10px; line-height: 1.5; font-size: 0.95rem; white-space: pre-wrap; }
        .msg.bot { background: #F1F5F9; color: var(--texto); align-self: flex-start; border-left: 3px solid var(--azul-claro); }
        .msg.user { background: var(--azul-udeg); color: var(--blanco); align-self: flex-end; }
        .typing { font-size: 0.85rem; color: #64748B; font-style: italic; padding: 0 24px 8px; display: none; }
        .input-area { padding: 16px 20px; background: var(--blanco); border-top: 1px solid #E2E8F0; display: flex; gap: 10px; }
        .input-area input { flex: 1; padding: 12px 16px; border: 1px solid #CBD5E1; border-radius: 8px; font-size: 0.95rem; outline: none; }
        .input-area input:focus { border-color: var(--azul-claro); }
        .input-area button { background: var(--azul-claro); color: var(--blanco); border: none; padding: 0 22px; border-radius: 8px; font-weight: 600; cursor: pointer; transition: 0.2s; }
        .input-area button:hover { background: var(--azul-udeg); }
    </style>
</head>
<body>
    <div class="chat-container">
        <div class="chat-header">
            <div>
                <h1>Tutor de Ética y Normativa Institucional</h1>
                <p>Centro Universitario de Tonalá - Universidad de Guadalajara</p>
            </div>
            <button class="btn-reset" onclick="reiniciarChat()">Nueva consulta</button>
        </div>
        <div class="chat-box" id="chatBox">
            <div class="msg bot">Hola. Soy el Tutor de Ética y Normativa del CUTonalá. Cuéntame tu situación o duda institucional para orientarte.</div>
        </div>
        <div class="typing" id="typingIndicator">Analizando normativa...</div>
        <div class="input-area">
            <input type="text" id="userInput" placeholder="Escribe tu mensaje aquí..." onkeypress="if(event.key==='Enter') enviarMensaje()">
            <button onclick="enviarMensaje()">Enviar</button>
        </div>
    </div>

    <script>
        const chatBox = document.getElementById("chatBox");
        const userInput = document.getElementById("userInput");
        const typing = document.getElementById("typingIndicator");

        function agregarMensaje(texto, clase) {
            const div = document.createElement("div");
            div.className = `msg ${clase}`;
            div.textContent = texto;
            chatBox.appendChild(div);
            chatBox.scrollTop = chatBox.scrollHeight;
        }

        async function enviarMensaje() {
            const texto = userInput.value.trim();
            if (!texto) return;

            agregarMensaje(texto, "user");
            userInput.value = "";
            typing.style.display = "block";

            try {
                const res = await fetch("/chat", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({ mensaje: texto })
                });
                const data = await res.json();
                typing.style.display = "none";
                agregarMensaje(data.respuesta, "bot");
            } catch (err) {
                typing.style.display = "none";
                agregarMensaje("Error de conexión con el servidor.", "bot");
            }
        }

        async function reiniciarChat() {
            const res = await fetch("/reiniciar", { method: "POST" });
            const data = await res.json();
            chatBox.innerHTML = "";
            agregarMensaje(data.respuesta, "bot");
        }
    </script>
</body>
</html>
"""

def format_docs(documentos):
    return "\n\n---\n\n".join(doc.page_content for doc in documentos)

@app.route("/")
def index():
    return render_template_string(HTML_INTERFAZ)

@app.route("/chat", methods=["POST"])
def chat():
    datos = request.get_json()
    pregunta = (datos.get("mensaje") or "").strip()

    if not pregunta:
        return jsonify({"respuesta": "Por favor escribe tu consulta."})

    if pregunta.lower() in {"hola", "buenas", "buenos dias", "buenas tardes", "buenas noches"}:
        resp = "¡Hola! Soy el Tutor de Ética y Normativa de CUTonalá. ¿En qué puedo orientarte hoy?"
        historial.extend([("Estudiante", pregunta), ("Tutor", resp)])
        return jsonify({"respuesta": resp})

    # Control silencioso del AFD (solo visible en la terminal de VS Code, nunca en la pantalla)
    if not automata.activo and any(p in pregunta.lower() for p in PALABRAS_TRIGGER):
        automata.activo = True
        print(f"[AFD] Iniciado en {automata.estado_actual}")

    if automata.activo:
        transicion = automata.clasificar_y_avanzar(llm, pregunta)
        print(f"[AFD] Transición: {transicion} -> Estado actual: {automata.estado_actual}")
        instruccion_afd = automata.obtener_instruccion_prompt()
        etapa = automata.NODOS[automata.estado_actual]["nombre"]
        query_faiss = f"{pregunta} {etapa} tutorías coordinación queja código civil jalisco 1391 1394 Bis"
    else:
        instruccion_afd = "AFD inactivo. Responde como consulta normativa general."
        query_faiss = pregunta

    try:
        documentos = retriever.invoke(query_faiss)
        contexto = format_docs(documentos)
        historial_txt = "\n".join(f"{r}: {t}" for r, t in historial[-8:])

        respuesta = cadena_respuesta.invoke({
            "input": pregunta,
            "history": historial_txt,
            "context": contexto,
            "estado_automata": instruccion_afd
        })

        if automata.activo and automata.estado_actual in automata.ESTADOS_FINALES:
            print(f"[AFD] Estado final {automata.estado_actual} alcanzado. Reiniciando AFD.")
            automata.reset()

        historial.extend([("Estudiante", pregunta), ("Tutor", respuesta)])
        if len(historial) > 20:
            del historial[:-20]

        return jsonify({"respuesta": respuesta})

    except Exception as e:
        print(f"[ERROR INTERNO]: {e}")
        return jsonify({
            "respuesta": "Ocurrió un problema temporal al consultar la normativa. Por favor intenta de nuevo en unos segundos."
        }), 500

@app.route("/reiniciar", methods=["POST"])
def reiniciar():
    historial.clear()
    automata.reset()
    return jsonify({"respuesta": "Conversación reiniciada. ¿En qué puedo ayudarte?"})

if __name__ == "__main__":
    app.run(debug=False, port=5000)