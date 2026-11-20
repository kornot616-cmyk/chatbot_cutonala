# automata_favoritismo.py

class AutomataFavoritismo:
    def __init__(self):
        self.reset()

    def reset(self):
        self.activo = False
        self.estado_actual = "q0"
        self.bitacora = []

    ESTADOS_FINALES = {"q4", "q5", "q6"}

    NODOS = {
        "q0": {
            "nombre": "Inicio",
            "objetivo": "Identificar si el caso corresponde a Favoritismo Asesor (e1) o Alumna Beneficiada (e2) para transicionar a q1.",
            "transiciones": {
                "e1_R1": ("e1 (Favoritismo Asesor) - R1: Más atención a alumnas sin justificación académica", "q1"),
                "e1_R2": ("e1 (Favoritismo Asesor) - R2: Preferencia explícita hacia compañeras mujeres", "q1"),
                "e1_R3": ("e1 (Favoritismo Asesor) - R3: Comentarios degradantes hacia los varones", "q1"),
                "e2_R1": ("e2 (Alumna Beneficiada) - R1: Plazos o segundas oportunidades solo para alumnas", "q1"),
                "e2_R2": ("e2 (Alumna Beneficiada) - R2: Puntos extras sin razón", "q1"),
                "e2_R3": ("e2 (Alumna Beneficiada) - R3: Proyecto donde se denigra el género", "q1"),
            }
        },
        "q1": {
            "nombre": "Favoritismo",
            "objetivo": "Indagar si existe Afectación Académica (e3 -> q2) o impacto emocional / necesidad de contención (e4 -> q4 Final).",
            "transiciones": {
                "e3_R1": ("e3 (Afectación Académica) - R1: Afectación en calificaciones", "q2"),
                "e3_R2": ("e3 (Afectación Académica) - R2: Exclusión de actividades", "q2"),
                "e3_R3": ("e3 (Afectación Académica) - R3: Afectación del aprendizaje", "q2"),
                "e4_R1": ("e4 (Emocional) - R1: Impacto emocional o necesidad de contención", "q4"),
            }
        },
        "q2": {
            "nombre": "Afectación",
            "objetivo": "Confirmar el impacto académico (e3) para avanzar hacia la evaluación de Evidencias (q3).",
            "transiciones": {
                "e3_R1": ("e3 (Afectación Académica) - R1: Confirmada afectación en calificaciones", "q3"),
                "e3_R2": ("e3 (Afectación Académica) - R2: Confirmada exclusión de actividades", "q3"),
                "e3_R3": ("e3 (Afectación Académica) - R3: Confirmada afectación del aprendizaje", "q3"),
            }
        },
        "q3": {
            "nombre": "Evidencias",
            "objetivo": "Determinar si el estudiante cuenta con pruebas (p1 -> q6 Legal), no tiene pruebas (p0 -> q5 Sin pruebas) o requiere contención emocional (e4 -> q4 Final).",
            "transiciones": {
                "p0_R1": ("p0 (Sin pruebas) - R1: Testimonio sin evidencias adicionales", "q5"),
                "p0_R2": ("p0 (Sin pruebas) - R2: Información insuficiente o contradictoria", "q5"),
                "p0_R3": ("p0 (Sin pruebas) - R3: Orientación para recopilar pruebas y valorar queja", "q5"),
                "p1_R1": ("p1 (Con pruebas) - R1: Documentos, calificaciones o testimonios", "q6"),
                "p1_R2": ("p1 (Con pruebas) - R2: Mensajes o publicaciones de acusaciones falsas", "q6"),
                "p1_R3": ("p1 (Con pruebas) - R3: Evidencias de amenazas, represalias o discriminación", "q6"),
                "e4_R1": ("e4 (Emocional) - R1: Impacto emocional o necesidad de contención", "q4"),
            }
        },
        "q4": {
            "nombre": "Emocional (Estado Final)",
            "objetivo": "ESTADO FINAL q4: Priorizar la contención emocional del estudiante y canalizarlo a Orientación Educativa / Tutorías en CUTonalá.",
            "transiciones": {}
        },
        "q5": {
            "nombre": "Sin pruebas (Estado Final)",
            "objetivo": "ESTADO FINAL q5: Orientar sobre cómo recopilar pruebas y documentar los hechos para poder valorar una queja formal ante Tutorías o Coordinación.",
            "transiciones": {}
        },
        "q6": {
            "nombre": "Legal (Estado Final)",
            "objetivo": (
                "ESTADO FINAL q6 (Reglas en Estado Final Legal):\n"
                "- R1: SI existen indicios de discriminación por género -> Valorar queja universitaria o denuncia ante autoridad competente.\n"
                "- R2: SI existe daño ilícito al honor o reputación -> Valorar demanda civil por daño moral (Arts. 1391 y 1394 Bis, Código Civil de Jalisco).\n"
                "- R3: SI existen amenazas o represalias -> Solicitar orientación jurídica y medidas de protección."
            ),
            "transiciones": {}
        }
    }

    def clasificar_y_avanzar(self, llm, mensaje_usuario: str) -> str:
        nodo = self.NODOS[self.estado_actual]
        if not nodo["transiciones"]:
            return "FINALIZADO"

        opciones = "\n".join([f"- {k}: {v[0]}" for k, v in nodo["transiciones"].items()])
        prompt = (
            f"Eres el motor de transición de un Autómata Finito Determinista (AFD).\n"
            f"Estado actual: {self.estado_actual} ({nodo['nombre']})\n"
            f"Mensaje del estudiante: \"{mensaje_usuario}\"\n\n"
            f"Transiciones posibles:\n{opciones}\n\n"
            f"Devuelve ÚNICAMENTE la clave exacta de la transición (ej. e1_R1, e2_R2, e3_R1, e4_R1, p0_R1, p1_R3). "
            f"Si el mensaje aún no responde a ninguna transición, devuelve 'NINGUNA'."
        )

        contenido = llm.invoke(prompt).content
        if isinstance(contenido, list):
            texto = " ".join(
                parte.get("text", "") if isinstance(parte, dict) else str(parte)
                for parte in contenido
            )
        else:
            texto = str(contenido)

        clave = texto.strip()
        if clave in nodo["transiciones"]:
            descripcion, siguiente = nodo["transiciones"][clave]
            self.bitacora.append((self.estado_actual, clave, descripcion, siguiente))
            self.estado_actual = siguiente
            return clave
        return "NINGUNA"

    def obtener_instruccion_prompt(self) -> str:
        nodo = self.NODOS[self.estado_actual]
        transiciones_txt = "\n".join([f"  * {v[0]}" for v in nodo["transiciones"].values()])
        recorrido = " -> ".join([f"{e0}({k})->{e1}" for e0, k, _, e1 in self.bitacora]) or "q0 (Inicio)"

        return (
            f"AFD ACTIVO | Estado actual: {self.estado_actual} ({nodo['nombre']})\n"
            f"Traza: {recorrido}\n"
            f"Directiva de este estado: {nodo['objetivo']}\n"
            f"Criterios a evaluar o presentar en este turno:\n"
            f"{transiciones_txt if transiciones_txt else 'Estado final alcanzado. Entrega la resolución clara y empática.'}"
        )