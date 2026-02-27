#!/usr/bin/env python3
"""
DataCorpus - Generador y validador de queries por tema.
Flujo: Selecciona temas → Genera con LLM → Valida con QueryShield → Guarda BD + JSONL
"""

import json
import uuid
import psycopg
import numpy as np
from typing import List, Dict, Optional
from openai import OpenAI
from sentence_transformers import SentenceTransformer
from query_shield import QueryShield, Estado

# ─────────────────────────────────────────────
# CONFIGURACIÓN
# ─────────────────────────────────────────────
DB_CONFIG = {
    "dbname": "datacorpus_bd", "user": "datacorpus",
    "password": "730822", "host": "127.0.0.1", "port": 5433
}

TOKEN_TOGETHER = "tgp_v1_35Ewiz4u1GT4huetCkSeITDZ9eyw-6tNcuYlSn5X7lY"

QUERIES_POR_TEMA = 1

TEMAS_VALIDOS = [
    "Medicina", "Legal / Derecho", "Finanzas", "Tecnología", "Educación / Académico",
    "Empresarial / Business", "Ciencia (General)", "Periodismo / Noticias",
    "Literatura / Humanidades", "Gaming / Entretenimiento", "E-commerce / Retail",
    "Gobierno / Política", "Ingeniería", "Arquitectura", "Marketing / Publicidad",
    "Recursos Humanos", "Contabilidad / Auditoría", "Bienes Raíces",
    "Turismo / Hospitalidad", "Agricultura", "Medio Ambiente", "Psicología",
    "Educación Física / Deportes", "Arte / Diseño", "Música", "Cine / Audiovisual",
    "Gastronomía / Culinaria", "Automoción", "Aviación", "Logística / Supply Chain"
]

# ─────────────────────────────────────────────
# MODELOS (lazy load)
# ─────────────────────────────────────────────
_embedding_model = None

def get_embedding_model() -> SentenceTransformer:
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
    return _embedding_model

def get_llm_client() -> OpenAI:
    return OpenAI(api_key=TOKEN_TOGETHER, base_url="https://api.together.xyz/v1")

# ─────────────────────────────────────────────
# SELECCIÓN DE TEMAS
# ─────────────────────────────────────────────
def seleccionar_temas() -> List[str]:
    """
    Consulta BD → calcula promedio de queries por tema
    → selecciona los 2 con menor conteo bajo el promedio.
    Fallback: si todos tienen el mismo conteo, toma los 2 primeros.
    """
    conn = psycopg.connect(**DB_CONFIG)
    cur = conn.cursor()
    cur.execute("SELECT tema, COUNT(*) FROM queries WHERE tema IS NOT NULL GROUP BY tema")
    stats = dict(cur.fetchall())
    cur.close()
    conn.close()

    conteos = {tema: stats.get(tema, 0) for tema in TEMAS_VALIDOS}
    promedio = sum(conteos.values()) / len(conteos)

    bajo_promedio = sorted(
        [(t, c) for t, c in conteos.items() if c < promedio],
        key=lambda x: x[1]
    ) or sorted(conteos.items(), key=lambda x: x[1])  # Fallback

    return [t for t, _ in bajo_promedio[:2]]

# ─────────────────────────────────────────────
# GENERACIÓN DE PREGUNTAS (LLM)
# ─────────────────────────────────────────────
def generar_preguntas_lm(tema: str, n: int) -> List[str]:
    """Genera n preguntas técnicas sobre el tema usando DeepSeek V3."""
    prompt = f"""Genera exactamente {n} preguntas únicas y específicas sobre {tema} en español.
Requisitos:
- Diversas en enfoque (cómo funciona, ventajas, desventajas, aplicaciones, etc.)
- Técnicas y específicas, no genéricas
- Una por línea, sin numeración
Evita: "¿Qué es {tema}?" o "¿Para qué sirve {tema}?"
Genera {n} preguntas sobre {tema}:"""

    response = get_llm_client().chat.completions.create(
        model="deepseek-ai/DeepSeek-V3",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=1000,
        temperature=0.7
    )

    texto = response.choices[0].message.content
    preguntas = [l.strip() for l in texto.split('\n') if l.strip() and '?' in l]
    preguntas = [p.split('. ', 1)[-1] if '. ' in p[:4] else p for p in preguntas]
    return preguntas[:n]

# ─────────────────────────────────────────────
# DECISIÓN ZONA AGENTE (LLM + fallback semántico)
# ─────────────────────────────────────────────
def decidir_agente(query_nueva: str, query_historica: str) -> bool:
    """
    Cuando QueryShield devuelve AGENTE (zona gris):
    1. Consulta LLM si son DIFERENTES o SIMILARES
    2. Fallback: similitud coseno (mantiene si ≤ 0.85)
    """
    model = get_embedding_model()
    e1, e2 = model.encode(query_nueva), model.encode(query_historica)
    similitud = float(np.dot(e1, e2) / (np.linalg.norm(e1) * np.linalg.norm(e2)))

    try:
        prompt = f"""Compara estas dos preguntas. ¿Son suficientemente diferentes o demasiado similares?
Pregunta existente: "{query_historica}"
Pregunta nueva: "{query_nueva}"
Responde SOLO con: DIFERENTES o SIMILARES"""

        response = get_llm_client().chat.completions.create(
            model="deepseek-ai/DeepSeek-V3",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=50,
            temperature=0.3
        )
        decision = response.choices[0].message.content.strip().upper()
        if "DIFERENTES" in decision: return True
        if "SIMILARES" in decision: return False

    except Exception as e:
        print(f"Fallback semántico (error LLM: {e})")

    return similitud <= 0.85  # Fallback

# ─────────────────────────────────────────────
# VALIDACIÓN Y PROCESAMIENTO (flujo principal)
# ─────────────────────────────────────────────
def validar_y_procesar(shield: QueryShield, query: str, tema: str, max_reintentos: int = 5) -> Optional[Dict]:
    """
    Flujo por query:
    - NUEVA     → Insertar en BD + log APROBADO
    - DUPLICADO → log RECHAZADO + regenerar nueva pregunta
    - AGENTE    → decidir_agente():
                    True  → Insertar + log APROBADO
                    False → log REGENERADO + nueva pregunta
    UUID se mantiene fijo a través de todos los reintentos (trazabilidad).
    """
    uid = str(uuid.uuid4())
    query_actual = query

    for _ in range(max_reintentos):
        resultado = shield.validar(uid, query_actual)

        if resultado.estado == Estado.NUEVA:
            #shield.agregar(uid, query_actual, tema)
            shield.log_validacion(uid, query_actual, resultado.uuid_parecida,
                                  Estado.NUEVA.value, resultado.score, "APROBADO", tema)
            return {"id": uid, "pregunta": query_actual, "tema": tema, "score": resultado.score}

        elif resultado.estado == Estado.DUPLICADO:
            shield.log_validacion(uid, query_actual, resultado.uuid_parecida,
                                  Estado.DUPLICADO.value, resultado.score, "RECHAZADO", tema)
            nuevas = generar_preguntas_lm(tema, n=1)
            if not nuevas: return None
            query_actual = nuevas[0]

        elif resultado.estado == Estado.AGENTE:
            if decidir_agente(query_actual, resultado.texto_historico):
                # shield.agregar(uid, query_actual, tema)
                shield.log_validacion(uid, query_actual, resultado.uuid_parecida,
                                      Estado.AGENTE.value, resultado.score, "APROBADO", tema)
                return {"id": uid, "pregunta": query_actual, "tema": tema, "score": resultado.score}
            else:
                shield.log_validacion(uid, query_actual, resultado.uuid_parecida,
                                      Estado.AGENTE.value, resultado.score, "REGENERADO", tema)
                nuevas = generar_preguntas_lm(tema, n=1)
                if not nuevas: return None
                query_actual = nuevas[0]

    # Agotó reintentos
    shield.log_validacion(uid, query_actual, None, "AGENTE", 0.0, "RECHAZADO", tema)
    return None

# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main():
    temas = seleccionar_temas()
    shield = QueryShield(DB_CONFIG)
    todas_las_queries = []

    for tema in temas:
        queries_aprobadas = []
        max_iteraciones = QUERIES_POR_TEMA * 5
        iteraciones = 0

        # Genera en lotes hasta completar QUERIES_POR_TEMA
        while len(queries_aprobadas) < QUERIES_POR_TEMA and iteraciones < max_iteraciones:
            n_generar = min((QUERIES_POR_TEMA - len(queries_aprobadas)) * 2, QUERIES_POR_TEMA)
            try:
                candidatas = generar_preguntas_lm(tema, n=n_generar)
                iteraciones += len(candidatas)

                for candidata in candidatas:
                    if len(queries_aprobadas) >= QUERIES_POR_TEMA:
                        break
                    resultado = validar_y_procesar(shield, candidata, tema)
                    if resultado:
                        queries_aprobadas.append(resultado)

            except Exception as e:
                print(f"ERROR [{tema}]: {type(e).__name__}: {e}")
                break

        todas_las_queries.extend(queries_aprobadas)

    # Guardar JSONL final
    with open("queries_validadas.jsonl", 'w', encoding='utf-8') as f:
        for q in todas_las_queries:
            f.write(json.dumps(q, ensure_ascii=False) + '\n')


if __name__ == "__main__":
    main()
