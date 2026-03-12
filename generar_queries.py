#!/usr/bin/env python3
"""
DataCorpus — Generador y validador de queries por tema.

Flujo:
  1. Seleccionar temas con menos datos en BD (balance)
  2. Generar preguntas con DeepSeek V3
  3. Validar con QueryShield (FAISS + LLM)
  4. Guardar aprobadas en BD + JSONL
"""

import os, warnings, logging
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'
os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = '1'
warnings.filterwarnings('ignore')
logging.getLogger('transformers').setLevel(logging.ERROR)
logging.getLogger('sentence_transformers').setLevel(logging.ERROR)

import json
import uuid
import random
import psycopg
from typing import Optional
from openai import OpenAI
from query_shield import QueryShield, Estado
from config import (DB_CONFIG, TOGETHER_API_KEY, TEMAS_VALIDOS,
                    QUERIES_FILE, es_pregunta_valida)

# ── Configuración ────────────────────────────────────────────────
QUERIES_POR_TEMA = 1


# ── Cliente LLM ──────────────────────────────────────────────────
def get_llm_client() -> OpenAI:
    return OpenAI(api_key=TOGETHER_API_KEY, base_url="https://api.together.xyz/v1")


# ── Selección de temas ────────────────────────────────────────────
def seleccionar_temas() -> list[str]:
    """
    Consulta BD → calcula balance de queries por tema.
    Selecciona los 2 temas con menor conteo.
    Si todos están iguales, elige 2 al azar.
    """
    conn = psycopg.connect(**DB_CONFIG)
    cur  = conn.cursor()
    cur.execute("SELECT tema, COUNT(*) FROM queries WHERE tema IS NOT NULL GROUP BY tema")
    stats = dict(cur.fetchall())
    cur.close(); conn.close()

    conteos = {tema: stats.get(tema, 0) for tema in TEMAS_VALIDOS}
    min_count = min(conteos.values())

    # Temas con el mínimo conteo
    minimos = [t for t, c in conteos.items() if c == min_count]

    if len(minimos) == len(TEMAS_VALIDOS):
        # Todos iguales → aleatorio
        return random.sample(TEMAS_VALIDOS, 2)

    # Ordenar por conteo ascendente y tomar los 2 primeros
    ordenados = sorted(conteos.items(), key=lambda x: x[1])
    return [t for t, _ in ordenados[:2]]


# ── Generación con LLM ────────────────────────────────────────────
def generar_preguntas_lm(tema: str, n: int) -> list[str]:
    """Genera n preguntas técnicas sobre el tema usando DeepSeek V3."""
    prompt_sistema = (
        "Eres un generador de preguntas técnicas especializadas. "
        "Tu tarea es crear preguntas específicas, profundas y diversas sobre temas especializados. "
        "SOLO genera las preguntas, una por línea, sin numeración ni explicaciones adicionales."
    )
    prompt_usuario = (
        f"Genera exactamente {n} preguntas técnicas sobre '{tema}' en español.\n\n"
        f"Requisitos:\n"
        f"- Cada pregunta debe ser específica y profunda, no genérica\n"
        f"- Diversas en enfoque: cómo funciona, ventajas, desventajas, aplicaciones, casos de uso, comparaciones, etc.\n"
        f"- Evita preguntas obvias como '¿Qué es {tema}?' o '¿Para qué sirve {tema}?'\n"
        f"- Formato: Una pregunta por línea, sin numeración\n\n"
        f"Genera {n} preguntas sobre {tema}:"
    )

    response = get_llm_client().chat.completions.create(
        model="deepseek-ai/DeepSeek-V3",
        messages=[
            {"role": "system", "content": prompt_sistema},
            {"role": "user",   "content": prompt_usuario},
        ],
        max_tokens=1000,
        temperature=0.8,
    )

    texto = response.choices[0].message.content.strip()
    lineas = []
    for l in texto.split('\n'):
        l = l.strip()
        if not l:
            continue
        # Quitar numeración tipo "1. Pregunta?"
        l = l.split('. ', 1)[-1] if '. ' in l[:4] else l
        lineas.append(l)

    return [l for l in lineas if es_pregunta_valida(l)][:n]


# ── Decisión LLM para zona AGENTE ─────────────────────────────────
def decidir_agente(query_nueva: str, query_historica: str) -> bool:
    """
    Devuelve True si el LLM dice que son DIFERENTES (aprobar).
    Devuelve False si dice SIMILARES (regenerar).
    """
    prompt = (
        f"Compara estas dos preguntas. ¿Son suficientemente diferentes o demasiado similares?\n"
        f"Pregunta existente: \"{query_historica}\"\n"
        f"Pregunta nueva: \"{query_nueva}\"\n"
        f"Responde SOLO con: DIFERENTES o SIMILARES"
    )
    try:
        response = get_llm_client().chat.completions.create(
            model="deepseek-ai/DeepSeek-V3",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=50,
            temperature=0.3,
        )
        decision = response.choices[0].message.content.strip().upper()
        if "DIFERENTES" in decision: return True
        if "SIMILARES"  in decision: return False
    except Exception as e:
        print(f"      ⚠️  Error LLM zona agente: {e} — usando fallback")

    return False   # Fallback conservador: regenerar


# ── Flujo de validación por query ─────────────────────────────────
def validar_y_procesar(shield: QueryShield, query: str, tema: str,
                       max_reintentos: int = 5,
                       uuid_forzado: Optional[str] = None) -> Optional[dict]:
    """
    Intenta aprobar una query con hasta max_reintentos intentos.
    El UUID se mantiene fijo a lo largo de todos los reintentos (trazabilidad).

    uuid_forzado: si se pasa, usa ese UUID en vez de generar uno nuevo.
                  Útil en el flujo reparador para mantener el UUID original.

    Retorna dict {id, pregunta, tema} si se aprobó, None si se agotaron reintentos.
    """
    uid          = uuid_forzado if uuid_forzado else str(uuid.uuid4())
    query_actual = query

    for intento in range(max_reintentos):
        resultado = shield.validar(uid, query_actual)

        if resultado.estado == Estado.NUEVA:
            # Solo insertar en queries + FAISS. El log APROBADO lo escribe
            # scrapear_queries cuando el documento queda guardado.
            shield.agregar(uid, query_actual, tema)
            return {"id": uid, "pregunta": query_actual, "tema": tema}

        elif resultado.estado == Estado.DUPLICADA:
            shield.log(uid, query_actual, resultado.score,
                       resultado.uuid_parecida, "DUPLICADA", tema)
            nuevas = generar_preguntas_lm(tema, n=1)
            if not nuevas:
                return None
            query_actual = nuevas[0]

        elif resultado.estado == Estado.AGENTE:
            es_diferente = decidir_agente(query_actual, resultado.texto_historico or "")
            if es_diferente:
                # Mismo criterio: solo agregar, el log APROBADO viene del scraper.
                shield.agregar(uid, query_actual, tema)
                return {"id": uid, "pregunta": query_actual, "tema": tema}
            else:
                shield.log(uid, query_actual, resultado.score,
                           resultado.uuid_parecida, "SIMILAR", tema)
                nuevas = generar_preguntas_lm(tema, n=1)
                if not nuevas:
                    return None
                query_actual = nuevas[0]

    # Agotó reintentos sin aprobar
    return None


# ── MAIN ──────────────────────────────────────────────────────────
def main(qshield_externo: Optional[QueryShield] = None):
    temas  = seleccionar_temas()
    shield = qshield_externo or QueryShield(DB_CONFIG)
    todas  = []

    print(f"   Temas seleccionados: {', '.join(temas)}")
    print(f"   Objetivo: {QUERIES_POR_TEMA} query(s) por tema\n")

    for tema in temas:
        print(f"   Procesando tema: {tema}")
        aprobadas       = []
        max_iter        = QUERIES_POR_TEMA * 6
        iter_count      = 0
        errores_consec  = 0

        while len(aprobadas) < QUERIES_POR_TEMA and iter_count < max_iter:
            n_generar = min((QUERIES_POR_TEMA - len(aprobadas)) * 2, QUERIES_POR_TEMA + 1)
            try:
                candidatas    = generar_preguntas_lm(tema, n=n_generar)
                errores_consec = 0
            except Exception as e:
                print(f"      ❌ Error generando candidatas: {e}")
                errores_consec += 1
                if errores_consec >= 3:
                    print(f"      ⚠️  3 errores consecutivos, abandonando tema")
                    break
                continue

            iter_count += len(candidatas)
            for candidata in candidatas:
                if len(aprobadas) >= QUERIES_POR_TEMA:
                    break
                try:
                    resultado = validar_y_procesar(shield, candidata, tema)
                except Exception as e:
                    print(f"      ❌ Error validando: {e}")
                    continue
                if resultado:
                    aprobadas.append(resultado)
                    print(f"      ✅ {len(aprobadas)}/{QUERIES_POR_TEMA} — {resultado['pregunta'][:70]}...")

        todas.extend(aprobadas)
        print(f"   Tema '{tema}' completado: {len(aprobadas)}/{QUERIES_POR_TEMA} aprobadas\n")

    # Guardar JSONL
    with open(QUERIES_FILE, 'w', encoding='utf-8') as f:
        for q in todas:
            if "score" in q and hasattr(q["score"], "item"):
                q["score"] = float(q["score"])
            f.write(json.dumps(q, ensure_ascii=False) + '\n')

    print(f"   Total queries guardadas: {len(todas)}")
    print(f"   Archivo: {QUERIES_FILE}")


if __name__ == "__main__":
    main()
