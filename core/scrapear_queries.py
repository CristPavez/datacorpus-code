#!/usr/bin/env python3
"""
DataCorpus — Scraper inteligente de queries.

Flujo por query:
  1. Buscar 3 URLs en Brave Search
  2. Por cada URL: extraer contenido → chunkear con NLTK
  3. Validar cada chunk con FAISS (DataShield):
       score == 1.0          → DUPLICADA (cuenta para umbral 50%)
       score >= 0.90 < 1.0   → LLM decide → SIMILAR (cuenta) o APROBADO
       score < 0.90          → APROBADO
  4. Si al procesar se alcanza 50% de chunks rechazados → stop, resto = OMITIDA
     → descartar URL, probar siguiente
  5. Si todas las URLs son descartadas → query OMITIDA en queries_logs
  6. Si ninguna URL tiene contenido   → query SIN_RESULTADOS en queries_logs
  7. Si se aprueba una URL → guardar documents + documents_logs + queries
"""

import json
import time
import psycopg
import requests
from urllib.parse import urlparse, parse_qs, unquote
from openai import OpenAI
from .data_shield import DataShield, ResultadoChunk
from .query_shield import QueryShield
from .config import DB_CONFIG, BRAVE_API_KEY, TOGETHER_API_KEY, QUERIES_FILE


# ── Cliente LLM ──────────────────────────────────────────────────
def get_llm_client() -> OpenAI:
    return OpenAI(api_key=TOGETHER_API_KEY, base_url="https://api.together.xyz/v1")


# ── Brave Search ──────────────────────────────────────────────────
def buscar_brave(query: str, max_results: int = 3) -> list[dict]:
    """Retorna lista de {'url': ...}. Reintentos con backoff en 429."""
    url     = "https://api.search.brave.com/res/v1/web/search"
    headers = {
        "Accept": "application/json",
        "Accept-Encoding": "gzip",
        "X-Subscription-Token": BRAVE_API_KEY,
    }
    params = {
        "q": query, "count": max_results,
        "search_lang": "es", "country": "es",
    }

    for intento in range(3):
        try:
            resp = requests.get(url, headers=headers, params=params, timeout=10)
            resp.raise_for_status()
            data = resp.json()
            results = []
            if "web" in data and "results" in data["web"]:
                for item in data["web"]["results"][:max_results]:
                    results.append({"url": item.get("url", "")})
            return results
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 429:
                espera = 2 ** intento
                print(f"      ⚠️  Rate limit. Reintentando en {espera}s...")
                time.sleep(espera)
                continue
            print(f"      ⚠️  HTTP {e.response.status_code}: {e}")
            return []
        except Exception as e:
            print(f"      ⚠️  Error Brave Search: {e}")
            return []

    return []


# ── Extracción de contenido ───────────────────────────────────────
def extraer_contenido(url: str) -> str | None:
    try:
        import trafilatura
        if url.startswith('//'):
            url = 'https:' + url
        if 'duckduckgo.com/l/' in url:
            params = parse_qs(urlparse(url).query)
            url = unquote(params['uddg'][0]) if 'uddg' in params else url
        downloaded = trafilatura.fetch_url(url)
        return trafilatura.extract(downloaded) if downloaded else None
    except Exception as e:
        print(f"      ⚠️  Error extrayendo {url[:60]}: {e}")
        return None


# ── Decisión LLM para chunk en zona AGENTE ────────────────────────
def decidir_chunk_llm(chunk_nuevo: str, chunk_existente: str) -> bool:
    """
    Retorna True si el chunk nuevo aporta información diferente (APROBADO).
    Retorna False si es redundante (SIMILAR).
    """
    try:
        prompt_sistema = (
            "Actúa como un analista de datos. Tu objetivo es identificar si un texto nuevo "
            "aporta información, métricas, pasos de proceso o conceptos que NO están en el texto existente."
        )
        prompt_usuario = (
            f"Texto Existente: \"\"\"{chunk_existente[:1500]}\"\"\"\n\n"
            f"Texto Nuevo: \"\"\"{chunk_nuevo[:1500]}\"\"\"\n\n"
            "Analiza: ¿El 'Texto Nuevo' contiene algún detalle específico, entidad o matiz "
            "que no esté en el 'Texto Existente'?\n"
            "Responde únicamente: 'NUEVO' si aporta algo diferente, o 'DUPLICADO' si es redundante."
        )
        resp = get_llm_client().chat.completions.create(
            model="deepseek-ai/DeepSeek-V3",
            messages=[
                {"role": "system", "content": prompt_sistema},
                {"role": "user",   "content": prompt_usuario},
            ],
            max_tokens=10,
            temperature=0.0,
        )
        resultado = resp.choices[0].message.content.strip().upper()
        return "NUEVO" in resultado
    except Exception as e:
        print(f"      ⚠️  Error LLM chunk: {e} — conservando (APROBADO)")
        return True  # Conservador: no perder datos por fallo de API


# ── Procesamiento de chunks con umbral 50% ────────────────────────
def procesar_chunks_url(dshield: DataShield, chunks: list[str]) -> list[ResultadoChunk]:
    """
    Procesa chunks uno a uno.
    En cuanto se alcanza 50% de rechazados, los restantes son OMITIDA.
    Retorna la lista completa de ResultadoChunk.
    """
    total     = len(chunks)
    umbral    = total * DataShield.RECHAZO_UMBRAL
    rechazados = 0
    resultados = []

    for i, chunk in enumerate(chunks):
        # ¿Ya se alcanzó el umbral?
        if rechazados >= umbral:
            # Resto = OMITIDA
            for j in range(i, total):
                resultados.append(ResultadoChunk(
                    chunk_numero=j, chunk_text=chunks[j], estado="OMITIDA",
                    score=0.0, uuid_chunk_similar=None,
                    chunk_numero_similar=None, embedding=None
                ))
            break

        emb = dshield.embed(chunk)
        uuid_sim, num_sim, score = dshield.buscar_similar(emb)

        if score >= DataShield.DUPLICADA_THRESHOLD:
            estado = "DUPLICADA"
            rechazados += 1
            print(f"         Chunk #{i} score={score:.3f} → DUPLICADA")

        elif score >= DataShield.AGENT_ZONE_LOW:
            # Zona AGENTE → LLM
            chunk_hist = dshield.get_chunk_text(uuid_sim, num_sim) if uuid_sim else ""
            print(f"         Chunk #{i} score={score:.3f} → LLM...", end=" ")
            es_diferente = decidir_chunk_llm(chunk, chunk_hist)
            if es_diferente:
                estado = "APROBADO"
                print("APROBADO")
            else:
                estado = "SIMILAR"
                rechazados += 1
                print("SIMILAR")

        else:
            estado = "APROBADO"

        resultados.append(ResultadoChunk(
            chunk_numero=i, chunk_text=chunk, estado=estado,
            score=score, uuid_chunk_similar=uuid_sim,
            chunk_numero_similar=num_sim, embedding=emb
        ))

    return resultados


# ── Procesar una query ────────────────────────────────────────────
def procesar_query(query: dict, dshield: DataShield, qshield: QueryShield,
                   idx: int, total: int):
    qid     = query["id"]
    qtexto  = query["pregunta"]
    tema    = query["tema"]

    print(f"\n{'#'*70}")
    print(f"Query {idx}/{total}")
    print(f"{'#'*70}")
    print(f"\n  {qtexto[:80]}")

    # ── 1. Buscar URLs ─────────────────────────────────────────────
    urls = buscar_brave(qtexto)
    if not urls:
        print(f"   ⚠️  Sin resultados de búsqueda")
        qshield.log(qid, qtexto, 0.0, None, "SIN_RESULTADOS", tema)
        return

    print(f"   {len(urls)} URL(s) encontradas\n")

    hubo_contenido  = False   # alguna URL tuvo texto extraíble
    url_aprobada    = False   # alguna URL pasó el 50%

    for url_data in urls:
        url = url_data["url"]
        print(f"   URL: {url[:70]}{'...' if len(url) > 70 else ''}")

        contenido = extraer_contenido(url)
        if not contenido or len(contenido) < 100:
            print(f"      ❌ Sin contenido extraíble\n")
            continue

        hubo_contenido = True
        print(f"      {len(contenido)} chars extraídos")

        chunks = dshield.split_text(contenido)
        print(f"      {len(chunks)} chunks generados")

        chunk_resultados = procesar_chunks_url(dshield, chunks)

        # Calcular rechazo final
        total_chunks   = len(chunks)
        rechazados_tot = sum(
            1 for c in chunk_resultados
            if c.estado in ("DUPLICADA", "SIMILAR")
        )
        pct_rechazo = rechazados_tot / total_chunks if total_chunks else 0

        print(f"      Rechazo: {rechazados_tot}/{total_chunks} ({pct_rechazo:.0%})")

        if pct_rechazo >= DataShield.RECHAZO_UMBRAL:
            print(f"      ❌ Umbral 50% alcanzado → URL descartada\n")
            dshield.log_rechazado(qid, url, chunk_resultados)
            continue

        # ── URL aprobada ───────────────────────────────────────────
        print(f"      ✅ URL aprobada → guardando documento...")
        dshield.agregar(qid, contenido, tema, url, chunk_resultados)
        qshield.agregar(qid, qtexto, tema)
        qshield.log(qid, qtexto, 0.0, None, "APROBADO", tema)
        url_aprobada = True
        print(f"      Documento y query guardados\n")
        break  # Con una URL aprobada es suficiente

    # ── Resultado final de la query ────────────────────────────────
    if not url_aprobada:
        if hubo_contenido:
            print(f"   ⚠️  Todas las URLs rechazadas por similitud → OMITIDA\n")
            qshield.log(qid, qtexto, 0.0, None, "OMITIDA", tema)
        else:
            print(f"   ⚠️  Ninguna URL tuvo contenido extraíble → SIN_RESULTADOS\n")
            qshield.log(qid, qtexto, 0.0, None, "SIN_RESULTADOS", tema)


# ── MAIN ──────────────────────────────────────────────────────────
def main(dshield_externo: DataShield = None, qshield_externo: QueryShield = None,
         stop_event=None):
    with open(QUERIES_FILE, encoding="utf-8") as f:
        queries = [json.loads(l) for l in f]

    dshield = dshield_externo or DataShield(DB_CONFIG)
    qshield = qshield_externo or QueryShield(DB_CONFIG)
    total   = len(queries)

    print(f"   Total queries a procesar: {total}\n")

    for i, query in enumerate(queries, 1):
        if stop_event and stop_event.is_set():
            print(f"   ⏹️  Detención solicitada — scraping interrumpido en query {i}/{total}")
            break
        procesar_query(query, dshield, qshield, i, total)
        if i < total:
            time.sleep(1)


if __name__ == "__main__":
    main()
