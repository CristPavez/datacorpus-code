#!/usr/bin/env python3
import json, uuid, time, requests, psycopg
from bs4 import BeautifulSoup
from urllib.parse import quote_plus, urlparse, parse_qs, unquote
from openai import OpenAI
from data_shield import DataShield, Estado
from query_shield import QueryShield

DB_CONFIG = {
    "dbname": "datacorpus_bd",
    "user": "datacorpus",
    "password": "730822",
    "host": "localhost"
}

def log_error_general(mensaje):
    import psycopg
    from datetime import datetime
    try:
        conn = psycopg.connect(**DB_CONFIG)
        cur = conn.cursor()
        cur.execute("""
            INSERT INTO logs_error (mensaje, fecha)
            VALUES (%s, %s)
        """, (mensaje, datetime.now()))
        conn.commit()
        cur.close(); conn.close()
    except Exception as e:
        print(f"[ERROR] No se pudo registrar en logs_error: {e}")

def leer_queries():
    with open("queries_validadas.jsonl", encoding="utf-8") as f:
        return [json.loads(l) for l in f]

def scrapear_duckduckgo(query: str, max_results=3):
    from datetime import datetime
    import tempfile, os
    q = query.replace('¿','').replace('?','')
    url = f"https://html.duckduckgo.com/html/?q={quote_plus(q)}"
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
        'Accept-Language': 'es-ES,es;q=0.9,en;q=0.8',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
        'Referer': 'https://duckduckgo.com/'
    }
    try:
        resp = requests.get(url, headers=headers, timeout=10)
        html = resp.text
        # Guardar HTML en archivo temporal
        with tempfile.NamedTemporaryFile(delete=False, suffix='.html', prefix='duckdbg_', mode='w', encoding='utf-8') as f:
            f.write(html)
            temp_html_path = f.name
        # Detectar posibles bloqueos/captcha
        if 'captcha' in html.lower() or 'Cloudflare' in html or 'Please verify' in html:
            print(f"      ⚠️  DuckDuckGo puede estar bloqueando el scraping o mostrando un captcha. Revisa {temp_html_path}")
            return []
        soup = BeautifulSoup(html, 'html.parser')
        results = []
        # Probar selectores alternativos
        for r in soup.select('div.result, div.web-result, div#links .result'):
            a = r.find('a', class_='result__a')
            if not a:
                a = r.find('a', href=True)
            if a:
                results.append({'url': a.get('href','')})
            if len(results) >= max_results:
                break
        if not results:
            print(f"      ⚠️  No se encontraron resultados en el HTML. Revisa {temp_html_path} para ajustar el selector.")
        return results
    except Exception as e:
        print(f"      ⚠️  Error scraping DuckDuckGo para query: {query} → {e}")
        return []

def extraer_contenido(url: str):
    try:
        import trafilatura
        if url.startswith('//'): url = 'https:' + url
        if 'duckduckgo.com/l/' in url:
            params = parse_qs(urlparse(url).query)
            url = unquote(params['uddg'][0]) if 'uddg' in params else url
        downloaded = trafilatura.fetch_url(url)
        return trafilatura.extract(downloaded) if downloaded else None
    except:
        return None

def decidir_con_llm(chunk_nuevo: str, chunk_existente: str) -> bool:
    try:
        client = OpenAI(api_key='tgp_v1_...', base_url="https://api.together.xyz/v1")
        resp = client.chat.completions.create(
            model="deepseek-ai/DeepSeek-V3",
            messages=[{"role": "user", "content":
                f"¿Son DIFERENTES o SIMILARES?\n\nExistente:\"\"\"{chunk_existente[:1000]}\"\"\"\nNuevo:\"\"\"{chunk_nuevo[:1000]}\"\"\"\n\nResponde solo: DIFERENTES o SIMILARES"}],
            max_tokens=20, temperature=0.1
        )
        return "DIFERENTES" in resp.choices[0].message.content.strip().upper()
    except:
        return False  # Rechazar por seguridad

def procesar_query(query: dict, shield: DataShield):
    qid, qtexto, tema = query['id'], query['pregunta'], query['tema']
    print(f"\n{'='*60}\n🔍 {qtexto[:80]}")

    resultados = scrapear_duckduckgo(qtexto)
    print(f"   [DEBUG] Resultados encontrados: {len(resultados)}")

    for idx, r in enumerate(resultados):
        url = r['url']
        print(f"   [DEBUG] URL resultado {idx+1}: {url}")
        contenido = extraer_contenido(url)
        if not contenido:
            print(f"   [DEBUG] No se pudo extraer contenido de la URL.")
            continue
        print(f"   [DEBUG] Longitud del contenido extraído: {len(contenido)}")
        if len(contenido) < 100:
            print(f"   [DEBUG] Contenido demasiado corto (<100 caracteres).")
            continue

        doc_id = qid  # Usar el uuid del JSONL para todo el proceso
        val = shield.validar(doc_id, contenido)
        print(f"   🌐 {url[:60]} | Estado: {val.estado.value} | Score: {val.score:.3f}")

        if val.estado == Estado.NUEVA:
            shield.agregar(doc_id, contenido, tema, url, "APROBADO")
            print("   ✅ GUARDADO"); return

        elif val.estado == Estado.DUPLICADO:
            shield.agregar_solo_logs(doc_id, contenido, url, "RECHAZADO", val.bloques_detalle)
            print("   🔴 DUPLICADO")

        elif val.estado == Estado.AGENTE:
            chunk_nuevo = val.bloque_match.get("bloque_nuevo_texto", "") if val.bloque_match else ""
            chunk_hist = ""
            if val.bloque_match and val.bloque_match.get("uuid_historico"):
                try:
                    conn = psycopg.connect(**DB_CONFIG)
                    cur = conn.cursor()
                    cur.execute("SELECT chunk FROM documents_logs WHERE uuid=%s AND chunk_numero=%s",
                                (val.bloque_match["uuid_historico"], val.bloque_match["bloque_historico"]))
                    row = cur.fetchone()
                    chunk_hist = row[0] if row else ""
                    cur.close(); conn.close()
                except: pass

            print(f"   [DEBUG] LLM comparación: chunk_nuevo({len(chunk_nuevo)}) vs chunk_hist({len(chunk_hist)})")
            if decidir_con_llm(chunk_nuevo, chunk_hist):
                shield.agregar(doc_id, contenido, tema, url, "APROBADO_LLM")
                print("   🟡 LLM → APROBADO"); return
            else:
                shield.agregar_solo_logs(doc_id, contenido, url, "RECHAZADO_LLM", val.bloques_detalle)
                print("   🟡 LLM → RECHAZADO")

    print("   ⚠️  Sin resultados aprobados")

def main():
    queries = leer_queries()
    shield = DataShield(DB_CONFIG, init_from_db=True)
    qshield = QueryShield(DB_CONFIG)

    def procesar_query_guardar_query(query, shield, qshield):
        qid, qtexto, tema = query['id'], query['pregunta'], query['tema']
        print(f"\n{'='*60}\n🔍 {qtexto[:80]}")
        resultados = scrapear_duckduckgo(qtexto)
        print(f"   [DEBUG] Resultados encontrados: {len(resultados)}")
        for idx, r in enumerate(resultados):
            url = r['url']
            print(f"   [DEBUG] URL resultado {idx+1}: {url}")
            contenido = extraer_contenido(url)
            if not contenido:
                print(f"   [DEBUG] No se pudo extraer contenido de la URL.")
                continue
            print(f"   [DEBUG] Longitud del contenido extraído: {len(contenido)}")
            if len(contenido) < 100:
                print(f"   [DEBUG] Contenido demasiado corto (<100 caracteres).")
                continue

            doc_id = qid  # Usar el uuid del JSONL
            val = shield.validar(doc_id, contenido)
            print(f"   🌐 {url[:60]} | Estado: {val.estado.value} | Score: {val.score:.3f}")

            if val.estado == Estado.NUEVA:
                shield.agregar(doc_id, contenido, tema, url, "APROBADO")
                print("   ✅ GUARDADO")
                # Guardar la query aprobada en la tabla queries
                qshield.agregar(doc_id, qtexto, tema)
                print("   ✅ Query registrada en tabla queries")
                return

            elif val.estado == Estado.DUPLICADO:
                shield.agregar_solo_logs(doc_id, contenido, url, "RECHAZADO", val.bloques_detalle)
                print("   🔴 DUPLICADO")
                # Guardar en logs de queries rechazadas (decision siempre 'RECHAZADO')
                qshield.log_validacion(doc_id, qtexto, val.bloque_match["uuid_historico"] if val.bloque_match else None, "DUPLICADO", val.score, "RECHAZADO_DOC", tema)
                print("   📝 Query registrada en logs de rechazadas")

            elif val.estado == Estado.AGENTE:
                chunk_nuevo = val.bloque_match.get("bloque_nuevo_texto", "") if val.bloque_match else ""
                chunk_hist = ""
                if val.bloque_match and val.bloque_match.get("uuid_historico"):
                    try:
                        conn = psycopg.connect(**DB_CONFIG)
                        cur = conn.cursor()
                        cur.execute("SELECT chunk FROM documents_logs WHERE uuid=%s AND chunk_numero=%s",
                                    (val.bloque_match["uuid_historico"], val.bloque_match["bloque_historico"]))
                        row = cur.fetchone()
                        chunk_hist = row[0] if row else ""
                        cur.close(); conn.close()
                    except: pass

                print(f"   [DEBUG] LLM comparación: chunk_nuevo({len(chunk_nuevo)}) vs chunk_hist({len(chunk_hist)})")
                if decidir_con_llm(chunk_nuevo, chunk_hist):
                    shield.agregar(doc_id, contenido, tema, url, "APROBADO_LLM")
                    print("   🟡 LLM → APROBADO")
                    # Guardar la query aprobada en la tabla queries
                    qshield.agregar(doc_id, qtexto, tema)
                    print("   ✅ Query registrada en tabla queries")
                    return
                else:
                    shield.agregar_solo_logs(doc_id, contenido, url, "RECHAZADO", val.bloques_detalle)
                    print("   🟡 LLM → RECHAZADO")
                    # Guardar en logs de queries rechazadas (decision siempre 'RECHAZADO')
                    qshield.log_validacion(doc_id, qtexto, val.bloque_match["uuid_historico"] if val.bloque_match else None, "AGENTE", val.score, "RECHAZADO_DOC", tema)
                    print("   📝 Query registrada en logs de rechazadas")

        print("   ⚠️  Sin resultados aprobados")

    for i, query in enumerate(queries, 1):
        print(f"\n{'#'*40} Query {i}/{len(queries)}")
        procesar_query_guardar_query(query, shield, qshield)
        time.sleep(2)

if __name__ == "__main__":
    main()
