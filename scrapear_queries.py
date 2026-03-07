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

# ── BRAVE SEARCH API ──────────────────────────────────────────────
# Coloca aquí tu API key de Brave Search
BRAVE_API_KEY = "BSA9TKRCDdLxynAieJYPeqx6A1BPcLH"

# ── TOGETHER AI API ───────────────────────────────────────────────
# API key para el modelo DeepSeek V3
TOGETHER_API_KEY = "tgp_v1_35Ewiz4u1GT4huetCkSeITDZ9eyw-6tNcuYlSn5X7lY"

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

def buscar_brave(query: str, max_results=3):
    """
    Busca usando la API de Brave Search.
    Más confiable que el scraping de DuckDuckGo.
    """
    url = "https://api.search.brave.com/res/v1/web/search"
    headers = {
        "Accept": "application/json",
        "Accept-Encoding": "gzip",
        "X-Subscription-Token": BRAVE_API_KEY
    }
    params = {
        "q": query,
        "count": max_results,
        "search_lang": "es",
        "country": "es"
    }
    
    try:
        resp = requests.get(url, headers=headers, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        
        results = []
        if "web" in data and "results" in data["web"]:
            for item in data["web"]["results"][:max_results]:
                results.append({'url': item.get('url', '')})
        
        if not results:
            print(f"      ⚠️  No se encontraron resultados para la query: {query}")
        
        return results
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 401:
            print(f"      ⚠️  Error de autenticación: Verifica tu BRAVE_API_KEY")
        elif e.response.status_code == 429:
            print(f"      ⚠️  Límite de requests alcanzado. Espera antes de continuar.")
        else:
            print(f"      ⚠️  Error HTTP {e.response.status_code}: {e}")
        return []
    except Exception as e:
        print(f"      ⚠️  Error buscando en Brave Search: {e}")
        return []

def scrapear_duckduckgo(query: str, max_results=3):
    """
    DEPRECADO: Usa buscar_brave() en su lugar.
    Se mantiene como fallback por compatibilidad.
    """
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
    """
    Evalúa si el chunk nuevo aporta valor informativo adicional 
    comparado con el chunk existente, incluso si el tema es el mismo.
    Retorna True si el chunk nuevo aporta información diferente (GUARDAR).
    Retorna False si es redundante (RECHAZAR).
    """
    try:
        # Usamos un prompt de "Novedad Informativa" en lugar de similitud simple
        prompt_sistema = (
            "Actúa como un analista de datos. Tu objetivo es identificar si un texto nuevo "
            "aporta información, métricas, pasos de proceso o conceptos que NO están en el texto existente."
        )
        
        prompt_usuario = (
            f"Texto Existente: \"\"\"{chunk_existente[:1500]}\"\"\"\n\n"
            f"Texto Nuevo: \"\"\"{chunk_nuevo[:1500]}\"\"\"\n\n"
            "Analiza: ¿El 'Texto Nuevo' contiene algún detalle específico, entidad o matiz que no esté en el 'Texto Existente'?\n"
            "Responde únicamente: 'NUEVO' si aporta algo diferente, o 'DUPLICADO' si es redundante."
        )

        client = OpenAI(api_key=TOGETHER_API_KEY, base_url="https://api.together.xyz/v1")
        resp = client.chat.completions.create(
            model="deepseek-ai/DeepSeek-V3",  # Excelente elección por su razonamiento
            messages=[
                {"role": "system", "content": prompt_sistema},
                {"role": "user", "content": prompt_usuario}
            ],
            max_tokens=10, 
            temperature=0.0  # Importante: Queremos determinismo total
        )
        
        resultado = resp.choices[0].message.content.strip().upper()
        
        # Logica: Si el LLM dice que es NUEVO, retornamos True para guardar
        es_nuevo = "NUEVO" in resultado
        return es_nuevo
        
    except Exception as e:
        print(f"      ⚠️  Error en validación LLM: {e}")
        # En caso de error, es mejor GUARDAR (True) para no perder datos por un fallo de API
        return True

def procesar_query(query: dict, shield: DataShield):
    qid, qtexto, tema = query['id'], query['pregunta'], query['tema']
    print(f"\n{'='*60}\n🔍 {qtexto[:80]}")

    resultados = buscar_brave(qtexto)
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
        print(f"   🌐 {url[:60]} | Estado: {val.estado.value} | Score: {val.score:.3f} | {val.razon}")

        if val.estado == Estado.NUEVA:
            shield.agregar(doc_id, contenido, tema, url, "APROBADO")
            print("   ✅ GUARDADO"); return

        elif val.estado == Estado.DUPLICADO:
            shield.agregar_solo_logs(doc_id, contenido, url, "RECHAZADO", val.bloques_detalle)
            print("   🔴 DUPLICADO → Probando siguiente URL")
            continue  # Pasar a la siguiente URL

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
                print("   🟡 LLM → RECHAZADO → Probando siguiente URL")
                continue  # Pasar a la siguiente URL

    print("   ⚠️  Sin resultados aprobados")

def main(dshield_externo=None, qshield_externo=None):
    """
    Args:
        dshield_externo: DataShield pre-inicializado (para evitar recargar modelo)
        qshield_externo: QueryShield pre-inicializado (para evitar recargar modelo)
    """
    queries = leer_queries()
    shield = dshield_externo if dshield_externo else DataShield(DB_CONFIG, init_from_db=True)
    qshield = qshield_externo if qshield_externo else QueryShield(DB_CONFIG)
    
    print(f"   📊 Total de queries a procesar: {len(queries)}\n")

    def procesar_query_guardar_query(query, shield, qshield):
        qid, qtexto, tema = query['id'], query['pregunta'], query['tema']
        print(f"\n{'#'*70}")
        print(f"Query {queries.index(query)+1}/{len(queries)}")
        print(f"{'#'*70}")
        print(f"\n🔍 {qtexto[:80]}")
        
        resultados = buscar_brave(qtexto)
        if not resultados:
            print(f"   ⚠️  Sin resultados - Query omitida")
            return
        
        print(f"   🎯 {len(resultados)} URLs encontradas\n")
        
        # Rastrear la mejor validación rechazada para registrar en logs si todas fallan
        mejor_rechazo = None
        
        for idx, r in enumerate(resultados):
            url = r['url']
            print(f"   🔗 URL {idx+1}: {url[:65]}{'...' if len(url) > 65 else ''}")
            
            contenido = extraer_contenido(url)
            if not contenido:
                print(f"      ❌ No se pudo extraer contenido\n")
                continue
            
            print(f"      📏 Contenido extraído: {len(contenido)} chars")
            
            if len(contenido) < 100:
                print(f"      ⚠️  Contenido muy corto (<100 chars)\n")
                continue

            doc_id = qid  # Usar el uuid del JSONL
            val = shield.validar(doc_id, contenido)
            
            # Construir mensaje de estado con colores
            if val.estado == Estado.NUEVA:
                icono = "✅"
                estado_msg = f"🎯 NUEVO"
            elif val.estado == Estado.AGENTE:
                icono = "🟡"
                estado_msg = f"⚠️  AGENTE (score: {val.score:.3f})"
            else:
                icono = "❌"
                estado_msg = f"❌ {val.estado.value.upper()}"
            
            print(f"      {icono} {estado_msg}")
            print(f"      📄 {val.razon}")

            if val.estado == Estado.NUEVA:
                shield.agregar(doc_id, contenido, tema, url, "APROBADO")
                print("      💾 Documento guardado")
                qshield.agregar(doc_id, qtexto, tema)
                print("      📝 Query registrada en BD")
                print(f"      ✅ APROBADO - Finalizando query\n")
                return  # ✅ Salir si se aprueba

            elif val.estado == Estado.DUPLICADO:
                shield.agregar_solo_logs(doc_id, contenido, url, "RECHAZADO", val.bloques_detalle)
                print("      ⏭️  Probando siguiente URL...\n")
                # Guardar info para registrar en logs si todas las URLs fallan
                if not mejor_rechazo or val.score > mejor_rechazo['val'].score:
                    mejor_rechazo = {'val': val, 'estado': 'DUPLICADO'}
                continue  # Pasar a la siguiente URL

            elif val.estado == Estado.AGENTE:
                # NUEVO: Validar TODOS los chunks en zona AGENTE con LLM
                chunks_agente = [b for b in val.bloques_detalle if b["estado"] == "AGENTE"]
                print(f"      🧠 Validando {len(chunks_agente)} chunk(s) en zona AGENTE con LLM...")
                
                # Obtener el texto original dividido en chunks para acceder a los textos nuevos
                bloques_originales = shield._split_text(contenido)
                
                aprobado_por_llm = True  # Asumimos aprobado hasta que LLM rechace
                
                for bloque in chunks_agente:
                    idx_chunk = bloque["indice"]
                    chunk_nuevo = bloques_originales[idx_chunk] if idx_chunk < len(bloques_originales) else ""
                    chunk_hist = ""
                    
                    # Obtener chunk histórico de BD
                    if bloque.get("uuid_historico"):
                        try:
                            conn = psycopg.connect(**DB_CONFIG)
                            cur = conn.cursor()
                            cur.execute(
                                "SELECT chunk FROM documents_logs WHERE uuid=%s AND chunk_numero=%s",
                                (bloque["uuid_historico"], bloque["chunk_num_historico"])
                            )
                            row = cur.fetchone()
                            chunk_hist = row[0] if row else ""
                            cur.close()
                            conn.close()
                        except Exception as e:
                            print(f"         ⚠️  Error obteniendo chunk histórico: {e}")
                    
                    # Validar con LLM
                    print(f"         🔄 Chunk #{idx_chunk} (score: {bloque['score']:.3f})...", end=" ")
                    decision_llm = decidir_con_llm(chunk_nuevo, chunk_hist)
                    
                    if decision_llm:  # TRUE → NUEVO
                        print("✅ DIFERENTE")
                        continue  # Pasar al siguiente chunk
                    else:  # FALSE → DUPLICADO
                        print("❌ DUPLICADO")
                        aprobado_por_llm = False
                        break  # Detener validación inmediatamente
                
                # Decisión final después de validar todos los chunks AGENTE
                if aprobado_por_llm:
                    shield.agregar(doc_id, contenido, tema, url, "APROBADO_LLM")
                    print("      ✅ TODOS los chunks aprobados por LLM")
                    print("      💾 Documento guardado")
                    qshield.agregar(doc_id, qtexto, tema)
                    print("      📝 Query registrada en BD")
                    print(f"      ✅ APROBADO - Finalizando query\n")
                    return  # ✅ Salir si LLM aprueba todos
                else:
                    shield.agregar_solo_logs(doc_id, contenido, url, "RECHAZADO_LLM", val.bloques_detalle)
                    print("      🧠 LLM rechazó al menos 1 chunk → Documento rechazado")
                    print("      ⏭️  Probando siguiente URL...\n")
                    # Guardar info para registrar en logs si todas las URLs fallan
                    if not mejor_rechazo or val.score > mejor_rechazo['val'].score:
                        mejor_rechazo = {'val': val, 'estado': 'AGENTE'}
                    continue  # Pasar a la siguiente URL

        # Si llegamos aquí, ninguna URL fue aprobada
        print("   ⚠️  Todas las URLs fueron rechazadas - Query sin contenido aprobado\n")
        
        # Registrar en logs de queries rechazadas con la mejor validación encontrada
        if mejor_rechazo:
            qshield.log_validacion(
                doc_id, qtexto, 
                mejor_rechazo['val'].bloque_match["uuid_historico"] if mejor_rechazo['val'].bloque_match else None, 
                mejor_rechazo['estado'], 
                mejor_rechazo['val'].score, 
                "RECHAZADO_DOC", 
                tema
            )

    # Procesar todas las queries
    for i, query in enumerate(queries, 1):
        procesar_query_guardar_query(query, shield, qshield)
        if i < len(queries):  # Pequeña pausa entre queries (excepto la última)
            time.sleep(1)

if __name__ == "__main__":
    main()
