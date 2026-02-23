#!/usr/bin/env python3
"""
Script de prueba: Scraping y validación de 5 queries
Flujo crítico: DATO → MAPEO → QUERY (solo si todo OK)
Con decisión LLM para estado AGENTE.
"""

import json
import uuid
import requests
import psycopg2
from bs4 import BeautifulSoup
from data_shield import DataShield
from urllib.parse import quote_plus
from typing import Optional, List, Dict
import time
from openai import OpenAI
import os
from sentence_transformers import SentenceTransformer

DB_CONFIG = {
    "dbname": "datacorpus_bd",
    "user": "datacorpus",
    "password": "730822",
    "host": "localhost"
}

# Modelo de embeddings para queries (384d - mismo que QueryShield)
query_embedding_model = None

def get_query_embedding_model():
    """Lazy load del modelo de embeddings de queries"""
    global query_embedding_model
    if query_embedding_model is None:
        print("   📥 Cargando modelo de embeddings de queries...")
        query_embedding_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
    return query_embedding_model


def cargar_token_together() -> str:
    """Carga el token de Together.ai desde las credenciales"""
    token_path = os.path.expanduser("~/.openclaw/credentials/together.token.json")
    try:
        with open(token_path, 'r') as f:
            data = json.load(f)
            return data['token']
    except FileNotFoundError:
        raise FileNotFoundError(
            f"No se encontró el archivo de token en {token_path}\n"
            "Crea el archivo con: echo '{{\"token\": \"tgp_v1_...\"}}' > {token_path}"
        )
    except KeyError:
        raise ValueError(f"El archivo {token_path} debe contener un campo 'token'")


def decidir_con_llm(texto_nuevo: str, texto_existente: str, num_chunks_similares: int = 1, total_chunks: int = 1) -> bool:
    """
    Consulta a DeepSeek V3.1 para decidir si dos chunks son suficientemente diferentes.
    
    Args:
        texto_nuevo: Chunk del documento nuevo
        texto_existente: Chunk del documento existente similar
        num_chunks_similares: Cantidad de chunks similares detectados
        total_chunks: Total de chunks en el documento
    
    Returns:
        True si son suficientemente diferentes (mantener documento)
        False si son demasiado similares (rechazar documento)
    """
    print(f"      🤖 Consultando a DeepSeek V3.1 para decidir...")
    print(f"      📊 Contexto: {num_chunks_similares}/{total_chunks} chunks requieren validación")
    
    try:
        token = cargar_token_together()
        client = OpenAI(
            api_key=token,
            base_url="https://api.together.xyz/v1"
        )
        
        # Limitar longitud de los textos para no exceder tokens
        max_chars = 1500
        texto_nuevo_preview = texto_nuevo[:max_chars] + ("..." if len(texto_nuevo) > max_chars else "")
        texto_existente_preview = texto_existente[:max_chars] + ("..." if len(texto_existente) > max_chars else "")
        
        # Agregar contexto sobre múltiples chunks si aplica
        contexto_adicional = ""
        if num_chunks_similares > 1:
            contexto_adicional = f"\n\nIMPORTANTE: Este documento tiene {num_chunks_similares} de {total_chunks} fragmentos con similitud alta. Si este fragmento es similar, probablemente el documento completo sea redundante."
        
        prompt = f"""Eres un experto en análisis de contenido. Compara estos dos fragmentos de texto y decide si contienen información SUFICIENTEMENTE DIFERENTE o si son DEMASIADO SIMILARES.

Fragmento existente:
\"\"\"{texto_existente_preview}\"\"\"

Fragmento nuevo:
\"\"\"{texto_nuevo_preview}\"\"\"{contexto_adicional}

Criterio:
- SUFICIENTEMENTE DIFERENTES: Si abordan aspectos distintos, tienen enfoques diferentes, o proporcionan información complementaria.
- DEMASIADO SIMILARES: Si contienen básicamente la misma información, aunque estén redactados de forma diferente.

Responde SOLO con una de estas dos palabras:
- DIFERENTES
- SIMILARES"""

        response = client.chat.completions.create(
            model="deepseek-ai/DeepSeek-V3",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=50,
            temperature=0.3
        )
        
        decision = response.choices[0].message.content.strip().upper()
        
        if "DIFERENTES" in decision:
            print(f"      ✅ Modelo dice: DIFERENTES → Mantener documento")
            return True
        elif "SIMILARES" in decision:
            print(f"      ❌ Modelo dice: SIMILARES → Rechazar documento")
            return False
        else:
            print(f"      ⚠️  Respuesta ambigua: '{decision}', rechazando por seguridad")
            return False
    
    except Exception as e:
        print(f"      ❌ Error consultando modelo: {e}")
        print(f"      ⚠️  Rechazando por seguridad (fallback conservador)")
        return False


def leer_queries(n=5) -> List[Dict]:
    """Lee las primeras N queries del JSONL (n=None para leer todas)"""
    
    # Primero intentar archivo de prueba local
    queries_path = "queries_test.jsonl"
    
    # Si no existe, intentar del workspace datacorpus-agente
    import os
    if not os.path.exists(queries_path):
        queries_path = "/home/datacorpus/.openclaw/workspace/datacorpus-agente/queries_validadas.jsonl"
    
    queries = []
    try:
        with open(queries_path, 'r') as f:
            for i, line in enumerate(f):
                if n is not None and i >= n:
                    break
                queries.append(json.loads(line))
        print(f"   Usando: {queries_path}")
        return queries
    except FileNotFoundError:
        print(f"❌ No se encontró archivo de queries")
        return []


def simplificar_query(query_texto: str) -> str:
    """
    Simplifica una query larga removiendo signos de interrogación y palabras comunes.
    Retorna una versión más corta para scraping que funciona mejor en motores de búsqueda.
    """
    # Remover signos de interrogación
    query_simple = query_texto.replace('¿', '').replace('?', '')
    
    # Palabras a remover (stopwords comunes en español)
    stopwords = ['cómo', 'qué', 'cuál', 'cuáles', 'dónde', 'cuándo', 'por qué', 
                 'para qué', 'en', 'el', 'la', 'los', 'las', 'un', 'una', 'de', 
                 'del', 'al', 'se', 'es', 'son', 'está', 'están', 'y', 'o', 'a']
    
    palabras = query_simple.lower().split()
    palabras_filtradas = [p for p in palabras if p not in stopwords and len(p) > 2]
    
    # Limitar a 6-8 palabras clave
    return ' '.join(palabras_filtradas[:8])


def scrapear_duckduckgo(query: str, max_results=3) -> List[Dict]:
    """
    Scrapea DuckDuckGo HTML y extrae URLs de resultados.
    Si no encuentra resultados con la query original, intenta con una versión simplificada.
    Retorna lista de dicts con {title, url, snippet}
    """
    # Intentar primero con query original
    query_simple = simplificar_query(query)
    url = f"https://html.duckduckgo.com/html/?q={quote_plus(query_simple)}"
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
    }
    
    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, 'html.parser')
        resultados = []
        
        # Buscar resultados en DuckDuckGo HTML
        for result in soup.find_all('div', class_='result')[:max_results]:
            title_elem = result.find('a', class_='result__a')
            snippet_elem = result.find('a', class_='result__snippet')
            
            if title_elem:
                title = title_elem.get_text(strip=True)
                href = title_elem.get('href', '')
                snippet = snippet_elem.get_text(strip=True) if snippet_elem else ''
                
                resultados.append({
                    'title': title,
                    'url': href,
                    'snippet': snippet
                })
        
        return resultados
    
    except Exception as e:
        print(f"   ❌ Error scraping DuckDuckGo: {e}")
        return []


def extraer_contenido(url: str) -> Optional[str]:
    """
    Extrae contenido limpio de una URL usando trafilatura.
    Maneja URLs de redirect de DuckDuckGo extrayendo el parámetro uddg.
    Retorna texto limpio o None si falla.
    """
    try:
        import trafilatura
        from urllib.parse import urlparse, parse_qs, unquote
        
        # Arreglar URLs que empiezan con // (agregar https:)
        if url.startswith('//'):
            url = 'https:' + url
        
        # Si es una URL de redirect de DuckDuckGo, extraer la URL real del parámetro uddg
        if 'duckduckgo.com/l/' in url:
            parsed = urlparse(url)
            params = parse_qs(parsed.query)
            if 'uddg' in params:
                url = unquote(params['uddg'][0])
                print(f"         → URL decodificada: {url[:80]}...")
        
        # Extraer contenido directamente con trafilatura
        downloaded = trafilatura.fetch_url(url)
        if not downloaded:
            return None
        
        text = trafilatura.extract(downloaded)
        return text
    
    except Exception as e:
        print(f"      ⚠️  Error extrayendo contenido: {e}")
        return None


def extraer_datos_chunk(bloque_match: Optional[Dict]) -> tuple:
    """
    Extrae información de chunks del bloque_match de ResultadoValidacion.
    
    Returns:
        tuple: (bloque_nuevo_idx, bloque_existente_idx, bloque_nuevo_texto, bloque_existente_texto)
    """
    if not bloque_match:
        return None, None, None, None
    
    return (
        bloque_match.get('bloque_nuevo'),
        bloque_match.get('bloque_historico'),
        bloque_match.get('bloque_nuevo_texto'),
        bloque_match.get('bloque_historico_texto')
    )


def enriquecer_bloques_con_ids(bloques_detalle: list, chunk_ids: list) -> list:
    """
    Vincula bloques_detalle con chunk_ids por índice.
    
    Args:
        bloques_detalle: Lista de dicts con análisis de cada bloque
        chunk_ids: Lista de tuplas (indice, chunk_id) retornada por shield.agregar()
    
    Returns:
        bloques_detalle enriquecido con campo 'chunk_id'
    """
    if not bloques_detalle or not chunk_ids:
        return bloques_detalle
    
    # Crear mapa indice -> chunk_id
    id_map = dict(chunk_ids)
    
    # Enriquecer cada bloque con su chunk_id
    for bloque in bloques_detalle:
        idx = bloque.get('indice')
        if idx is not None:
            bloque['chunk_id'] = id_map.get(idx)
    
    return bloques_detalle


def registrar_chunks_analisis(query_id: str, documento_id: str, query_texto: str,
                              url: str, tema: str, decision: str,
                              bloques_detalle: list,
                              uuid_similar: Optional[str] = None):
    """
    Registra cada chunk individualmente en chunk_analysis_log.
    1 chunk = 1 fila en la base de datos.
    
    Args:
        query_id: UUID de la query
        documento_id: UUID del documento procesado
        query_texto: Texto de la query
        url: URL origen del contenido
        tema: Tema de la query
        decision: Decisión final del documento (APROBADO, RECHAZADO, etc)
        bloques_detalle: Lista con info de cada chunk
        uuid_similar: UUID del documento similar (si aplica)
    """
    if not bloques_detalle:
        print(f"      ⚠️  Sin chunks para registrar")
        return
    
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        cur = conn.cursor()
        
        chunks_guardados = 0
        
        for chunk in bloques_detalle:
            # Extraer datos del chunk
            chunk_id = chunk.get('chunk_id')  # Puede ser None si no fue guardado
            chunk_idx = chunk.get('indice')
            chunk_score = float(chunk.get('similitud', 0.0)) if chunk.get('similitud') is not None else None
            chunk_estado = chunk.get('estado', 'NUEVO')
            chunk_similar_idx = chunk.get('bloque_historico_idx')
            
            # Insertar chunk individual
            cur.execute("""
                INSERT INTO chunk_analysis_log 
                (chunk_id, chunk_idx, documento_id, chunk_score, chunk_estado,
                 documento_similar_id, chunk_similar_idx, decision,
                 query_id, query_texto, url_origen, tema)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """, (chunk_id, chunk_idx, documento_id, chunk_score, chunk_estado,
                  uuid_similar, chunk_similar_idx, decision,
                  query_id, query_texto, url, tema))
            
            chunks_guardados += 1
        
        conn.commit()
        cur.close()
        conn.close()
        
        print(f"      ✅ {chunks_guardados} chunks registrados en análisis")
        
    except Exception as e:
        print(f"      ❌ Error registrando chunks: {e}")


def guardar_mapeo(query_id: str, documento_id: str, query_texto: str, url: str) -> bool:
    """Guarda el mapeo query ↔ dato. Retorna True si éxito."""
    
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        cur = conn.cursor()
        
        cur.execute("""
            INSERT INTO query_document_mapping (query_id, documento_id, query_texto, url)
            VALUES (%s, %s, %s, %s)
            ON CONFLICT (query_id, documento_id) DO NOTHING
        """, (query_id, documento_id, query_texto, url))
        
        conn.commit()
        cur.close()
        conn.close()
        
        return True
    
    except Exception as e:
        print(f"      ❌ Error guardando mapeo: {e}")
        return False


def guardar_query_en_preguntas(query_id: str, query_texto: str, tema: str) -> bool:
    """
    Guarda la query en la tabla queries CON su embedding.
    SOLO se llama si el dato fue guardado exitosamente.
    Retorna True si éxito.
    """
    
    try:
        # Generar embedding de la query (384d)
        model = get_query_embedding_model()
        embedding = model.encode(query_texto, show_progress_bar=False)
        embedding_list = embedding.tolist()
        
        conn = psycopg2.connect(**DB_CONFIG)
        cur = conn.cursor()
        
        # Insertar en queries con embedding
        cur.execute("""
            INSERT INTO queries (id, texto, tema, embedding)
            VALUES (%s, %s, %s, %s)
            ON CONFLICT (id) DO NOTHING
        """, (query_id, query_texto, tema, embedding_list))
        
        conn.commit()
        cur.close()
        conn.close()
        
        print(f"      ✅ Query guardada con embedding ({len(embedding_list)}d)")
        return True
    
    except Exception as e:
        print(f"      ❌ Error guardando query en preguntas: {e}")
        return False


def procesar_query(query: Dict, shield: DataShield):
    """
    Procesa una query: scrape → validate → save (con flujo correcto)
    """
    query_id = query['id']
    query_texto = query['texto']
    tema = query['tema']
    
    print(f"\n{'='*80}")
    print(f"🔍 Query: {query_texto}")
    print(f"   UUID: {query_id}")
    print(f"   Tema: {tema}")
    print(f"{'='*80}")
    
    # Paso 1: Scrapear DuckDuckGo
    print("\n📡 Scrapeando DuckDuckGo...")
    resultados = scrapear_duckduckgo(query_texto, max_results=3)
    
    if not resultados:
        print("   ❌ No se encontraron resultados")
        return
    
    print(f"   ✅ {len(resultados)} resultados encontrados")
    
    # Paso 2: Para cada resultado, extraer y validar
    for i, resultado in enumerate(resultados, 1):
        url = resultado['url']
        print(f"\n   🌐 Resultado {i}: {resultado['title'][:60]}...")
        print(f"      URL: {url}")
        
        # Extraer contenido
        print(f"      📥 Extrayendo contenido...")
        contenido = extraer_contenido(url)
        
        if not contenido or len(contenido.strip()) < 100:
            print(f"      ⚠️  Contenido vacío o muy corto, saltando...")
            continue
        
        print(f"      ✅ Extraído: {len(contenido)} caracteres")
        
        # Generar UUID para el DATO
        id = str(uuid.uuid4())
        
        # Validar con DataShield
        print(f"      🛡️  Validando con DataShield...")
        resultado_val = shield.validar(id, contenido)
        
        print(f"      Estado: {resultado_val.estado.value}")
        print(f"      Score: {resultado_val.score:.3f}")
        
        # Contar bloques
        num_dupl = sum(1 for b in resultado_val.bloques_detalle if b['estado'] == 'DUPLICADO')
        num_agente = sum(1 for b in resultado_val.bloques_detalle if b['estado'] == 'AGENTE')
        num_nuevos = sum(1 for b in resultado_val.bloques_detalle if b['estado'] == 'NUEVO')
        
        # Extraer datos de chunks para seguimiento
        chunk_nuevo_idx, chunk_hist_idx, chunk_nuevo_txt, chunk_hist_txt = extraer_datos_chunk(resultado_val.bloque_match)
        
        # DECISIÓN CRÍTICA
        if resultado_val.estado.value == "NUEVA":
            print(f"      ✅ NUEVA - Guardando...")
            
            # 1. Guardar DATO
            try:
                chunk_ids = shield.agregar(id, contenido, tema)
                num_bloques = len(chunk_ids)
                print(f"      ✅ Dato guardado ({num_bloques} bloques)")
                
                # 2. Guardar MAPEO
                if guardar_mapeo(query_id, id, query_texto, url):
                    print(f"      ✅ Mapeo guardado")
                    
                    # 3. Liberar modelo BGE-M3 antes de cargar modelo de queries
                    print(f"      🧹 Liberando modelo de validación...")
                    shield.liberar_modelo()
                    
                    # 4. Guardar QUERY (SOLO si 1 y 2 OK)
                    if guardar_query_en_preguntas(query_id, query_texto, tema):
                        print(f"      ✅ Query guardada en 'preguntas'")
                    else:
                        print(f"      ⚠️  Query NO guardada (error)")
                    
                    # 5. Restaurar modelo BGE-M3 para siguiente validación
                    print(f"      🔄 Restaurando modelo de validación...")
                    shield.restaurar_modelo()
                else:
                    print(f"      ⚠️  Mapeo NO guardado (error)")
                
                # Enriquecer bloques_detalle con chunk_ids
                bloques_enriquecidos = enriquecer_bloques_con_ids(
                    resultado_val.bloques_detalle, 
                    chunk_ids
                )
                
                # Registrar cada chunk individualmente
                registrar_chunks_analisis(
                    query_id, id, query_texto, url, tema,
                    "APROBADO",
                    bloques_enriquecidos
                )
                
                print(f"\n      🎯 DATO PROCESADO COMPLETAMENTE")
                return  # Éxito, pasar a siguiente query
                
            except Exception as e:
                print(f"      ❌ Error guardando: {e}")
                
                try:
                    # Intentar guardar al menos los chunks para trazabilidad
                    chunk_ids = shield.agregar_solo_chunks(id, contenido, tema)
                    print(f"      ⚠️  Chunks guardados para trazabilidad ({len(chunk_ids)} chunks)")
                    
                    # Enriquecer bloques_detalle con chunk_ids
                    bloques_enriquecidos = enriquecer_bloques_con_ids(
                        resultado_val.bloques_detalle,
                        chunk_ids
                    )
                    
                    registrar_chunks_analisis(
                        query_id, id, query_texto, url, tema,
                        "ERROR",
                        bloques_enriquecidos
                    )
                except Exception as e2:
                    print(f"      ❌ Error guardando chunks: {e2}")
                    # Registrar sin chunk_ids
                    registrar_chunks_analisis(
                        query_id, id, query_texto, url, tema,
                        "ERROR",
                        resultado_val.bloques_detalle
                    )
        
        elif resultado_val.estado.value == "DUPLICADO":
            print(f"      🔴 DUPLICADO - No guardar documento, solo chunks para trazabilidad")
            
            try:
                # Guardar SOLO chunks (no documento, no FAISS)
                chunk_ids = shield.agregar_solo_chunks(id, contenido, tema)
                print(f"      ✅ Chunks guardados para trazabilidad ({len(chunk_ids)} chunks)")
                
                # Enriquecer bloques_detalle con chunk_ids
                bloques_enriquecidos = enriquecer_bloques_con_ids(
                    resultado_val.bloques_detalle,
                    chunk_ids
                )
                
                # Registrar análisis con chunk_ids válidos
                print(f"      📝 Registrando chunks en análisis...")
                registrar_chunks_analisis(
                    query_id, id, query_texto, url, tema,
                    "RECHAZADO",
                    bloques_enriquecidos,
                    uuid_similar=resultado_val.uuid_historico
                )
            except Exception as e:
                print(f"      ❌ Error guardando chunks para trazabilidad: {e}")
                # Registrar sin chunk_ids si falla
                registrar_chunks_analisis(
                    query_id, id, query_texto, url, tema,
                    "RECHAZADO",
                    resultado_val.bloques_detalle,
                    uuid_similar=resultado_val.uuid_historico
                )
        
        elif resultado_val.estado.value == "AGENTE":
            print(f"      🟡 AGENTE - Consultando modelo LLM para decidir...")
            
            # Obtener estadísticas de chunks
            total_chunks = len(resultado_val.bloques_detalle)
            chunks_similares = sum(1 for b in resultado_val.bloques_detalle if b['estado'] == 'AGENTE')
            
            print(f"      📊 {chunks_similares}/{total_chunks} chunks similares detectados")
            
            # Obtener el chunk más similar del bloque_match
            if resultado_val.bloque_match and resultado_val.bloque_match.get('bloque_historico_texto'):
                chunk_nuevo = resultado_val.bloque_match.get('bloque_nuevo_texto', '')
                chunk_existente = resultado_val.bloque_match.get('bloque_historico_texto', '')
                
                # Consultar al modelo LLM con contexto completo
                mantener = decidir_con_llm(chunk_nuevo, chunk_existente, chunks_similares, total_chunks)
                
                if mantener:
                    print(f"      ✅ LLM decidió: SUFICIENTEMENTE DIFERENTE → Guardando...")
                    
                    # Guardar el documento
                    try:
                        chunk_ids = shield.agregar(id, contenido, tema)
                        num_bloques = len(chunk_ids)
                        print(f"      ✅ Dato guardado ({num_bloques} bloques)")
                        
                        # Guardar mapeo
                        if guardar_mapeo(query_id, id, query_texto, url):
                            print(f"      ✅ Mapeo guardado")
                            
                            # Liberar modelo BGE-M3 antes de cargar modelo de queries
                            print(f"      🧹 Liberando modelo de validación...")
                            shield.liberar_modelo()
                            
                            # Guardar query
                            if guardar_query_en_preguntas(query_id, query_texto, tema):
                                print(f"      ✅ Query guardada en 'preguntas'")
                            
                            # Restaurar modelo BGE-M3 para siguiente validación
                            print(f"      🔄 Restaurando modelo de validación...")
                            shield.restaurar_modelo()
                        
                        # Enriquecer bloques_detalle con chunk_ids
                        bloques_enriquecidos = enriquecer_bloques_con_ids(
                            resultado_val.bloques_detalle,
                            chunk_ids
                        )
                        
                        # Registrar cada chunk individualmente
                        print(f"      📝 Registrando chunks en análisis...")
                        registrar_chunks_analisis(
                            query_id, id, query_texto, url, tema,
                            "APROBADO_LLM",
                            bloques_enriquecidos,
                            uuid_similar=resultado_val.uuid_historico
                        )
                        
                        print(f"\n      🎯 DATO PROCESADO COMPLETAMENTE (aprobado por LLM)")
                        return  # Éxito
                    
                    except Exception as e:
                        print(f"      ❌ Error guardando: {e}")
                        
                        try:
                            # Intentar guardar al menos los chunks para trazabilidad
                            chunk_ids = shield.agregar_solo_chunks(id, contenido, tema)
                            print(f"      ⚠️  Chunks guardados para trazabilidad ({len(chunk_ids)} chunks)")
                            
                            # Enriquecer bloques_detalle con chunk_ids
                            bloques_enriquecidos = enriquecer_bloques_con_ids(
                                resultado_val.bloques_detalle,
                                chunk_ids
                            )
                            
                            registrar_chunks_analisis(
                                query_id, id, query_texto, url, tema,
                                "ERROR",
                                bloques_enriquecidos,
                                uuid_similar=resultado_val.uuid_historico
                            )
                        except Exception as e2:
                            print(f"      ❌ Error guardando chunks: {e2}")
                            # Registrar sin chunk_ids
                            registrar_chunks_analisis(
                                query_id, id, query_texto, url, tema,
                                "ERROR",
                                resultado_val.bloques_detalle,
                                uuid_similar=resultado_val.uuid_historico
                            )
                else:
                    print(f"      ❌ LLM decidió: DEMASIADO SIMILAR → Rechazando documento")
                    
                    try:
                        # Guardar SOLO chunks para trazabilidad
                        chunk_ids = shield.agregar_solo_chunks(id, contenido, tema)
                        print(f"      ✅ Chunks guardados para trazabilidad ({len(chunk_ids)} chunks)")
                        
                        # Enriquecer bloques_detalle con chunk_ids
                        bloques_enriquecidos = enriquecer_bloques_con_ids(
                            resultado_val.bloques_detalle,
                            chunk_ids
                        )
                        
                        # Registrar análisis con chunk_ids válidos
                        print(f"      📝 Registrando chunks en análisis...")
                        registrar_chunks_analisis(
                            query_id, id, query_texto, url, tema,
                            "RECHAZADO_LLM",
                            bloques_enriquecidos,
                            uuid_similar=resultado_val.uuid_historico
                        )
                    except Exception as e:
                        print(f"      ❌ Error guardando chunks para trazabilidad: {e}")
                        # Registrar sin chunk_ids si falla
                        registrar_chunks_analisis(
                            query_id, id, query_texto, url, tema,
                            "RECHAZADO_LLM",
                            resultado_val.bloques_detalle,
                            uuid_similar=resultado_val.uuid_historico
                        )
            else:
                # No hay información suficiente para comparar, rechazar por seguridad
                print(f"      ⚠️  Sin información para comparar, rechazando por seguridad")
                
                try:
                    # Guardar SOLO chunks para trazabilidad
                    chunk_ids = shield.agregar_solo_chunks(id, contenido, tema)
                    print(f"      ✅ Chunks guardados para trazabilidad ({len(chunk_ids)} chunks)")
                    
                    # Enriquecer bloques_detalle con chunk_ids
                    bloques_enriquecidos = enriquecer_bloques_con_ids(
                        resultado_val.bloques_detalle,
                        chunk_ids
                    )
                    
                    # Registrar análisis con chunk_ids válidos
                    registrar_chunks_analisis(
                        query_id, id, query_texto, url, tema,
                        "RECHAZADO",
                        bloques_enriquecidos,
                        uuid_similar=resultado_val.uuid_historico
                    )
                except Exception as e:
                    print(f"      ❌ Error guardando chunks para trazabilidad: {e}")
                    # Registrar sin chunk_ids si falla
                    registrar_chunks_analisis(
                        query_id, id, query_texto, url, tema,
                        "RECHAZADO",
                        resultado_val.bloques_detalle,
                        uuid_similar=resultado_val.uuid_historico
                    )
    
    print(f"\n   ⚠️  Ningún resultado fue aprobado para esta query")


def main():
    print("🚀 DataCorpus Scraper - Procesamiento de queries")
    print("="*80)
    
    # Paso 1: Leer queries
    print("\n📋 Cargando queries...")
    queries = leer_queries(n=None)  # Procesar todas las queries disponibles
    
    if not queries:
        print("❌ No se pudieron cargar queries")
        return
    
    print(f"✅ {len(queries)} queries cargadas")
    
    # Paso 2: Inicializar DataShield
    print("\n🛡️  Inicializando DataShield...")
    shield = DataShield(DB_CONFIG, init_from_db=True)
    stats = shield.stats()
    print(f"   Textos en BD: {stats['total_textos']}")
    print(f"   Bloques en FAISS: {stats['faiss_ntotal']}")
    
    # Paso 3: Procesar cada query
    print("\n" + "="*80)
    print("INICIANDO PROCESAMIENTO")
    print("="*80)
    
    for i, query in enumerate(queries, 1):
        print(f"\n\n{'#'*80}")
        print(f"# QUERY {i}/{len(queries)}")
        print(f"{'#'*80}")
        
        procesar_query(query, shield)
        
        # Pausa entre queries para no saturar
        if i < len(queries):
            print("\n⏸️  Pausa de 2 segundos...")
            time.sleep(2)
    
    # Resumen final
    print("\n\n" + "="*80)
    print("📊 RESUMEN FINAL")
    print("="*80)
    
    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor()
    
    cur.execute("SELECT COUNT(*) FROM documents")
    total_datos = cur.fetchone()[0]
    
    cur.execute("SELECT COUNT(*) FROM query_document_mapping")
    total_mapeos = cur.fetchone()[0]
    
    cur.execute("SELECT COUNT(*) FROM queries")
    total_queries = cur.fetchone()[0]
    
    cur.execute("SELECT COUNT(*) FROM chunk_analysis_log")
    total_seguimiento = cur.fetchone()[0]
    
    cur.close()
    conn.close()
    
    print(f"\n📊 Datos guardados: {total_datos}")
    print(f"🔗 Mapeos creados: {total_mapeos}")
    print(f"❓ Queries en 'queries': {total_queries}")
    print(f"📋 Chunks analizados: {total_seguimiento}")
    
    print("\n✅ Procesamiento completado")
    print("="*80)


if __name__ == "__main__":
    main()


def simplificar_query(query_texto: str) -> str:
    """
    Simplifica una query larga removiendo signos de interrogación y palabras comunes.
    Retorna una versión más corta para scraping.
    """
    # Remover signos de interrogación
    query_simple = query_texto.replace('¿', '').replace('?', '')
    
    # Palabras a remover (stopwords comunes en español)
    stopwords = ['cómo', 'qué', 'cuál', 'cuáles', 'dónde', 'cuándo', 'por qué', 
                 'para qué', 'en', 'el', 'la', 'los', 'las', 'un', 'una', 'de', 
                 'del', 'al', 'se', 'es', 'son', 'está', 'están', 'y', 'o', 'a']
    
    palabras = query_simple.lower().split()
    palabras_filtradas = [p for p in palabras if p not in stopwords and len(p) > 2]
    
    # Limitar a 6-8 palabras clave
    return ' '.join(palabras_filtradas[:8])
