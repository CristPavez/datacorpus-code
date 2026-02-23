#!/usr/bin/env python3
"""
Generador Inteligente de Queries para DataCorpus
Flujo limpio:
1. Analizar desbalance en PostgreSQL
2. Generar preguntas SOLO con LLM (DeepSeek V3.1)
3. Validar con query_shield:
   - NUEVA → Guardar
   - DUPLICADO → Eliminar y regenerar
   - AGENTE → Comparar semánticamente:
     * Mismo sentido → Regenerar
     * Diferente sentido → Guardar
4. Guardar directamente en JSONL
"""

import json
import uuid
import psycopg2
from collections import Counter
from query_shield import QueryShield, Estado
from typing import List, Dict, Optional
from openai import OpenAI
from sentence_transformers import SentenceTransformer
import os
import numpy as np

# Configuración
DB_CONFIG = {
    "dbname": "datacorpus_bd",
    "user": "datacorpus",
    "password": "730822",
    "host": "localhost"
}

TEMAS_PRIORITARIOS = ["Tecnología", "Ciencia (General)", "Literatura / Humanidades", "Medicina"]

# Modelo de embeddings para comparación semántica (mismo que query_shield)
embedding_model = None

def get_embedding_model():
    """Lazy load del modelo de embeddings"""
    global embedding_model
    if embedding_model is None:
        print("   📥 Cargando modelo de embeddings...")
        embedding_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
    return embedding_model


def analizar_desbalance() -> Dict[str, int]:
    """Analiza el desbalance de datos por tema en PostgreSQL"""
    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor()
    
    cur.execute("""
        SELECT tema, COUNT(*) as cantidad
        FROM queries
        WHERE tema IS NOT NULL
        GROUP BY tema
        ORDER BY cantidad ASC
    """)
    
    resultado = dict(cur.fetchall())
    cur.close()
    conn.close()
    
    return resultado


def seleccionar_temas_objetivo(stats: Dict[str, int]) -> List[str]:
    """
    Selecciona temas para generar queries.
    Si hay desbalance claro (3x), toma los menos representados.
    Si no, usa temas prioritarios.
    """
    if not stats:
        print("📊 Base de datos vacía, usando temas prioritarios")
        return TEMAS_PRIORITARIOS
    
    valores = list(stats.values())
    if not valores:
        return TEMAS_PRIORITARIOS
    
    max_val = max(valores)
    min_val = min(valores)
    
    if max_val > min_val * 3:
        print(f"📊 Desbalance detectado: max={max_val}, min={min_val}")
        temas_ordenados = sorted(stats.items(), key=lambda x: x[1])
        return [tema for tema, _ in temas_ordenados[:4]]
    else:
        print(f"📊 Datos balanceados, usando temas prioritarios")
        return TEMAS_PRIORITARIOS


def cargar_token_together() -> str:
    """Carga el token de Together.ai desde las credenciales"""
    token_path = os.path.expanduser("~/.openclaw/credentials/together.token.json")
    try:
        with open(token_path, 'r') as f:
            data = json.load(f)
            token = data.get('token', '')
            if not token or token == "TU_TOKEN_AQUI":
                raise Exception("Token no configurado")
            return token
    except Exception as e:
        raise Exception(f"Error cargando token Together.ai: {e}")


def generar_preguntas_lm(tema: str, n: int = 10) -> List[str]:
    """
    Genera preguntas usando DeepSeek V3.1 via Together.ai
    SOLO usa LLM - sin fallback sintético
    """
    token = cargar_token_together()
    
    client = OpenAI(
        api_key=token,
        base_url="https://api.together.xyz/v1"
    )
    
    prompt = f"""Genera exactamente {n} preguntas únicas y específicas sobre {tema} en español.

Requisitos:
- Preguntas diversas en enfoque (qué es, cómo funciona, ventajas, desventajas, aplicaciones, etc.)
- Específicas y técnicas, no genéricas
- Útiles para investigación o aprendizaje profesional
- Una pregunta por línea, sin numeración

Ejemplos de BUENAS preguntas:
- ¿Cuáles son los efectos secundarios de la metformina?
- ¿Cómo funciona el aprendizaje por refuerzo en IA?
- ¿Qué diferencias hay entre leasing y renting empresarial?

Evita preguntas genéricas como "¿Qué es {tema}?" o "¿Para qué sirve {tema}?"

Genera {n} preguntas sobre {tema}:"""
    
    response = client.chat.completions.create(
        model="deepseek-ai/DeepSeek-V3",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=1000,
        temperature=0.7
    )
    
    texto = response.choices[0].message.content
    preguntas = [line.strip() for line in texto.split('\n') if line.strip() and '?' in line]
    
    # Debug: mostrar lo que devolvió el LLM
    if not preguntas:
        print(f"   ⚠️  LLM no generó preguntas válidas. Respuesta completa:")
        print(f"   {texto[:200]}...")
    
    # Limpiar numeración si existe (1. pregunta → pregunta)
    preguntas = [p.split('. ', 1)[-1] if '. ' in p[:4] else p for p in preguntas]
    
    return preguntas[:n]


def registrar_seguimiento(uuid_nueva: str, uuid_existente: str, texto_nueva: str, 
                         texto_existente: str, tema: str, estado_deteccion: str,
                         score_shield: float, similitud_semantica: Optional[float],
                         decision: str):
    """
    Registra en la tabla de seguimiento los duplicados y decisiones de AGENTE.
    
    Args:
        uuid_nueva: UUID de la pregunta candidata
        uuid_existente: UUID de la pregunta con la que se comparó
        texto_nueva: Texto de la pregunta candidata
        texto_existente: Texto de la pregunta existente
        tema: Tema de la pregunta
        estado_deteccion: 'DUPLICADO' o 'AGENTE'
        score_shield: Score de QueryShield
        similitud_semantica: Similitud coseno (None para DUPLICADO)
        decision: 'RECHAZADO', 'REGENERADO' o 'APROBADO'
    """
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        cur = conn.cursor()
        
        cur.execute("""
            INSERT INTO queries_seguimiento 
            (uuid_nueva, uuid_existente, texto_nueva, texto_existente, tema, 
             estado_deteccion, score_queryshield, similitud_semantica, decision)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
        """, (uuid_nueva, uuid_existente, texto_nueva, texto_existente, tema,
              estado_deteccion, score_shield, similitud_semantica, decision))
        
        conn.commit()
        cur.close()
        conn.close()
        
    except Exception as e:
        print(f"   ⚠️  Error registrando seguimiento: {e}")


def comparar_semanticamente(pregunta1: str, pregunta2: str) -> float:
    """
    Compara el sentido semántico de dos preguntas usando embeddings.
    Retorna similitud coseno (0.0 = diferentes, 1.0 = idénticas)
    """
    model = get_embedding_model()
    
    emb1 = model.encode(pregunta1, convert_to_tensor=False)
    emb2 = model.encode(pregunta2, convert_to_tensor=False)
    
    # Similitud coseno
    similitud = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
    
    return float(similitud)


def decidir_estado_agente(query_nueva: str, query_historica: str, score_shield: float) -> tuple[bool, float]:
    """
    Decide si mantener o regenerar cuando query_shield devuelve AGENTE.
    
    Estrategia con LLM:
    1. Calcula similitud semántica para registro
    2. Consulta a DeepSeek V3.1 para que decida si son suficientemente diferentes
    3. Fallback a comparación semántica si LLM falla
    
    Returns:
        (mantener: bool, similitud: float)
    """
    # Calcular similitud para registro
    similitud = comparar_semanticamente(query_nueva, query_historica)
    print(f"   📊 Similitud semántica: {similitud:.3f}")
    
    # Consultar al modelo LLM
    print(f"   🤖 Consultando a DeepSeek V3.1 para decidir...")
    
    try:
        token = cargar_token_together()
        client = OpenAI(
            api_key=token,
            base_url="https://api.together.xyz/v1"
        )
        
        prompt = f"""Eres un experto en análisis de preguntas. Compara estas dos preguntas y decide si son SUFICIENTEMENTE DIFERENTES o DEMASIADO SIMILARES.

Pregunta existente: "{query_historica}"

Pregunta nueva: "{query_nueva}"

Criterio:
- SUFICIENTEMENTE DIFERENTES: Si abordan aspectos diferentes del tema, tienen enfoques distintos, o buscan información complementaria.
- DEMASIADO SIMILARES: Si buscan básicamente la misma información, aunque estén redactadas diferente.

Responde SOLO con una de estas dos palabras:
- DIFERENTES
- SIMILARES"""

        response = client.chat.completions.create(
            model="deepseek-ai/DeepSeek-V3",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=50,
            temperature=0.3  # Baja temperatura para decisión consistente
        )
        
        decision = response.choices[0].message.content.strip().upper()
        
        # Parse de la respuesta
        if "DIFERENTES" in decision:
            print(f"   ✅ MANTENER: Modelo dice DIFERENTES")
            return True, similitud
        elif "SIMILARES" in decision:
            print(f"   🔄 REGENERAR: Modelo dice SIMILARES")
            return False, similitud
        else:
            # Fallback si respuesta ambigua
            print(f"   ⚠️  Respuesta ambigua: '{decision}', usando fallback semántico")
            mantener = similitud <= 0.85
            if mantener:
                print(f"   ✅ MANTENER: Fallback (similitud={similitud:.3f} ≤ 0.85)")
            else:
                print(f"   🔄 REGENERAR: Fallback (similitud={similitud:.3f} > 0.85)")
            return mantener, similitud
    
    except Exception as e:
        print(f"   ❌ Error consultando modelo: {e}")
        print(f"   ⚠️  Usando fallback semántico (umbral 0.85)")
        mantener = similitud <= 0.85
        if mantener:
            print(f"   ✅ MANTENER: Fallback (similitud={similitud:.3f} ≤ 0.85)")
        else:
            print(f"   🔄 REGENERAR: Fallback (similitud={similitud:.3f} > 0.85)")
        return mantener, similitud


def validar_y_procesar(shield: QueryShield, query: str, tema: str, max_reintentos: int = 5) -> Optional[Dict]:
    """
    Valida una query y aplica el flujo completo:
    - NUEVA → Aceptar
    - DUPLICADO → Registrar y regenerar
    - AGENTE → Comparar semánticamente, registrar y decidir
    
    El UUID se mantiene constante a través de todas las regeneraciones,
    permitiendo rastrear el historial completo en query_deduplication_log.
    
    Returns:
        Dict con metadata si se aprueba, None si no se logra aprobar
    """
    query_actual = query
    
    # UUID único que se mantiene a través de todas las regeneraciones
    uid = str(uuid.uuid4())
    
    for intento in range(max_reintentos):
        resultado = shield.validar(uid, query_actual)
        
        if resultado.estado == Estado.NUEVA:
            # ✅ Aceptar directamente
            print(f"✅ NUEVA [{resultado.score:.3f}]: {query_actual[:80]}...")
            return {
                "id": uid,
                "texto": query_actual,
                "tema": tema,
                "estado": "nueva",
                "score": resultado.score
            }
        
        elif resultado.estado == Estado.DUPLICADO:
            # 🔴 Registrar duplicado y regenerar
            print(f"🔴 DUPLICADO: {query_actual[:60]}...")
            
            # Registrar en tabla de seguimiento
            registrar_seguimiento(
                uuid_nueva=uid,
                uuid_existente=resultado.uuid_historico,
                texto_nueva=query_actual,
                texto_existente=resultado.texto_historico,
                tema=tema,
                estado_deteccion='DUPLICADO',
                score_shield=resultado.score,
                similitud_semantica=None,
                decision='REGENERADO'
            )
            
            print(f"   Regenerando...")
            nuevas = generar_preguntas_lm(tema, n=1)
            if nuevas and len(nuevas) > 0:
                query_actual = nuevas[0]
            else:
                print(f"   ❌ Error generando nueva pregunta (lista vacía)")
                return None
        
        elif resultado.estado == Estado.AGENTE:
            # 🟡 Comparar semánticamente y decidir
            print(f"🟡 AGENTE [{resultado.score:.3f}]: {query_actual[:60]}...")
            print(f"   Comparando con: {resultado.texto_historico[:60]}...")
            
            mantener, similitud = decidir_estado_agente(query_actual, resultado.texto_historico, resultado.score)
            
            # Registrar en tabla de seguimiento
            decision = 'APROBADO' if mantener else 'REGENERADO'
            registrar_seguimiento(
                uuid_nueva=uid,
                uuid_existente=resultado.uuid_historico,
                texto_nueva=query_actual,
                texto_existente=resultado.texto_historico,
                tema=tema,
                estado_deteccion='AGENTE',
                score_shield=resultado.score,
                similitud_semantica=similitud,
                decision=decision
            )
            
            if mantener:
                return {
                    "id": uid,
                    "texto": query_actual,
                    "tema": tema,
                    "estado": "agente_aprobada",
                    "score": resultado.score
                }
            else:
                # Regenerar
                print(f"   Regenerando...")
                nuevas = generar_preguntas_lm(tema, n=1)
                if nuevas and len(nuevas) > 0:
                    query_actual = nuevas[0]
                else:
                    print(f"   ❌ Error generando nueva pregunta (lista vacía)")
                    return None
    
    print(f"⚠️  Límite de reintentos alcanzado")
    return None


def main():
    print("🚀 Generador de Queries DataCorpus\n")
    
    # Paso 1: Analizar desbalance
    print("📊 Paso 1: Analizando base de datos...")
    stats = analizar_desbalance()
    
    if stats:
        print(f"   Temas en BD: {len(stats)}")
        top_3 = sorted(stats.items(), key=lambda x: x[1])[:3]
        print(f"   Menos representados: {top_3}")
    else:
        print(f"   Base de datos vacía")
    
    # Paso 2: Seleccionar temas
    temas = seleccionar_temas_objetivo(stats)
    print(f"\n🎯 Temas objetivo: {temas}\n")
    
    # Paso 3: Inicializar QueryShield
    print("🛡️  Paso 2: Inicializando QueryShield...")
    shield = QueryShield(DB_CONFIG)
    print(f"   {shield.stats()}\n")
    
    # Paso 4: Generar y validar 50 queries
    print("📝 Paso 3: Generando 50 queries con LLM...\n")
    
    QUERIES_TOTAL = 50
    queries_por_tema = QUERIES_TOTAL // len(temas)
    resto = QUERIES_TOTAL % len(temas)
    
    todas_las_queries = []
    
    for i, tema in enumerate(temas):
        n_objetivo = queries_por_tema + (1 if i < resto else 0)
        print(f"--- {tema} ({n_objetivo} queries objetivo) ---")
        
        queries_aprobadas = []
        candidatas_generadas = 0
        
        # Generar en lotes hasta completar n_objetivo
        while len(queries_aprobadas) < n_objetivo and candidatas_generadas < n_objetivo * 3:
            # Generar lote de candidatas
            n_generar = min(n_objetivo * 2, (n_objetivo - len(queries_aprobadas)) * 2)
            print(f"   Generando {n_generar} candidatas con DeepSeek V3...")
            
            try:
                candidatas = generar_preguntas_lm(tema, n=n_generar)
                candidatas_generadas += len(candidatas)
                print(f"   Recibidas {len(candidatas)} preguntas")
                
                # Validar cada candidata
                for idx_candidata, candidata in enumerate(candidatas):
                    if len(queries_aprobadas) >= n_objetivo:
                        break
                    
                    print(f"   Validando candidata {idx_candidata+1}/{len(candidatas)}: {candidata[:60]}...")
                    resultado = validar_y_procesar(shield, candidata, tema)
                    if resultado:
                        queries_aprobadas.append(resultado)
                    else:
                        print(f"   ⚠️  Candidata rechazada o agotada reintentos")
                
            except Exception as e:
                print(f"   ❌ Error generando preguntas: {e}")
                break
        
        todas_las_queries.extend(queries_aprobadas)
        print(f"   ✅ Completado: {len(queries_aprobadas)}/{n_objetivo}\n")
    
    # Paso 5: Guardar en JSONL
    output_file = "queries_validadas.jsonl"
    print(f"💾 Paso 4: Guardando en {output_file}...")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for query in todas_las_queries:
            f.write(json.dumps(query, ensure_ascii=False) + '\n')
    
    # Resumen final
    print(f"\n✅ Completado!")
    print(f"   Total: {len(todas_las_queries)} queries")
    print(f"   Archivo: {output_file}")
    
    estados = Counter([q['estado'] for q in todas_las_queries])
    print(f"   - Nuevas: {estados.get('nueva', 0)}")
    print(f"   - Agente aprobadas: {estados.get('agente_aprobada', 0)}")


if __name__ == "__main__":
    main()
