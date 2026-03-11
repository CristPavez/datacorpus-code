#!/usr/bin/env python3
"""
Script para reconstruir el índice FAISS desde la base de datos.
Asegura sincronización completa entre BD y el índice de búsqueda.
"""

import faiss
import numpy as np
import psycopg
import pickle
from pathlib import Path
from pgvector.psycopg import register_vector

# ── CONFIGURACIÓN ─────────────────────────────────────────────────
DB_CONFIG = {
    "dbname": "datacorpus_bd",
    "user": "datacorpus",
    "password": "730822",
    "host": "localhost"
}

FAISS_INDEX_PATH = Path("faiss_index_bge3.bin")
FAISS_MAPPING_PATH = Path("faiss_index_bge3.mapping")
VECTOR_DIMENSION = 1024  # Dimensión del modelo BAAI/bge-m3


def get_connection():
    """Crea conexión a la base de datos."""
    conn = psycopg.connect(**DB_CONFIG)
    register_vector(conn)
    return conn


def contar_embeddings_disponibles():
    """Cuenta cuántos embeddings válidos hay en la BD."""
    conn = get_connection()
    cur = conn.cursor()
    
    # Contar embeddings no nulos de chunks procesados
    cur.execute("""
        SELECT COUNT(*) 
        FROM documents_logs 
        WHERE chunk_embedding IS NOT NULL 
        AND estado = 'procesado'
    """)
    count_procesados = cur.fetchone()[0]
    
    # Contar total de embeddings no nulos
    cur.execute("""
        SELECT COUNT(*) 
        FROM documents_logs 
        WHERE chunk_embedding IS NOT NULL
    """)
    count_total = cur.fetchone()[0]
    
    cur.close()
    conn.close()
    
    return count_procesados, count_total


def rebuild_faiss_index(solo_procesados=True, max_records=None):
    """
    Reconstruye el índice FAISS desde cero usando la base de datos.
    
    Args:
        solo_procesados: Si True, solo incluye chunks con estado='procesado'
        max_records: Límite de registros a cargar (None = todos)
    
    Returns:
        Tupla (índice FAISS, mapping list)
    """
    print("🔄 Iniciando reconstrucción del índice FAISS...")
    
    conn = get_connection()
    cur = conn.cursor()
    
    # Construir query según parámetros
    where_clause = "WHERE chunk_embedding IS NOT NULL"
    if solo_procesados:
        where_clause += " AND estado = 'procesado'"
    
    limit_clause = f"LIMIT {max_records}" if max_records else ""
    
    query = f"""
        SELECT uuid, chunk_numero, chunk_embedding 
        FROM documents_logs 
        {where_clause}
        ORDER BY id
        {limit_clause}
    """
    
    print(f"📊 Ejecutando query: {query}")
    cur.execute(query)
    rows = cur.fetchall()
    cur.close()
    conn.close()
    
    if not rows:
        print("⚠️  No se encontraron embeddings en la base de datos")
        return None, []
    
    print(f"✅ Encontrados {len(rows)} embeddings en la BD")
    
    # Crear nuevo índice FAISS
    print(f"🔨 Creando índice FAISS (dim={VECTOR_DIMENSION})...")
    index = faiss.IndexHNSWFlat(VECTOR_DIMENSION, 32)
    mapping = []
    embeddings_list = []
    
    # Procesar embeddings
    print("📦 Procesando embeddings...")
    for i, (doc_uuid, chunk_num, emb_raw) in enumerate(rows):
        if i % 1000 == 0:
            print(f"   Procesados: {i}/{len(rows)}")
        
        # Convertir y normalizar embedding
        emb = np.array(emb_raw, dtype=np.float32)
        faiss.normalize_L2(emb.reshape(1, -1))
        
        embeddings_list.append(emb.flatten())
        mapping.append((str(doc_uuid), chunk_num))
    
    # Agregar todos los embeddings al índice
    print("➕ Agregando embeddings al índice FAISS...")
    embeddings_array = np.stack(embeddings_list).astype("float32")
    index.add(embeddings_array)
    
    print(f"✅ Índice FAISS creado: {index.ntotal} vectores indexados")
    
    return index, mapping


def save_faiss_index(index, mapping, backup=True):
    """
    Guarda el índice FAISS y el mapping en disco.
    
    Args:
        index: Índice FAISS
        mapping: Lista de tuplas (uuid, chunk_numero)
        backup: Si True, crea backup de archivos existentes
    """
    # Crear backup si existen archivos previos
    if backup:
        if FAISS_INDEX_PATH.exists():
            backup_path = FAISS_INDEX_PATH.with_suffix(".bin.backup")
            print(f"💾 Creando backup: {backup_path}")
            FAISS_INDEX_PATH.rename(backup_path)
        
        if FAISS_MAPPING_PATH.exists():
            backup_path = FAISS_MAPPING_PATH.with_suffix(".mapping.backup")
            print(f"💾 Creando backup: {backup_path}")
            FAISS_MAPPING_PATH.rename(backup_path)
    
    # Guardar nuevo índice
    print(f"💾 Guardando índice FAISS: {FAISS_INDEX_PATH}")
    faiss.write_index(index, str(FAISS_INDEX_PATH))
    
    print(f"💾 Guardando mapping: {FAISS_MAPPING_PATH}")
    with open(FAISS_MAPPING_PATH, "wb") as f:
        pickle.dump(mapping, f)
    
    print("✅ Archivos guardados exitosamente")


def verificar_indice():
    """Verifica que el índice guardado se pueda cargar correctamente."""
    print("\n🔍 Verificando índice guardado...")
    
    try:
        # Cargar índice
        index = faiss.read_index(str(FAISS_INDEX_PATH))
        with open(FAISS_MAPPING_PATH, "rb") as f:
            mapping = pickle.load(f)
        
        print(f"✅ Índice cargado correctamente")
        print(f"   - Vectores en índice: {index.ntotal}")
        print(f"   - Entradas en mapping: {len(mapping)}")
        
        if index.ntotal == len(mapping):
            print("✅ Sincronización correcta: índice y mapping coinciden")
            return True
        else:
            print("⚠️  WARNING: El tamaño del índice y mapping no coinciden")
            return False
            
    except Exception as e:
        print(f"❌ Error al verificar índice: {e}")
        return False


def mostrar_estadisticas():
    """Muestra estadísticas de la base de datos."""
    print("\n📊 ESTADÍSTICAS DE LA BASE DE DATOS")
    print("=" * 60)
    
    conn = get_connection()
    cur = conn.cursor()
    
    # Total de chunks
    cur.execute("SELECT COUNT(*) FROM documents_logs")
    total_chunks = cur.fetchone()[0]
    print(f"Total de chunks en BD: {total_chunks}")
    
    # Chunks por estado
    cur.execute("""
        SELECT estado, COUNT(*) 
        FROM documents_logs 
        GROUP BY estado
    """)
    for estado, count in cur.fetchall():
        print(f"  - {estado}: {count}")
    
    # Chunks con embedding
    cur.execute("""
        SELECT COUNT(*) 
        FROM documents_logs 
        WHERE chunk_embedding IS NOT NULL
    """)
    con_embedding = cur.fetchone()[0]
    print(f"Chunks con embedding: {con_embedding}")
    
    # Chunks sin embedding
    sin_embedding = total_chunks - con_embedding
    print(f"Chunks sin embedding: {sin_embedding}")
    
    # Documentos únicos
    cur.execute("SELECT COUNT(DISTINCT uuid) FROM documents_logs")
    docs_unicos = cur.fetchone()[0]
    print(f"Documentos únicos: {docs_unicos}")
    
    # Queries únicas
    cur.execute("SELECT COUNT(*) FROM queries")
    queries = cur.fetchone()[0]
    print(f"Queries en tabla queries: {queries}")
    
    cur.close()
    conn.close()
    print("=" * 60)


def rebuild_queries_faiss():
    """
    Reconstruye el índice FAISS de queries (QueryShield) desde la tabla queries.
    Usa el mismo modelo y lógica que query_shield.py._sync_pg_faiss().
    """
    import warnings, logging, os
    os.environ['TRANSFORMERS_VERBOSITY'] = 'error'
    warnings.filterwarnings('ignore')
    logging.getLogger('transformers').setLevel(logging.ERROR)
    logging.getLogger('sentence_transformers').setLevel(logging.ERROR)

    from sentence_transformers import SentenceTransformer
    from pgvector.psycopg import register_vector

    QUERY_INDEX_PATH   = Path("faiss_index.bin")
    QUERY_MAPPING_PATH = Path("faiss_index.mapping")
    QUERY_DIM          = 384

    print("\n🔄 Reconstruyendo FAISS de queries (QueryShield)...")
    print("⚙️  Cargando modelo paraphrase-multilingual-MiniLM-L12-v2...")
    model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2", device='cpu')
    model.tokenizer.model_max_length = 512
    print("✅ Modelo cargado")

    conn = psycopg.connect(**DB_CONFIG)
    register_vector(conn)
    cur = conn.cursor()
    cur.execute("SELECT uuid, embedding FROM queries ORDER BY fecha_creacion")
    rows = cur.fetchall()
    cur.close(); conn.close()

    if not rows:
        print("⚠️  No hay queries en BD. Índice vacío creado.")
        index = faiss.IndexHNSWFlat(QUERY_DIM, 32)
        mapping = []
    else:
        print(f"📊 {len(rows)} queries encontradas en BD")
        index = faiss.IndexHNSWFlat(QUERY_DIM, 32)
        mapping = []
        embeddings = []

        for uuid_val, emb_raw in rows:
            emb = np.array(emb_raw, dtype=np.float32).flatten()
            faiss.normalize_L2(emb.reshape(1, -1))
            embeddings.append(emb)
            mapping.append(str(uuid_val))

        index.add(np.stack(embeddings).astype('float32'))
        print(f"✅ Índice creado: {index.ntotal} vectores")

    # Backup y guardar
    for path in [QUERY_INDEX_PATH, QUERY_MAPPING_PATH]:
        if path.exists():
            path.rename(path.with_suffix(path.suffix + ".backup"))

    faiss.write_index(index, str(QUERY_INDEX_PATH))
    with open(QUERY_MAPPING_PATH, "wb") as f:
        pickle.dump(mapping, f)

    print(f"💾 Guardado: {QUERY_INDEX_PATH} ({index.ntotal} vectores)")

    # Verificar
    idx2 = faiss.read_index(str(QUERY_INDEX_PATH))
    with open(QUERY_MAPPING_PATH, "rb") as f:
        map2 = pickle.load(f)
    if idx2.ntotal == len(map2):
        print("✅ Verificación OK: índice y mapping coinciden")
    else:
        print("⚠️  Índice y mapping no coinciden")


def main():
    """Función principal."""
    print("🚀 REBUILD FAISS INDEX")
    print("=" * 60)

    mostrar_estadisticas()

    count_procesados, count_total = contar_embeddings_disponibles()
    print(f"\n📈 Embeddings disponibles:")
    print(f"   - Procesados: {count_procesados}")
    print(f"   - Total: {count_total}")

    print("\n⚙️  ¿QUÉ ÍNDICE RECONSTRUIR?")
    print("1. FAISS de chunks/documentos  (DataShield  — faiss_index_bge3)")
    print("2. FAISS de queries            (QueryShield — faiss_index)")
    print("3. Ambos")

    opcion_indice = input("\nSelecciona [1/2/3]: ").strip()
    if opcion_indice not in ("1", "2", "3"):
        print("❌ Opción inválida"); return

    confirmacion = input("\n⚠️  Esto sobrescribirá los archivos existentes. ¿Continuar? [s/N]: ").strip().lower()
    if confirmacion != 's':
        print("❌ Operación cancelada"); return

    # ── FAISS de documentos ───────────────────────────────────────
    if opcion_indice in ("1", "3"):
        print("\n⚙️  OPCIONES PARA FAISS DE CHUNKS:")
        print("1. Solo chunks PROCESADOS (recomendado)")
        print("2. Todos los chunks con embedding")
        sub = input("Selecciona [1/2]: ").strip()
        solo_procesados = sub != "2"

        index, mapping = rebuild_faiss_index(solo_procesados, max_records=None)
        if index is None:
            print("❌ No se pudo crear el índice de chunks"); return
        save_faiss_index(index, mapping, backup=True)
        verificar_indice()

    # ── FAISS de queries ──────────────────────────────────────────
    if opcion_indice in ("2", "3"):
        rebuild_queries_faiss()

    print("\n✅ RECONSTRUCCIÓN COMPLETADA")


if __name__ == "__main__":
    main()
