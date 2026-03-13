#!/usr/bin/env python3
"""
Reconstruye los índices FAISS desde la base de datos.
Útil cuando el archivo .bin se corrompe o queda desincronizado.

Índice de queries  → faiss_queries.bin  (dim=384, MiniLM)
Índice de docs     → faiss_docs.bin     (dim=1024, BGE-M3)
"""

import os, warnings, logging
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'
warnings.filterwarnings('ignore')
logging.getLogger('transformers').setLevel(logging.ERROR)
logging.getLogger('sentence_transformers').setLevel(logging.ERROR)

import faiss
import numpy as np
import psycopg
import pickle
from pgvector.psycopg import register_vector
from .config import (DB_CONFIG,
                    FAISS_DOCS_PATH, FAISS_DOCS_MAPPING,
                    FAISS_DOCS_DIM, FAISS_DOCS_HNSW_M,
                    FAISS_QUERY_PATH, FAISS_QUERY_MAPPING,
                    FAISS_QUERY_DIM, FAISS_QUERY_HNSW_M,
                    MODEL_QUERY, MODEL_MAX_LENGTH)


def get_conn():
    conn = psycopg.connect(**DB_CONFIG)
    register_vector(conn)
    return conn


# ── FAISS de documentos ───────────────────────────────────────────
def rebuild_docs_faiss():
    """
    Reconstruye faiss_docs.bin desde documents_logs.
    Mapping: [(documents_logs.id, chunk_numero), ...]
    Solo chunks con estado = 'APROBADO'.
    """
    print("\n  Reconstruyendo FAISS de documentos (BGE-M3)...")

    conn = get_conn()
    cur  = conn.cursor()
    cur.execute("""
        SELECT dl.uuid, dl.chunk_numero, dl.embedding
        FROM documents_logs dl
        INNER JOIN documents d ON d.uuid = dl.uuid
        WHERE dl.embedding IS NOT NULL
          AND dl.estado = 'APROBADO'
        ORDER BY dl.id
    """)
    rows = cur.fetchall()

    # Contar total para informar cuántos huérfanos se omitieron
    cur.execute("""
        SELECT COUNT(*) FROM documents_logs
        WHERE embedding IS NOT NULL AND estado = 'APROBADO'
    """)
    total_logs = cur.fetchone()[0]
    cur.close(); conn.close()

    huerfanos = total_logs - len(rows)
    if huerfanos > 0:
        print(f"  ⚠️  {huerfanos} chunk(s) huérfano(s) omitidos (documento padre no existe en documents)")

    if not rows:
        print("  ⚠️  No hay chunks APROBADO con embedding. Creando índice vacío.")
        index   = faiss.IndexHNSWFlat(FAISS_DOCS_DIM, FAISS_DOCS_HNSW_M)
        mapping = []
    else:
        print(f"  {len(rows)} chunks encontrados")
        index   = faiss.IndexHNSWFlat(FAISS_DOCS_DIM, FAISS_DOCS_HNSW_M)
        mapping = []
        embeddings = []

        for doc_uuid, chunk_num, emb_raw in rows:
            emb = np.array(emb_raw, dtype=np.float32)
            faiss.normalize_L2(emb.reshape(1, -1))
            embeddings.append(emb.flatten())
            mapping.append((str(doc_uuid), int(chunk_num)))

        index.add(np.stack(embeddings).astype("float32"))
        print(f"  ✅ {index.ntotal} vectores indexados")

    _guardar(index, mapping, FAISS_DOCS_PATH, FAISS_DOCS_MAPPING)


# ── FAISS de queries ──────────────────────────────────────────────
def rebuild_queries_faiss():
    """
    Reconstruye faiss_queries.bin desde queries.
    Mapping: [uuid_str, ...]
    """
    print("\n  Reconstruyendo FAISS de queries (MiniLM)...")

    conn = get_conn()
    cur  = conn.cursor()
    cur.execute("SELECT uuid, embedding FROM queries ORDER BY fecha_creacion")
    rows = cur.fetchall()
    cur.close(); conn.close()

    if not rows:
        print("  ⚠️  No hay queries en BD. Creando índice vacío.")
        index   = faiss.IndexHNSWFlat(FAISS_QUERY_DIM, FAISS_QUERY_HNSW_M)
        mapping = []
    else:
        print(f"  {len(rows)} queries encontradas")
        index   = faiss.IndexHNSWFlat(FAISS_QUERY_DIM, FAISS_QUERY_HNSW_M)
        mapping = []
        embeddings = []

        for uuid_val, emb_raw in rows:
            emb = np.array(emb_raw, dtype=np.float32).flatten()
            faiss.normalize_L2(emb.reshape(1, -1))
            embeddings.append(emb)
            mapping.append(str(uuid_val))

        index.add(np.stack(embeddings).astype("float32"))
        print(f"  ✅ {index.ntotal} vectores indexados")

    _guardar(index, mapping, FAISS_QUERY_PATH, FAISS_QUERY_MAPPING)


# ── Guardar ───────────────────────────────────────────────────────
def _guardar(index, mapping, bin_path, map_path):
    faiss.write_index(index, str(bin_path))
    with open(map_path, "wb") as f:
        pickle.dump(mapping, f)

    # Verificación
    idx2 = faiss.read_index(str(bin_path))
    with open(map_path, "rb") as f:
        map2 = pickle.load(f)

    if idx2.ntotal == len(map2):
        print(f"  ✅ Guardado y verificado: {bin_path.name} ({idx2.ntotal} vectores)")
    else:
        print(f"  ⚠️  Índice y mapping no coinciden: {idx2.ntotal} vs {len(map2)}")


# ── MAIN ──────────────────────────────────────────────────────────
def main():
    print("\n" + "="*60)
    print("REBUILD FAISS INDEX".center(60))
    print("="*60)

    print("\n  ¿Qué índice reconstruir?")
    print("  1. FAISS documentos  (faiss_docs.bin    — BGE-M3)")
    print("  2. FAISS queries     (faiss_queries.bin — MiniLM)")
    print("  3. Ambos")

    opcion = input("\n  Selecciona [1/2/3]: ").strip()
    if opcion not in ("1", "2", "3"):
        print("  ❌ Opción inválida"); return

    confirm = input("\n  ⚠️  Esto sobrescribirá los archivos. ¿Continuar? [s/N]: ").strip().lower()
    if confirm != "s":
        print("  ❌ Cancelado"); return

    if opcion in ("1", "3"):
        rebuild_docs_faiss()
    if opcion in ("2", "3"):
        rebuild_queries_faiss()

    print("\n" + "="*60)
    print("RECONSTRUCCION COMPLETADA".center(60))
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
