#!/usr/bin/env python3
"""
PASO 5 — Salud del índice FAISS
Compara el índice FAISS en disco contra la BD y detecta desincronizaciones.

Uso:
  python 05_faiss.py           → solo diagnóstico
  python 05_faiss.py --rebuild → reconstruye el índice si hay desincronización
"""

import sys
import os
import pickle
import numpy as np
import faiss
import psycopg
from pathlib import Path
from datetime import datetime
from pgvector.psycopg import register_vector
from config import DB_CONFIG, ok, warn, error, info, titulo, BOLD, RESET, AZUL

FAISS_DOCS_PATH    = Path("../faiss_index_bge3.bin")
MAPPING_DOCS_PATH  = Path("../faiss_index_bge3.mapping")
FAISS_QUERY_PATH   = Path("../faiss_index.bin")
MAPPING_QUERY_PATH = Path("../faiss_index.mapping")


def cargar_faiss(faiss_path: Path, mapping_path: Path):
    """Carga índice FAISS y su mapping. Retorna (index, mapping) o (None, None)."""
    if not faiss_path.exists():
        return None, []
    try:
        index = faiss.read_index(str(faiss_path))
        with open(mapping_path, "rb") as f:
            mapping = pickle.load(f)
        return index, mapping
    except Exception as e:
        return None, []


def main():
    rebuild = "--rebuild" in sys.argv

    print(f"\n{BOLD}{AZUL}{'='*65}{RESET}")
    print(f"{BOLD}  PASO 5 — SALUD DEL ÍNDICE FAISS{RESET}")
    print(f"  {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}")
    print(f"{BOLD}{AZUL}{'='*65}{RESET}")

    conn = psycopg.connect(**DB_CONFIG)
    register_vector(conn)
    cur  = conn.cursor()
    problemas = 0

    # ── FAISS DOCUMENTOS (DataShield) ─────────────────────────────
    titulo("FAISS Documentos (faiss_index_bge3)")

    index_d, mapping_d = cargar_faiss(FAISS_DOCS_PATH, MAPPING_DOCS_PATH)

    if index_d is None:
        warn("Índice FAISS de documentos no encontrado en disco")
        problemas += 1
    else:
        faiss_total = index_d.ntotal
        info(f"Vectores en FAISS:     {faiss_total:,}")

        cur.execute("""
            SELECT COUNT(*) FROM documents_logs
            WHERE chunk_embedding IS NOT NULL AND estado = 'procesado'
        """)
        bd_total = cur.fetchone()[0]
        info(f"Chunks procesados BD:  {bd_total:,}")
        info(f"Tamaño mapping:        {len(mapping_d):,}")

        diff = abs(faiss_total - bd_total)
        if diff == 0:
            ok("FAISS y BD están sincronizados")
        elif diff <= bd_total * 0.02:
            warn(f"Diferencia de {diff} vectores (<2%) — aceptable pero recomendable rebuild")
            problemas += 1
        else:
            error(f"Desincronización: {diff} vectores de diferencia — rebuild necesario")
            problemas += 1

        if len(mapping_d) != faiss_total:
            error(f"Mapping inconsistente: {len(mapping_d)} entradas vs {faiss_total} vectores")
            problemas += 1
        else:
            ok("Mapping consistente con índice")

        # UUIDs en FAISS que no están en BD
        cur.execute("""
            SELECT ARRAY_AGG(DISTINCT uuid::text)
            FROM documents_logs WHERE estado = 'procesado'
        """)
        uuids_bd = set(cur.fetchone()[0] or [])
        uuids_faiss = {str(m[0]) for m in mapping_d}
        huerfanos_faiss = uuids_faiss - uuids_bd
        if huerfanos_faiss:
            warn(f"{len(huerfanos_faiss)} UUIDs en FAISS que no están en BD")
            problemas += 1
        else:
            ok("Todos los UUIDs del FAISS existen en BD")

    # ── FAISS QUERIES (QueryShield) ────────────────────────────────
    titulo("FAISS Queries (faiss_index)")

    index_q, mapping_q = cargar_faiss(FAISS_QUERY_PATH, MAPPING_QUERY_PATH)

    if index_q is None:
        warn("Índice FAISS de queries no encontrado en disco")
        problemas += 1
    else:
        faiss_total = index_q.ntotal
        info(f"Vectores en FAISS:  {faiss_total:,}")

        cur.execute("SELECT COUNT(*) FROM queries")
        bd_total = cur.fetchone()[0]
        info(f"Queries en BD:      {bd_total:,}")
        info(f"Tamaño mapping:     {len(mapping_q):,}")

        diff = abs(faiss_total - bd_total)
        if diff == 0:
            ok("FAISS y BD están sincronizados")
        elif diff <= bd_total * 0.02:
            warn(f"Diferencia de {diff} queries (<2%) — aceptable")
            problemas += 1
        else:
            error(f"Desincronización: {diff} queries de diferencia — rebuild necesario")
            problemas += 1

        if len(mapping_q) != faiss_total:
            error(f"Mapping inconsistente: {len(mapping_q)} entradas vs {faiss_total} vectores")
            problemas += 1
        else:
            ok("Mapping consistente con índice")

    # ── ARCHIVOS EN DISCO ─────────────────────────────────────────
    titulo("Archivos en disco")
    archivos = [
        (FAISS_DOCS_PATH,   "faiss_index_bge3.bin   (DataShield)"),
        (MAPPING_DOCS_PATH, "faiss_index_bge3.mapping"),
        (FAISS_QUERY_PATH,  "faiss_index.bin        (QueryShield)"),
        (MAPPING_QUERY_PATH,"faiss_index.mapping"),
    ]
    for path, label in archivos:
        if path.exists():
            size_mb = path.stat().st_size / 1024 / 1024
            ok(f"{label:<40} {size_mb:.1f} MB")
        else:
            warn(f"{label:<40} NO EXISTE")

    # ── REBUILD ───────────────────────────────────────────────────
    print(f"\n  {'─'*60}")
    if problemas == 0:
        ok("Índices FAISS saludables\n")
    else:
        warn(f"{problemas} problema(s) detectado(s)")
        if not rebuild:
            print(f"\n  Para reconstruir: python 05_faiss.py --rebuild")
        else:
            print()
            _rebuild(cur, conn)

    cur.close(); conn.close()


def _rebuild(cur, conn):
    titulo("Reconstruyendo índices FAISS")
    print(f"  Ejecuta manualmente según lo que necesites:\n")
    print(f"    python ../rebuild_faiss.py   → menú interactivo (chunks / queries / ambos)")
    print(f"\n  Opciones rápidas sin menú:")
    print(f"    Opción 1 (solo chunks procesados) → seleccionar 1 → 1")
    print(f"    Opción 2 (solo queries)            → seleccionar 2")
    print(f"    Opción 3 (ambos)                   → seleccionar 3 → 1\n")


if __name__ == "__main__":
    main()
