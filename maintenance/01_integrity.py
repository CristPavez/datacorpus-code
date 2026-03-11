#!/usr/bin/env python3
"""
PASO 1 — Integridad de datos
Verifica consistencia entre todas las tablas y detecta anomalías estructurales.
"""

import psycopg
from datetime import datetime
from config import (DB_CONFIG, DECISION_VALIDAS_QUERIES_LOGS,
                    DECISION_VALIDAS_DOCUMENTS_LOGS, ESTADO_VALIDOS_QUERIES_LOGS,
                    ESTADO_VALIDOS_DOCUMENTS_LOGS, ok, warn, error, info, titulo, BOLD, RESET, AZUL)


def run(cur, label, query, params=None):
    cur.execute(query, params or ())
    rows = cur.fetchall()
    if not rows:
        ok(label)
    return rows


def main():
    print(f"\n{BOLD}{AZUL}{'='*65}{RESET}")
    print(f"{BOLD}  PASO 1 — INTEGRIDAD DE DATOS{RESET}")
    print(f"  {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}")
    print(f"{BOLD}{AZUL}{'='*65}{RESET}")

    conn = psycopg.connect(**DB_CONFIG)
    cur  = conn.cursor()
    problemas = 0

    # ── VOLUMEN ───────────────────────────────────────────────────
    titulo("Volumen general")
    cur.execute("""
        SELECT (SELECT COUNT(*) FROM queries),
               (SELECT COUNT(*) FROM queries_logs),
               (SELECT COUNT(*) FROM documents),
               (SELECT COUNT(*) FROM documents_logs)
    """)
    q, ql, d, dl = cur.fetchone()
    print(f"  queries:        {q:>8,}")
    print(f"  queries_logs:   {ql:>8,}")
    print(f"  documents:      {d:>8,}")
    print(f"  documents_logs: {dl:>8,}")

    # ── QUERIES ───────────────────────────────────────────────────
    titulo("Queries")

    for label, query in [
        ("Todas las queries tienen embedding",
         "SELECT id FROM queries WHERE embedding IS NULL"),
        ("No hay preguntas duplicadas exactas",
         "SELECT pregunta FROM queries GROUP BY pregunta HAVING COUNT(*) > 1"),
        ("No hay uuid duplicados en queries",
         "SELECT uuid FROM queries GROUP BY uuid HAVING COUNT(*) > 1"),
    ]:
        rows = run(cur, label, query)
        if rows:
            error(f"{label} — {len(rows)} problema(s)")
            problemas += len(rows)

    # ── QUERIES_LOGS ──────────────────────────────────────────────
    titulo("Queries logs")

    cur.execute("SELECT DISTINCT decision FROM queries_logs WHERE decision IS NOT NULL")
    inv = {r[0] for r in cur.fetchall()} - DECISION_VALIDAS_QUERIES_LOGS
    if inv: error(f"Decisiones inválidas: {inv}"); problemas += 1
    else:   ok("Todas las decisiones son válidas")

    cur.execute("SELECT DISTINCT estado FROM queries_logs WHERE estado IS NOT NULL")
    inv = {r[0] for r in cur.fetchall()} - ESTADO_VALIDOS_QUERIES_LOGS
    if inv: error(f"Estados inválidos: {inv}"); problemas += 1
    else:   ok("Todos los estados son válidos")

    rows = run(cur, "Scores en rango [0, 1]",
               "SELECT id FROM queries_logs WHERE score < 0 OR score > 1")
    if rows: error(f"Scores fuera de rango: {len(rows)}"); problemas += len(rows)

    # APROBADO sin queries ni RECHAZADO_DOC = pipeline interrumpido
    cur.execute("""
        SELECT COUNT(*) FROM queries_logs ql
        WHERE ql.decision = 'APROBADO'
          AND NOT EXISTS (SELECT 1 FROM queries q WHERE q.uuid = ql.uuid)
          AND NOT EXISTS (SELECT 1 FROM queries_logs ql2
                          WHERE ql2.uuid = ql.uuid AND ql2.decision = 'RECHAZADO_DOC')
    """)
    n = cur.fetchone()[0]
    if n: warn(f"{n} queries APROBADAS sin scraping registrado (pipeline interrumpido)"); problemas += n
    else: ok("No hay queries con pipeline interrumpido")

    # APROBADO → RECHAZADO_DOC (flujo correcto, solo informativo)
    cur.execute("""
        SELECT COUNT(DISTINCT uuid) FROM queries_logs
        WHERE decision = 'APROBADO'
          AND NOT EXISTS (SELECT 1 FROM queries q WHERE q.uuid = queries_logs.uuid)
          AND EXISTS (SELECT 1 FROM queries_logs ql2
                      WHERE ql2.uuid = queries_logs.uuid AND ql2.decision = 'RECHAZADO_DOC')
    """)
    n = cur.fetchone()[0]
    if n > 0: ok(f"{n} queries con flujo APROBADO → RECHAZADO_DOC (correcto)")

    # ── DOCUMENTS ─────────────────────────────────────────────────
    titulo("Documents")

    for label, query in [
        ("Todos los uuid_queries existen en queries",
         "SELECT id FROM documents WHERE NOT EXISTS (SELECT 1 FROM queries q WHERE q.uuid = documents.uuid_queries)"),
        ("No hay uuid_queries duplicados",
         "SELECT uuid_queries FROM documents GROUP BY uuid_queries HAVING COUNT(*) > 1"),
        ("Todos los documents tienen logs",
         "SELECT id FROM documents d WHERE NOT EXISTS (SELECT 1 FROM documents_logs dl WHERE dl.uuid = d.uuid_queries)"),
        ("Todos los documents tienen contenido > 100 chars",
         "SELECT id FROM documents WHERE length(dato) < 100"),
    ]:
        rows = run(cur, label, query)
        if rows: error(f"{label} — {len(rows)} problema(s)"); problemas += len(rows)

    # ── DOCUMENTS_LOGS ────────────────────────────────────────────
    titulo("Documents logs")

    cur.execute("SELECT DISTINCT decision FROM documents_logs WHERE decision IS NOT NULL")
    inv = {r[0] for r in cur.fetchall()} - DECISION_VALIDAS_DOCUMENTS_LOGS
    if inv: error(f"Decisiones inválidas: {inv}"); problemas += 1
    else:   ok("Todas las decisiones son válidas")

    cur.execute("SELECT DISTINCT estado FROM documents_logs WHERE estado IS NOT NULL")
    inv = {r[0] for r in cur.fetchall()} - ESTADO_VALIDOS_DOCUMENTS_LOGS
    if inv: error(f"Estados inválidos: {inv}"); problemas += 1
    else:   ok("Todos los estados son válidos")

    rows = run(cur, "Chunks procesados tienen embedding",
               "SELECT id FROM documents_logs WHERE estado='procesado' AND chunk_embedding IS NULL")
    if rows: error(f"Chunks procesados sin embedding: {len(rows)}"); problemas += len(rows)

    rows = run(cur, "No hay chunks APROBADOS duplicados",
               """SELECT uuid, chunk_numero FROM documents_logs
                  WHERE decision IN ('APROBADO','APROBADO_LLM')
                  GROUP BY uuid, chunk_numero HAVING COUNT(*) > 1""")
    if rows: error(f"Chunks APROBADOS duplicados: {len(rows)}"); problemas += len(rows)

    # Rechazos múltiples (esperado)
    cur.execute("""
        SELECT COUNT(*) FROM (
            SELECT uuid, chunk_numero FROM documents_logs
            WHERE decision IN ('RECHAZADO','RECHAZADO_LLM')
            GROUP BY uuid, chunk_numero HAVING COUNT(*) > 1
        ) s
    """)
    n = cur.fetchone()[0]
    if n > 0: ok(f"{n} combinaciones con múltiples rechazos (múltiples URLs por query — correcto)")

    rows = run(cur, "Chunks APROBADOS tienen documento correspondiente",
               """SELECT dl.id FROM documents_logs dl
                  WHERE dl.decision IN ('APROBADO','APROBADO_LLM')
                    AND NOT EXISTS (SELECT 1 FROM documents d WHERE d.uuid_queries = dl.uuid)""")
    if rows: error(f"Chunks APROBADOS sin documento: {len(rows)}"); problemas += len(rows)

    cur.execute("""
        SELECT COUNT(DISTINCT dl.uuid) FROM documents_logs dl
        WHERE dl.decision IN ('RECHAZADO','RECHAZADO_LLM')
          AND NOT EXISTS (SELECT 1 FROM documents d WHERE d.uuid_queries = dl.uuid)
    """)
    n = cur.fetchone()[0]
    if n > 0: ok(f"{n} UUIDs con solo rechazos sin documento (correcto)")

    rows = run(cur, "score_similar en rango [0, 1]",
               "SELECT id FROM documents_logs WHERE score_similar IS NOT NULL AND (score_similar < 0 OR score_similar > 1)")
    if rows: error(f"Scores fuera de rango: {len(rows)}"); problemas += len(rows)

    # ── CONSISTENCIA ──────────────────────────────────────────────
    titulo("Consistencia entre tablas")

    cur.execute("""
        SELECT COUNT(DISTINCT ql.uuid) FROM queries_logs ql
        WHERE ql.decision = 'APROBADO'
          AND EXISTS (SELECT 1 FROM documents d WHERE d.uuid_queries = ql.uuid)
    """)
    print(f"  Queries con documento aprobado: {cur.fetchone()[0]:,}")

    cur.execute("""
        SELECT COUNT(DISTINCT uuid_queries) FROM documents d
        WHERE NOT EXISTS (
            SELECT 1 FROM documents_logs dl
            WHERE dl.uuid = d.uuid_queries AND dl.decision IN ('APROBADO','APROBADO_LLM')
        )
    """)
    n = cur.fetchone()[0]
    if n > 0: warn(f"{n} documents sin ningún chunk aprobado"); problemas += n
    else:     ok("Todos los documents tienen al menos 1 chunk aprobado")

    # ── RESUMEN ───────────────────────────────────────────────────
    print(f"\n  {'─'*60}")
    if problemas == 0:
        ok(f"Sin problemas detectados ✨")
    else:
        warn(f"{problemas} problema(s) detectado(s) — revisar arriba")
    print()

    cur.close(); conn.close()
    return problemas


if __name__ == "__main__":
    main()
