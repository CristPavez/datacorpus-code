#!/usr/bin/env python3
"""
PASO 2 — Dashboard de estadísticas
Visión general del estado del corpus: volumen, temas, tasas, evolución temporal.
"""

import psycopg
from datetime import datetime
from config import DB_CONFIG, titulo, BOLD, RESET, AZUL, VERDE, AMARILLO, GRIS


def barra(valor, maximo, ancho=30):
    filled = int((valor / maximo) * ancho) if maximo > 0 else 0
    return f"{VERDE}{'█' * filled}{GRIS}{'░' * (ancho - filled)}{RESET}"


def main():
    print(f"\n{BOLD}{AZUL}{'='*65}{RESET}")
    print(f"{BOLD}  PASO 2 — DASHBOARD DE ESTADÍSTICAS{RESET}")
    print(f"  {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}")
    print(f"{BOLD}{AZUL}{'='*65}{RESET}")

    conn = psycopg.connect(**DB_CONFIG)
    cur  = conn.cursor()

    # ── RESUMEN GLOBAL ────────────────────────────────────────────
    titulo("Resumen global")
    cur.execute("""
        SELECT
            (SELECT COUNT(*) FROM queries)                                            AS queries,
            (SELECT COUNT(*) FROM documents)                                          AS docs,
            (SELECT COUNT(*) FROM documents_logs WHERE estado = 'procesado')          AS chunks,
            (SELECT COUNT(*) FROM queries_logs WHERE decision = 'RECHAZADO_DOC')      AS rechazados_doc,
            (SELECT COUNT(*) FROM queries_logs WHERE decision = 'RECHAZADO')          AS rechazados_q,
            (SELECT COUNT(*) FROM documents_logs WHERE decision IN ('APROBADO','APROBADO_LLM')) AS chunks_ok,
            (SELECT COUNT(*) FROM documents_logs WHERE decision IN ('RECHAZADO','RECHAZADO_LLM')) AS chunks_ko
    """)
    queries, docs, chunks, rech_doc, rech_q, chunks_ok, chunks_ko = cur.fetchone()

    total_intentos_q  = queries + rech_doc + rech_q
    tasa_aprobacion_q = (queries / total_intentos_q * 100) if total_intentos_q > 0 else 0
    total_chunks      = chunks_ok + chunks_ko
    tasa_chunks_ok    = (chunks_ok / total_chunks * 100) if total_chunks > 0 else 0

    print(f"  {'Queries aprobadas:':<30} {queries:>7,}")
    print(f"  {'Documentos guardados:':<30} {docs:>7,}")
    print(f"  {'Chunks indexados:':<30} {chunks:>7,}")
    print(f"  {'Tasa aprobación queries:':<30} {tasa_aprobacion_q:>6.1f}%")
    print(f"  {'Tasa aprobación chunks:':<30} {tasa_chunks_ok:>6.1f}%")

    # ── QUERIES POR TEMA ──────────────────────────────────────────
    titulo("Queries aprobadas por tema")
    cur.execute("""
        SELECT tema, COUNT(*) as total
        FROM queries
        GROUP BY tema
        ORDER BY total DESC
    """)
    rows = cur.fetchall()
    max_val = rows[0][1] if rows else 1
    for tema, total in rows:
        bar = barra(total, max_val)
        print(f"  {tema:<35} {bar} {total:>4}")

    # ── TASA DE ÉXITO POR TEMA ────────────────────────────────────
    titulo("Tasa de éxito scraping por tema (aprobados vs intentos)")
    cur.execute("""
        SELECT
            q.tema,
            COUNT(DISTINCT d.uuid_queries)                                    AS aprobados,
            COUNT(DISTINCT dl_r.uuid)                                         AS rechazados
        FROM queries q
        LEFT JOIN documents d ON d.uuid_queries = q.uuid
        LEFT JOIN documents_logs dl_r
            ON dl_r.uuid = q.uuid
           AND dl_r.decision IN ('RECHAZADO','RECHAZADO_LLM')
        GROUP BY q.tema
        ORDER BY aprobados DESC
    """)
    rows = cur.fetchall()
    print(f"  {'Tema':<35} {'Aprobados':>10} {'Rechazados':>12} {'Tasa':>7}")
    print(f"  {'─'*63}")
    for tema, aprobados, rechazados in rows:
        total = aprobados + rechazados
        tasa  = (aprobados / total * 100) if total > 0 else 0
        color = VERDE if tasa >= 70 else (AMARILLO if tasa >= 40 else "\033[91m")
        print(f"  {tema:<35} {aprobados:>10,} {rechazados:>12,} {color}{tasa:>6.1f}%{RESET}")

    # ── MÉTODO DE APROBACIÓN ──────────────────────────────────────
    titulo("Método de aprobación de chunks")
    cur.execute("""
        SELECT decision, COUNT(*) FROM documents_logs
        WHERE decision IN ('APROBADO','APROBADO_LLM')
        GROUP BY decision
    """)
    rows = cur.fetchall()
    total_ap = sum(r[1] for r in rows)
    for decision, cnt in rows:
        pct = cnt / total_ap * 100 if total_ap > 0 else 0
        print(f"  {decision:<20} {cnt:>8,}  ({pct:.1f}%)")

    # ── TOP URLs FUENTE ───────────────────────────────────────────
    titulo("Top 10 dominios fuente (chunks aprobados)")
    cur.execute("""
        SELECT
            REGEXP_REPLACE(url_busqueda, '^https?://([^/]+).*', '\\1') AS dominio,
            COUNT(*) AS chunks
        FROM documents_logs
        WHERE decision IN ('APROBADO','APROBADO_LLM')
          AND url_busqueda IS NOT NULL
        GROUP BY dominio
        ORDER BY chunks DESC
        LIMIT 10
    """)
    rows = cur.fetchall()
    max_v = rows[0][1] if rows else 1
    for dominio, chunks_cnt in rows:
        bar = barra(chunks_cnt, max_v, 20)
        print(f"  {dominio:<40} {bar} {chunks_cnt:>5,}")

    # ── EVOLUCIÓN ÚLTIMOS 14 DÍAS ─────────────────────────────────
    titulo("Documentos aprobados — últimos 14 días")
    cur.execute("""
        SELECT
            DATE(fecha_creacion) AS dia,
            COUNT(DISTINCT uuid) AS docs
        FROM documents_logs
        WHERE decision IN ('APROBADO','APROBADO_LLM')
          AND fecha_creacion >= NOW() - INTERVAL '14 days'
        GROUP BY dia
        ORDER BY dia
    """)
    rows = cur.fetchall()
    if not rows:
        print(f"  {GRIS}Sin actividad en los últimos 14 días{RESET}")
    else:
        max_v = max(r[1] for r in rows)
        for dia, docs_cnt in rows:
            bar = barra(docs_cnt, max_v, 25)
            print(f"  {str(dia)} {bar} {docs_cnt:>4}")

    # ── DISTRIBUCIÓN DE SCORES ────────────────────────────────────
    titulo("Distribución de scores (documents_logs procesados)")
    cur.execute("""
        SELECT
            CASE
                WHEN score_similar < 0.70 THEN '0.00 – 0.69 (nuevo)'
                WHEN score_similar < 0.85 THEN '0.70 – 0.84 (nuevo)'
                WHEN score_similar < 0.95 THEN '0.85 – 0.94 (zona agente)'
                WHEN score_similar < 1.00 THEN '0.95 – 0.99 (zona agente)'
                ELSE                           '1.00        (duplicado)'
            END AS rango,
            COUNT(*) AS cnt
        FROM documents_logs
        WHERE estado = 'procesado' AND score_similar IS NOT NULL
        GROUP BY rango
        ORDER BY rango
    """)
    rows = cur.fetchall()
    max_v = max(r[1] for r in rows) if rows else 1
    for rango, cnt in rows:
        bar = barra(cnt, max_v)
        print(f"  {rango:<30} {bar} {cnt:>6,}")

    print()
    cur.close(); conn.close()


if __name__ == "__main__":
    main()
