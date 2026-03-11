#!/usr/bin/env python3
"""
PASO 4 — Queries pendientes (pipeline interrumpido)
Detecta queries APROBADAS que nunca llegaron al scraper.

Las queries pendientes pasaron el filtro de deduplicación en su momento,
pero nunca se insertaron en la tabla queries ni en FAISS (bug de agregar()
comentado). Por eso --recuperar las re-valida contra el FAISS actual antes
de incluirlas en el JSONL, descartando las que ahora sean duplicadas.

Uso:
  python 04_pending.py             → muestra las pendientes
  python 04_pending.py --recuperar → re-valida y genera JSONL para re-scrapeado
  python 04_pending.py --limpiar   → marca como RECHAZADO_DOC (descarta)
"""

import os
import sys
import json
import psycopg
from datetime import datetime
from config import DB_CONFIG, ok, warn, error, info, titulo, BOLD, RESET, AZUL

# QueryShield está en el directorio padre
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

OUTPUT_FILE   = "./queries_pendientes.jsonl"
TEMA_FALLBACK = "General"


def obtener_pendientes(cur):
    cur.execute("""
        SELECT ql.uuid, ql.pregunta, ql.uuid_similar, ql.score, ql.fecha_creacion
        FROM queries_logs ql
        WHERE ql.decision = 'APROBADO'
          AND NOT EXISTS (SELECT 1 FROM queries q WHERE q.uuid = ql.uuid)
          AND NOT EXISTS (
              SELECT 1 FROM queries_logs ql2
              WHERE ql2.uuid = ql.uuid AND ql2.decision = 'RECHAZADO_DOC'
          )
        ORDER BY ql.fecha_creacion
    """)
    return cur.fetchall()


def inferir_temas(cur, pendientes):
    uuids_similar = list({str(r[2]) for r in pendientes if r[2]})
    if not uuids_similar:
        return {}
    cur.execute("SELECT uuid::text, tema FROM queries WHERE uuid::text = ANY(%s)", (uuids_similar,))
    return {r[0]: r[1] for r in cur.fetchall()}


def main():
    args      = sys.argv[1:]
    recuperar = "--recuperar" in args
    limpiar   = "--limpiar"   in args

    print(f"\n{BOLD}{AZUL}{'='*65}{RESET}")
    print(f"{BOLD}  PASO 4 — QUERIES PENDIENTES{RESET}")
    print(f"  {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}")
    print(f"{BOLD}{AZUL}{'='*65}{RESET}")

    conn = psycopg.connect(**DB_CONFIG)
    cur  = conn.cursor()
    pendientes = obtener_pendientes(cur)

    titulo("Estado")
    if not pendientes:
        ok("No hay queries con pipeline interrumpido.\n")
        cur.close(); conn.close(); return

    warn(f"{len(pendientes)} queries APROBADAS sin scraping registrado")
    print()
    for i, (uuid_val, pregunta, *_) in enumerate(pendientes[:5], 1):
        info(f"{i}. {pregunta[:75]}{'...' if len(pregunta) > 75 else ''}")
    if len(pendientes) > 5:
        info(f"... y {len(pendientes) - 5} más")

    if not recuperar and not limpiar:
        print(f"\n  Opciones:")
        print(f"    --recuperar  Re-valida contra FAISS actual y genera JSONL")
        print(f"    --limpiar    Marca como RECHAZADO_DOC (descarta)\n")
        cur.close(); conn.close(); return

    if recuperar:
        _modo_recuperar(cur, conn, pendientes)
    elif limpiar:
        _modo_limpiar(cur, conn, pendientes)

    cur.close(); conn.close()


def _modo_recuperar(cur, conn, pendientes):
    titulo("Recuperar → re-validar contra FAISS actual")

    # Importar QueryShield del proyecto principal
    try:
        from query_shield import QueryShield, Estado
    except ImportError as e:
        error(f"No se pudo importar QueryShield: {e}")
        error("Asegúrate de ejecutar desde el directorio maintenance/\n")
        return

    print("  ⚙️  Cargando QueryShield (puede tardar unos segundos)...")
    shield = QueryShield(DB_CONFIG)
    ok("QueryShield listo\n")

    tema_por_uuid = inferir_temas(cur, pendientes)

    aprobadas   = []
    duplicadas  = []
    sin_tema    = 0

    for uuid_val, pregunta, uuid_similar, score, _ in pendientes:
        uuid_str = str(uuid_val)
        tema     = tema_por_uuid.get(str(uuid_similar) if uuid_similar else "", TEMA_FALLBACK)
        if tema == TEMA_FALLBACK:
            sin_tema += 1

        # Re-validar contra el FAISS actual
        resultado = shield.validar(uuid_str, pregunta)

        if resultado.estado == Estado.DUPLICADO:
            duplicadas.append((uuid_str, pregunta, tema, resultado.score))
            info(f"DUPLICADA (score={resultado.score:.3f}): {pregunta[:60]}...")
        else:
            # NUEVA o AGENTE → incluir en JSONL
            aprobadas.append({
                "id":       uuid_str,
                "pregunta": pregunta,
                "tema":     tema,
                "score":    float(resultado.score)
            })

    # Marcar duplicadas como RECHAZADO en queries_logs
    if duplicadas:
        print()
        warn(f"{len(duplicadas)} queries ahora duplicadas → se marcarán como RECHAZADO en logs")
        for uuid_str, pregunta, tema, score in duplicadas:
            try:
                cur.execute("""
                    INSERT INTO queries_logs (uuid, pregunta, uuid_similar, estado, score, decision)
                    VALUES (%s, %s, %s, 'DUPLICADO', %s, 'RECHAZADO')
                """, (uuid_str, pregunta, uuid_str, score))
            except Exception as e:
                error(f"Error marcando uuid={uuid_str}: {e}")
        conn.commit()

    # Guardar JSONL con las que sí pasan
    print()
    if not aprobadas:
        warn("Ninguna query pendiente sobrevivió la re-validación.")
        warn("Todas eran duplicadas de queries ya existentes.\n")
        return

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for q in aprobadas:
            f.write(json.dumps(q, ensure_ascii=False) + "\n")

    ok(f"{len(aprobadas)} queries válidas guardadas en '{OUTPUT_FILE}'")
    if duplicadas:
        info(f"{len(duplicadas)} descartadas por ser ahora duplicadas")
    if sin_tema > 0:
        warn(f"{sin_tema} queries usaron tema fallback '{TEMA_FALLBACK}'")

    print(f"\n  Próximo paso:")
    print(f"    cp {OUTPUT_FILE} ../queries_validadas.jsonl")
    print(f"    python ../scrapear_queries.py\n")


def _modo_limpiar(cur, conn, pendientes):
    titulo("Limpiar → marcar como RECHAZADO_DOC")
    warn(f"Se marcarán {len(pendientes)} queries como RECHAZADO_DOC")
    if input("  ¿Confirmar? (s/N): ").strip().lower() != "s":
        print("  Cancelado.\n"); return

    insertadas = 0
    for uuid_val, pregunta, uuid_similar, score, _ in pendientes:
        uuid_str     = str(uuid_val)
        uuid_sim_str = str(uuid_similar) if uuid_similar else uuid_str
        try:
            cur.execute("""
                INSERT INTO queries_logs (uuid, pregunta, uuid_similar, estado, score, decision)
                VALUES (%s, %s, %s, 'AGENTE', %s, 'RECHAZADO_DOC')
            """, (uuid_str, pregunta, uuid_sim_str, float(score)))
            insertadas += 1
        except Exception as e:
            error(f"uuid={uuid_str}: {e}")

    conn.commit()
    ok(f"{insertadas} queries marcadas como RECHAZADO_DOC\n")


if __name__ == "__main__":
    main()
