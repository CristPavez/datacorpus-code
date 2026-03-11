#!/usr/bin/env python3
"""
PASO 3 — Detección y limpieza de ruido
Encuentra queries que son respuestas del LLM en vez de preguntas reales.

Uso:
  python 03_noise.py                        → solo muestra (BD)
  python 03_noise.py --borrar               → borra de BD
  python 03_noise.py --jsonl archivo.jsonl  → solo muestra (JSONL)
  python 03_noise.py --jsonl archivo.jsonl --borrar → limpia JSONL
"""

import sys
import json
import os
import psycopg
from datetime import datetime
from config import DB_CONFIG, es_pregunta_valida, ok, warn, error, info, titulo, BOLD, RESET, AZUL


def main():
    args   = sys.argv[1:]
    borrar = "--borrar" in args

    jsonl_path = None
    if "--jsonl" in args:
        idx = args.index("--jsonl")
        if idx + 1 < len(args):
            jsonl_path = args[idx + 1]
        else:
            print("  ❌ --jsonl requiere una ruta de archivo.\n"); sys.exit(1)

    print(f"\n{BOLD}{AZUL}{'='*65}{RESET}")
    print(f"{BOLD}  PASO 3 — DETECCIÓN DE RUIDO{RESET}")
    print(f"  {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}")
    print(f"{BOLD}{AZUL}{'='*65}{RESET}")

    if jsonl_path:
        _revisar_jsonl(jsonl_path, borrar)
    else:
        _revisar_bd(borrar)


def _revisar_bd(borrar: bool):
    titulo("Análisis en base de datos")

    conn = psycopg.connect(**DB_CONFIG)
    cur  = conn.cursor()

    # Escanear queries (pipeline completo)
    cur.execute("SELECT uuid, pregunta, tema FROM queries ORDER BY fecha_creacion")
    en_queries = cur.fetchall()

    # Escanear queries_logs (pueden tener ruido que nunca llegó a queries)
    cur.execute("""
        SELECT DISTINCT ON (uuid) uuid, pregunta, 'queries_logs' as fuente
        FROM queries_logs
        WHERE NOT EXISTS (SELECT 1 FROM queries q WHERE q.uuid = queries_logs.uuid)
        ORDER BY uuid, fecha_creacion
    """)
    solo_en_logs = cur.fetchall()

    ruido_queries = [(str(u), p, t, "queries",      _razones(p)) for u, p, t    in en_queries   if not es_pregunta_valida(p)]
    ruido_logs    = [(str(u), p, f, "queries_logs", _razones(p)) for u, p, f    in solo_en_logs if not es_pregunta_valida(p)]
    ruido         = ruido_queries + ruido_logs

    total = len(en_queries) + len(solo_en_logs)
    print(f"  Analizadas: {len(en_queries):,} en queries + {len(solo_en_logs):,} solo en logs = {total:,} total")
    print(f"  Ruido:      {len(ruido_queries)} en queries | {len(ruido_logs)} en queries_logs\n")

    if not ruido:
        ok("No se detectó ruido en la BD.\n")
        cur.close(); conn.close(); return

    titulo("Queries de ruido detectadas")
    for i, (uuid_str, pregunta, fuente_o_tema, tabla, razones) in enumerate(ruido, 1):
        print(f"\n  [{i}] [{tabla}]  {fuente_o_tema}")
        info(f"uuid:   {uuid_str}")
        info(f"razón:  {', '.join(razones)}")
        info(f"texto:  {pregunta[:120]}{'...' if len(pregunta) > 120 else ''}")

    if not borrar:
        print(f"\n  ℹ️  Modo lectura. Para eliminar: python 03_noise.py --borrar\n")
        cur.close(); conn.close(); return

    print(f"\n  ⚠️  Se eliminarán {len(ruido)} queries y todos sus registros")
    print(f"      (queries, queries_logs, documents, documents_logs)")
    if input("  ¿Confirmar? (s/N): ").strip().lower() != "s":
        print("  Cancelado.\n"); cur.close(); conn.close(); return

    eliminadas = 0
    for uuid_str, pregunta, _, tabla, razones in ruido:
        try:
            cur.execute("DELETE FROM documents_logs WHERE uuid = %s", (uuid_str,)); dl = cur.rowcount
            cur.execute("DELETE FROM documents WHERE uuid_queries = %s", (uuid_str,)); d = cur.rowcount
            cur.execute("DELETE FROM queries_logs WHERE uuid = %s", (uuid_str,)); ql = cur.rowcount
            cur.execute("DELETE FROM queries WHERE uuid = %s", (uuid_str,)); q = cur.rowcount
            conn.commit()
            eliminadas += 1
            ok(f"{pregunta[:60]}...")
            info(f"queries:{q} | queries_logs:{ql} | documents:{d} | documents_logs:{dl}")
        except Exception as e:
            conn.rollback()
            error(f"Error uuid={uuid_str}: {e}")

    print(f"\n  Eliminadas: {eliminadas}/{len(ruido)}")
    if any(t == "queries" for _, _, _, t, _ in ruido):
        print(f"  ⚠️  Reconstruye el índice FAISS: python ../rebuild_faiss.py")
    print()


def _revisar_jsonl(ruta: str, borrar: bool):
    titulo(f"Análisis en JSONL: {os.path.basename(ruta)}")

    if not os.path.exists(ruta):
        error(f"Archivo no encontrado: {ruta}\n"); return

    with open(ruta, encoding="utf-8") as f:
        todas = [json.loads(l) for l in f if l.strip()]

    ruido   = [(q, _razones(q["pregunta"])) for q in todas if not es_pregunta_valida(q["pregunta"])]
    validas = [q for q in todas if es_pregunta_valida(q["pregunta"])]

    print(f"  Total: {len(todas):,}  |  Válidas: {len(validas):,}  |  Ruido: {len(ruido):,}\n")

    if not ruido:
        ok("No se detectó ruido en el JSONL.\n"); return

    for i, (q, razones) in enumerate(ruido, 1):
        print(f"\n  [{i}] {q.get('tema','?')}")
        info(f"razón: {', '.join(razones)}")
        info(f"texto: {q['pregunta'][:120]}{'...' if len(q['pregunta']) > 120 else ''}")

    if not borrar:
        print(f"\n  ℹ️  Para limpiar: python 03_noise.py --jsonl {ruta} --borrar\n"); return

    if input(f"\n  ¿Eliminar {len(ruido)} entradas? (s/N): ").strip().lower() != "s":
        print("  Cancelado.\n"); return

    with open(ruta, "w", encoding="utf-8") as f:
        for q in validas:
            f.write(json.dumps(q, ensure_ascii=False) + "\n")
    ok(f"JSONL limpiado: {len(validas)} válidas, {len(ruido)} eliminadas.\n")


def _razones(texto: str) -> list:
    from config import FRASES_RUIDO, PALABRAS_INGLES
    p, p_lower = texto.strip(), texto.strip().lower()
    razones = []
    if "?" not in p:        razones.append("sin '?'")
    if len(p) > 500:        razones.append(f"muy larga ({len(p)} chars)")
    for f in FRASES_RUIDO:
        if f in p_lower:    razones.append(f"frase meta: '{f}'"); break
    palabras = p_lower.split()
    if len(palabras) > 5:
        hits = sum(1 for w in palabras if w in PALABRAS_INGLES)
        if hits / len(palabras) > 0.25:
            razones.append(f"posible inglés ({hits}/{len(palabras)} palabras)")
    return razones or ["criterio desconocido"]


if __name__ == "__main__":
    main()
