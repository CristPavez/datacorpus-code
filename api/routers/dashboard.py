import psycopg
from fastapi import APIRouter, HTTPException
from pgvector.psycopg import register_vector

from core.config import DB_CONFIG

router = APIRouter(prefix="/dashboard", tags=["Dashboard"])


def get_conn():
    conn = psycopg.connect(**DB_CONFIG)
    register_vector(conn)
    return conn


def _query(sql: str, params=()) -> list[dict]:
    conn = get_conn()
    cur  = conn.cursor()
    try:
        cur.execute(sql, params)
        cols = [d[0] for d in cur.description]
        return [dict(zip(cols, row)) for row in cur.fetchall()]
    finally:
        cur.close(); conn.close()


@router.get("/resumen", summary="Resumen general del corpus")
def resumen():
    rows = _query(
        "SELECT total_documents, total_queries, total_uuids_documents_logs, total_uuids_queries_logs, "
        "documents_aprobados, documents_duplicados, documents_omitidos, documents_similares, "
        "queries_aprobadas, queries_similares, queries_sin_resultados, queries_omitidas, queries_duplicadas "
        "FROM public.v_resumen_totales"
    )
    return rows[0] if rows else {}


@router.get("/estados", summary="Distribución de estados en documents_logs y queries_logs")
def estados():
    return _query("SELECT * FROM v_estados_resumen")


@router.get("/temas", summary="Queries por tema")
def temas():
    return _query("SELECT * FROM v_queries_por_tema")


@router.get("/ultimas-queries", summary="Últimas queries generadas")
def ultimas_queries():
    rows = _query("SELECT * FROM v_ultimas_queries")
    for r in rows:
        if "uuid" in r:
            r["uuid"] = str(r["uuid"])
        if "fecha_creacion" in r and r["fecha_creacion"]:
            r["fecha_creacion"] = r["fecha_creacion"].isoformat()
    return rows
