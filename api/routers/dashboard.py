import psycopg
from fastapi import APIRouter, HTTPException
from pgvector.psycopg import register_vector

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from config import DB_CONFIG

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
    rows = _query("SELECT * FROM v_resumen")
    return rows[0] if rows else {}


@router.get("/temas", summary="Queries aprobadas por tema")
def temas():
    return _query("SELECT * FROM v_queries_por_tema")


@router.get("/estados-queries", summary="Distribución de estados en queries_logs")
def estados_queries():
    return _query("SELECT * FROM v_estados_queries")


@router.get("/estados-chunks", summary="Distribución de estados en documents_logs")
def estados_chunks():
    return _query("SELECT * FROM v_estados_chunks")


@router.get("/actividad", summary="Actividad diaria por estado")
def actividad():
    return _query("SELECT * FROM v_actividad_diaria LIMIT 90")


@router.get("/tasa-exito", summary="Tasa de éxito por tema")
def tasa_exito():
    return _query("SELECT * FROM v_tasa_exito_temas")


@router.get("/ultimas-queries", summary="Últimas 50 queries aprobadas")
def ultimas_queries():
    rows = _query("SELECT * FROM v_ultimas_queries")
    # Convertir UUID a string para serialización
    for r in rows:
        if "uuid" in r:
            r["uuid"] = str(r["uuid"])
        if "fecha_creacion" in r and r["fecha_creacion"]:
            r["fecha_creacion"] = r["fecha_creacion"].isoformat()
    return rows
