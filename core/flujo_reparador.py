#!/usr/bin/env python3
"""
DataCorpus — Flujo Reparador.

Mantiene siempre el UUID original de cada intento para trazabilidad completa.
Divide la reparación en 2 flujos:

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
FLUJO 1 — Reparación de vacíos (pipeline interrumpido)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Detecta queries en `queries` sin documento en `documents` Y sin entrada
en `queries_logs` (el pipeline se cayó antes de que se pudiera loguear).
Acción: re-scrapear con pregunta original, mismo UUID.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
FLUJO 2 — Reprocesamiento de estados fallidos (con log registrado)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Detecta queries cuyo ÚLTIMO estado en `queries_logs` es no-APROBADO.
Siempre usa el UUID original para que el mapeo quede completo.

  DUPLICADA / SIMILAR  (UUID NO está en queries — falló en QueryShield)
    → generar nueva pregunta con mismo tema
    → validar con QueryShield forzando el UUID original
    → insertar en queries con UUID original → scrapear → guardar documento

  SIN_RESULTADOS / OMITIDA  (UUID SÍ está en queries — falló en scraping)
    → generar pregunta alternativa con mismo tema (solo para Brave Search)
    → scrapear con pregunta alternativa, guardar documento con UUID original
    → la entrada en queries NO se modifica (el embedding es del original aprobado)
"""

import os, warnings, logging
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'
os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = '1'
warnings.filterwarnings('ignore')
logging.getLogger('transformers').setLevel(logging.ERROR)
logging.getLogger('sentence_transformers').setLevel(logging.ERROR)

import time
import psycopg
from .config import DB_CONFIG

MAX_VACIOS_POR_RUN   = 20
MAX_FALLIDOS_POR_RUN = 20


# ══════════════════════════════════════════════════════════════════
# FLUJO 1 — Reparación de vacíos (sin log, pipeline interrumpido)
# ══════════════════════════════════════════════════════════════════

def obtener_vacios() -> list[dict]:
    """
    queries que están en `queries` pero:
      - NO tienen documento en `documents`
      - NO tienen ninguna entrada en `queries_logs`
    → el pipeline se interrumpió entre la validación y el scraping.
    """
    conn = psycopg.connect(**DB_CONFIG)
    cur  = conn.cursor()
    cur.execute("""
        SELECT q.uuid, q.pregunta, q.tema
        FROM queries q
        LEFT JOIN documents   d  ON d.uuid  = q.uuid
        LEFT JOIN queries_logs ql ON ql.uuid = q.uuid
        WHERE d.id  IS NULL
          AND ql.id IS NULL
        ORDER BY q.fecha_creacion ASC
        LIMIT %s
    """, (MAX_VACIOS_POR_RUN,))
    rows = cur.fetchall()
    cur.close(); conn.close()
    return [{"uuid": str(r[0]), "pregunta": r[1], "tema": r[2]} for r in rows]


def _limpiar_logs_huerfanos(uuid_str: str):
    """
    Elimina entradas en documents_logs para un UUID que no tiene documento en documents.
    Se llama antes de re-scrapear en Flujo 1 para evitar mezclar logs de intentos distintos.
    """
    conn = psycopg.connect(**DB_CONFIG)
    cur  = conn.cursor()
    cur.execute(
        "DELETE FROM documents_logs WHERE uuid = %s AND NOT EXISTS "
        "(SELECT 1 FROM documents WHERE uuid = documents_logs.uuid)",
        (uuid_str,)
    )
    eliminados = cur.rowcount
    conn.commit()
    cur.close(); conn.close()
    if eliminados:
        print(f"   🧹 {eliminados} log(s) huérfano(s) eliminados de documents_logs")


def reparar_vacio(vacio: dict, dshield, qshield) -> bool:
    """
    Re-scrapea con la pregunta original manteniendo el UUID.
    Limpia documents_logs huérfanos antes de intentarlo.
    Si tiene éxito guarda el documento. Si falla loguea el estado final.
    """
    from .scrapear_queries import buscar_brave, extraer_contenido, procesar_chunks_url
    from .data_shield import DataShield

    uuid_original = vacio["uuid"]
    pregunta      = vacio["pregunta"]
    tema          = vacio["tema"]

    print(f"   UUID:     {uuid_original}")
    print(f"   Pregunta: {pregunta[:70]}...")
    _limpiar_logs_huerfanos(uuid_original)

    urls = buscar_brave(pregunta)
    if not urls:
        print(f"   ⚠️  Sin URLs → SIN_RESULTADOS")
        qshield.log(uuid_original, pregunta, 0.0, None, "SIN_RESULTADOS", tema)
        return False

    hubo_contenido = False
    url_aprobada   = False

    for url_data in urls:
        url       = url_data["url"]
        contenido = extraer_contenido(url)
        if not contenido or len(contenido) < 100:
            continue

        hubo_contenido = True
        chunks           = dshield.split_text(contenido)
        chunk_resultados = procesar_chunks_url(dshield, chunks)
        rechazados       = sum(1 for c in chunk_resultados if c.estado in ("DUPLICADA", "SIMILAR"))
        pct              = rechazados / len(chunks) if chunks else 0

        print(f"   URL: {url[:60]}...  rechazo={pct:.0%}")

        if pct >= DataShield.RECHAZO_UMBRAL:
            dshield.log_rechazado(uuid_original, url, chunk_resultados)
            continue

        dshield.agregar(uuid_original, contenido, tema, url, chunk_resultados)
        qshield.log(uuid_original, pregunta, 0.0, None, "APROBADO", tema)
        print(f"   ✅ Vacío reparado → documento guardado")
        url_aprobada = True
        break

    if not url_aprobada:
        estado_final = "OMITIDA" if hubo_contenido else "SIN_RESULTADOS"
        qshield.log(uuid_original, pregunta, 0.0, None, estado_final, tema)
        print(f"   ❌ No se pudo reparar → {estado_final}")

    return url_aprobada


def ejecutar_reparacion_vacios(qshield, dshield, stop_event=None):
    print("\n" + "─"*70)
    print("FLUJO 1 — REPARACION DE VACIOS".center(70))
    print("─"*70)

    vacios = obtener_vacios()
    if not vacios:
        print("\n   ✅ Sin vacíos detectados\n")
        return 0

    print(f"\n   {len(vacios)} query(s) sin documento ni log\n")
    aprobadas = 0

    for i, vacio in enumerate(vacios, 1):
        if stop_event and stop_event.is_set():
            print(f"   ⏹️  Detención solicitada — reparación interrumpida en vacío {i}/{len(vacios)}")
            break
        print(f"\n   [{i}/{len(vacios)}]")
        try:
            if reparar_vacio(vacio, dshield, qshield):
                aprobadas += 1
        except Exception as e:
            print(f"   ❌ Error: {e}")
        if i < len(vacios):
            time.sleep(1)

    print(f"\n   Resultado: {aprobadas}/{len(vacios)} vacíos reparados")
    return aprobadas


# ══════════════════════════════════════════════════════════════════
# FLUJO 2 — Reprocesamiento de estados fallidos
# ══════════════════════════════════════════════════════════════════

def obtener_fallidos() -> list[dict]:
    """
    queries cuyo ÚLTIMO estado en queries_logs es DUPLICADA, SIMILAR,
    SIN_RESULTADOS u OMITIDA.

    Usa DISTINCT ON para tomar solo el último registro por UUID,
    así un UUID con APROBADO (QueryShield) + SIN_RESULTADOS (scraping)
    queda correctamente clasificado como SIN_RESULTADOS.
    """
    conn = psycopg.connect(**DB_CONFIG)
    cur  = conn.cursor()
    cur.execute("""
        WITH ultimo AS (
            SELECT DISTINCT ON (uuid)
                uuid, pregunta, tema, estado
            FROM queries_logs
            WHERE tema IS NOT NULL
            ORDER BY uuid, fecha_creacion DESC
        )
        SELECT uuid, pregunta, tema, estado
        FROM ultimo
        WHERE estado IN ('DUPLICADA', 'SIMILAR', 'SIN_RESULTADOS', 'OMITIDA')
        ORDER BY
            CASE estado
                WHEN 'SIN_RESULTADOS' THEN 1
                WHEN 'OMITIDA'        THEN 2
                WHEN 'SIMILAR'        THEN 3
                WHEN 'DUPLICADA'      THEN 4
            END,
            uuid
        LIMIT %s
    """, (MAX_FALLIDOS_POR_RUN,))
    rows = cur.fetchall()
    cur.close(); conn.close()
    return [
        {"uuid": str(r[0]), "pregunta": r[1], "tema": r[2], "estado": r[3]}
        for r in rows
    ]


def _uuid_en_queries(uuid_str: str) -> bool:
    """Verifica si el UUID existe en la tabla queries."""
    conn = psycopg.connect(**DB_CONFIG)
    cur  = conn.cursor()
    cur.execute("SELECT 1 FROM queries WHERE uuid = %s", (uuid_str,))
    existe = cur.fetchone() is not None
    cur.close(); conn.close()
    return existe


def reparar_duplicada_similar(fallido: dict, qshield, dshield) -> bool:
    """
    DUPLICADA / SIMILAR: UUID no está en queries.
    → Genera nueva pregunta con mismo tema.
    → Valida con QueryShield forzando el UUID original.
    → Si pasa, inserta en queries y scrapea.
    """
    from .generar_queries import generar_preguntas_lm, validar_y_procesar
    from .scrapear_queries import buscar_brave, extraer_contenido, procesar_chunks_url
    from .data_shield import DataShield

    uuid_original = fallido["uuid"]
    tema          = fallido["tema"]
    estado        = fallido["estado"]

    print(f"   Estado: {estado}  |  Tema: {tema}")
    print(f"   Pregunta original: {fallido['pregunta'][:65]}...")

    # Generar candidatas con mismo tema
    candidatas = generar_preguntas_lm(tema, n=3)
    if not candidatas:
        print(f"   ❌ No se generaron candidatas")
        return False

    # Validar forzando el UUID original
    resultado = None
    for candidata in candidatas:
        resultado = validar_y_procesar(
            qshield, candidata, tema,
            max_reintentos=3,
            uuid_forzado=uuid_original   # ← mantiene UUID original
        )
        if resultado:
            break

    if not resultado:
        print(f"   ❌ QueryShield rechazó todas las candidatas")
        return False

    nueva_preg = resultado["pregunta"]
    print(f"   QueryShield ✅ {nueva_preg[:65]}...")

    # Scrapear con nueva pregunta, guardar con UUID original
    return _scrapear_y_guardar(uuid_original, nueva_preg, tema, qshield, dshield)


def reparar_sin_resultados_omitida(fallido: dict, qshield, dshield) -> bool:
    """
    SIN_RESULTADOS / OMITIDA: UUID ya está en queries.
    → Genera pregunta alternativa con mismo tema (solo para Brave Search).
    → Scrapea con pregunta alternativa, guarda documento con UUID original.
    → La entrada en queries NO se modifica (embedding del original intacto).
    """
    from .generar_queries import generar_preguntas_lm
    from .data_shield import DataShield

    uuid_original = fallido["uuid"]
    tema          = fallido["tema"]
    estado        = fallido["estado"]

    print(f"   Estado: {estado}  |  Tema: {tema}")
    print(f"   Pregunta original: {fallido['pregunta'][:65]}...")

    # Generar pregunta alternativa (solo para búsqueda, no reemplaza la original)
    candidatas = generar_preguntas_lm(tema, n=3)
    pregunta_busqueda = candidatas[0] if candidatas else fallido["pregunta"]
    print(f"   Pregunta alternativa: {pregunta_busqueda[:65]}...")

    return _scrapear_y_guardar(uuid_original, pregunta_busqueda, tema, qshield, dshield)


def _scrapear_y_guardar(uuid_original: str, pregunta_busqueda: str,
                         tema: str, qshield, dshield) -> bool:
    """
    Lógica común de scraping para ambos casos del flujo 2.
    Usa siempre el UUID original para guardar.
    """
    from .scrapear_queries import buscar_brave, extraer_contenido, procesar_chunks_url
    from .data_shield import DataShield

    urls = buscar_brave(pregunta_busqueda)
    if not urls:
        print(f"   ⚠️  Sin URLs → SIN_RESULTADOS")
        qshield.log(uuid_original, pregunta_busqueda, 0.0, None, "SIN_RESULTADOS", tema)
        return False

    hubo_contenido = False
    url_aprobada   = False

    for url_data in urls:
        url       = url_data["url"]
        contenido = extraer_contenido(url)
        if not contenido or len(contenido) < 100:
            continue

        hubo_contenido = True
        chunks           = dshield.split_text(contenido)
        chunk_resultados = procesar_chunks_url(dshield, chunks)
        rechazados       = sum(1 for c in chunk_resultados if c.estado in ("DUPLICADA", "SIMILAR"))
        pct              = rechazados / len(chunks) if chunks else 0

        print(f"   URL: {url[:60]}...  rechazo={pct:.0%}")

        if pct >= DataShield.RECHAZO_UMBRAL:
            dshield.log_rechazado(uuid_original, url, chunk_resultados)
            continue

        # Guardar con UUID original
        dshield.agregar(uuid_original, contenido, tema, url, chunk_resultados)
        qshield.log(uuid_original, pregunta_busqueda, 0.0, None, "APROBADO", tema)
        print(f"   ✅ Reparado → documento guardado con UUID original")
        url_aprobada = True
        break

    if not url_aprobada:
        estado_final = "OMITIDA" if hubo_contenido else "SIN_RESULTADOS"
        qshield.log(uuid_original, pregunta_busqueda, 0.0, None, estado_final, tema)
        print(f"   ❌ Reparación fallida → {estado_final}")

    return url_aprobada


def ejecutar_reprocesamiento_fallidos(qshield, dshield, stop_event=None):
    print("\n" + "─"*70)
    print("FLUJO 2 — REPROCESAMIENTO DE ESTADOS FALLIDOS".center(70))
    print("─"*70)

    fallidos = obtener_fallidos()
    if not fallidos:
        print("\n   ✅ Sin estados fallidos pendientes\n")
        return 0

    from collections import Counter
    conteo = Counter(f["estado"] for f in fallidos)
    print(f"\n   {len(fallidos)} query(s) a reprocesar:\n")
    for estado, n in sorted(conteo.items()):
        print(f"      {estado:<15} {n}")
    print()

    aprobadas = 0

    for i, fallido in enumerate(fallidos, 1):
        if stop_event and stop_event.is_set():
            print(f"   ⏹️  Detención solicitada — reprocesamiento interrumpido en {i}/{len(fallidos)}")
            break
        print(f"\n   [{i}/{len(fallidos)}]")
        try:
            estado = fallido["estado"]
            if estado in ("DUPLICADA", "SIMILAR"):
                exito = reparar_duplicada_similar(fallido, qshield, dshield)
            else:
                exito = reparar_sin_resultados_omitida(fallido, qshield, dshield)

            if exito:
                aprobadas += 1
        except Exception as e:
            print(f"   ❌ Error inesperado: {e}")

        if i < len(fallidos):
            time.sleep(1)

    print(f"\n   Resultado: {aprobadas}/{len(fallidos)} reprocesadas exitosamente")
    return aprobadas


# ══════════════════════════════════════════════════════════════════
# ORQUESTADOR
# ══════════════════════════════════════════════════════════════════

def ejecutar_flujo_reparador(qshield=None, dshield=None, stop_event=None):
    from .query_shield import QueryShield
    from .data_shield  import DataShield

    print("\n" + "="*70)
    print("FLUJO REPARADOR".center(70))
    print("="*70)

    qshield = qshield or QueryShield(DB_CONFIG)
    dshield = dshield or DataShield(DB_CONFIG)

    aprobados_vacios   = ejecutar_reparacion_vacios(qshield, dshield, stop_event)
    if stop_event and stop_event.is_set():
        print("   ⏹️  Detenido tras Flujo 1")
        return
    aprobados_fallidos = ejecutar_reprocesamiento_fallidos(qshield, dshield, stop_event)

    print("\n" + "="*70)
    print("RESUMEN REPARADOR".center(70))
    print("─"*70)
    print(f"   Vacíos reparados:           {aprobados_vacios}")
    print(f"   Estados fallidos reparados: {aprobados_fallidos}")
    print(f"   Total aprobados:            {aprobados_vacios + aprobados_fallidos}")
    print("="*70 + "\n")


def run_reparador(stop_event=None, qshield=None, dshield=None):
    """Punto de entrada para la API FastAPI.
    qshield y dshield se pasan ya cargados desde el lifespan del servidor.
    """
    ejecutar_flujo_reparador(qshield=qshield, dshield=dshield, stop_event=stop_event)


if __name__ == "__main__":
    ejecutar_flujo_reparador()
