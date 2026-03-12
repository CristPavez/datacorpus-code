#!/usr/bin/env python3
"""
DataCorpus Pipeline — Orquestador principal.

Pasos:
  0. Cargar modelos una sola vez
  1. Generar queries (generar_queries.py)
  2. Validar JSONL
  3. Scraping y deduplicación (scrapear_queries.py)
  4. Enviar reporte por correo
"""

import os
import sys
import smtplib
import threading
import psycopg
from datetime import datetime
from email.message import EmailMessage
from typing import Optional
from .config import (DB_CONFIG, QUERIES_FILE,
                    EMAIL_FROM, EMAIL_TO, EMAIL_PASSWORD, SMTP_HOST, SMTP_PORT)


# ── Validar JSONL ─────────────────────────────────────────────────
def validar_jsonl() -> bool:
    if not os.path.exists(QUERIES_FILE):
        print(f"   ❌ No existe '{QUERIES_FILE}'")
        return False
    with open(QUERIES_FILE, encoding="utf-8") as f:
        count = sum(1 for l in f if l.strip())
    if count == 0:
        print(f"   ❌ '{QUERIES_FILE}' está vacío")
        return False
    print(f"   ✅ {count} queries listas para scraping")
    return True


# ── Estadísticas para el correo ───────────────────────────────────
def obtener_resumen(timestamp_inicio: datetime) -> dict:
    resumen = {}

    try:
        with open(QUERIES_FILE, encoding="utf-8") as f:
            resumen["generadas"] = sum(1 for l in f if l.strip())
    except Exception:
        resumen["generadas"] = 0

    try:
        conn = psycopg.connect(**DB_CONFIG)
        cur  = conn.cursor()

        # Esta ejecución
        cur.execute("""
            SELECT COUNT(DISTINCT uuid)
            FROM documents_logs
            WHERE estado = 'APROBADO'
              AND fecha_creacion >= %s
        """, (timestamp_inicio,))
        resumen["docs_aprobados"] = cur.fetchone()[0]

        cur.execute("""
            SELECT COUNT(DISTINCT uuid)
            FROM queries_logs
            WHERE estado IN ('DUPLICADA','SIMILAR','OMITIDA','SIN_RESULTADOS')
              AND fecha_creacion >= %s
        """, (timestamp_inicio,))
        resumen["queries_rechazadas"] = cur.fetchone()[0]

        # Histórico total
        cur.execute("SELECT COUNT(*) FROM queries")
        resumen["queries_total"] = cur.fetchone()[0]

        cur.execute("SELECT COUNT(*) FROM documents")
        resumen["docs_total"] = cur.fetchone()[0]

        cur.execute("SELECT COUNT(*) FROM documents_logs WHERE estado = 'APROBADO'")
        resumen["chunks_total"] = cur.fetchone()[0]

        cur.close(); conn.close()
    except Exception as e:
        print(f"   ⚠️  Error estadísticas: {e}")
        for k in ("docs_aprobados", "queries_rechazadas", "queries_total",
                  "docs_total", "chunks_total"):
            resumen.setdefault(k, "?")

    return resumen


# ── Envío de correo ───────────────────────────────────────────────
def enviar_correo(timestamp_inicio: datetime):
    r   = obtener_resumen(timestamp_inicio)
    now = datetime.now().strftime("%d/%m/%Y %H:%M")

    html = f"""<!DOCTYPE html>
<html><head><meta charset="UTF-8"></head>
<body style="font-family:Arial,sans-serif;background:#f5f5f5;padding:30px 10px;">
  <div style="max-width:500px;margin:0 auto;background:#fff;border-radius:8px;overflow:hidden;box-shadow:0 2px 8px rgba(0,0,0,.1);">
    <div style="background:#4CAF50;padding:24px;text-align:center;">
      <h1 style="color:#fff;margin:0;font-size:22px;">✅ Pipeline Completado</h1>
      <p style="color:#e8f5e9;margin:8px 0 0;font-size:13px;">{now}</p>
    </div>
    <div style="padding:28px 24px;">
      <div style="background:#f0f7f0;border-left:4px solid #4CAF50;padding:16px;margin-bottom:20px;border-radius:4px;">
        <h2 style="color:#2e7d32;font-size:14px;margin:0 0 12px;text-transform:uppercase;">Proceso actual</h2>
        <table style="width:100%;font-size:14px;">
          <tr><td style="color:#555;padding:6px 0;">Queries generadas</td>
              <td style="text-align:right;font-weight:700;color:#2e7d32;">{r.get('generadas',0)}</td></tr>
          <tr><td style="color:#555;padding:6px 0;">Documentos aprobados</td>
              <td style="text-align:right;font-weight:700;color:#4CAF50;">{r.get('docs_aprobados',0)}</td></tr>
          <tr><td style="color:#555;padding:6px 0;">Queries rechazadas/omitidas</td>
              <td style="text-align:right;font-weight:700;color:#f44336;">{r.get('queries_rechazadas',0)}</td></tr>
        </table>
      </div>
      <div style="background:#f0f4ff;border-left:4px solid #2196F3;padding:16px;border-radius:4px;">
        <h2 style="color:#1565c0;font-size:14px;margin:0 0 12px;text-transform:uppercase;">Histórico total</h2>
        <table style="width:100%;font-size:14px;">
          <tr><td style="color:#555;padding:6px 0;">Total queries</td>
              <td style="text-align:right;font-weight:700;color:#1565c0;">{r.get('queries_total',0)}</td></tr>
          <tr><td style="color:#555;padding:6px 0;">Total documentos</td>
              <td style="text-align:right;font-weight:700;color:#1976d2;">{r.get('docs_total',0)}</td></tr>
          <tr><td style="color:#555;padding:6px 0;">Total chunks indexados</td>
              <td style="text-align:right;font-weight:700;color:#2196F3;">{r.get('chunks_total',0)}</td></tr>
        </table>
      </div>
    </div>
    <div style="background:#fafafa;padding:16px;text-align:center;border-top:1px solid #eee;">
      <small style="color:#999;font-size:12px;">DataCorpus Pipeline</small>
    </div>
  </div>
</body></html>"""

    msg = EmailMessage()
    msg["From"]    = EMAIL_FROM
    msg["To"]      = EMAIL_TO
    msg["Subject"] = "DataCorpus - Pipeline Completado ✅"
    msg.set_content("DataCorpus Pipeline completado exitosamente.")
    msg.add_alternative(html, subtype="html")

    try:
        with smtplib.SMTP_SSL(SMTP_HOST, SMTP_PORT) as smtp:
            smtp.login(EMAIL_FROM, EMAIL_PASSWORD)
            smtp.send_message(msg)
        print("   ✅ Correo enviado")
    except Exception as e:
        print(f"   ❌ Error enviando correo: {e}")


# ── Función callable desde la API ────────────────────────────────
def run_pipeline(stop_event: Optional[threading.Event] = None, qshield=None, dshield=None):
    """Punto de entrada para la API FastAPI. Ejecuta el pipeline completo.
    qshield y dshield se pasan ya cargados desde el lifespan del servidor.
    Si no se pasan (ejecución por terminal), se cargan aquí.
    """
    timestamp_inicio = datetime.now()

    print("\n" + "="*70)
    print("DATACORPUS PIPELINE - INICIO".center(70))
    print("="*70 + "\n")

    if qshield is None or dshield is None:
        print("\n" + "─"*70)
        print("CARGANDO MODELOS DE IA".center(70))
        print("─"*70)

        from .query_shield import QueryShield
        from .data_shield  import DataShield

        print("   Cargando QueryShield...")
        qshield = qshield or QueryShield(DB_CONFIG)
        print("   ✅ QueryShield listo")
        print("   Cargando DataShield...")
        dshield = dshield or DataShield(DB_CONFIG)
        print("   ✅ DataShield listo\n")

    if stop_event and stop_event.is_set():
        print("   ⏹️  Detención solicitada antes de iniciar"); return

    print("\n" + "─"*70)
    print("GENERACION DE QUERIES".center(70))
    print("─"*70 + "\n")

    from .generar_queries import main as main_queries
    main_queries(qshield_externo=qshield)

    if stop_event and stop_event.is_set():
        print("   ⏹️  Detención solicitada tras generación"); return

    if not validar_jsonl():
        print("\n❌ ABORTANDO: No hay queries válidas para procesar\n"); return

    print("\n" + "─"*70)
    print("SCRAPING Y VALIDACION DE CONTENIDO".center(70))
    print("─"*70 + "\n")

    from .scrapear_queries import main as main_scraper
    main_scraper(dshield_externo=dshield, qshield_externo=qshield, stop_event=stop_event)

    print("\n" + "─"*70)
    print("ENVIANDO REPORTE".center(70))
    print("─"*70 + "\n")
    enviar_correo(timestamp_inicio)

    print("\n" + "="*70)
    print("PIPELINE COMPLETADO".center(70))
    print("="*70 + "\n")


# ── MAIN ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Modos disponibles:
    #   (sin args)   → pipeline completo
    #   --reparar    → solo flujo reparador
    #   --reparar-al-final → pipeline completo + reparador al terminar
    modo_solo_reparar    = "--reparar"        in sys.argv[1:]
    modo_reparar_al_final = "--reparar-al-final" in sys.argv[1:]

    timestamp_inicio = datetime.now()

    print("\n" + "="*70)
    print("DATACORPUS PIPELINE - INICIO".center(70))
    print("="*70 + "\n")

    # ── 0. Cargar modelos ─────────────────────────────────────────
    print("\n" + "─"*70)
    print("CARGANDO MODELOS DE IA".center(70))
    print("─"*70)

    from .query_shield import QueryShield
    from .data_shield  import DataShield

    print("   Cargando QueryShield...")
    qshield = QueryShield(DB_CONFIG)
    print("   ✅ QueryShield listo")

    print("   Cargando DataShield...")
    dshield = DataShield(DB_CONFIG)
    print("   ✅ DataShield listo\n")

    # ── Modo reparador solamente ──────────────────────────────────
    if modo_solo_reparar:
        from .flujo_reparador import ejecutar_flujo_reparador
        ejecutar_flujo_reparador(qshield, dshield)
        enviar_correo(timestamp_inicio)
        sys.exit(0)

    # ── 1. Generar queries ────────────────────────────────────────
    print("\n" + "─"*70)
    print("GENERACION DE QUERIES".center(70))
    print("─"*70 + "\n")

    from .generar_queries import main as main_queries
    main_queries(qshield_externo=qshield)

    # ── 2. Validar JSONL ──────────────────────────────────────────
    if not validar_jsonl():
        print("\n❌ ABORTANDO: No hay queries válidas para procesar\n")
        sys.exit(1)

    # ── 3. Scraping ───────────────────────────────────────────────
    print("\n" + "─"*70)
    print("SCRAPING Y VALIDACION DE CONTENIDO".center(70))
    print("─"*70 + "\n")

    from .scrapear_queries import main as main_scraper
    main_scraper(dshield_externo=dshield, qshield_externo=qshield)

    # ── 4. Flujo reparador (opcional) ─────────────────────────────
    if modo_reparar_al_final:
        from .flujo_reparador import ejecutar_flujo_reparador
        ejecutar_flujo_reparador(qshield, dshield)

    # ── 5. Correo ─────────────────────────────────────────────────
    print("\n" + "─"*70)
    print("ENVIANDO REPORTE".center(70))
    print("─"*70 + "\n")
    enviar_correo(timestamp_inicio)

    print("\n" + "="*70)
    print("PIPELINE COMPLETADO EXITOSAMENTE".center(70))
    print("="*70 + "\n")
