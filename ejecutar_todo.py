#!/usr/bin/env python3
import os, sys

QUERIES_FILE = "queries_validadas.jsonl"

def validar_jsonl() -> bool:
    if not os.path.exists(QUERIES_FILE):
        print(f"❌ No existe '{QUERIES_FILE}'")
        return False
    count = sum(1 for l in open(QUERIES_FILE, encoding="utf-8") if l.strip())
    if count == 0:
        print(f"❌ '{QUERIES_FILE}' está vacío")
        return False
    print(f"✅ '{QUERIES_FILE}' listo → {count} queries encontradas")
    return True

if __name__ == "__main__":
    # 1. Generar queries → espera a que termine
    print("🧠 Generando queries...")
    from generar_queries import main as main_queries
    main_queries()
    print("✅ Queries generadas")

    # 2. Validar JSONL antes de scraper
    if not validar_jsonl():
        print("🛑 Abortando: no hay queries válidas")
        sys.exit(1)

    # 3. Scraper → solo si validación OK
    print("🕷️  Iniciando scraper...")
    from scrapear_queries import main as main_scraper
    main_scraper()
        enviar_correo_finalizado()
    
    import smtplib
    from email.message import EmailMessage
    import json
    import psycopg

    def obtener_resumen_ejecucion():
        resumen = {}
        # Preguntas totales
        try:
            with open("queries_validadas.jsonl", encoding="utf-8") as f:
                preguntas = [json.loads(l) for l in f]
            resumen["preguntas_totales"] = len(preguntas)
        except Exception:
            resumen["preguntas_totales"] = "Error al leer queries_validadas.jsonl"

        # Chunks procesados, errores, etc. desde la base de datos
        try:
            conn = psycopg.connect(dbname="datacorpus_bd", user="datacorpus", password="730822", host="127.0.0.1", port=5433)
            cur = conn.cursor()
            cur.execute("SELECT COUNT(*) FROM documents_logs WHERE estado='procesado'")
            resumen["chunks_procesados"] = cur.fetchone()[0]
            cur.execute("SELECT COUNT(*) FROM documents_logs WHERE estado='rechazado'")
            resumen["chunks_rechazados"] = cur.fetchone()[0]
            cur.execute("SELECT COUNT(*) FROM logs_error")
            resumen["errores"] = cur.fetchone()[0]
            cur.close(); conn.close()
        except Exception:
            resumen["chunks_procesados"] = resumen["chunks_rechazados"] = resumen["errores"] = "Error al consultar BD"

        # Datos totales
        resumen["datos_totales"] = resumen.get("chunks_procesados",0) + resumen.get("chunks_rechazados",0) if isinstance(resumen.get("chunks_procesados",0), int) else "-"
        return resumen

    def enviar_correo_finalizado():
        remitente = "cristian.pavez.medina@gmail.com"
        destinatario = "cristian.pavez.medina@gmail.com"
        asunto = "✅ Ejecución completada: ejecutar_todo.py"
        resumen = obtener_resumen_ejecucion()
        cuerpo = f"""
    <html>
    <body style='font-family:sans-serif;'>
      <h2 style='color:#2e6c80;'>¡Proceso finalizado exitosamente!</h2>
      <p>Resumen de la ejecución:</p>
      <ul>
        <li><b>Preguntas totales:</b> {resumen['preguntas_totales']}</li>
        <li><b>Chunks procesados:</b> {resumen['chunks_procesados']}</li>
        <li><b>Chunks rechazados:</b> {resumen['chunks_rechazados']}</li>
        <li><b>Errores registrados:</b> {resumen['errores']}</li>
        <li><b>Datos totales (chunks):</b> {resumen['datos_totales']}</li>
      </ul>
      <hr>
      <small>Enviado automáticamente por ejecutar_todo.py</small>
    </body>
    </html>
    """

        msg = EmailMessage()
        msg["From"] = remitente
        msg["To"] = destinatario
        msg["Subject"] = asunto
        msg.set_content("El proceso ha finalizado. Ver resumen en HTML.")
        msg.add_alternative(cuerpo, subtype="html")

        try:
            with smtplib.SMTP_SSL("smtp.gmail.com", 465) as smtp:
                smtp.login(remitente, "nxstohbvicjwthbl")
                smtp.send_message(msg)
            print("Correo de notificación enviado.")
        except Exception as e:
            print(f"[ERROR] No se pudo enviar el correo: {e}")
