#!/usr/bin/env python3
import os, sys, smtplib, json, psycopg
from email.message import EmailMessage

QUERIES_FILE = "queries_validadas.jsonl"

def validar_jsonl() -> bool:
    if not os.path.exists(QUERIES_FILE):
        print(f"No existe '{QUERIES_FILE}'")
        return False
    count = sum(1 for l in open(QUERIES_FILE, encoding="utf-8") if l.strip())
    if count == 0:
        print(f"'{QUERIES_FILE}' esta vacio")
        return False
    print(f"'{QUERIES_FILE}' listo {count} queries encontradas")
    return True

def obtener_resumen():
    resumen = {}
    try:
        with open(QUERIES_FILE, encoding="utf-8") as f:
            resumen["preguntas_totales"] = sum(1 for l in f if l.strip())
    except:
        resumen["preguntas_totales"] = "Error"

    try:
        conn = psycopg.connect(dbname="datacorpus_bd", user="datacorpus",
                               password="730822", host="localhost")
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM documents_logs WHERE estado='procesado'")
        resumen["chunks_procesados"] = cur.fetchone()[0]
        cur.execute("SELECT COUNT(*) FROM documents_logs WHERE estado='rechazado'")
        resumen["chunks_rechazados"] = cur.fetchone()[0]
        cur.close(); conn.close()
    except:
        resumen["chunks_procesados"] = resumen["chunks_rechazados"] = "Error"

    return resumen

def enviar_correo():
    resumen = obtener_resumen()
    msg = EmailMessage()
    msg["From"] = "cristian.pavez.medina@gmail.com"
    msg["To"]   = "cristian.pavez.medina@gmail.com"
    msg["Subject"] = "DataCorpus - Ejecucion Completada"
    
    html_template = f"""
    <!DOCTYPE html>
    <html>
    <head><meta charset="UTF-8"></head>
    <body style="font-family: -apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,Arial,sans-serif; line-height:1.6; color:#333;">
        <div style="max-width:600px; margin:0 auto; padding:20px; background:#f8f9fa; border-radius:12px;">
            <h1 style="color:#2e7d32; margin:0 0 20px;">Proceso Finalizado</h1>
            
            <div style="background:#fff; padding:25px; border-radius:8px; box-shadow:0 2px 10px rgba(0,0,0,0.1);">
                <h2 style="color:#1976d2; margin-top:0;">Resumen de Ejecucion</h2>
                <table style="width:100%; border-collapse:collapse; margin:20px 0;">
                    <tr><td style="padding:12px; font-weight:600; background:#e3f2fd; border:1px solid #ddd;">Preguntas Procesadas</td><td style="padding:12px; background:#f5f5f5; border:1px solid #ddd;">{resumen.get('preguntas_totales', 'N/A')}</td></tr>
                    <tr><td style="padding:12px; font-weight:600; background:#e8f5e8; border:1px solid #ddd;">Chunks Procesados </td><td style="padding:12px; background:#f5f5f5; border:1px solid #ddd;">{resumen.get('chunks_procesados', 'Error')}</td></tr>
                    <tr><td style="padding:12px; font-weight:600; background:#fff3e0; border:1px solid #ddd;">Chunks Rechazados </td><td style="padding:12px; background:#f5f5f5; border:1px solid #ddd;">{resumen.get('chunks_rechazados', 'Error')}</td></tr>
                </table>
                
                <div style="margin:20px 0; padding:15px; background:#e8f5e8; border-radius:6px; border-left:4px solid #4caf50;">
                    <strong>exito:</strong> Pipeline completado correctamente
                </div>
            </div>
            
            <div style="text-align:center; margin-top:25px; padding:15px; color:#666; font-size:12px;">
                <small>DataCorpus Pipeline | {obtener_fecha_hora()}</small>
            </div>
        </div>
    </body>
    </html>
    """
    
    msg.set_content("DataCorpus Pipeline completado.")
    msg.add_alternative(html_template, subtype="html")
    
    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as smtp:
            smtp.login("cristian.pavez.medina@gmail.com", "nxstohbvicjwthbl")
            smtp.send_message(msg)
        print("Correo enviado")
    except Exception as e:
        print(f"Error correo: {e}")

def obtener_fecha_hora():
    from datetime import datetime
    return datetime.now().strftime("%d/%m/%Y %H:%M")


if __name__ == "__main__":
    # 1. Generar queries
    print("Generando queries...")
    from generar_queries import main as main_queries
    main_queries()
    print("Queries generadas")

    # 2. Validar JSONL
    if not validar_jsonl():
        print("Abortando: no hay queries validas")
        sys.exit(1)

    # 3. Scraper
    print("Iniciando scraper...")
    from scrapear_queries import main as main_scraper
    main_scraper()

    # 4. Notificacion por correo
    enviar_correo()
