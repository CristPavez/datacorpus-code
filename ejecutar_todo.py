#!/usr/bin/env python3
import os, sys, smtplib, json, psycopg
from email.message import EmailMessage
import faiss, numpy as np, pickle
from pathlib import Path
from pgvector.psycopg import register_vector

QUERIES_FILE = "queries_validadas.jsonl"

# ── CONFIGURACIÓN FAISS ───────────────────────────────────────────
DB_CONFIG = {
    "dbname": "datacorpus_bd",
    "user": "datacorpus",
    "password": "730822",
    "host": "localhost",
    "port": 5433
}

FAISS_INDEX_PATH = Path("faiss_index_bge3.bin")
FAISS_MAPPING_PATH = Path("faiss_index_bge3.mapping")
VECTOR_DIMENSION = 1024


def rebuild_faiss_automatico():
    """
    Reconstruye el índice FAISS automáticamente desde la BD.
    Solo incluye chunks con estado='procesado'.
    """
    print("\n" + "─"*70)
    print("🔄 SINCRONIZACIÓN FAISS".center(70))
    print("─"*70 + "\n")
    
    try:
        # Conectar a BD
        conn = psycopg.connect(**DB_CONFIG)
        register_vector(conn)
        cur = conn.cursor()
        
        # Obtener embeddings de chunks procesados
        cur.execute("""
            SELECT uuid, chunk_numero, chunk_embedding 
            FROM documents_logs 
            WHERE chunk_embedding IS NOT NULL 
            AND estado = 'procesado'
            ORDER BY id
        """)
        rows = cur.fetchall()
        cur.close()
        conn.close()
        
        if not rows:
            print("   ⚠️  No hay embeddings procesados en la BD")
            print("   🔨 Creando índice vacío...")
            index = faiss.IndexHNSWFlat(VECTOR_DIMENSION, 32)
            mapping = []
        else:
            print(f"   📊 Encontrados {len(rows)} embeddings procesados")
            
            # Crear nuevo índice
            print("   🔨 Reconstruyendo índice FAISS...")
            index = faiss.IndexHNSWFlat(VECTOR_DIMENSION, 32)
            mapping = []
            embeddings_list = []
            
            # Procesar embeddings
            for doc_uuid, chunk_num, emb_raw in rows:
                emb = np.array(emb_raw, dtype=np.float32)
                faiss.normalize_L2(emb.reshape(1, -1))
                embeddings_list.append(emb.flatten())
                mapping.append((str(doc_uuid), chunk_num))
            
            # Agregar al índice
            embeddings_array = np.stack(embeddings_list).astype("float32")
            index.add(embeddings_array)
            print(f"   ✅ Índice creado: {index.ntotal} vectores")
        
        # Crear backup del índice anterior (solo mantener 1 backup)
        if FAISS_INDEX_PATH.exists():
            backup_path = FAISS_INDEX_PATH.with_suffix(".bin.backup")
            # Eliminar backup anterior si existe
            if backup_path.exists():
                backup_path.unlink()
            FAISS_INDEX_PATH.rename(backup_path)
            print(f"   💾 Backup índice anterior guardado")
        
        if FAISS_MAPPING_PATH.exists():
            backup_path = FAISS_MAPPING_PATH.with_suffix(".mapping.backup")
            # Eliminar backup anterior si existe
            if backup_path.exists():
                backup_path.unlink()
            FAISS_MAPPING_PATH.rename(backup_path)
        
        # Guardar nuevos archivos
        faiss.write_index(index, str(FAISS_INDEX_PATH))
        with open(FAISS_MAPPING_PATH, "wb") as f:
            pickle.dump(mapping, f)
        
        print(f"   💾 Nuevo índice guardado: {FAISS_INDEX_PATH.name}")
        print("   ✅ Sincronización completada\n")
        return True
        
    except Exception as e:
        print(f"   ❌ Error en sincronización FAISS: {e}")
        print("   ⚠️  Continuando con índice existente...\n")
        return False

def validar_jsonl() -> bool:
    if not os.path.exists(QUERIES_FILE):
        print(f"   ❌ No existe '{QUERIES_FILE}'")
        return False
    count = sum(1 for l in open(QUERIES_FILE, encoding="utf-8") if l.strip())
    if count == 0:
        print(f"   ❌ '{QUERIES_FILE}' está vacío")
        return False
    print(f"   ✅ {count} queries listas para scraping")
    return True

def obtener_resumen(timestamp_inicio=None):
    resumen = {}
    
    # Queries del JSONL (esta ejecución)
    try:
        with open(QUERIES_FILE, encoding="utf-8") as f:
            resumen["preguntas_generadas"] = sum(1 for l in f if l.strip())
    except:
        resumen["preguntas_generadas"] = 0

    try:
        conn = psycopg.connect(dbname="datacorpus_bd", user="datacorpus",
                               password="730822", host="localhost", port=5433)
        cur = conn.cursor()
        
        # === ESTA EJECUCIÓN ===
        # Si hay timestamp, filtrar solo registros después de ese momento
        if timestamp_inicio:
            cur.execute("""
                SELECT COUNT(DISTINCT uuid) 
                FROM documents_logs 
                WHERE decision IN ('APROBADO', 'APROBADO_LLM')
                AND fecha_creacion >= %s
            """, (timestamp_inicio,))
            resumen["docs_aprobados_run"] = cur.fetchone()[0]
            
            cur.execute("""
                SELECT COUNT(DISTINCT uuid) 
                FROM documents_logs 
                WHERE decision LIKE 'RECHAZADO%%'
                AND fecha_creacion >= %s
            """, (timestamp_inicio,))
            resumen["docs_rechazados_run"] = cur.fetchone()[0]
        else:
            # Fallback: últimos 1000 registros (comportamiento antiguo)
            cur.execute("""
                SELECT COUNT(DISTINCT uuid) 
                FROM documents_logs 
                WHERE decision IN ('APROBADO', 'APROBADO_LLM')
                AND id > (SELECT COALESCE(MAX(id), 0) - 1000 FROM documents_logs)
            """)
            resumen["docs_aprobados_run"] = cur.fetchone()[0]
            
            cur.execute("""
                SELECT COUNT(DISTINCT uuid) 
                FROM documents_logs 
                WHERE decision LIKE 'RECHAZADO%%'
                AND id > (SELECT COALESCE(MAX(id), 0) - 1000 FROM documents_logs)
            """)
            resumen["docs_rechazados_run"] = cur.fetchone()[0]
        
        # === HISTÓRICO TOTAL ===
        cur.execute("SELECT COUNT(*) FROM queries")
        resumen["queries_total"] = cur.fetchone()[0]
        
        cur.execute("SELECT COUNT(*) FROM documents")
        resumen["documentos_total"] = cur.fetchone()[0]
        
        cur.execute("SELECT COUNT(*) FROM documents_logs WHERE estado='procesado'")
        resumen["chunks_total"] = cur.fetchone()[0]
        
        cur.close(); conn.close()
    except Exception as e:
        print(f"Error obteniendo estadísticas: {e}")
        resumen["docs_aprobados_run"] = "Error"
        resumen["docs_rechazados_run"] = "Error"
        resumen["queries_total"] = "Error"
        resumen["documentos_total"] = "Error"
        resumen["chunks_total"] = "Error"

    return resumen

def enviar_correo(timestamp_inicio=None):
    resumen = obtener_resumen(timestamp_inicio)
    msg = EmailMessage()
    msg["From"] = "cristian.pavez.medina@gmail.com"
    msg["To"]   = "cristian.pavez.medina@gmail.com"
    msg["Subject"] = "DataCorpus - Pipeline Completado ✅"
    
    html_template = f"""
    <!DOCTYPE html>
    <html>
    <head><meta charset="UTF-8"></head>
    <body style="font-family: -apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,Arial,sans-serif; background:#f5f5f5; padding:30px 10px; margin:0;">
        <div style="max-width:500px; margin:0 auto; background:#fff; border-radius:8px; overflow:hidden; box-shadow:0 2px 8px rgba(0,0,0,0.1);">
            
            <!-- Header -->
            <div style="background:#4CAF50; padding:24px; text-align:center;">
                <h1 style="color:#fff; margin:0; font-size:22px; font-weight:600;">✅ Pipeline Completado</h1>
                <p style="color:#e8f5e9; margin:8px 0 0; font-size:13px;">{obtener_fecha_hora()}</p>
            </div>
            
            <!-- Content -->
            <div style="padding:28px 24px;">
                
                <!-- Proceso Actual -->
                <div style="background:#f0f7f0; border-left:4px solid #4CAF50; padding:16px; margin-bottom:20px; border-radius:4px;">
                    <h2 style="color:#2e7d32; font-size:14px; margin:0 0 12px; font-weight:600; text-transform:uppercase; letter-spacing:0.5px;">
                        ⚡ Proceso Actual
                    </h2>
                    <table style="width:100%; font-size:14px;">
                        <tr>
                            <td style="padding:6px 0; color:#555;">Queries Procesadas</td>
                            <td style="padding:6px 0; text-align:right; font-weight:700; color:#2e7d32; font-size:16px;">{resumen.get('preguntas_generadas', 0)}</td>
                        </tr>
                        <tr>
                            <td style="padding:6px 0; color:#555;">Documentos Aprobados</td>
                            <td style="padding:6px 0; text-align:right; font-weight:700; color:#4CAF50; font-size:16px;">{resumen.get('docs_aprobados_run', 0)}</td>
                        </tr>
                        <tr>
                            <td style="padding:6px 0; color:#555;">Documentos Rechazados</td>
                            <td style="padding:6px 0; text-align:right; font-weight:700; color:#f44336; font-size:16px;">{resumen.get('docs_rechazados_run', 0)}</td>
                        </tr>
                    </table>
                </div>
                
                <!-- Histórico Total -->
                <div style="background:#f0f4ff; border-left:4px solid #2196F3; padding:16px; border-radius:4px;">
                    <h2 style="color:#1565c0; font-size:14px; margin:0 0 12px; font-weight:600; text-transform:uppercase; letter-spacing:0.5px;">
                        📊 Histórico Total en Base de Datos
                    </h2>
                    <table style="width:100%; font-size:14px;">
                        <tr>
                            <td style="padding:6px 0; color:#555;">Total Queries</td>
                            <td style="padding:6px 0; text-align:right; font-weight:700; color:#1565c0; font-size:16px;">{resumen.get('queries_total', 0)}</td>
                        </tr>
                        <tr>
                            <td style="padding:6px 0; color:#555;">Total Documentos</td>
                            <td style="padding:6px 0; text-align:right; font-weight:700; color:#1976d2; font-size:16px;">{resumen.get('documentos_total', 0)}</td>
                        </tr>
                        <tr>
                            <td style="padding:6px 0; color:#555;">Total Chunks Indexados</td>
                            <td style="padding:6px 0; text-align:right; font-weight:700; color:#2196F3; font-size:16px;">{resumen.get('chunks_total', 0)}</td>
                        </tr>
                    </table>
                </div>
                
            </div>
            
            <!-- Footer -->
            <div style="background:#fafafa; padding:16px; text-align:center; border-top:1px solid #eee;">
                <small style="color:#999; font-size:12px;">DataCorpus Pipeline</small>
            </div>
            
        </div>
    </body>
    </html>
    """
    
    msg.set_content("DataCorpus Pipeline completado exitosamente.")
    msg.add_alternative(html_template, subtype="html")
    
    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as smtp:
            smtp.login("cristian.pavez.medina@gmail.com", "nxstohbvicjwthbl")
            smtp.send_message(msg)
        print("✅ Correo enviado exitosamente")
    except Exception as e:
        print(f"❌ Error enviando correo: {e}")

def obtener_fecha_hora():
    from datetime import datetime
    return datetime.now().strftime("%d/%m/%Y %H:%M")


if __name__ == "__main__":
    from datetime import datetime
    
    # Marcar inicio de ejecución para filtrado preciso
    timestamp_inicio = datetime.now()
    
    print("\n" + "="*70)
    print("🚀 DATACORPUS PIPELINE - INICIO".center(70))
    print("="*70 + "\n")
    
    # 0. Sincronizar índice FAISS con BD
    rebuild_faiss_automatico()
    
    # 1. Cargar modelos una sola vez (OPTIMIZACIÓN)
    print("\n" + "─"*70)
    print("📦 CARGANDO MODELOS DE IA".center(70))
    print("─"*70)
    
    from query_shield import QueryShield
    from data_shield import DataShield
    
    print("   ⚙️  Cargando QueryShield (query deduplication)...")
    qshield = QueryShield(DB_CONFIG)
    print("   ✅ QueryShield listo")
    
    print("   ⚙️  Cargando DataShield (content deduplication)...")
    dshield = DataShield(DB_CONFIG, init_from_db=True)
    print("   ✅ DataShield listo")
    
    print("   🎯 Modelos cargados y reutilizables en todo el pipeline\n")
    
    # 2. Generar queries
    print("\n" + "─"*70)
    print("📝 GENERACIÓN DE QUERIES".center(70))
    print("─"*70 + "\n")
    
    from generar_queries import main as main_queries
    main_queries(qshield_externo=qshield)
    
    # 3. Validar JSONL
    if not validar_jsonl():
        print("\n❌ ABORTANDO: No hay queries válidas para procesar\n")
        sys.exit(1)
    
    # 4. Scraper
    print("\n" + "─"*70)
    print("🌐 SCRAPING Y VALIDACIÓN DE CONTENIDO".center(70))
    print("─"*70 + "\n")
    
    from scrapear_queries import main as main_scraper
    main_scraper(dshield_externo=dshield, qshield_externo=qshield)
    
    # 5. Notificación por correo
    print("\n" + "─"*70)
    print("📧 ENVIANDO REPORTE".center(70))
    print("─"*70 + "\n")
    enviar_correo(timestamp_inicio)
    
    print("\n" + "="*70)
    print("✅ PIPELINE COMPLETADO EXITOSAMENTE".center(70))
    print("="*70 + "\n")
