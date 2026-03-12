"""Configuración central del proyecto DataCorpus."""

# ssh -L 5433:localhost:5432 datacorpus@192.168.1.87 -N -f

from pathlib import Path

# ── BASE DIR ──────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent.parent

# ── BASE DE DATOS ─────────────────────────────────────────────────
DB_CONFIG = {
    "dbname":   "datacorpus_bd_test",
    "user":     "datacorpus",
    "password": "730822",
    "host":     "100.116.167.76",
}

# ── API KEYS ──────────────────────────────────────────────────────
TOGETHER_API_KEY = "tgp_v1_35Ewiz4u1GT4huetCkSeITDZ9eyw-6tNcuYlSn5X7lY"
BRAVE_API_KEY    = "BSA9TKRCDdLxynAieJYPeqx6A1BPcLH"

# ── EMAIL (notificaciones) ────────────────────────────────────────
EMAIL_FROM     = "cristian.pavez.medina@gmail.com"
EMAIL_TO       = "cristian.pavez.medina@gmail.com"
EMAIL_PASSWORD = "nxstohbvicjwthbl"
SMTP_HOST      = "smtp.gmail.com"
SMTP_PORT      = 465

# ── MODELOS DE EMBEDDINGS ─────────────────────────────────────────
MODEL_QUERY      = "paraphrase-multilingual-MiniLM-L12-v2"  # dim=384, para queries
MODEL_DOCS       = "BAAI/bge-m3"                             # dim=1024, para documentos
MODEL_MAX_LENGTH = 512

# ── FAISS — DOCUMENTOS (DataShield, BGE-M3) ───────────────────────
FAISS_DOCS_PATH    = BASE_DIR / "data" / "faiss" / "faiss_docs.bin"
FAISS_DOCS_MAPPING = BASE_DIR / "data" / "faiss" / "faiss_docs.mapping"
FAISS_DOCS_DIM     = 1024
FAISS_DOCS_HNSW_M  = 32

# ── FAISS — QUERIES (QueryShield, MiniLM) ────────────────────────
FAISS_QUERY_PATH    = BASE_DIR / "data" / "faiss" / "faiss_queries.bin"
FAISS_QUERY_MAPPING = BASE_DIR / "data" / "faiss" / "faiss_queries.mapping"
FAISS_QUERY_DIM     = 384
FAISS_QUERY_HNSW_M  = 32

# ── ARCHIVOS DE DATOS ─────────────────────────────────────────────
QUERIES_FILE = BASE_DIR / "data" / "queries_validadas.jsonl"

# ── TEMAS VÁLIDOS ─────────────────────────────────────────────────
TEMAS_VALIDOS = [
    "Medicina", "Legal / Derecho", "Finanzas", "Tecnología", "Educación / Académico",
    "Empresarial / Business", "Ciencia (General)", "Periodismo / Noticias",
    "Literatura / Humanidades", "Gaming / Entretenimiento", "E-commerce / Retail",
    "Gobierno / Política", "Ingeniería", "Arquitectura", "Marketing / Publicidad",
    "Recursos Humanos", "Contabilidad / Auditoría", "Bienes Raíces",
    "Turismo / Hospitalidad", "Agricultura", "Medio Ambiente", "Psicología",
    "Educación Física / Deportes", "Arte / Diseño", "Música", "Cine / Audiovisual",
    "Gastronomía / Culinaria", "Automoción", "Aviación", "Logística / Supply Chain",
]

# ── ESTADOS VÁLIDOS ───────────────────────────────────────────────
ESTADOS_QUERIES    = {"APROBADO", "SIN_RESULTADOS", "DUPLICADA", "SIMILAR", "OMITIDA"}
ESTADOS_DOCUMENTOS = {"APROBADO", "DUPLICADA", "SIMILAR", "OMITIDA"}

# ── FILTROS DE RUIDO ──────────────────────────────────────────────
FRASES_RUIDO = [
    "i'm sorry", "i am sorry", "lo siento",
    "could you clarify", "could you please", "please clarify",
    "i'm not sure", "i am not sure", "let me know",
    "entiendo que", "quieres información", "solicitas", "me pides",
    "if you can provide", "si puedes proporcionar",
    "i'll do my best", "i will do my best",
    "as an ai", "como ia", "como modelo de",
    "here are", "aquí tienes", "a continuación",
    "seem to be asking", "seems like you",
    "necesitas", "pides",
]

PALABRAS_INGLES = {
    "the", "and", "for", "that", "with", "this", "are", "you",
    "what", "how", "why", "when", "where", "which", "can", "but",
    "about", "your", "have", "from", "they", "will", "more", "also",
    "some", "just", "into", "than", "been", "would", "could", "should",
}

def es_pregunta_valida(texto: str) -> bool:
    p = texto.strip()
    p_lower = p.lower()
    if "?" not in p:                              return False
    if len(p) > 500:                              return False
    if any(f in p_lower for f in FRASES_RUIDO):  return False
    palabras = p_lower.split()
    if len(palabras) > 5:
        hits = sum(1 for w in palabras if w in PALABRAS_INGLES)
        if hits / len(palabras) > 0.25:           return False
    return True

# ── COLORES CONSOLA ───────────────────────────────────────────────
ROJO     = "\033[91m"
VERDE    = "\033[92m"
AMARILLO = "\033[93m"
AZUL     = "\033[94m"
GRIS     = "\033[90m"
RESET    = "\033[0m"
BOLD     = "\033[1m"

def ok(texto):     print(f"  {VERDE}✅ {texto}{RESET}")
def warn(texto):   print(f"  {AMARILLO}⚠️  {texto}{RESET}")
def error(texto):  print(f"  {ROJO}❌ {texto}{RESET}")
def info(texto):   print(f"     {GRIS}{texto}{RESET}")
def titulo(texto):
    print(f"\n{BOLD}{AZUL}{'─'*65}{RESET}")
    print(f"{BOLD}{AZUL}  {texto}{RESET}")
    print(f"{BOLD}{AZUL}{'─'*65}{RESET}")
