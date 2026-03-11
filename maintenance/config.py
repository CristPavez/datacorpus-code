"""Configuración compartida para todos los scripts de mantenimiento."""

DB_CONFIG = {
    "dbname":   "datacorpus_bd",
    "user":     "datacorpus",
    "password": "730822",
    "host":     "localhost",
    "port":     5433
}

DECISION_VALIDAS_QUERIES_LOGS   = {"APROBADO", "RECHAZADO", "REGENERADO", "RECHAZADO_DOC"}
DECISION_VALIDAS_DOCUMENTS_LOGS = {"APROBADO", "APROBADO_LLM", "RECHAZADO", "RECHAZADO_LLM"}
ESTADO_VALIDOS_QUERIES_LOGS     = {"NUEVA", "DUPLICADO", "AGENTE"}
ESTADO_VALIDOS_DOCUMENTS_LOGS   = {"procesado", "rechazado", "omitido"}

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

# Colores consola
ROJO     = "\033[91m"
VERDE    = "\033[92m"
AMARILLO = "\033[93m"
AZUL     = "\033[94m"
GRIS     = "\033[90m"
RESET    = "\033[0m"
BOLD     = "\033[1m"


def ok(texto):    print(f"  {VERDE}✅ {texto}{RESET}")
def warn(texto):  print(f"  {AMARILLO}⚠️  {texto}{RESET}")
def error(texto): print(f"  {ROJO}❌ {texto}{RESET}")
def info(texto):  print(f"     {GRIS}{texto}{RESET}")
def titulo(texto):
    print(f"\n{BOLD}{AZUL}{'─'*65}{RESET}")
    print(f"{BOLD}{AZUL}  {texto}{RESET}")
    print(f"{BOLD}{AZUL}{'─'*65}{RESET}")


def es_pregunta_valida(texto: str) -> bool:
    p = texto.strip()
    p_lower = p.lower()
    if "?" not in p:                          return False
    if len(p) > 500:                          return False
    if any(f in p_lower for f in FRASES_RUIDO): return False
    palabras = p_lower.split()
    if len(palabras) > 5:
        hits = sum(1 for w in palabras if w in PALABRAS_INGLES)
        if hits / len(palabras) > 0.25:       return False
    return True
