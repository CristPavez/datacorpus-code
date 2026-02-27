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
