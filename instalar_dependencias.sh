#!/bin/bash

# Script de instalación de dependencias para datacorpus-scraper

echo "🔧 Instalando dependencias..."

pip install --quiet psycopg2-binary pgvector sentence-transformers faiss-cpu

# Para scraping
pip install --quiet requests beautifulsoup4 lxml

# Para PDFs
pip install --quiet pypdf2

# Para HTML cleaning
pip install --quiet html2text trafilatura

echo "✅ Dependencias instaladas exitosamente"
