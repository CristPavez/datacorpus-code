#!/bin/bash
# Script de instalación de dependencias

echo "📦 Instalando dependencias para DataCorpus Query Generator..."

pip3 install --user psycopg2-binary pgvector sentence-transformers faiss-cpu openai

echo ""
echo "✅ Dependencias instaladas"
echo ""
echo "🔍 Verificando instalación..."
python3 -c "import psycopg2; import faiss; import sentence_transformers; from openai import OpenAI; print('✅ Todas las librerías importadas correctamente')"

echo ""
echo "🚀 Todo listo. Ejecuta:"
echo "   python3 generar_queries.py"
