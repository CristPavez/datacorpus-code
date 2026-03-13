#!/usr/bin/env python3
"""
DataShield — Deduplicación de chunks de documentos con FAISS.

Estados en documents_logs:
  APROBADO  → chunk nuevo (score < 0.90) o aprobado por LLM
  DUPLICADA → score coseno == 1.0 (idéntico)
  SIMILAR   → LLM confirmó que es similar a uno existente
  OMITIDA   → no procesado porque ya se alcanzó el 50% de rechazo

Lógica 50%:
  Si (DUPLICADA + SIMILAR) >= 50% del total de chunks → URL descartada.
  Los chunks que quedaron sin procesar se marcan como OMITIDA.
"""

import os, warnings, logging
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'
os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = '1'
warnings.filterwarnings('ignore')
logging.getLogger('transformers').setLevel(logging.ERROR)
logging.getLogger('sentence_transformers').setLevel(logging.ERROR)

import faiss
import numpy as np
import psycopg
import pickle
import nltk
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional
from pgvector.psycopg import register_vector
from sentence_transformers import SentenceTransformer
from nltk.tokenize import sent_tokenize
from .config import (MODEL_DOCS, MODEL_MAX_LENGTH,
                    FAISS_DOCS_DIM, FAISS_DOCS_HNSW_M, FAISS_DOCS_PATH)

nltk.download('punkt_tab', quiet=True)


# ─────────────────────────────────────────────
# TIPOS
# ─────────────────────────────────────────────
@dataclass
class ResultadoChunk:
    chunk_numero:         int
    chunk_text:           str
    estado:               str              # APROBADO | DUPLICADA | SIMILAR | OMITIDA
    score:                float
    uuid_chunk_similar:   Optional[str]   # UUID del documento similar
    chunk_numero_similar: Optional[int]
    embedding:            Optional[np.ndarray]


@dataclass
class ResultadoValidacion:
    aprobada: bool                       # True → URL vale, guardar
    chunks:   list = field(default_factory=list)   # list[ResultadoChunk]


# ─────────────────────────────────────────────
# DATASHIELD
# ─────────────────────────────────────────────
class DataShield:
    DUPLICADA_THRESHOLD = 1.0
    AGENT_ZONE_LOW      = 0.90
    RECHAZO_UMBRAL      = 0.50           # 50% de chunks rechazados → URL descartada

    def __init__(self, db_config: dict, faiss_path=None):
        self.db_config  = db_config
        self.faiss_path = Path(faiss_path) if faiss_path else FAISS_DOCS_PATH
        # mapping: [(uuid, chunk_numero), ...]
        self.mapping: list[tuple[str, int]] = []
        self.model   = SentenceTransformer(MODEL_DOCS, device='cpu')
        self.model.tokenizer.model_max_length = MODEL_MAX_LENGTH
        self.faiss_index = self._load_faiss()
        self._reconciliar_con_bd()          # Issue 7: sincroniza gaps BD↔FAISS

    # ── Conexión ──────────────────────────────────────────────────
    def _get_conn(self):
        conn = psycopg.connect(**self.db_config)
        register_vector(conn)
        return conn

    # ── FAISS ─────────────────────────────────────────────────────
    def _load_faiss(self):
        try:
            idx = faiss.read_index(str(self.faiss_path))
            with open(self.faiss_path.with_suffix('.mapping'), 'rb') as f:
                self.mapping = pickle.load(f)
            return idx
        except FileNotFoundError:
            # Primera ejecución — índice vacío, _reconciliar_con_bd() lo poblará si hay datos en BD
            self.mapping = []
            return faiss.IndexHNSWFlat(FAISS_DOCS_DIM, FAISS_DOCS_HNSW_M)
        except Exception as e:
            # Issue 8: corrupción detectada — reconstruir desde BD en vez de arrancar vacío
            print(f"   ❌ FAISS de documentos corrupto ({e}) — reconstruyendo desde BD...")
            return self._rebuild_desde_bd()

    def _rebuild_desde_bd(self):
        """Reconstruye el índice FAISS de documentos completo desde los embeddings en BD."""
        conn = self._get_conn()
        cur  = conn.cursor()
        cur.execute("""
            SELECT dl.uuid, dl.chunk_numero, dl.embedding
            FROM documents_logs dl
            INNER JOIN documents d ON d.uuid = dl.uuid
            WHERE dl.embedding IS NOT NULL
              AND dl.estado = 'APROBADO'
            ORDER BY dl.id
        """)
        rows = cur.fetchall()
        cur.close(); conn.close()

        index = faiss.IndexHNSWFlat(FAISS_DOCS_DIM, FAISS_DOCS_HNSW_M)
        self.mapping = []

        if not rows:
            print("   ℹ️  Sin chunks aprobados en BD — índice vacío creado")
            return index

        embeddings = []
        for uuid_val, chunk_num, emb_raw in rows:
            emb = np.array(emb_raw, dtype=np.float32).flatten()
            faiss.normalize_L2(emb.reshape(1, -1))
            embeddings.append(emb)
            self.mapping.append((str(uuid_val), int(chunk_num)))

        index.add(np.stack(embeddings).astype('float32'))
        faiss.write_index(index, str(self.faiss_path))
        with open(self.faiss_path.with_suffix('.mapping'), 'wb') as f:
            pickle.dump(self.mapping, f)
        print(f"   ✅ Índice reconstruido: {index.ntotal} chunks")
        return index

    def _reconciliar_con_bd(self):
        """Issue 7: detecta chunks en BD que falten en FAISS por crash y los agrega."""
        conn = self._get_conn()
        cur  = conn.cursor()
        cur.execute("""
            SELECT dl.uuid, dl.chunk_numero, dl.embedding
            FROM documents_logs dl
            INNER JOIN documents d ON d.uuid = dl.uuid
            WHERE dl.embedding IS NOT NULL
              AND dl.estado = 'APROBADO'
            ORDER BY dl.id
        """)
        rows = cur.fetchall()
        cur.close(); conn.close()

        claves_faiss = set(self.mapping)
        faltantes    = [
            (str(u), int(n), e) for u, n, e in rows
            if (str(u), int(n)) not in claves_faiss
        ]

        if not faltantes:
            return

        print(f"   ⚠️  {len(faltantes)} chunk(s) en BD sin entrada en FAISS — reconciliando...")
        for uuid_val, chunk_num, emb_raw in faltantes:
            emb = np.array(emb_raw, dtype=np.float32).flatten()
            faiss.normalize_L2(emb.reshape(1, -1))
            self.faiss_index.add(emb.reshape(1, -1).astype('float32'))
            self.mapping.append((uuid_val, chunk_num))

        self._save_faiss()
        print(f"   ✅ Reconciliación completada: {len(faltantes)} chunk(s) agregado(s)")

    def _save_faiss(self):
        faiss.write_index(self.faiss_index, str(self.faiss_path))
        with open(self.faiss_path.with_suffix('.mapping'), 'wb') as f:
            pickle.dump(self.mapping, f)

    # ── Texto → chunks ────────────────────────────────────────────
    def split_text(self, texto: str) -> list[str]:
        oraciones = sent_tokenize(texto)
        chunks, chunk = [], []
        for sent in oraciones:
            chunk.append(sent)
            if len(chunk) >= 5:
                chunks.append(' '.join(chunk))
                chunk = []
        if chunk:
            chunks.append(' '.join(chunk))
        return chunks or [texto]

    # ── Embed + normalizar ────────────────────────────────────────
    def embed(self, texto: str) -> np.ndarray:
        emb = self.model.encode(texto).astype(np.float32)
        faiss.normalize_L2(emb.reshape(1, -1))
        return emb

    # ── Buscar similar en FAISS ───────────────────────────────────
    def buscar_similar(self, emb: np.ndarray) -> tuple[Optional[str], Optional[int], float]:
        """Devuelve (uuid_chunk_similar, chunk_numero_similar, score)."""
        if self.faiss_index.ntotal == 0:
            return None, None, 0.0
        D, I = self.faiss_index.search(emb.reshape(1, -1).astype('float32'), 1)
        idx = I[0][0]
        if idx < 0 or idx >= len(self.mapping):
            return None, None, 0.0
        score = float(max(0.0, min(1.0, 1 - (D[0][0] ** 2 / 2))))
        uuid_sim, chunk_num = self.mapping[idx]
        return uuid_sim, chunk_num, score

    # ── Obtener texto de un chunk histórico ───────────────────────
    def get_chunk_text(self, uuid_sim: str, chunk_num: int) -> str:
        conn = self._get_conn()
        cur  = conn.cursor()
        cur.execute(
            "SELECT chunk_text FROM documents_logs WHERE uuid = %s AND chunk_numero = %s",
            (uuid_sim, chunk_num)
        )
        row = cur.fetchone()
        cur.close(); conn.close()
        return row[0] if row else ""

    # ── MÉTODO PÚBLICO: validar (sin LLM, retorna AGENTE para zona gris) ──
    def validar_chunks(self, chunks: list[str]) -> list[ResultadoChunk]:
        """
        Valida similitud de cada chunk contra FAISS.
        Los chunks en zona gris retornan estado=AGENTE para que el caller decida con LLM.
        No modifica BD ni FAISS.
        """
        resultados = []
        for i, chunk in enumerate(chunks):
            emb = self.embed(chunk)
            id_sim, num_sim, score = self.buscar_similar(emb)

            if score >= self.DUPLICADA_THRESHOLD:
                estado = "DUPLICADA"
            elif score >= self.AGENT_ZONE_LOW:
                estado = "AGENTE"
            else:
                estado = "APROBADO"

            resultados.append(ResultadoChunk(
                chunk_numero=i, chunk_text=chunk, estado=estado,
                score=score, id_chunk_similar=id_sim,
                chunk_numero_similar=num_sim, embedding=emb
            ))
        return resultados

    # ── Helper: prepara valores para INSERT ──────────────────────
    @staticmethod
    def _vals_chunk(c: ResultadoChunk) -> tuple:
        """
        Normaliza score y uuid_chunk_similar antes de insertar en BD:
          - score redondeado a 2 decimales
          - uuid_chunk_similar solo si score >= 0.90, sino NULL
        """
        score_db      = round(float(c.score), 2)
        uuid_sim_db   = c.uuid_chunk_similar if score_db >= DataShield.AGENT_ZONE_LOW else None
        chunk_num_sim = c.chunk_numero_similar if uuid_sim_db is not None else None
        return score_db, uuid_sim_db, chunk_num_sim

    # ── MÉTODO PÚBLICO: agregar (URL aprobada) ────────────────────
    def agregar(self, uuid: str, texto: str, tema: str,
                url: str, chunk_resultados: list[ResultadoChunk]):
        """
        Guarda el documento aprobado en documents + documents_logs.
        Solo los chunks APROBADO entran al índice FAISS.
        """
        conn = self._get_conn()
        cur  = conn.cursor()

        cur.execute(
            "INSERT INTO documents (uuid, texto) VALUES (%s, %s) ON CONFLICT DO NOTHING",
            (uuid, texto)
        )

        embeddings_nuevos = []
        for c in chunk_resultados:
            emb_db                       = None if c.estado == "OMITIDA" else (c.embedding.tolist() if c.embedding is not None else None)
            score_db, uuid_sim, num_sim  = self._vals_chunk(c)

            cur.execute("""
                INSERT INTO documents_logs
                    (uuid, chunk_numero, chunk_text, url, estado, score,
                     uuid_chunk_similar, chunk_numero_similar, embedding)
                VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)
            """, (uuid, c.chunk_numero, c.chunk_text, url, c.estado,
                  score_db, uuid_sim, num_sim, emb_db))

            # Solo chunks APROBADO entran al índice FAISS
            if c.estado == "APROBADO" and c.embedding is not None:
                embeddings_nuevos.append((uuid, c.chunk_numero, c.embedding))

        conn.commit()
        cur.close(); conn.close()

        for doc_uuid, chunk_num, emb in embeddings_nuevos:
            self.faiss_index.add(emb.reshape(1, -1).astype('float32'))
            self.mapping.append((doc_uuid, chunk_num))

        if embeddings_nuevos:
            self._save_faiss()

    # ── MÉTODO PÚBLICO: log_rechazado (URL descartada) ────────────
    def log_rechazado(self, uuid: str, url: str,
                      chunk_resultados: list[ResultadoChunk]):
        """
        Registra en documents_logs los chunks de una URL descartada.
        No modifica FAISS.
        """
        conn = self._get_conn()
        cur  = conn.cursor()

        for c in chunk_resultados:
            emb_db                      = None if c.estado == "OMITIDA" else (c.embedding.tolist() if c.embedding is not None else None)
            score_db, uuid_sim, num_sim = self._vals_chunk(c)

            cur.execute("""
                INSERT INTO documents_logs
                    (uuid, chunk_numero, chunk_text, url, estado, score,
                     uuid_chunk_similar, chunk_numero_similar, embedding)
                VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)
            """, (uuid, c.chunk_numero, c.chunk_text, url, c.estado,
                  score_db, uuid_sim, num_sim, emb_db))

        conn.commit()
        cur.close(); conn.close()
