#!/usr/bin/env python3
"""
QueryShield — Deduplicación de queries con FAISS + LLM.

Estados en queries_logs:
  APROBADO      → query nueva, guardada en BD e índice
  DUPLICADA     → score coseno == 1.0 (idéntica)
  SIMILAR       → LLM confirmó que es similar a una existente
  SIN_RESULTADOS→ scraping no encontró contenido (lo registra scrapear_queries)
  OMITIDA       → DataShield rechazó todos los documentos (lo registra scrapear_queries)
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
from pathlib import Path
from dataclasses import dataclass
from enum import Enum
from typing import Optional
from pgvector.psycopg import register_vector
from sentence_transformers import SentenceTransformer
from .config import (MODEL_QUERY, MODEL_MAX_LENGTH,
                    FAISS_QUERY_DIM, FAISS_QUERY_HNSW_M, FAISS_QUERY_PATH)


# ─────────────────────────────────────────────
# TIPOS
# ─────────────────────────────────────────────
class Estado(Enum):
    NUEVA     = "NUEVA"
    DUPLICADA = "DUPLICADA"   # score == 1.0
    AGENTE    = "AGENTE"      # 0.90 <= score < 1.0 → pasa por LLM


@dataclass
class ResultadoValidacion:
    estado: Estado
    score: float
    uuid_parecida:   Optional[str] = None
    texto_historico: Optional[str] = None


# ─────────────────────────────────────────────
# QUERYSHIELD
# ─────────────────────────────────────────────
class QueryShield:
    DUPLICADA_THRESHOLD = 1.0
    AGENT_ZONE_LOW      = 0.90

    def __init__(self, db_config: dict, faiss_path=None):
        self.db_config  = db_config
        self.faiss_path = Path(faiss_path) if faiss_path else FAISS_QUERY_PATH
        self.model      = SentenceTransformer(MODEL_QUERY, device='cpu')
        self.model.tokenizer.model_max_length = MODEL_MAX_LENGTH
        self.uuid_mapping: list[str] = []
        self.faiss_index = self._load_or_create_faiss()
        self._reconciliar_con_bd()          # Issue 7: sincroniza gaps BD↔FAISS

    # ── Conexión ──────────────────────────────────────────────────
    def _get_conn(self):
        conn = psycopg.connect(**self.db_config)
        register_vector(conn)
        return conn

    # ── FAISS ─────────────────────────────────────────────────────
    def _load_or_create_faiss(self):
        try:
            index = faiss.read_index(str(self.faiss_path))
            with open(self.faiss_path.with_suffix('.mapping'), 'rb') as f:
                self.uuid_mapping = pickle.load(f)
            return index
        except FileNotFoundError:
            # Primera ejecución — índice vacío, _reconciliar_con_bd() lo poblará si hay datos en BD
            return faiss.IndexHNSWFlat(FAISS_QUERY_DIM, FAISS_QUERY_HNSW_M)
        except Exception as e:
            # Issue 8: corrupción detectada — reconstruir desde BD en vez de arrancar vacío
            print(f"   ❌ FAISS de queries corrupto ({e}) — reconstruyendo desde BD...")
            return self._rebuild_desde_bd()

    def _rebuild_desde_bd(self):
        """Reconstruye el índice FAISS de queries completo desde los embeddings en BD."""
        conn = self._get_conn()
        cur  = conn.cursor()
        cur.execute("SELECT uuid, embedding FROM queries ORDER BY fecha_creacion")
        rows = cur.fetchall()
        cur.close(); conn.close()

        index = faiss.IndexHNSWFlat(FAISS_QUERY_DIM, FAISS_QUERY_HNSW_M)
        self.uuid_mapping = []

        if not rows:
            print("   ℹ️  Sin queries en BD — índice vacío creado")
            return index

        embeddings = []
        for uuid_val, emb_raw in rows:
            emb = np.array(emb_raw, dtype=np.float32).flatten()
            faiss.normalize_L2(emb.reshape(1, -1))
            embeddings.append(emb)
            self.uuid_mapping.append(str(uuid_val))

        index.add(np.stack(embeddings).astype('float32'))
        faiss.write_index(index, str(self.faiss_path))
        with open(self.faiss_path.with_suffix('.mapping'), 'wb') as f:
            pickle.dump(self.uuid_mapping, f)
        print(f"   ✅ Índice reconstruido: {index.ntotal} queries")
        return index

    def _reconciliar_con_bd(self):
        """Issue 7: detecta UUIDs en BD que falten en FAISS por crash y los agrega."""
        conn = self._get_conn()
        cur  = conn.cursor()
        cur.execute("SELECT uuid, embedding FROM queries ORDER BY fecha_creacion")
        rows = cur.fetchall()
        cur.close(); conn.close()

        uuids_faiss = set(self.uuid_mapping)
        faltantes   = [(str(u), e) for u, e in rows if str(u) not in uuids_faiss]

        if not faltantes:
            return

        print(f"   ⚠️  {len(faltantes)} query(s) en BD sin entrada en FAISS — reconciliando...")
        for uuid_val, emb_raw in faltantes:
            emb = np.array(emb_raw, dtype=np.float32).flatten()
            faiss.normalize_L2(emb.reshape(1, -1))
            self.faiss_index.add(emb.reshape(1, -1).astype('float32'))
            self.uuid_mapping.append(uuid_val)

        self._save_faiss()
        print(f"   ✅ Reconciliación completada: {len(faltantes)} query(s) agregada(s)")

    def _save_faiss(self):
        faiss.write_index(self.faiss_index, str(self.faiss_path))
        with open(self.faiss_path.with_suffix('.mapping'), 'wb') as f:
            pickle.dump(self.uuid_mapping, f)

    # ── Helpers BD ────────────────────────────────────────────────
    def _obtener_info_query(self, uuid_str: str) -> Optional[str]:
        """Devuelve pregunta de la tabla queries por UUID."""
        conn = self._get_conn()
        cur  = conn.cursor()
        cur.execute("SELECT pregunta FROM queries WHERE uuid = %s", (uuid_str,))
        row = cur.fetchone()
        cur.close(); conn.close()
        return row[0] if row else None

    # ── MÉTODO PÚBLICO: agregar ────────────────────────────────────
    def agregar(self, uuid: str, pregunta: str, tema: str):
        """Inserta query aprobada en BD + actualiza FAISS incrementalmente."""
        embedding = self.model.encode(pregunta).astype(np.float32)
        faiss.normalize_L2(embedding.reshape(1, -1))

        conn = self._get_conn()
        cur  = conn.cursor()
        cur.execute("""
            INSERT INTO queries (uuid, pregunta, tema, embedding)
            VALUES (%s, %s, %s, %s)
            ON CONFLICT (uuid) DO UPDATE
            SET pregunta  = EXCLUDED.pregunta,
                tema      = EXCLUDED.tema,
                embedding = EXCLUDED.embedding
        """, (uuid, pregunta, tema, embedding.tolist()))
        conn.commit()
        cur.close(); conn.close()

        if uuid not in self.uuid_mapping:
            self.faiss_index.add(embedding.reshape(1, -1).astype('float32'))
            self.uuid_mapping.append(uuid)
            self._save_faiss()

    # ── MÉTODO PÚBLICO: log ────────────────────────────────────────
    def log(self, uuid: str, pregunta: str, score: float,
            uuid_similar: Optional[str], estado: str, tema: Optional[str] = None):
        """
        Registra resultado en queries_logs.
        estado ∈ {APROBADO, SIN_RESULTADOS, DUPLICADA, SIMILAR, OMITIDA}
        tema se guarda para que el flujo reparador pueda reformular con el mismo tema.
        """
        # uuid_similar solo se registra si el score supera la zona de similitud
        score_db           = round(float(score), 2)
        uuid_similar_final = uuid_similar if score_db >= self.AGENT_ZONE_LOW else None

        conn = self._get_conn()
        cur  = conn.cursor()
        cur.execute("""
            INSERT INTO queries_logs (uuid, pregunta, tema, score, uuid_similar, estado)
            VALUES (%s, %s, %s, %s, %s, %s)
        """, (uuid, pregunta, tema, score_db, uuid_similar_final, estado))
        conn.commit()
        cur.close(); conn.close()

    # ── MÉTODO PÚBLICO: validar ────────────────────────────────────
    def validar(self, uuid: str, pregunta: str) -> ResultadoValidacion:
        """
        Busca similitud en FAISS.
        Retorna estado NUEVA / DUPLICADA / AGENTE + score + info del parecido.
        """
        if self.faiss_index.ntotal == 0 or not self.uuid_mapping:
            return ResultadoValidacion(Estado.NUEVA, 0.0)

        emb = self.model.encode(pregunta).astype(np.float32)
        faiss.normalize_L2(emb.reshape(1, -1))

        D, I = self.faiss_index.search(
            emb.reshape(1, -1), min(5, self.faiss_index.ntotal)
        )

        top_idx = I[0][0]
        if top_idx == -1 or top_idx >= len(self.uuid_mapping):
            return ResultadoValidacion(Estado.NUEVA, 0.0)

        # Distancia L2 → similitud coseno (vectores normalizados)
        score = float(max(0.0, min(1.0, 1 - (D[0][0] ** 2 / 2))))
        uuid_parecida = self.uuid_mapping[top_idx]
        texto_hist    = self._obtener_info_query(uuid_parecida)

        if score >= self.DUPLICADA_THRESHOLD:
            return ResultadoValidacion(Estado.DUPLICADA, score, uuid_parecida, texto_hist)
        elif score >= self.AGENT_ZONE_LOW:
            return ResultadoValidacion(Estado.AGENTE, score, uuid_parecida, texto_hist)
        else:
            return ResultadoValidacion(Estado.NUEVA, score, uuid_parecida, texto_hist)
