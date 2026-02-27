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

# ─────────────────────────────────────────────
# TIPOS
# ─────────────────────────────────────────────
class Estado(Enum):
    NUEVA = "NUEVA"
    DUPLICADO = "DUPLICADO"
    AGENTE = "AGENTE"


@dataclass
class ResultadoValidacion:
    estado: Estado
    score: float
    uuid_parecida: Optional[str] = None
    texto_historico: Optional[str] = None


# ─────────────────────────────────────────────
# QUERYSHIELD
# ─────────────────────────────────────────────
class QueryShield:
    AGENT_ZONE_LOW = 0.90

    def __init__(self, db_config: dict, faiss_path: str = "faiss_index.bin"):
        self.db_config = db_config
        self.faiss_path = Path(faiss_path)
        self.model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
        self.uuid_mapping = []
        self.faiss_index = self._load_or_create_faiss()
        self._sync_pg_faiss()

    # ── Conexión ──────────────────────────────
    def _get_conn(self):
        conn = psycopg.connect(**self.db_config)
        register_vector(conn)
        return conn

    # ── FAISS: cargar o crear ─────────────────
    def _load_or_create_faiss(self):
        try:
            index = faiss.read_index(str(self.faiss_path))
            with open(self.faiss_path.with_suffix('.mapping'), 'rb') as f:
                self.uuid_mapping = pickle.load(f)
            return index
        except:
            return faiss.IndexHNSWFlat(384, 32)

    # ── FAISS: guardar ────────────────────────
    def _save_faiss(self):
        faiss.write_index(self.faiss_index, str(self.faiss_path))
        with open(self.faiss_path.with_suffix('.mapping'), 'wb') as f:
            pickle.dump(self.uuid_mapping, f)

    # ── FAISS: sincronizar desde BD ───────────
    def _sync_pg_faiss(self):
        """Reconstruye el índice FAISS completo desde los embeddings en BD."""
        conn = self._get_conn()
        cur = conn.cursor()
        cur.execute("SELECT uuid, embedding FROM queries ORDER BY fecha_creacion")

        embeddings, self.uuid_mapping = [], []
        for uuid, emb_raw in cur.fetchall():
            emb = np.array(emb_raw, dtype=np.float32).flatten()
            faiss.normalize_L2(emb.reshape(1, -1))
            embeddings.append(emb)
            self.uuid_mapping.append(str(uuid))

        cur.close()
        conn.close()

        if embeddings:
            self.faiss_index.reset()
            self.faiss_index.add(np.stack(embeddings).astype('float32'))
        self._save_faiss()

    # ── Helpers BD ───────────────────────────
    def _insertar_query(self, cur, uuid: str, pregunta: str, tema: str, embedding):
        """Inserta o actualiza query en BD."""
        cur.execute("""
            INSERT INTO queries (uuid, pregunta, tema, embedding)
            VALUES (%s, %s, %s, %s)
            ON CONFLICT (uuid) DO UPDATE
            SET pregunta = EXCLUDED.pregunta,
                tema = EXCLUDED.tema,
                embedding = EXCLUDED.embedding
        """, (uuid, pregunta, tema, embedding.tolist()))

    def _obtener_texto(self, uuid: str) -> Optional[str]:
        """Obtiene pregunta de BD por UUID."""
        conn = self._get_conn()
        cur = conn.cursor()
        cur.execute("SELECT pregunta FROM queries WHERE uuid = %s", (uuid,))
        row = cur.fetchone()
        cur.close()
        conn.close()
        return row[0] if row else None

    def _uuid_existe(self, cur, uuid: str) -> bool:
        cur.execute("SELECT 1 FROM queries WHERE uuid = %s", (uuid,))
        return cur.fetchone() is not None

    # ── MÉTODO PÚBLICO: agregar ───────────────
    def agregar(self, uuid: str, pregunta: str, tema: str):
        """Inserta query aprobada en BD + sincroniza FAISS."""
        embedding = self.model.encode(pregunta)
        faiss.normalize_L2(embedding.reshape(1, -1))

        conn = self._get_conn()
        cur = conn.cursor()
        self._insertar_query(cur, uuid, pregunta, tema, embedding)
        conn.commit()
        cur.close()
        conn.close()

        self._sync_pg_faiss()

    # ── MÉTODO PÚBLICO: log_validacion ────────
    def log_validacion(self, uuid: str, pregunta: str, uuid_similar: Optional[str],
                       estado: str, score: float, decision: Optional[str] = None,
                       tema: Optional[str] = None):
        """
        Registra resultado en queries_logs.
        Si uuid no existe en queries (rechazadas/regeneradas), lo inserta primero (FK).
        Solo agregar() sincroniza FAISS → logs de rechazadas NO entran al índice.
        """
        conn = self._get_conn()
        cur = conn.cursor()

        # uuid_similar debe existir en queries; si no, usa el propio uuid
        if uuid_similar:
            cur.execute("SELECT 1 FROM queries WHERE uuid = %s", (uuid_similar,))
            uuid_similar_final = uuid_similar if cur.fetchone() else uuid
        else:
            uuid_similar_final = uuid

        cur.execute("""
            INSERT INTO queries_logs (uuid, pregunta, uuid_similar, estado, score, decision)
            VALUES (%s, %s, %s, %s, %s, %s)
        """, (uuid, pregunta, uuid_similar_final, estado, score, decision))

        conn.commit()
        cur.close()
        conn.close()

    # ── MÉTODO PÚBLICO: validar ───────────────
    def validar(self, uuid: str, pregunta: str) -> ResultadoValidacion:
        """
        Busca similitud en FAISS (rápido).
        Retorna estado: NUEVA / DUPLICADO / AGENTE + score + uuid_parecida + texto_historico.
        """
        if self.faiss_index.ntotal == 0 or not self.uuid_mapping:
            return ResultadoValidacion(Estado.NUEVA, 0.0)

        emb = self.model.encode(pregunta)
        faiss.normalize_L2(emb.reshape(1, -1))

        D, I = self.faiss_index.search(emb.reshape(1, -1).astype('float32'), min(5, self.faiss_index.ntotal))

        top_idx = I[0][0]
        if top_idx == -1 or top_idx >= len(self.uuid_mapping):
            return ResultadoValidacion(Estado.NUEVA, 0.0)

        # Distancia L2 → similitud coseno (vectores normalizados)
        max_sim = max(0.0, min(1.0, 1 - (D[0][0] ** 2 / 2)))
        uuid_parecida = self.uuid_mapping[top_idx]
        texto_historico = self._obtener_texto(uuid_parecida)

        if max_sim >= 1.0:
            return ResultadoValidacion(Estado.DUPLICADO, max_sim, uuid_parecida, texto_historico)
        elif max_sim >= self.AGENT_ZONE_LOW:
            return ResultadoValidacion(Estado.AGENTE, max_sim, uuid_parecida, texto_historico)
        else:
            return ResultadoValidacion(Estado.NUEVA, max_sim, uuid_parecida, texto_historico)
