#!/usr/bin/env python3
import faiss, numpy as np, psycopg, pickle
from pgvector.psycopg import register_vector
from sentence_transformers import SentenceTransformer
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional, List
import nltk

nltk.download('punkt_tab', quiet=True)
from nltk.tokenize import sent_tokenize

class Estado(Enum):
    NUEVA     = "NUEVA"
    DUPLICADO = "DUPLICADO"
    AGENTE    = "AGENTE"

@dataclass
class ResultadoValidacion:
    estado: Estado
    score: float
    uuid_historico: Optional[str] = None
    razon: str = ""
    bloque_match: Optional[dict] = None
    bloques_detalle: List[dict] = field(default_factory=list)

class DataShield:
    IDENTICAL_SCORE = 1.0
    AGENT_ZONE_LOW  = 0.90

    def __init__(self, db_config: dict, faiss_path="faiss_index_bge3.bin", init_from_db=True):
        self.db_config   = db_config
        self.faiss_path  = Path(faiss_path)
        self.mapping     = []
        self.model       = SentenceTransformer("BAAI/bge-m3")
        self.faiss_index = self._load_faiss()
        if init_from_db and self.faiss_index.ntotal == 0:
            self._load_from_db()

    def _get_conn(self):
        conn = psycopg.connect(**self.db_config)
        register_vector(conn)
        return conn

    # ── FAISS ─────────────────────────────────────────────────────
    def _load_faiss(self):
        try:
            idx = faiss.read_index(str(self.faiss_path))
            with open(self.faiss_path.with_suffix(".mapping"), "rb") as f:
                self.mapping = pickle.load(f)
            return idx
        except:
            self.mapping = []
            return faiss.IndexHNSWFlat(1024, 32)

    def _save_faiss(self):
        faiss.write_index(self.faiss_index, str(self.faiss_path))
        with open(self.faiss_path.with_suffix(".mapping"), "wb") as f:
            pickle.dump(self.mapping, f)

    def _load_from_db(self):
        conn = self._get_conn(); cur = conn.cursor()
        cur.execute("SELECT uuid, chunk_numero, chunk_embedding FROM documents_logs ORDER BY id LIMIT 50000")
        rows = cur.fetchall()
        cur.close(); conn.close()
        if not rows: return

        embeddings = []
        for doc_uuid, chunk_num, emb_raw in rows:
            emb = np.array(emb_raw, dtype=np.float32)
            faiss.normalize_L2(emb.reshape(1, -1))
            embeddings.append(emb.flatten())
            self.mapping.append((str(doc_uuid), chunk_num))

        self.faiss_index.add(np.stack(embeddings).astype("float32"))
        self._save_faiss()

    # ── Split ─────────────────────────────────────────────────────
    def _split_text(self, texto: str) -> List[str]:
        oraciones = sent_tokenize(texto)
        chunks, chunk = [], []
        for sent in oraciones:
            chunk.append(sent)
            if len(chunk) >= 5:
                chunks.append(' '.join(chunk)); chunk = []
        if chunk: chunks.append(' '.join(chunk))
        return chunks or [texto]

    # ── Embed helper ──────────────────────────────────────────────
    def _embed(self, texto: str) -> np.ndarray:
        emb = self.model.encode(texto).astype(np.float32)
        faiss.normalize_L2(emb.reshape(1, -1))
        return emb

    def _buscar_similar(self, emb: np.ndarray):
        if self.faiss_index.ntotal == 0: return None, None, 0.0
        D, I = self.faiss_index.search(emb.reshape(1, -1), 1)
        idx = I[0][0]
        score = 1 - (D[0][0] ** 2 / 2) if idx >= 0 else 0.0
        if idx >= 0 and idx < len(self.mapping):
            return *self.mapping[idx], score
        return None, None, 0.0

    # ── Validar ───────────────────────────────────────────────────
    def validar(self, doc_uuid: str, texto: str) -> ResultadoValidacion:
        blocks = self._split_text(texto)
        if self.faiss_index.ntotal == 0:
            return ResultadoValidacion(Estado.NUEVA, 0.0, razon="Sin datos en FAISS")

        bloques_detalle, max_score, best_match = [], 0.0, None

        for i, block in enumerate(blocks):
            emb = self._embed(block)
            uuid_hist, chunk_num_hist, score = self._buscar_similar(emb)
            estado_chunk = ("DUPLICADO" if score >= self.IDENTICAL_SCORE else
                            "AGENTE"    if score >= self.AGENT_ZONE_LOW  else "NUEVO")
            bloques_detalle.append({"indice": i, "estado": estado_chunk, "score": score,
                                    "uuid_historico": uuid_hist, "chunk_num_historico": chunk_num_hist})
            if score > max_score:
                max_score = score
                best_match = {"bloque_nuevo": i, "bloque_nuevo_texto": block,
                              "uuid_historico": uuid_hist, "bloque_historico": chunk_num_hist}

        duplicados = sum(1 for b in bloques_detalle if b["estado"] == "DUPLICADO")
        agentes    = sum(1 for b in bloques_detalle if b["estado"] == "AGENTE")

        if duplicados > 0:
            return ResultadoValidacion(Estado.DUPLICADO, max_score, best_match["uuid_historico"],
                                       f"{duplicados} chunk(s) duplicados", best_match, bloques_detalle)
        elif agentes > 0:
            return ResultadoValidacion(Estado.AGENTE, max_score, best_match["uuid_historico"],
                                       f"{agentes} chunk(s) en zona AGENTE", best_match, bloques_detalle)
        return ResultadoValidacion(Estado.NUEVA, max_score, None, "Contenido nuevo", best_match, bloques_detalle)

    # ── Agregar (aprobado) ────────────────────────────────────────
    def agregar(self, doc_uuid: str, texto: str, tema: str,
                url: str = None, decision: str = "APROBADO") -> List[tuple]:
        blocks = self._split_text(texto)
        conn = self._get_conn(); cur = conn.cursor()

        cur.execute("INSERT INTO documents (uuid_queries, dato) VALUES (%s, %s) ON CONFLICT DO NOTHING",
                    (doc_uuid, texto))

        chunk_ids, embeddings = [], []
        for i, block in enumerate(blocks):
            emb = self._embed(block)
            uuid_sim, chunk_num_sim, score_sim = self._buscar_similar(emb)

            cur.execute("""
                INSERT INTO documents_logs
                    (uuid, chunk_numero, chunk, url_busqueda, estado, score_similar,
                     decision, uuid_chunk_similar, chunk_num_similar, chunk_embedding)
                VALUES (%s,%s,%s,%s,'procesado',%s,%s,%s,%s,%s) RETURNING id
            """, (doc_uuid, i, block, url, score_sim, decision, uuid_sim, chunk_num_sim, emb.tolist()))

            chunk_ids.append((i, cur.fetchone()[0]))
            embeddings.append(emb.flatten())
            self.mapping.append((doc_uuid, i))

        conn.commit(); cur.close(); conn.close()
        self.faiss_index.add(np.stack(embeddings).astype("float32"))
        self._save_faiss()
        return chunk_ids

    # ── Solo logs (rechazado) ─────────────────────────────────────
    def agregar_solo_logs(self, doc_uuid: str, texto: str, url: str = None,
                          decision: str = "RECHAZADO", bloques_detalle: list = None) -> List[tuple]:
        blocks = self._split_text(texto)
        conn = self._get_conn(); cur = conn.cursor()

        chunk_ids = []
        for i, block in enumerate(blocks):
            emb  = self._embed(block)
            score = bloques_detalle[i].get("score", 0.0) if bloques_detalle and i < len(bloques_detalle) else 0.0

            cur.execute("""
                INSERT INTO documents_logs
                    (uuid, chunk_numero, chunk, url_busqueda, estado, score_similar, decision, chunk_embedding)
                VALUES (%s,%s,%s,%s,'rechazado',%s,%s,%s) RETURNING id
            """, (doc_uuid, i, block, url, score, decision, emb.tolist()))

            chunk_ids.append((i, cur.fetchone()[0]))

        conn.commit(); cur.close(); conn.close()
        return chunk_ids

 