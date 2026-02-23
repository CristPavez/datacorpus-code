import faiss
import numpy as np
import psycopg2
from pgvector.psycopg2 import register_vector
from sentence_transformers import SentenceTransformer
from dataclasses import dataclass
from enum import Enum
from typing import Optional, List
import pickle
from pathlib import Path

class Estado(Enum):
    NUEVA = "NUEVA"
    DUPLICADO = "DUPLICADO"
    AGENTE = "AGENTE"

@dataclass
class ResultadoValidacion:
    estado: Estado
    score: float
    uuid_historico: Optional[str]
    texto_historico: Optional[str]
    razon: str
    top_matches: List[tuple] = None

class QueryShield:
    IDENTICAL_SCORE = 1.0
    AGENT_ZONE_LOW = 0.90
    
    def __init__(self, db_config: dict, faiss_path: str = "faiss_index.bin"):
        self.db_config = db_config
        self.faiss_path = Path(faiss_path)
        self.model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
        self.faiss_index = self._load_faiss()
        self.uuid_mapping = []
        self._sync_pg_faiss()
    
    def _get_conn(self):
        conn = psycopg2.connect(**self.db_config)
        register_vector(conn)
        return conn
    
    def _load_faiss(self):
        try:
            index = faiss.read_index(str(self.faiss_path))
            with open(self.faiss_path.with_suffix('.mapping'), 'rb') as f:
                self.uuid_mapping = pickle.load(f)
            return index
        except:
            d = 384
            index = faiss.IndexHNSWFlat(d, 32)
            self.uuid_mapping = []
            return index
    
    def _save_faiss(self):
        faiss.write_index(self.faiss_index, str(self.faiss_path))
        with open(self.faiss_path.with_suffix('.mapping'), 'wb') as f:
            pickle.dump(self.uuid_mapping, f)
    
    def _sync_pg_faiss(self):
        conn = self._get_conn()
        cur = conn.cursor()
        cur.execute("SELECT id, embedding FROM queries ORDER BY created_at")
        
        embeddings = []
        self.uuid_mapping = []
        
        for row in cur.fetchall():
            uuid, emb_raw = row
            emb = np.array(emb_raw, dtype=np.float32)
            faiss.normalize_L2(emb.reshape(1, -1))
            embeddings.append(emb.flatten())
            self.uuid_mapping.append(uuid)
        
        cur.close()
        conn.close()
        
        if embeddings:
            self.faiss_index.reset()
            self.faiss_index.add(np.stack(embeddings).astype('float32'))
            self._save_faiss()
    
    def agregar(self, uuid: str, texto: str):
        embedding = self.model.encode(texto)
        faiss.normalize_L2(embedding.reshape(1, -1))
        
        conn = self._get_conn()
        cur = conn.cursor()
        cur.execute("""
            INSERT INTO preguntas (id, texto, embedding) 
            VALUES (%s, %s, %s) 
            ON CONFLICT (id) DO UPDATE 
            SET texto = EXCLUDED.texto, embedding = EXCLUDED.embedding
        """, (uuid, texto, embedding.tolist()))
        conn.commit()
        cur.close()
        conn.close()
        
        self._sync_pg_faiss()
    
    def validar(self, uuid: str, texto: str) -> ResultadoValidacion:
        emb_nuevo = self.model.encode(texto)
        faiss.normalize_L2(emb_nuevo.reshape(1, -1))
        
        if self.faiss_index.ntotal == 0:
            return ResultadoValidacion(Estado.NUEVA, 0.0, None, None, "FAISS vacio")
        
        k = min(10, self.faiss_index.ntotal)
        D, I = self.faiss_index.search(
            emb_nuevo.reshape(1, -1).astype('float32'), 
            k
        )
        
        # Convertir distancias L2 a similitud coseno (vectores normalizados)
        # similarity = 1 - (distance^2 / 2)
        similarities = [1 - (d**2 / 2) for d in D[0]]
        top_matches = [(self.uuid_mapping[i], float(sim)) for i, sim in zip(I[0], similarities)]
        max_sim_pair = max(top_matches, key=lambda x: x[1])
        uuid_top, max_sim = max_sim_pair
        texto_top = self._get_texto(uuid_top)
        
        if max_sim >= self.IDENTICAL_SCORE:
            return ResultadoValidacion(
                estado=Estado.DUPLICADO, score=max_sim, 
                uuid_historico=uuid_top, texto_historico=texto_top,
                razon="FAISS identico", top_matches=top_matches
            )
        elif max_sim >= self.AGENT_ZONE_LOW:
            return ResultadoValidacion(
                estado=Estado.AGENTE, score=max_sim,
                uuid_historico=uuid_top, texto_historico=texto_top,
                razon="FAISS agente", top_matches=top_matches
            )
        else:
            return ResultadoValidacion(
                estado=Estado.NUEVA, score=max_sim,
                uuid_historico=uuid_top, texto_historico=None,
                razon=f"FAISS {max_sim:.3f}", top_matches=top_matches
            )
    
    def _get_texto(self, uuid: str) -> Optional[str]:
        conn = self._get_conn()
        cur = conn.cursor()
        cur.execute("SELECT texto FROM queries WHERE id = %s", (uuid,))
        row = cur.fetchone()
        cur.close()
        conn.close()
        return row[0] if row else None
    
    def stats(self):
        return {
            'faiss_ntotal': self.faiss_index.ntotal,
            'uuid_mapping_len': len(self.uuid_mapping),
            'faiss_path': str(self.faiss_path)
        }

