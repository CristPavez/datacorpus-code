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
import unicodedata


# TEMAS VALIDOS EXACTOS
TEMAS_VALIDOS = {
    "Medicina", "Legal / Derecho", "Finanzas", "Tecnología", "Educación / Académico",
    "Empresarial / Business", "Ciencia (General)", "Periodismo / Noticias",
    "Literatura / Humanidades", "Gaming / Entretenimiento", "E-commerce / Retail",
    "Gobierno / Política", "Ingeniería", "Arquitectura", "Marketing / Publicidad",
    "Recursos Humanos", "Contabilidad / Auditoría", "Bienes Raíces",
    "Turismo / Hospitalidad", "Agricultura", "Medio Ambiente", "Psicología",
    "Educación Física / Deportes", "Arte / Diseño", "Música", "Cine / Audiovisual",
    "Gastronomía / Culinaria", "Automoción", "Aviación", "Logística / Supply Chain"
}


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
    bloque_match: Optional[dict] = None
    bloques_detalle: List[dict] = None  # Nuevo: detalle de todos los bloques


class DataShield:
    IDENTICAL_SCORE = 1.0
    AGENT_ZONE_LOW = 0.90
    CHUNK_SIZE = 500
    MIN_PARAGRAPH_WORDS = 15  # Solo combinar parrafos muy pequenos (titulos, fragmentos)
    MAX_PARAGRAPH_WORDS = 1000  # Maximo de palabras antes de dividir un parrafo

    def __init__(
        self,
        db_config: dict,
        faiss_path: str = "faiss_index_bge3.bin",
        init_from_db: bool = True,
    ):
        self.db_config = db_config
        self.faiss_path = Path(faiss_path)
        self.model = SentenceTransformer("BAAI/bge-m3")
        self.faiss_index = self._load_faiss()
        self.block_mapping = []

        if init_from_db and self.faiss_index.ntotal == 0:
            self._load_from_db_batch()

    def _get_conn(self):
        conn = psycopg2.connect(**self.db_config)
        register_vector(conn)
        return conn

    @staticmethod
    def normalizar_texto(texto: str) -> str:
        """
        Normaliza texto para evitar problemas de encoding y caracteres especiales.
        - Maneja diferentes encodings (UTF-8, Latin-1, Windows-1252)
        - Corrige strings corruptos por encoding incorrecto
        - Convierte a UTF-8 limpio
        - Normaliza espacios en blanco
        - Preserva la estructura del texto (saltos de linea)
        """
        if not texto:
            return ""
        
        # Paso 1: Si es bytes, intentar decodificar
        if isinstance(texto, bytes):
            # Intentar diferentes encodings comunes
            for encoding in ['utf-8', 'latin-1', 'windows-1252', 'iso-8859-1', 'cp1252']:
                try:
                    texto = texto.decode(encoding)
                    break
                except (UnicodeDecodeError, AttributeError):
                    continue
        
        # Paso 2: Safe decode para strings corruptos (decode/encode cycle)
        if isinstance(texto, str):
            try:
                # Intentar detectar y corregir encoding incorrecto
                # Si el string tiene caracteres raros, puede ser UTF-8 interpretado como Latin-1
                texto_bytes = texto.encode('latin-1', errors='ignore')
                try:
                    # Intentar decodificar como UTF-8
                    texto = texto_bytes.decode('utf-8')
                except UnicodeDecodeError:
                    # Si falla, usar latin-1 con reemplazo
                    texto = texto_bytes.decode('latin-1', errors='replace')
            except (UnicodeDecodeError, UnicodeEncodeError):
                # Si todo falla, limpiar con ignore
                texto = texto.encode('utf-8', errors='ignore').decode('utf-8', errors='ignore')
        
        # Paso 3: Asegurar UTF-8 valido final
        try:
            texto = texto.encode('utf-8', errors='ignore').decode('utf-8', errors='ignore')
        except:
            pass
        
        # Paso 4: Normalizar caracteres Unicode (convierte caracteres compuestos a su forma canonica)
        try:
            texto = unicodedata.normalize('NFC', texto)
        except:
            pass
        
        # Paso 5: Limpiar espacios multiples pero preservar estructura
        lines = texto.split('\n')
        lines = [' '.join(line.split()) for line in lines]
        texto = '\n'.join(lines)
        
        # Paso 6: Eliminar espacios al inicio y final
        texto = texto.strip()
        
        return texto

    def _load_faiss(self):
        try:
            index = faiss.read_index(str(self.faiss_path))
            with open(self.faiss_path.with_suffix(".mapping"), "rb") as f:
                self.block_mapping = pickle.load(f)
            return index
        except:
            d = 1024
            index = faiss.IndexHNSWFlat(d, 32)
            self.block_mapping = []
            return index

    def _save_faiss(self):
        faiss.write_index(self.faiss_index, str(self.faiss_path))
        with open(self.faiss_path.with_suffix(".mapping"), "wb") as f:
            pickle.dump(self.block_mapping, f)

    def _load_from_db_batch(self):
        conn = self._get_conn()
        cur = conn.cursor()

        cur.execute("SELECT COUNT(*) FROM chunks")
        total_bloques = cur.fetchone()[0]

        if total_bloques == 0:
            cur.close()
            conn.close()
            return

        offset = 0
        BATCH_SIZE = 10000
        while True:
            cur.execute(
                """
                SELECT id, indice, embedding 
                FROM chunks 
                ORDER BY id, indice 
                LIMIT %s OFFSET %s
                """,
                (BATCH_SIZE, offset),
            )

            rows = cur.fetchall()
            if not rows:
                break

            embeddings_batch = []
            for id, indice, emb_raw in rows:
                emb = np.array(emb_raw, dtype=np.float32)
                faiss.normalize_L2(emb.reshape(1, -1))
                embeddings_batch.append(emb.flatten())
                self.block_mapping.append((id, indice))

            self.faiss_index.add(np.stack(embeddings_batch).astype("float32"))
            offset += len(rows)

        cur.close()
        conn.close()
        self._save_faiss()

    def _split_into_blocks(self, texto: str) -> List[str]:
        """
        Segmentacion semantica por parrafos.
        Cada parrafo es tratado como una unidad logica independiente.
        Solo combina parrafos muy pequenos o divide parrafos extremadamente largos.
        """
        # Separar por doble salto de linea (parrafos)
        paragraphs = [p.strip() for p in texto.split("\n\n") if p.strip()]
        
        if not paragraphs:
            return [texto] if texto.strip() else []
        
        bloques = []
        buffer_small_paragraphs = []
        buffer_words = 0
        
        for paragraph in paragraphs:
            words = paragraph.split()
            word_count = len(words)
            
            # Caso 1: Parrafo extremadamente largo - dividir (muy raro)
            if word_count > self.MAX_PARAGRAPH_WORDS:
                # Primero, vaciar el buffer de parrafos pequenos
                if buffer_small_paragraphs:
                    bloques.append(" ".join(buffer_small_paragraphs))
                    buffer_small_paragraphs = []
                    buffer_words = 0
                
                # Dividir el parrafo largo en chunks
                for i in range(0, word_count, self.CHUNK_SIZE):
                    chunk = " ".join(words[i : i + self.CHUNK_SIZE])
                    bloques.append(chunk)
            
            # Caso 2: Parrafo muy pequeno - considerar combinar
            elif word_count < self.MIN_PARAGRAPH_WORDS:
                # Si agregar este parrafo al buffer supera CHUNK_SIZE, vaciar buffer primero
                if buffer_words > 0 and buffer_words + word_count > self.CHUNK_SIZE:
                    bloques.append(" ".join(buffer_small_paragraphs))
                    buffer_small_paragraphs = [paragraph]
                    buffer_words = word_count
                else:
                    buffer_small_paragraphs.append(paragraph)
                    buffer_words += word_count
            
            # Caso 3: Parrafo de tamano normal - mantener como bloque independiente
            else:
                # Primero vaciar el buffer
                if buffer_small_paragraphs:
                    bloques.append(" ".join(buffer_small_paragraphs))
                    buffer_small_paragraphs = []
                    buffer_words = 0
                
                # Agregar este parrafo como bloque independiente
                bloques.append(paragraph)
        
        # Vaciar buffer final
        if buffer_small_paragraphs:
            bloques.append(" ".join(buffer_small_paragraphs))
        
        return bloques if bloques else [texto]

    def _get_bloque_texto(self, uuid: str, indice: int) -> str:
        conn = self._get_conn()
        cur = conn.cursor()
        cur.execute(
            """
            SELECT texto FROM chunks 
            WHERE documento_id = %s AND indice = %s
            """,
            (uuid, indice),
        )
        row = cur.fetchone()
        cur.close()
        conn.close()
        return row[0] if row else ""

    def agregar(self, uuid: str, texto: str, tema: Optional[str] = None) -> list:
        """
        Agrega documento y sus chunks al índice FAISS y PostgreSQL.
        
        Returns:
            Lista de tuplas (indice, chunk_id) con los IDs de chunks insertados
        """
        # Validar tema
        if tema and tema not in TEMAS_VALIDOS:
            raise ValueError(f"Tema inválido: '{tema}'. Use: {', '.join(TEMAS_VALIDOS)}")
        
        # Normalizar texto
        texto = self.normalizar_texto(texto)
        
        bloques = self._split_into_blocks(texto)

        conn = self._get_conn()
        cur = conn.cursor()

        # INSERT con tema
        cur.execute(
            """
            INSERT INTO documents (id, texto, tema) 
            VALUES (%s, %s, %s) 
            ON CONFLICT (id) DO UPDATE 
            SET texto = EXCLUDED.texto,
                tema = EXCLUDED.tema
            """,
            (uuid, texto, tema),
        )

        embeddings_batch = []
        chunk_ids = []  # Lista de (indice, chunk_id)
        
        for idx, bloque in enumerate(bloques):
            emb = self.model.encode(bloque)
            faiss.normalize_L2(emb.reshape(1, -1))

            # Usar RETURNING id para obtener el chunk_id insertado
            cur.execute(
                """
                INSERT INTO chunks (documento_id, indice, texto, embedding)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT (documento_id, indice) DO UPDATE
                SET texto = EXCLUDED.texto, embedding = EXCLUDED.embedding
                RETURNING id
                """,
                (uuid, idx, bloque, emb.tolist()),
            )
            
            chunk_id = cur.fetchone()[0]
            chunk_ids.append((idx, chunk_id))

            embeddings_batch.append(emb.flatten())
            self.block_mapping.append((uuid, idx))

        conn.commit()
        cur.close()
        conn.close()

        if embeddings_batch:
            self.faiss_index.add(np.stack(embeddings_batch).astype("float32"))
            self._save_faiss()

        return chunk_ids

    def agregar_solo_chunks(self, uuid: str, texto: str, tema: Optional[str] = None) -> list:
        """
        Agrega SOLO los chunks a PostgreSQL para trazabilidad.
        NO inserta en documents ni en FAISS.
        Útil para documentos rechazados/duplicados donde se quiere mantener registro.
        
        Returns:
            Lista de tuplas (indice, chunk_id) con los IDs de chunks insertados
        """
        # Validar tema
        if tema and tema not in TEMAS_VALIDOS:
            raise ValueError(f"Tema inválido: '{tema}'. Use: {', '.join(TEMAS_VALIDOS)}")
        
        # Normalizar texto
        texto = self.normalizar_texto(texto)
        
        bloques = self._split_into_blocks(texto)

        conn = self._get_conn()
        cur = conn.cursor()

        chunk_ids = []  # Lista de (indice, chunk_id)
        
        for idx, bloque in enumerate(bloques):
            # Generar embedding (necesario para la tabla chunks)
            emb = self.model.encode(bloque)
            faiss.normalize_L2(emb.reshape(1, -1))

            # Insertar chunk en PostgreSQL
            cur.execute(
                """
                INSERT INTO chunks (documento_id, indice, texto, embedding)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT (documento_id, indice) DO UPDATE
                SET texto = EXCLUDED.texto, embedding = EXCLUDED.embedding
                RETURNING id
                """,
                (uuid, idx, bloque, emb.tolist()),
            )
            
            chunk_id = cur.fetchone()[0]
            chunk_ids.append((idx, chunk_id))

        conn.commit()
        cur.close()
        conn.close()

        # NO agregar a FAISS
        # NO agregar a block_mapping
        # NO insertar en documents

        return chunk_ids

    def validar(self, uuid: str, texto: str) -> ResultadoValidacion:
        # Normalizar texto para evitar problemas de encoding
        texto = self.normalizar_texto(texto)
        
        bloques_nuevos = self._split_into_blocks(texto)

        if self.faiss_index.ntotal == 0:
            bloque_info = {
                "bloque_nuevo": 0,
                "bloque_historico": 0,
                "similitud": 0.0,
                "bloque_nuevo_texto": bloques_nuevos[0] if bloques_nuevos else "",
                "bloque_historico_texto": "SIN HISTORICO"
            }
            return ResultadoValidacion(
                Estado.NUEVA, 0.0, None, None, "FAISS vacio - sin comparacion",
                bloque_match=bloque_info,
                bloques_detalle=[]
            )

        # Lista para almacenar el detalle de TODOS los bloques
        bloques_detalle = []
        max_similitud = 0.0
        best_match = None

        # Analizar TODOS los bloques (no detener al encontrar duplicado)
        for idx_nuevo, bloque_nuevo in enumerate(bloques_nuevos):
            emb_nuevo = self.model.encode(bloque_nuevo)
            faiss.normalize_L2(emb_nuevo.reshape(1, -1))
            k = min(10, self.faiss_index.ntotal)

            D, I = self.faiss_index.search(
                emb_nuevo.reshape(1, -1).astype("float32"), k
            )

            # Buscar la mejor coincidencia para este bloque específico
            max_sim_bloque = 0.0
            mejor_match_bloque = None

            for dist, idx_faiss in zip(D[0], I[0]):
                # FAISS devuelve -1 si no hay match válido → filtrar
                if idx_faiss < 0 or idx_faiss >= len(self.block_mapping):
                    continue
                    
                similitud = 1 - (dist**2 / 2)
                if similitud > max_sim_bloque:
                    max_sim_bloque = similitud
                    uuid_historico, indice = self.block_mapping[idx_faiss]
                    bloque_historico_texto = self._get_bloque_texto(uuid_historico, indice)
                    
                    mejor_match_bloque = {
                        "uuid": uuid_historico,
                        "bloque_historico": indice,
                        "similitud": similitud,
                        "bloque_historico_texto": bloque_historico_texto,
                    }

            # Determinar el estado de este bloque espec�fico
            if max_sim_bloque >= self.IDENTICAL_SCORE:
                estado_bloque = "DUPLICADO"
            elif max_sim_bloque >= self.AGENT_ZONE_LOW:
                estado_bloque = "AGENTE"
            else:
                estado_bloque = "NUEVO"

            # Agregar informacion detallada de este bloque
            detalle_bloque = {
                "indice": idx_nuevo,
                "bloque_texto": bloque_nuevo[:200] + "..." if len(bloque_nuevo) > 200 else bloque_nuevo,
                "estado": estado_bloque,
                "similitud": max_sim_bloque,
                "uuid_historico": mejor_match_bloque["uuid"] if mejor_match_bloque else None,
                "bloque_historico_idx": mejor_match_bloque["bloque_historico"] if mejor_match_bloque else None,
                "bloque_historico_texto": (mejor_match_bloque["bloque_historico_texto"][:200] + "..." 
                                          if mejor_match_bloque and len(mejor_match_bloque["bloque_historico_texto"]) > 200 
                                          else mejor_match_bloque["bloque_historico_texto"] if mejor_match_bloque else "SIN MATCH")
            }
            bloques_detalle.append(detalle_bloque)

            # Actualizar el m�ximo global
            if max_sim_bloque > max_similitud:
                max_similitud = max_sim_bloque
                best_match = {
                    "uuid": mejor_match_bloque["uuid"] if mejor_match_bloque else None,
                    "bloque_historico": mejor_match_bloque["bloque_historico"] if mejor_match_bloque else 0,
                    "bloque_nuevo": idx_nuevo,
                    "similitud": max_sim_bloque,
                    "bloque_nuevo_texto": bloque_nuevo,
                    "bloque_historico_texto": mejor_match_bloque["bloque_historico_texto"] if mejor_match_bloque else "SIN MATCH",
                }

        # SIEMPRE hay bloque_match
        if best_match is None:
            best_match = {
                "bloque_nuevo": 0,
                "bloque_historico": 0,
                "similitud": 0.0,
                "bloque_nuevo_texto": bloques_nuevos[0] if bloques_nuevos else "",
                "bloque_historico_texto": "SIN CANDIDATOS",
            }

        # Contar estadísticas de TODOS los bloques analizados
        duplicados = sum(1 for b in bloques_detalle if b["estado"] == "DUPLICADO")
        agentes = sum(1 for b in bloques_detalle if b["estado"] == "AGENTE")
        nuevos = sum(1 for b in bloques_detalle if b["estado"] == "NUEVO")

        # LÓGICA CORRECTA: Prioridad basada en el peor caso de TODOS los chunks
        # 1. Si hay 1+ chunk DUPLICADO → rechazar todo el documento
        if duplicados > 0:
            uuid_top = best_match["uuid"]
            texto_top = self._get_texto(uuid_top)
            
            return ResultadoValidacion(
                estado=Estado.DUPLICADO,
                score=max_similitud,
                uuid_historico=uuid_top,
                texto_historico=texto_top,
                razon=f"RECHAZADO: {duplicados} chunk(s) duplicado(s) - No se puede guardar documento con contenido duplicado",
                bloque_match=best_match,
                bloques_detalle=bloques_detalle
            )
        
        # 2. Si no hay duplicados pero hay 1+ chunk AGENTE → requiere decisión LLM
        elif agentes > 0:
            uuid_top = best_match["uuid"]
            texto_top = self._get_texto(uuid_top)
            
            return ResultadoValidacion(
                estado=Estado.AGENTE,
                score=max_similitud,
                uuid_historico=uuid_top,
                texto_historico=texto_top,
                razon=f"REQUIERE LLM: {agentes} chunk(s) similares - LLM decidirá si es suficientemente diferente",
                bloque_match=best_match,
                bloques_detalle=bloques_detalle
            )
        
        # 3. Solo si TODOS los chunks son NUEVOS → aceptar documento
        else:
            return ResultadoValidacion(
                estado=Estado.NUEVA,
                score=max_similitud,
                uuid_historico=None,
                texto_historico=None,
                razon=f"ACEPTADO: Todos los {nuevos} chunks son nuevos (similitud máxima: {max_similitud:.3f})",
                bloque_match=best_match,
                bloques_detalle=bloques_detalle
            )

    def _get_texto(self, uuid: str) -> Optional[str]:
        conn = self._get_conn()
        cur = conn.cursor()
        cur.execute("SELECT texto FROM documents WHERE id = %s", (uuid,))
        row = cur.fetchone()
        cur.close()
        conn.close()
        return row[0] if row else None

    def stats(self):
        conn = self._get_conn()
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM documents")
        total_textos = cur.fetchone()[0]
        cur.execute("SELECT COUNT(*) FROM chunks")
        total_bloques = cur.fetchone()[0]
        cur.close()
        conn.close()

        return {
            "faiss_ntotal": self.faiss_index.ntotal,
            "block_mapping_len": len(self.block_mapping),
            "total_textos": total_textos,
            "total_bloques": total_bloques,
            "faiss_path": str(self.faiss_path),
            "promedio_bloques_por_texto": total_bloques / total_textos if total_textos > 0 else 0,
        }

    def liberar_modelo(self):
        """
        Libera la memoria del modelo de embeddings BGE-M3.
        Útil cuando se necesita cargar otro modelo grande (ej: modelo de queries).
        """
        import gc
        if hasattr(self, 'model') and self.model is not None:
            del self.model
            self.model = None
            gc.collect()  # Forzar garbage collection
            
    def restaurar_modelo(self):
        """
        Restaura el modelo de embeddings si fue liberado.
        """
        if self.model is None:
            self.model = SentenceTransformer("BAAI/bge-m3")
