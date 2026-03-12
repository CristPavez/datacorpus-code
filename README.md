# DataCorpus

Constructor automático de corpus en español de alta calidad. Genera preguntas técnicas con LLM, busca contenido en la web, lo extrae y deduplica semánticamente usando FAISS y pgvector.

---

## Tabla de contenidos

- [Descripción general](#descripción-general)
- [Stack tecnológico](#stack-tecnológico)
- [Estructura del proyecto](#estructura-del-proyecto)
- [Esquema de base de datos](#esquema-de-base-de-datos)
- [Temas válidos](#temas-válidos)
- [Configuración e instalación](#configuración-e-instalación)
- [Flujo principal del pipeline](#flujo-principal-del-pipeline)
- [Flujo de reparación](#flujo-de-reparación)
- [Umbrales de similitud](#umbrales-de-similitud)
- [API Backend](#api-backend)
  - [Iniciar el servidor](#iniciar-el-servidor)
  - [Endpoints del sistema](#endpoints-del-sistema)
  - [Endpoints del pipeline](#endpoints-del-pipeline)
  - [Endpoints del reparador](#endpoints-del-reparador)
  - [WebSocket de logs](#websocket-de-logs)
  - [Endpoints del dashboard](#endpoints-del-dashboard)
  - [Endpoints de mantenimiento](#endpoints-de-mantenimiento)
  - [Sistema de logs (log_manager.py)](#sistema-de-logs-log_managerpy)
  - [Runner de flujos (runner.py)](#runner-de-flujos-runnerpy)
  - [Servicio systemd (Linux)](#servicio-systemd-linux)
- [Vistas SQL del dashboard](#vistas-sql-del-dashboard)
- [Decisiones de diseño](#decisiones-de-diseño)

---

## Descripción general

DataCorpus automatiza la construcción de un corpus documental en español organizado por tema. El proceso completo es:

1. Selecciona los temas con menos cobertura en la base de datos.
2. Genera preguntas técnicas mediante DeepSeek V3 (Together AI).
3. Valida cada pregunta contra el corpus existente usando similitud semántica (QueryShield + FAISS).
4. Busca URLs relevantes con Brave Search API.
5. Extrae el texto de cada URL con trafilatura.
6. Divide el contenido en fragmentos (chunks) de 5 oraciones con NLTK.
7. Valida cada chunk contra el corpus existente (DataShield + FAISS).
8. Guarda los documentos aprobados en PostgreSQL y actualiza los índices FAISS.

---

## Stack tecnológico

| Componente | Detalle |
|---|---|
| Lenguaje | Python 3.11+ |
| Base de datos | PostgreSQL + extensión pgvector |
| Búsqueda de similitud | FAISS con índice HNSW |
| Modelo de queries | `paraphrase-multilingual-MiniLM-L12-v2` (dim=384) |
| Modelo de documentos | `BAAI/bge-m3` (dim=1024) |
| LLM | DeepSeek V3 vía Together AI API |
| Búsqueda web | Brave Search API |
| Scraping | trafilatura |
| Tokenización | NLTK (oraciones, 5 por chunk) |
| Driver PostgreSQL | psycopg v3 |
| Backend API | FastAPI + uvicorn |

---

## Estructura del proyecto

```
datacorpus-code/
├── api/
│   ├── main.py                 # App FastAPI: CORS, WebSocket /ws/logs, /health
│   ├── routers/
│   │   ├── pipeline.py         # POST /pipeline/start|stop, GET /pipeline/status
│   │   ├── reparador.py        # POST /reparador/start|stop, GET /reparador/status
│   │   ├── dashboard.py        # GET /dashboard/* (7 endpoints)
│   │   └── maintenance.py      # POST /maintenance/rebuild-faiss/{modo}
│   └── services/
│       ├── log_manager.py      # Captura de stdout + broadcast a WebSocket
│       └── runner.py           # FlowRunner con parada controlada via threading.Event
├── config.py                   # DB_CONFIG, claves API, nombres de modelos, rutas FAISS, TEMAS_VALIDOS
├── setup.sql                   # DROP + CREATE de las 4 tablas
├── views.sql                   # 7 vistas PostgreSQL para el dashboard
├── data_shield.py              # Clase DataShield - deduplicación de chunks con FAISS
├── query_shield.py             # Clase QueryShield - deduplicación de queries con FAISS
├── generar_queries.py          # Generación de preguntas con LLM + validación
├── scrapear_queries.py         # Brave Search + scraping + procesamiento de chunks
├── flujo_reparador.py          # Flujo de reparación (2 sub-flujos)
├── ejecutar_todo.py            # Orquestador principal del pipeline
├── rebuild_faiss.py            # Reconstrucción de índices FAISS desde la DB
├── queries_validadas.jsonl     # Archivo intermedio entre generación y scraping
└── test_reparador.py           # Script temporal para simulación del flujo reparador
```

---

## Esquema de base de datos

### Tabla `queries`
Almacena las preguntas aprobadas y únicas en el corpus.

```sql
CREATE TABLE queries (
    id              SERIAL PRIMARY KEY,
    uuid            UUID UNIQUE,
    pregunta        TEXT,
    tema            TEXT,
    embedding       vector(384),
    fecha_creacion  TIMESTAMP
);
```

### Tabla `queries_logs`
Log de auditoría del proceso de validación de preguntas.

```sql
CREATE TABLE queries_logs (
    id              SERIAL PRIMARY KEY,
    uuid            UUID,
    pregunta        TEXT,
    tema            TEXT,
    score           FLOAT,
    uuid_similar    UUID,
    estado          VARCHAR(20),   -- APROBADO | DUPLICADA | SIMILAR | SIN_RESULTADOS | OMITIDA
    fecha_creacion  TIMESTAMP
);
```

### Tabla `documents`
Almacena los documentos aprobados tras el scraping.

```sql
CREATE TABLE documents (
    id              SERIAL PRIMARY KEY,
    uuid            UUID REFERENCES queries(uuid),
    texto           TEXT,
    fecha_creacion  TIMESTAMP
);
```

### Tabla `documents_logs`
Log de auditoría por chunk durante el proceso de validación de documentos.

```sql
CREATE TABLE documents_logs (
    id                   SERIAL PRIMARY KEY,
    uuid                 UUID,
    chunk_numero         INT,
    chunk_text           TEXT,
    url                  TEXT,
    estado               VARCHAR(20),   -- APROBADO | DUPLICADA | SIMILAR | OMITIDA
    score                FLOAT,
    uuid_chunk_similar   UUID,
    chunk_numero_similar INT,
    embedding            vector(1024),
    fecha_creacion       TIMESTAMP
);
```

---

## Temas válidos

El corpus cubre 30 dominios temáticos:

| # | Tema | # | Tema |
|---|---|---|---|
| 1 | Medicina | 16 | Recursos Humanos |
| 2 | Legal/Derecho | 17 | Contabilidad/Auditoría |
| 3 | Finanzas | 18 | Bienes Raíces |
| 4 | Tecnología | 19 | Turismo/Hospitalidad |
| 5 | Educación/Académico | 20 | Agricultura |
| 6 | Empresarial/Business | 21 | Medio Ambiente |
| 7 | Ciencia (General) | 22 | Psicología |
| 8 | Periodismo/Noticias | 23 | Educación Física/Deportes |
| 9 | Literatura/Humanidades | 24 | Arte/Diseño |
| 10 | Gaming/Entretenimiento | 25 | Música |
| 11 | E-commerce/Retail | 26 | Cine/Audiovisual |
| 12 | Gobierno/Política | 27 | Gastronomía/Culinaria |
| 13 | Ingeniería | 28 | Automoción |
| 14 | Arquitectura | 29 | Aviación |
| 15 | Marketing/Publicidad | 30 | Logística/Supply Chain |

---

## Configuración e instalación

### 1. Clonar el repositorio

```bash
git clone <url-del-repo>
cd datacorpus-code
```

### 2. Crear entorno virtual e instalar dependencias

```bash
python -m venv venv
source venv/bin/activate        # Linux/macOS
venv\Scripts\activate           # Windows

pip install -r requirements.txt
```

### 3. Configurar `config.py`

Editar el archivo `config.py` con los valores del entorno:

```python
DB_CONFIG = {
    "host": "localhost",
    "dbname": "datacorpus",
    "user": "usuario",
    "password": "contraseña",
}

TOGETHER_API_KEY = "..."
BRAVE_API_KEY    = "..."

FAISS_QUERIES_PATH = "faiss_queries.bin"
FAISS_DOCS_PATH    = "faiss_docs.bin"
```

### 4. Crear las tablas

```bash
psql -U usuario -d datacorpus -f setup.sql
```

### 5. Crear las vistas del dashboard

```bash
psql -U usuario -d datacorpus -f views.sql
```

### 6. Iniciar el servidor API

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

---

## Flujo principal del pipeline

El orquestador `ejecutar_todo.py` ejecuta la función `run_pipeline()` que sigue estos pasos:

### Paso 1 — Carga de modelos

Se instancian una vez `QueryShield` (modelo de queries, dim=384) y `DataShield` (modelo de documentos, dim=1024), cargando los índices FAISS desde disco.

### Paso 2 — Generación de preguntas (`generar_queries.py`)

1. Consulta la DB para identificar los 2 temas con menor cantidad de documentos (balanceo).
2. Llama a DeepSeek V3 para generar N preguntas técnicas en español.
3. Por cada pregunta, `QueryShield` calcula la similitud coseno contra el índice FAISS:

| Score | Clasificación | Acción |
|---|---|---|
| `score == 1.0` | DUPLICADA | Se registra en log y se regenera |
| `0.90 ≤ score < 1.0` | AGENTE | El LLM decide: DIFERENTES o SIMILARES |
| `score < 0.90` | NUEVA | Se guarda en `queries` y en FAISS |

4. Las preguntas aprobadas se escriben en `queries_validadas.jsonl`.

### Paso 3 — Scraping (`scrapear_queries.py`)

Para cada pregunta aprobada:

1. Se buscan 3 URLs con Brave Search API.
2. Se extrae el texto de cada URL con trafilatura.
3. El texto se divide en chunks de 5 oraciones (NLTK).
4. Cada chunk se valida con `DataShield`:

| Score | Clasificación | Acción |
|---|---|---|
| `score == 1.0` | DUPLICADA | Cuenta para el umbral del 50% |
| `0.90 ≤ score < 1.0` | AGENTE | El LLM decide: NUEVO o DUPLICADO |
| `score < 0.90` | APROBADO | Se acepta directamente |

5. Si el 50% o más de los chunks son rechazados, se descarta la URL y se intenta la siguiente.
6. La primera URL con suficientes chunks aprobados: el documento se guarda en `documents` y los embeddings se añaden a FAISS.
7. El estado final de la query se registra en `queries_logs`: `APROBADO`, `SIN_RESULTADOS` u `OMITIDA`.

### Paso 4 — Reporte por email

Al finalizar el ciclo se envía un resumen por correo electrónico.

---

## Flujo de reparación

El archivo `flujo_reparador.py` implementa dos sub-flujos independientes para recuperar queries que fallaron en ciclos anteriores.

### Sub-flujo 1 — Reparación de queries vacías

**Detecta:** queries en la tabla `queries` sin documento asociado Y sin entrada en `queries_logs`. Esto indica que el pipeline se interrumpió entre la validación y el scraping.

**Acción:**
1. Limpia entradas huérfanas en `documents_logs` asociadas al UUID.
2. Re-ejecuta el scraping usando el UUID original de la query.

### Sub-flujo 2 — Re-procesamiento de estados fallidos

| Estado original | UUID en `queries` | Acción |
|---|---|---|
| `DUPLICADA` / `SIMILAR` | No | Genera nueva pregunta → valida con QueryShield (forzando UUID original) → scraping |
| `SIN_RESULTADOS` / `OMITIDA` | Sí | Genera pregunta alternativa solo para la búsqueda Brave → scraping con UUID original |

Ambos sub-flujos **preservan siempre el UUID original** para mantener la trazabilidad completa en los logs.

---

## Umbrales de similitud

Los scores se redondean a 2 decimales antes de comparar con los umbrales, para evitar problemas de precisión de punto flotante.

| Rango de score | Nombre | Decisión |
|---|---|---|
| `score == 1.0` | DUPLICADA | Automática (solo el modelo) |
| `0.90 ≤ score < 1.0` | Zona AGENTE | El LLM toma la decisión final |
| `score < 0.90` | NUEVA / APROBADO | Aprobación automática |

---

## API Backend

El backend expone un servidor FastAPI que permite controlar el pipeline, consultar el dashboard y transmitir logs en tiempo real.

### Iniciar el servidor

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

---

### Endpoints del sistema

#### `GET /health`

Verifica el estado del servidor y de los flujos activos.

**Respuesta:**
```json
{
  "ok": true,
  "pipeline_status": "idle",
  "reparador_status": "idle"
}
```

---

### Endpoints del pipeline

#### `POST /pipeline/start`

Inicia el pipeline principal en un hilo en segundo plano.

- No acepta cuerpo.
- Retorna error si el pipeline o el reparador ya están en ejecución.

#### `POST /pipeline/stop`

Solicita una parada controlada (graceful stop). El pipeline termina el ciclo de la query actual y luego se detiene.

#### `GET /pipeline/status`

Retorna el estado actual del pipeline.

**Respuesta:**
```json
{
  "flow": "pipeline",
  "status": "idle"
}
```

Los valores posibles de `status` son: `idle`, `running`, `stopping`.

---

### Endpoints del reparador

#### `POST /reparador/start`

Inicia el flujo de reparación manualmente en segundo plano.

#### `POST /reparador/stop`

Solicita parada controlada del reparador.

#### `GET /reparador/status`

```json
{
  "flow": "reparador",
  "status": "idle"
}
```

---

### WebSocket de logs

#### `WS /ws/logs`

Transmite en tiempo real los mensajes del flujo activo como objetos JSON. Soporta múltiples clientes simultáneos.

**Formato de cada mensaje:**
```json
{
  "ts": "2026-03-12T15:30:00",
  "level": "SUCCESS",
  "msg": "Query aprobada y guardada correctamente."
}
```

**Niveles de log:**

| Level | Significado |
|---|---|
| `SUCCESS` | Chunk o query aprobado (✅) |
| `ERROR` | Errores de scraping, DB o red (❌) |
| `WARN` | Duplicados u omitidos (⚠️) |
| `SECTION` | Cabeceras de sección del flujo |
| `LOG` | Mensajes informativos regulares |

---

### Endpoints del dashboard

Todos los endpoints del dashboard devuelven datos provenientes de las vistas SQL definidas en `views.sql`.

#### `GET /dashboard/resumen`

Totales globales del corpus: queries aprobadas, documentos, chunks por estado, etc.

#### `GET /dashboard/temas`

Número de queries por tema, ordenado de mayor a menor.

#### `GET /dashboard/estados-queries`

Distribución de estados de queries con porcentajes.

**Ejemplo de respuesta:**
```json
[
  { "estado": "APROBADO", "cantidad": 1240, "porcentaje": 68.5 },
  { "estado": "SIN_RESULTADOS", "cantidad": 310, "porcentaje": 17.1 },
  ...
]
```

#### `GET /dashboard/estados-chunks`

Distribución de estados de chunks con porcentajes.

#### `GET /dashboard/actividad`

Actividad diaria agrupada por estado durante los últimos 90 días. Útil para gráficos de series temporales.

#### `GET /dashboard/tasa-exito`

Tasa de éxito por tema: proporción de queries que tienen al menos un documento aprobado sobre el total de queries de ese tema.

#### `GET /dashboard/ultimas-queries`

Las últimas 50 queries aprobadas, incluyendo el número de chunks asociados.

---

### Endpoints de mantenimiento

#### `POST /maintenance/rebuild-faiss/docs`

Reconstruye el índice FAISS de documentos (`faiss_docs.bin`) leyendo todos los embeddings aprobados desde `documents_logs`.

#### `POST /maintenance/rebuild-faiss/queries`

Reconstruye el índice FAISS de queries (`faiss_queries.bin`) leyendo todos los embeddings desde la tabla `queries`.

#### `POST /maintenance/rebuild-faiss/ambos`

Reconstruye ambos índices en secuencia.

#### `GET /maintenance/rebuild-faiss/status`

```json
{ "running": false }
```

---

### Sistema de logs (`api/services/log_manager.py`)

`LogManager` intercepta `sys.stdout` mediante una clase `_CaptureStream` personalizada que reemplaza el flujo estándar de salida durante la ejecución de un flujo.

**Funcionamiento interno:**

1. Cuando un flujo arranca, `stdout` se redirige al stream de captura.
2. Cada llamada a `print()` es interceptada y procesada.
3. El nivel del mensaje se detecta por la presencia de emojis y palabras clave en el texto.
4. El mensaje se encola en un `asyncio.Queue` individual por cada cliente WebSocket conectado.
5. El puente entre el hilo del flujo (síncrono) y el event loop de asyncio se realiza con `loop.call_soon_threadsafe()`.
6. Al finalizar el flujo, `stdout` se restaura al original.

Este mecanismo permite que los `print()` del código de scraping y generación se transmitan en tiempo real al frontend sin modificar nada en esos módulos.

---

### Runner de flujos (`api/services/runner.py`)

La clase `FlowRunner` gestiona la ejecución de un único flujo a la vez.

**Métodos principales:**

| Método | Descripción |
|---|---|
| `start(fn)` | Lanza `fn` en un hilo daemon, captura stdout, establece status=RUNNING |
| `stop()` | Activa `threading.Event` (stop_event), establece status=STOPPING |

**Control de parada:**
El flujo verifica `stop_event.is_set()` entre cada query o iteración. Cuando detecta la señal, termina el ciclo actual y sale limpiamente. Al completarse el hilo, el status vuelve a `IDLE`.

**Singletons disponibles:**
- `pipeline_runner` — para el flujo principal
- `reparador_runner` — para el flujo de reparación

**Exclusión mutua:** el pipeline y el reparador no pueden ejecutarse simultáneamente. Si uno está activo, el intento de iniciar el otro retorna un error `409 Conflict`.

---

### Servicio systemd (Linux)

Para ejecutar el pipeline de forma automática en un servidor Linux, se puede crear un servicio oneshot que llame al endpoint de arranque:

**`/etc/systemd/system/datacorpus-pipeline.service`:**
```ini
[Unit]
Description=DataCorpus Pipeline
After=network.target

[Service]
Type=oneshot
ExecStart=/usr/bin/curl -s -X POST http://localhost:8000/pipeline/start
RemainAfterExit=no

[Install]
WantedBy=multi-user.target
```

Para ejecutarlo periódicamente, crear un timer asociado (`datacorpus-pipeline.timer`).

**Parar el pipeline desde CLI:**
```bash
curl -X POST http://localhost:8000/pipeline/stop
```

> El servidor uvicorn debe estar levantado previamente. Se recomienda gestionarlo con un servicio systemd separado o con un supervisor de procesos.

---

## Vistas SQL del dashboard

El archivo `views.sql` define 7 vistas que encapsulan las consultas del dashboard:

| Vista | Descripción |
|---|---|
| `v_resumen` | Totales globales por tipo de registro y estado |
| `v_queries_por_tema` | Queries aprobadas agrupadas por tema, ordenadas por cantidad |
| `v_estados_queries` | Distribución de estados de queries con porcentaje sobre el total |
| `v_estados_chunks` | Distribución de estados de chunks con porcentaje sobre el total |
| `v_actividad_diaria` | Actividad diaria por estado en los últimos 90 días |
| `v_tasa_exito_temas` | Tasa de éxito por tema (queries con documento / total queries del tema) |
| `v_ultimas_queries` | Últimas 50 queries aprobadas con su conteo de chunks |

---

## Decisiones de diseño

- **FAISS con HNSW:** el índice HNSW (Hierarchical Navigable Small World) ofrece búsqueda aproximada de vecinos más cercanos con alta velocidad y buen recall, adecuado para búsquedas de similitud a escala.
- **Dos índices FAISS separados:** uno para queries (dim=384) y otro para documentos (dim=1024), usando modelos distintos optimizados para cada caso de uso.
- **Mapeo por UUID:** la sincronización entre FAISS y la base de datos se hace mediante UUIDs (no IDs enteros), lo que permite reconstruir los índices sin depender del orden de inserción.
- **Único punto de escritura de APROBADO:** el log de estado `APROBADO` solo lo escribe `scrapear_queries.py` después de confirmar que el documento fue guardado en la DB, evitando inconsistencias.
- **Umbral del 50% por URL:** si la mitad o más de los chunks de una URL son rechazados, se descarta esa URL y se intenta la siguiente. Esto evita incorporar documentos de baja calidad semántica.
- **Redondeo a 2 decimales en scores:** las comparaciones con los umbrales se hacen sobre el score redondeado para evitar problemas de precisión de punto flotante (e.g., `0.8999999... < 0.90`).
- **Preservación del UUID en el reparador:** ambos sub-flujos de reparación reutilizan el UUID original de la query para mantener la trazabilidad completa en los logs históricos.
- **Limpieza de documentos_logs huérfanos:** en el Sub-flujo 1 del reparador, se eliminan primero las entradas huérfanas de `documents_logs` antes de re-scrapear, para evitar contaminación del log de auditoría.
