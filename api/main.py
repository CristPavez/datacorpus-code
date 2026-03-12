"""
DataCorpus API — FastAPI backend.

Ejecutar:
    uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload

Endpoints:
    POST  /pipeline/start       → inicia pipeline principal
    POST  /pipeline/stop        → detiene pipeline (graceful)
    GET   /pipeline/status      → estado actual
    POST  /reparador/start      → inicia flujo reparador
    POST  /reparador/stop       → detiene reparador (graceful)
    GET   /reparador/status     → estado actual
    WS    /ws/logs              → stream de logs en tiempo real
    GET   /dashboard/*          → métricas y estadísticas
"""

import asyncio
import json
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from .services.log_manager import log_manager
from .services import shields
from .routers import pipeline, reparador, dashboard, maintenance


@asynccontextmanager
async def lifespan(app: FastAPI):
    loop = asyncio.get_event_loop()
    log_manager.set_loop(loop)

    print("\n   Iniciando DataCorpus API — cargando modelos de IA...")
    try:
        await loop.run_in_executor(None, shields.load)
    except Exception:
        print("   ⚠️  Los modelos no se cargaron. Los flujos no estarán disponibles.")

    yield


app = FastAPI(
    title="DataCorpus API",
    version="1.0.0",
    description="Backend para monitorizar y controlar el pipeline DataCorpus.",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://datacorpus.tail78f56e.ts.net:3000",
        "http://localhost:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(pipeline.router)
app.include_router(reparador.router)
app.include_router(dashboard.router)
app.include_router(maintenance.router)


# ── WebSocket — stream de logs ────────────────────────────────────
@app.websocket("/ws/logs")
async def ws_logs(websocket: WebSocket):
    await websocket.accept()
    queue = log_manager.add_client()
    try:
        while True:
            entry = await queue.get()
            await websocket.send_text(json.dumps(entry, ensure_ascii=False))
    except WebSocketDisconnect:
        pass
    finally:
        log_manager.remove_client(queue)


# ── Health check ──────────────────────────────────────────────────
@app.get("/health", tags=["Sistema"])
def health():
    from .services.runner import pipeline_runner, reparador_runner
    models = shields.status()
    return {
        "ok":        models["loaded"],
        "modelos":   models,
        "pipeline":  pipeline_runner.status,
        "reparador": reparador_runner.status,
    }
