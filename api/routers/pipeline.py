from typing import Optional
from fastapi import APIRouter, HTTPException, Query
from ..services.runner import pipeline_runner, reparador_runner
from ..services import shields
from core.pipeline import run_pipeline

router = APIRouter(prefix="/pipeline", tags=["Pipeline"])


@router.post("/start", summary="Inicia el pipeline principal")
def start_pipeline(
    iterations: Optional[int] = Query(
        default=None,
        ge=1,
        description="Número de iteraciones a ejecutar. Sin valor = bucle infinito hasta /stop."
    )
):
    if not shields.status()["loaded"]:
        raise HTTPException(503, "Los modelos aún no están cargados. Espera a que el servidor termine de iniciar.")
    if reparador_runner.is_running:
        raise HTTPException(409, "El flujo reparador está en ejecución. Detenlo primero.")
    if pipeline_runner.is_running:
        raise HTTPException(409, "El pipeline ya está en ejecución.")

    ok = pipeline_runner.start(
        run_pipeline,
        max_iterations=iterations,
        qshield=shields.get_qshield(),
        dshield=shields.get_dshield(),
    )
    if not ok:
        raise HTTPException(409, "No se pudo iniciar el pipeline.")
    return {
        "started":        True,
        "status":         pipeline_runner.status,
        "max_iterations": iterations,
        "modo":           f"{iterations} iteraciones" if iterations else "bucle infinito",
    }


@router.post("/stop", summary="Detiene el pipeline después de la iteración actual")
def stop_pipeline():
    ok = pipeline_runner.stop()
    if not ok:
        raise HTTPException(400, "El pipeline no está en ejecución.")
    return {"stopping": True, "status": pipeline_runner.status}


@router.get("/status", summary="Estado actual del pipeline")
def status_pipeline():
    return pipeline_runner.to_dict()
