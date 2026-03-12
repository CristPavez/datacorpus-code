from fastapi import APIRouter, HTTPException
from ..services.runner import pipeline_runner, reparador_runner

router = APIRouter(prefix="/pipeline", tags=["Pipeline"])


@router.post("/start", summary="Inicia el pipeline principal")
def start_pipeline():
    if reparador_runner.is_running:
        raise HTTPException(409, "El flujo reparador está en ejecución. Detenlo primero.")
    if pipeline_runner.is_running:
        raise HTTPException(409, "El pipeline ya está en ejecución.")

    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
    from ejecutar_todo import run_pipeline

    ok = pipeline_runner.start(run_pipeline)
    if not ok:
        raise HTTPException(409, "No se pudo iniciar el pipeline.")
    return {"started": True, "status": pipeline_runner.status}


@router.post("/stop", summary="Detiene el pipeline después de la iteración actual")
def stop_pipeline():
    ok = pipeline_runner.stop()
    if not ok:
        raise HTTPException(400, "El pipeline no está en ejecución.")
    return {"stopping": True, "status": pipeline_runner.status}


@router.get("/status", summary="Estado actual del pipeline")
def status_pipeline():
    return pipeline_runner.to_dict()
