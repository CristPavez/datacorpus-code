from fastapi import APIRouter, HTTPException
from ..services.runner import pipeline_runner, reparador_runner
from ..services import shields
from core.flujo_reparador import run_reparador

router = APIRouter(prefix="/reparador", tags=["Reparador"])


@router.post("/start", summary="Inicia el flujo reparador manualmente")
def start_reparador():
    if not shields.status()["loaded"]:
        raise HTTPException(503, "Los modelos aún no están cargados. Espera a que el servidor termine de iniciar.")
    if pipeline_runner.is_running:
        raise HTTPException(409, "El pipeline principal está en ejecución. Detenlo primero.")
    if reparador_runner.is_running:
        raise HTTPException(409, "El reparador ya está en ejecución.")

    ok = reparador_runner.start(
        run_reparador,
        max_iterations=1,
        qshield=shields.get_qshield(),
        dshield=shields.get_dshield(),
    )
    if not ok:
        raise HTTPException(409, "No se pudo iniciar el reparador.")
    return {"started": True, "status": reparador_runner.status}


@router.post("/stop", summary="Detiene el reparador después de la iteración actual")
def stop_reparador():
    ok = reparador_runner.stop()
    if not ok:
        raise HTTPException(400, "El reparador no está en ejecución.")
    return {"stopping": True, "status": reparador_runner.status}


@router.get("/status", summary="Estado actual del reparador")
def status_reparador():
    return reparador_runner.to_dict()
