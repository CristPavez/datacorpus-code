import threading
from fastapi import APIRouter, HTTPException
from ..services.log_manager import log_manager

router = APIRouter(prefix="/maintenance", tags=["Mantenimiento"])

_rebuild_lock = threading.Lock()
_rebuild_running = False


def _run_rebuild(modo: str):
    global _rebuild_running
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
    from rebuild_faiss import rebuild_docs_faiss, rebuild_queries_faiss

    log_manager.start_capture()
    try:
        if modo in ("docs", "ambos"):
            rebuild_docs_faiss()
        if modo in ("queries", "ambos"):
            rebuild_queries_faiss()
        log_manager.emit("✅ Reconstrucción FAISS completada.", "SUCCESS")
    except Exception as e:
        log_manager.emit(f"❌ Error reconstruyendo FAISS: {e}", "ERROR")
    finally:
        log_manager.stop_capture()
        global _rebuild_running
        _rebuild_running = False


@router.post("/rebuild-faiss/{modo}", summary="Reconstruye índice(s) FAISS desde la BD")
def rebuild_faiss(modo: str):
    """
    modo:
      - `docs`    → solo faiss_docs.bin   (BGE-M3)
      - `queries` → solo faiss_queries.bin (MiniLM)
      - `ambos`   → ambos índices
    """
    global _rebuild_running

    if modo not in ("docs", "queries", "ambos"):
        raise HTTPException(400, "modo debe ser: docs | queries | ambos")

    with _rebuild_lock:
        if _rebuild_running:
            raise HTTPException(409, "Ya hay una reconstrucción en curso.")
        _rebuild_running = True

    t = threading.Thread(target=_run_rebuild, args=(modo,), daemon=True)
    t.start()
    return {"started": True, "modo": modo, "mensaje": "Reconstrucción iniciada — sigue el progreso en /ws/logs"}


@router.get("/rebuild-faiss/status", summary="Estado de la reconstrucción FAISS")
def rebuild_status():
    return {"running": _rebuild_running}
