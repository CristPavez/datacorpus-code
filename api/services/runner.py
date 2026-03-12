"""
Gestiona la ejecución en background de los flujos y la señal de parada.
"""

import sys
import threading
from enum import Enum
from typing import Optional
from .log_manager import log_manager


class FlowStatus(str, Enum):
    IDLE     = "idle"
    RUNNING  = "running"
    STOPPING = "stopping"


class FlowRunner:
    def __init__(self, name: str):
        self.name         = name
        self._status      = FlowStatus.IDLE
        self._thread:     Optional[threading.Thread] = None
        self.stop_event   = threading.Event()
        self._lock        = threading.Lock()

    # ── Estado ────────────────────────────────────────────────────
    @property
    def status(self) -> FlowStatus:
        return self._status

    @property
    def is_running(self) -> bool:
        return self._status == FlowStatus.RUNNING

    # ── Iniciar ───────────────────────────────────────────────────
    def start(self, target_fn, **kwargs) -> bool:
        with self._lock:
            if self._status != FlowStatus.IDLE:
                return False
            self.stop_event.clear()
            self._status = FlowStatus.RUNNING

        def _run():
            log_manager.start_capture()
            try:
                log_manager.emit(f"▶  [{self.name}] Iniciando...", "SECTION")
                target_fn(stop_event=self.stop_event, **kwargs)
                log_manager.emit(f"⏹  [{self.name}] Finalizado.", "SECTION")
            except Exception as e:
                log_manager.emit(f"❌ [{self.name}] Error inesperado: {e}", "ERROR")
            finally:
                log_manager.stop_capture()
                with self._lock:
                    self._status = FlowStatus.IDLE

        self._thread = threading.Thread(target=_run, daemon=True, name=self.name)
        self._thread.start()
        return True

    # ── Detener ───────────────────────────────────────────────────
    def stop(self) -> bool:
        with self._lock:
            if self._status != FlowStatus.RUNNING:
                return False
            self._status = FlowStatus.STOPPING
            self.stop_event.set()
        log_manager.emit(f"⏸  [{self.name}] Detención solicitada — esperando fin de iteración...", "WARN")
        return True

    def to_dict(self) -> dict:
        return {
            "flow":   self.name,
            "status": self._status.value,
        }


# Singletons
pipeline_runner  = FlowRunner("PIPELINE")
reparador_runner = FlowRunner("REPARADOR")
