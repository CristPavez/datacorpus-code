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
    def start(self, target_fn, max_iterations: Optional[int] = None, **kwargs) -> bool:
        """
        max_iterations=None  → bucle infinito hasta que stop() sea llamado.
        max_iterations=N     → ejecuta N veces y para solo (o antes si stop() es llamado).
        """
        with self._lock:
            if self._status != FlowStatus.IDLE:
                return False
            self.stop_event.clear()
            self._status   = FlowStatus.RUNNING
            self._iteration       = 0
            self._max_iterations  = max_iterations

        def _run():
            log_manager.start_capture()
            modo = f"máx {max_iterations} iteraciones" if max_iterations else "bucle infinito"
            try:
                log_manager.emit(f"▶  [{self.name}] Iniciando ({modo})...", "SECTION")

                while not self.stop_event.is_set():
                    with self._lock:
                        self._iteration += 1
                        iteracion_actual = self._iteration

                    if max_iterations:
                        log_manager.emit(
                            f"── [{self.name}] Iteración {iteracion_actual}/{max_iterations}", "SECTION"
                        )
                    else:
                        log_manager.emit(
                            f"── [{self.name}] Iteración {iteracion_actual}", "SECTION"
                        )

                    target_fn(stop_event=self.stop_event, **kwargs)

                    if self.stop_event.is_set():
                        break
                    if max_iterations and iteracion_actual >= max_iterations:
                        log_manager.emit(
                            f"✅ [{self.name}] Completadas {max_iterations} iteraciones.", "SUCCESS"
                        )
                        break

                log_manager.emit(f"⏹  [{self.name}] Finalizado tras {self._iteration} iteración(es).", "SECTION")
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
        d = {
            "flow":      self.name,
            "status":    self._status.value,
            "iteration": getattr(self, "_iteration", 0),
        }
        if getattr(self, "_max_iterations", None):
            d["max_iterations"] = self._max_iterations
        return d


# Singletons
pipeline_runner  = FlowRunner("PIPELINE")
reparador_runner = FlowRunner("REPARADOR")
