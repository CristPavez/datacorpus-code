"""
Captura los print() de los flujos y los transmite por WebSocket.
Funciona con hilos: usa loop.call_soon_threadsafe para cruzar la barrera thread→async.
"""

import sys
import asyncio
from datetime import datetime
from typing import Optional


# Patrones de ruido de librerías externas que no deben llegar al WebSocket.
# Aplica a: tqdm (progress bars), Python warnings, sentence_transformers, trafilatura.
_NOISE_PATTERNS = (
    "%|",               # barra de progreso tqdm
    "it/s",             # velocidad tqdm
    "B/s",              # velocidad tqdm (bytes)
    "UserWarning",
    "DeprecationWarning",
    "FutureWarning",
    "RuntimeWarning",
    "warnings.warn",
)


class _CaptureStream:
    """Reemplaza sys.stdout para interceptar print()."""

    def __init__(self, manager: "LogManager", original):
        self._mgr  = manager
        self._orig = original
        self._buf  = ""

    def write(self, text: str):
        self._orig.write(text)          # sigue mostrando en terminal
        self._buf += text
        while "\n" in self._buf:
            line, self._buf = self._buf.split("\n", 1)
            if line.strip() and not _es_ruido_externo(line):
                self._mgr._broadcast(line)

    def flush(self):
        self._orig.flush()


def _es_ruido_externo(line: str) -> bool:
    """Devuelve True si la línea parece output de una librería externa."""
    return any(p in line for p in _NOISE_PATTERNS)


class LogManager:
    def __init__(self):
        self._clients: set[asyncio.Queue] = set()
        self._loop:    Optional[asyncio.AbstractEventLoop] = None
        self._orig     = sys.stdout

    # ── Lifecycle ─────────────────────────────────────────────────
    def set_loop(self, loop: asyncio.AbstractEventLoop):
        self._loop = loop

    def start_capture(self):
        sys.stdout = _CaptureStream(self, self._orig)

    def stop_capture(self):
        sys.stdout = self._orig

    # ── Clientes WebSocket ────────────────────────────────────────
    def add_client(self) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue(maxsize=500)
        self._clients.add(q)
        return q

    def remove_client(self, q: asyncio.Queue):
        self._clients.discard(q)

    # ── Broadcast ────────────────────────────────────────────────
    def _broadcast(self, message: str):
        if not self._loop or self._loop.is_closed():
            return
        entry = {
            "ts":    datetime.now().isoformat(timespec="seconds"),
            "level": _parse_level(message),
            "msg":   message,
        }
        for q in list(self._clients):
            try:
                self._loop.call_soon_threadsafe(q.put_nowait, entry)
            except asyncio.QueueFull:
                pass

    def emit(self, message: str, level: str = "INFO"):
        """Emite un mensaje de sistema (no viene de print)."""
        if not self._loop or self._loop.is_closed():
            return
        entry = {"ts": datetime.now().isoformat(timespec="seconds"),
                 "level": level, "msg": message}
        for q in list(self._clients):
            try:
                self._loop.call_soon_threadsafe(q.put_nowait, entry)
            except asyncio.QueueFull:
                pass


def _parse_level(msg: str) -> str:
    m = msg
    if any(x in m for x in ("✅", "aprobado", "Aprobado", "APROBADO", "completado", "guardado", "Completado")):
        return "SUCCESS"
    if any(x in m for x in ("❌", "Error", "error", "ERROR", "fallo", "Fallo")):
        return "ERROR"
    if any(x in m for x in ("⚠️", "warning", "Warning", "DUPLICADA", "SIMILAR",
                              "SIN_RESULTADOS", "OMITIDA", "rechazado", "Umbral")):
        return "WARN"
    if any(x in m for x in ("===", "───", "###", "PIPELINE", "REPARADOR",
                              "FLUJO", "SCRAPING", "GENERACION")):
        return "SECTION"
    return "LOG"


log_manager = LogManager()
