"""
Singletons de QueryShield y DataShield.
Se cargan una sola vez al iniciar el servidor (lifespan de FastAPI).
"""

from typing import Optional

_qshield = None
_dshield = None
_loaded  = False
_error: Optional[str] = None


def load():
    """Carga ambos modelos. Llamar desde el lifespan del servidor."""
    global _qshield, _dshield, _loaded, _error
    try:
        from core.query_shield import QueryShield
        from core.data_shield  import DataShield
        from core.config       import DB_CONFIG

        print("   Cargando QueryShield (MiniLM L12)...")
        _qshield = QueryShield(DB_CONFIG)
        print("   ✅ QueryShield listo")

        print("   Cargando DataShield (BGE-M3)...")
        _dshield = DataShield(DB_CONFIG)
        print("   ✅ DataShield listo")

        _loaded = True
    except Exception as e:
        _error = str(e)
        print(f"   ❌ Error cargando modelos: {e}")
        raise


def get_qshield():
    return _qshield


def get_dshield():
    return _dshield


def status() -> dict:
    return {
        "loaded": _loaded,
        "error":  _error,
    }
