#!/usr/bin/env python3
"""
DataCorpus — Punto de entrada principal de mantenimiento.
Ejecuta todos los pasos en modo lectura (sin borrar nada).

Uso:
  python revisar.py          → ejecuta todos los pasos
  python revisar.py --paso 2 → ejecuta solo el paso indicado

Pasos disponibles:
  1  Integridad de datos
  2  Dashboard de estadísticas
  3  Detección de ruido (BD)
  4  Queries pendientes
  5  Salud del índice FAISS
"""

import sys
import os
from datetime import datetime

# Asegurarse de que los imports funcionen desde este directorio
os.chdir(os.path.dirname(os.path.abspath(__file__)))

BOLD  = "\033[1m"
AZUL  = "\033[94m"
VERDE = "\033[92m"
RESET = "\033[0m"

PASOS = {
    1: ("Integridad de datos",       "01_integrity"),
    2: ("Dashboard de estadísticas", "02_dashboard"),
    3: ("Detección de ruido",        "03_noise"),
    4: ("Queries pendientes",        "04_pending"),
    5: ("Salud del índice FAISS",    "05_faiss"),
}


def ejecutar_paso(num, nombre, modulo):
    print(f"\n{BOLD}{AZUL}{'━'*65}{RESET}")
    print(f"{BOLD}  [{num}/5] {nombre}{RESET}")
    print(f"{BOLD}{AZUL}{'━'*65}{RESET}")
    try:
        mod = __import__(modulo)
        mod.main()
    except Exception as e:
        print(f"\n  ❌ Error en paso {num}: {e}\n")


def main():
    args = sys.argv[1:]

    # Modo paso único
    if "--paso" in args:
        idx = args.index("--paso")
        if idx + 1 < len(args):
            try:
                num = int(args[idx + 1])
                if num not in PASOS:
                    raise ValueError
                nombre, modulo = PASOS[num]
                ejecutar_paso(num, nombre, modulo)
                return
            except ValueError:
                print(f"  ❌ Paso inválido. Usa un número del 1 al 5.\n")
                sys.exit(1)

    # Todos los pasos
    print(f"\n{BOLD}{AZUL}{'='*65}{RESET}")
    print(f"{BOLD}  DATACORPUS — REVISIÓN COMPLETA{RESET}".center(72))
    print(f"  {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}")
    print(f"{BOLD}{AZUL}{'='*65}{RESET}")
    print(f"\n  Pasos: {', '.join(str(k) for k in PASOS)}")
    print(f"  Modo:  solo lectura (ningún paso modifica datos)\n")

    resultados = {}
    for num, (nombre, modulo) in PASOS.items():
        try:
            mod = __import__(modulo)
            resultado = mod.main()
            resultados[num] = ("ok" if not resultado else "warn", nombre)
        except Exception as e:
            resultados[num] = ("error", nombre)
            print(f"\n  ❌ Error en paso {num} ({nombre}): {e}\n")

    # Resumen final
    print(f"\n{BOLD}{AZUL}{'='*65}{RESET}")
    print(f"{BOLD}  RESUMEN DE REVISIÓN{RESET}")
    print(f"{BOLD}{AZUL}{'='*65}{RESET}\n")

    for num, (estado, nombre) in resultados.items():
        if estado == "ok":
            icono = f"{VERDE}✅{RESET}"
        elif estado == "warn":
            icono = "⚠️ "
        else:
            icono = "❌"
        print(f"  {icono} Paso {num}: {nombre}")

    print(f"\n  Revisión completada — {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}")
    print(f"\n  Acciones disponibles si se detectaron problemas:")
    print(f"    python 03_noise.py --borrar              Eliminar ruido de BD")
    print(f"    python 03_noise.py --jsonl archivo.jsonl  Limpiar JSONL")
    print(f"    python 04_pending.py --recuperar          Re-scrapedar pendientes")
    print(f"    python 04_pending.py --limpiar            Descartar pendientes")
    print(f"    python 05_faiss.py --rebuild              Reconstruir FAISS\n")


if __name__ == "__main__":
    main()
