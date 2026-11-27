"""Actividad 3: drift emulado del lote de inferencia. Solo pandas y NumPy.

Regla (nivel entre 0 y 1):
1. Drift del objetivo: se reemplaza una fracción `nivel` del lote por establecimientos REALES de 2024
   con producción mayor al P75 del entrenamiento. Sus etiquetas son las observadas (no se inventan).
2. Drift de entradas: SALARPER y EELEC se multiplican por (1 + nivel).
Con nivel 0.0 el lote queda intacto.
"""
import math

import pandas as pd

from scripts.load_data import OBJETIVO

COLUMNAS_ESCALADAS = ["SALARPER", "EELEC"]


def emular_drift(lote, pool, nivel, semilla, p75):
    """Devuelve una copia del lote con drift emulado. No modifica los argumentos."""
    if not isinstance(nivel, (int, float)) or not math.isfinite(nivel) or not 0 <= nivel <= 1:
        raise ValueError(f"El nivel de drift debe ser un número entre 0 y 1, no {nivel!r}")
    if nivel == 0:
        return lote.copy()

    n = round(nivel * len(lote))
    candidatos = pool[(pool[OBJETIVO] > p75) & ~pool["nordest"].isin(lote["nordest"])]
    if len(candidatos) < n:
        raise ValueError(f"Hay {len(candidatos)} candidatos sobre el P75 y se necesitan {n}")

    # Orden fijo por semilla: el nivel 0.6 incluye los reemplazos del 0.3.
    nuevos = candidatos.sample(frac=1, random_state=semilla).head(n)
    quitar = lote.sample(frac=1, random_state=semilla).index[:n]
    out = pd.concat([lote.drop(index=quitar), nuevos], ignore_index=True)
    for columna in COLUMNAS_ESCALADAS:
        out[columna] = out[columna] * (1 + nivel)
    return out
