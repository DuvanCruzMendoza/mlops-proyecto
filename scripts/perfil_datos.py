"""Actividad 3: perfil de los datos de la EAM para responder la retroalimentación del profesor.

Cuenta nulos, ceros y negativos por variable y describe la distribución de PRODBIND
(media, mediana, percentiles y extremos). Lee el CSV procesado y, para el registro descartado,
los Excel originales.

Ejecución (desde la raíz):  poetry run python scripts/perfil_datos.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from scripts.load_data import NUMERICAS, OBJETIVO, cargar_anio, cargar_procesados, leer_config


def perfil(df):
    columnas = NUMERICAS + [OBJETIVO]
    return pd.DataFrame({
        "nulos": df[columnas].isna().sum(),
        "ceros": (df[columnas] == 0).sum(),
        "ceros_%": ((df[columnas] == 0).mean() * 100).round(1),
        "negativos": (df[columnas] < 0).sum(),
    })


def main():
    cfg = leer_config()
    df = cargar_procesados(cfg["datos"])
    print(f"Base procesada: {len(df)} filas\n")
    print(perfil(df).to_string(), "\n")

    y = df[OBJETIVO]
    q = y.quantile([0.25, 0.5, 0.75, 0.9, 0.99])
    print(f"{OBJETIVO}: media {y.mean():,.0f} | mediana {y.median():,.0f} | media/mediana {y.mean() / y.median():.1f}")
    print("Percentiles: " + " | ".join(f"P{int(p * 100)} {v:,.0f}" for p, v in q.items()))
    print(f"Máximo {y.max():,.0f} ({y.max() / y.median():,.0f} veces la mediana)\n")

    d = cfg["datos"]
    crudos = pd.concat([cargar_anio(Path(d["carpeta"]) / d["patron_archivo"].format(anio=a), a) for a in d["anios"]])
    no_positivos = crudos[crudos[OBJETIVO] <= 0]
    print(f"Registros con {OBJETIVO} <= 0 en los Excel: {len(no_positivos)} "
          f"(ceros: {(no_positivos[OBJETIVO] == 0).sum()}, negativos: {(no_positivos[OBJETIVO] < 0).sum()})")


if __name__ == "__main__":
    main()
