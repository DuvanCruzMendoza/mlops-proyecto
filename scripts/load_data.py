"""Carga la EAM anonimizada (DANE) de varios años, valida sus columnas y la unifica.

Adaptado de scripts/load_data.py del repositorio del curso (actividad 1).

Caso de uso: imputar la producción bruta (PRODBIND) de un establecimiento industrial
a partir de variables que suelen estar disponibles (personal, salarios, energía,
activos, actividad CIIU y departamento).

Fuente: Encuesta Anual Manufacturera (EAM), microdatos anonimizados del DANE,
Archivo Nacional de Datos (ANDA): https://microdatos.dane.gov.co
Los Excel no se versionan; data/README.md explica cómo descargarlos.

Ejecución (desde la raíz del repositorio):
    poetry run python scripts/load_data.py

Para usarla desde un flow:
    from scripts.load_data import cargar_datos
"""
from pathlib import Path

import pandas as pd
import yaml

RUTA_CONFIG = "configs/config.yaml"

# Identificadores y variables de agrupación. nordest (establecimiento anonimizado)
# permite seguir al mismo establecimiento entre años.
IDENTIFICADORES = ["nordemp", "nordest", "periodo"]
CATEGORICAS = ["dpto", "ciiu4"]

# Variables explicativas candidatas. No se incluyen consumo intermedio, ventas ni
# valor agregado: forman parte de la producción por identidad contable y, en una
# imputación real, suelen faltar junto con ella.
NUMERICAS = [
    "PERSOCU",   # personal ocupado
    "PERTOTAL",  # personal total
    "SALARPER",  # salarios del personal permanente
    "PRESSPER",  # prestaciones del personal permanente
    "REMUTEMP",  # remuneración del personal temporal
    "PERAPREN",  # aprendices
    "EELEC",     # consumo de energía eléctrica
    "ACTIVFI",   # activos fijos
    "DEPRECIA",  # depreciación
    "INVEBRTA",  # inversión bruta
]

OBJETIVO = "PRODBIND"  # producción bruta industrial

COLUMNAS = IDENTIFICADORES + CATEGORICAS + NUMERICAS + [OBJETIVO]


def leer_config(ruta_config=RUTA_CONFIG):
    with open(ruta_config, encoding="utf-8") as f:
        return yaml.safe_load(f)


def cargar_anio(ruta, anio):
    """Lee el Excel de un año, valida columnas y periodo, y devuelve solo COLUMNAS."""
    df = pd.read_excel(ruta, usecols=lambda c: c in COLUMNAS)
    faltan = [c for c in COLUMNAS if c not in df.columns]
    if faltan:
        raise ValueError(f"A {Path(ruta).name} le faltan columnas: {faltan}")
    if set(df["periodo"].unique()) != {anio}:
        raise ValueError(f"{Path(ruta).name}: 'periodo' debería ser {anio} y trae {sorted(df['periodo'].unique())}")
    if df["nordest"].duplicated().any():
        raise ValueError(f"{Path(ruta).name}: hay establecimientos (nordest) repetidos")
    return df[COLUMNAS]


def cargar_datos(carpeta=None, anios=None, patron=None):
    """Devuelve un DataFrame con todos los años unidos (una fila por establecimiento y año).

    Sin argumentos usa datos.carpeta, datos.anios y datos.patron_archivo de configs/config.yaml.
    Falla si falta un archivo o una columna. Descarta los registros con producción <= 0
    porque el objetivo se modela en logaritmo.
    """
    if carpeta is None or anios is None or patron is None:
        cfg = leer_config()["datos"]
        carpeta = carpeta or cfg["carpeta"]
        anios = anios or cfg["anios"]
        patron = patron or cfg["patron_archivo"]

    partes = []
    for anio in anios:
        ruta = Path(carpeta) / patron.format(anio=anio)
        if not ruta.exists():
            raise FileNotFoundError(f"No existe {ruta}. Descárgalo del ANDA (ver data/README.md).")
        partes.append(cargar_anio(ruta, anio))

    df = pd.concat(partes, ignore_index=True)
    no_validos = int((df[OBJETIVO] <= 0).sum())
    if no_validos:
        print(f"Se descartan {no_validos} registros con {OBJETIVO} <= 0")
        df = df[df[OBJETIVO] > 0].reset_index(drop=True)
    return df


def resumen(df):
    """Datos básicos del dataset, para imprimir o registrar en Prefect y MLflow."""
    return {
        "filas": len(df),
        "columnas": df.shape[1],
        "nulos": int(df.isna().sum().sum()),
        "establecimientos": int(df["nordest"].nunique()),
        "filas_por_anio": {int(k): int(v) for k, v in df["periodo"].value_counts().sort_index().items()},
    }


def main():
    cfg = leer_config()["datos"]
    df = cargar_datos()
    r = resumen(df)
    print(f"Filas: {r['filas']}  columnas: {r['columnas']}  nulos: {r['nulos']}  establecimientos: {r['establecimientos']}")
    print(f"Filas por año: {r['filas_por_anio']}")

    # Guarda un CSV unificado: más liviano y mucho más rápido de leer que los Excel.
    salida = Path(cfg["salida"])
    salida.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(salida, index=False)
    print(f"CSV unificado guardado en {salida}")


if __name__ == "__main__":
    main()
