"""Actividad 3: utilidades compartidas de monitoreo. Sin MLflow ni Prefect.

La referencia del drift son las filas de ENTRENAMIENTO (años anteriores a division.anio_prueba),
con las mismas variables que usa el modelo. Se calcula aquí, con las funciones del entrenamiento,
para que inferencia y flows de drift usen exactamente la misma.
"""
from pathlib import Path

from scripts.features import RUTA_FEATURES, construir_variables, dividir_por_anio, leer_features, resolver_features
from scripts.load_data import OBJETIVO

IDENTIFICADORES_LOTE = ["nordest", "periodo"]


def columnas_modelo(cfg, ruta_features=RUTA_FEATURES):
    """(numéricas, categóricas) que usa el modelo según configs/features.yaml."""
    return resolver_features(leer_features(ruta_features), cfg["variables"]["usar_rezago"])


def construir_referencia(df, cfg, ruta_features=RUTA_FEATURES):
    """Filas de entrenamiento con las columnas del modelo y el objetivo PRODBIND."""
    numericas, categoricas = columnas_modelo(cfg, ruta_features)
    datos = construir_variables(df, cfg["variables"]["usar_rezago"])
    train, _ = dividir_por_anio(datos, cfg["division"]["anio_prueba"])
    return train[numericas + categoricas + [OBJETIVO]].reset_index(drop=True)


def ruta_lote(cfg, nivel):
    """data/monitoreo/lotes/lote_nivel_0.6.csv"""
    return Path(cfg["monitoreo"]["carpeta"]) / "lotes" / f"lote_nivel_{float(nivel):.1f}.csv"


def carpeta_reportes(cfg, nivel):
    """data/monitoreo/reportes/nivel_0.6/"""
    return Path(cfg["monitoreo"]["carpeta"]) / "reportes" / f"nivel_{float(nivel):.1f}"
