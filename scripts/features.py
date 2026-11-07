"""Variables del modelo de imputación, selección de features, línea base y métricas.

Este módulo no usa Prefect ni MLflow: lo comparten el entrenamiento, la inferencia y la API,
para que las tres calculen las variables exactamente igual.

- Las variables monetarias y de personal se pasan a logaritmo (tienen colas muy largas).
- El objetivo es log(PRODBIND); las predicciones se devuelven a pesos con exp().
- Las variables de rezago usan el año anterior del mismo establecimiento (nordest):
  producción, personal, salarios y energía.
"""
import numpy as np
import pandas as pd
import yaml

from scripts.load_data import NUMERICAS, OBJETIVO

RUTA_FEATURES = "configs/features.yaml"

CATEGORICAS_BASE = ["dpto", "ciiu4"]
# Columnas del año anterior que necesita el rezago (las agrega agregar_rezago o las envía la API).
COLUMNAS_ANTERIOR = ["prod_ant", "persocu_ant", "salarper_ant", "eelec_ant"]
REZAGO = ["l_prod_ant", "dl_persocu", "dl_salarper", "dl_eelec"]
SELECCIONES = ("ALL", "NUMERIC", "CUSTOM")


def log_con_signo(x):
    """log(1 + |x|) conservando el signo: funciona con ceros y valores negativos (p. ej. inversión)."""
    return np.sign(x) * np.log1p(np.abs(x))


def agregar_rezago(df):
    """Une a cada fila los datos del año anterior del mismo establecimiento.

    Solo quedan las filas que tienen año anterior (la del primer año de cada establecimiento se pierde).
    """
    anterior = df[["nordest", "periodo", OBJETIVO, "PERSOCU", "SALARPER", "EELEC"]].copy()
    anterior["periodo"] = anterior["periodo"] + 1  # el dato de 2023 es el "año anterior" de 2024
    anterior.columns = ["nordest", "periodo"] + COLUMNAS_ANTERIOR
    return df.merge(anterior, on=["nordest", "periodo"], how="inner")


def calcular_variables(df):
    """Agrega las variables que usa el modelo a partir de las columnas originales.

    Sirve tanto para la base completa como para un solo establecimiento (API). Las variables
    de rezago se calculan solo si vienen las columnas del año anterior.
    """
    out = df.copy()
    for c in NUMERICAS:
        out["l_" + c] = log_con_signo(out[c])
    out["ciiu2"] = out["ciiu4"] // 100
    if "prod_ant" in out:
        out["l_prod_ant"] = np.log(out["prod_ant"])
        out["dl_persocu"] = log_con_signo(out["PERSOCU"]) - log_con_signo(out["persocu_ant"])
        out["dl_salarper"] = log_con_signo(out["SALARPER"]) - log_con_signo(out["salarper_ant"])
        out["dl_eelec"] = log_con_signo(out["EELEC"]) - log_con_signo(out["eelec_ant"])
    if OBJETIVO in out:
        out["y_log"] = np.log(out[OBJETIVO])
    return out


def construir_variables(df, usar_rezago=True):
    """Variables de la base completa: rezago (opcional) + transformaciones."""
    return calcular_variables(agregar_rezago(df) if usar_rezago else df)


def leer_features(ruta=RUTA_FEATURES):
    with open(ruta, encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolver_features(cfg_features, usar_rezago=True):
    """Traduce la selección de configs/features.yaml a (numéricas, categóricas) del modelo.

    - ALL: las 10 numéricas, dpto y CIIU.
    - NUMERIC: solo las 10 numéricas.
    - CUSTOM: las columnas listadas en `custom`, con los nombres del CSV (p. ej. PERSOCU, ciiu4).
    Con usar_rezago, en los tres casos se agregan las variables del año anterior.
    """
    seleccion = cfg_features["seleccion"]
    if seleccion == "ALL":
        originales = NUMERICAS + CATEGORICAS_BASE
    elif seleccion == "NUMERIC":
        originales = list(NUMERICAS)
    elif seleccion == "CUSTOM":
        originales = cfg_features.get("custom") or []
        desconocidas = [c for c in originales if c not in NUMERICAS + CATEGORICAS_BASE]
        if not originales or desconocidas:
            raise ValueError(f"CUSTOM debe listar columnas de {NUMERICAS + CATEGORICAS_BASE}; no válidas: {desconocidas}")
    else:
        raise ValueError(f"seleccion debe ser una de {SELECCIONES}, no '{seleccion}'")

    numericas = ["l_" + c for c in NUMERICAS if c in originales]
    categoricas = [c for c in ["dpto"] if c in originales]
    if "ciiu4" in originales:
        categoricas += ["ciiu2", "ciiu4"]
    if usar_rezago:
        numericas += REZAGO
    return numericas, categoricas


def dividir_por_anio(df, anio_prueba):
    """Entrenamiento: años anteriores a anio_prueba. Prueba: anio_prueba."""
    train = df[df["periodo"] < anio_prueba]
    test = df[df["periodo"] == anio_prueba]
    if train.empty or test.empty:
        raise ValueError(f"No hay datos para entrenar antes de {anio_prueba} o para probar en {anio_prueba}")
    return train, test


def linea_base(train, test, usar_rezago):
    """Predicción ingenua en log: la producción del año anterior, o la mediana del CIIU sin rezago."""
    if usar_rezago:
        return test["l_prod_ant"].to_numpy()
    medianas = train.groupby("ciiu4")["y_log"].median()
    return test["ciiu4"].map(medianas).fillna(train["y_log"].median()).to_numpy()


def calcular_metricas(y_log, pred_log):
    """Métricas en escala log y en pesos (error porcentual absoluto)."""
    y_log, pred_log = np.asarray(y_log), np.asarray(pred_log)
    error_pct = np.abs(np.exp(pred_log - y_log) - 1)
    return {
        "mae_log": float(np.mean(np.abs(pred_log - y_log))),
        "rmse_log": float(np.sqrt(np.mean((pred_log - y_log) ** 2))),
        "mdape": float(np.median(error_pct)),                    # error porcentual mediano
        "pct_error_menor_20": float(np.mean(error_pct < 0.20)),  # % de establecimientos con error < 20 %
    }
