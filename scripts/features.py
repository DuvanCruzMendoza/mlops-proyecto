"""Construye las variables del modelo de imputación a partir de los datos de load_data.py.

- Las variables monetarias y de personal se pasan a logaritmo (tienen colas muy largas).
- El objetivo es log(PRODBIND); las predicciones se devuelven a pesos con exp().
- Con usar_rezago=True se agrega la información del año anterior del mismo
  establecimiento (nordest). Solo quedan las filas que tienen año anterior.
"""
import numpy as np
import pandas as pd

from scripts.load_data import NUMERICAS, OBJETIVO

CATEGORICAS_MODELO = ["dpto", "ciiu2", "ciiu4"]
NUMERICAS_MODELO = ["l_" + c for c in NUMERICAS]
REZAGO_MODELO = ["l_prod_ant", "dl_persocu", "dl_salarper", "dl_eelec"]


def log_con_signo(x):
    """log(1 + |x|) conservando el signo: funciona con ceros y valores negativos (p. ej. inversión)."""
    return np.sign(x) * np.log1p(np.abs(x))


def construir_variables(df, usar_rezago=True):
    """Devuelve (datos con variables, lista de columnas numéricas, lista de categóricas)."""
    out = df.copy()
    for c in NUMERICAS:
        out["l_" + c] = log_con_signo(out[c])
    out["ciiu2"] = out["ciiu4"] // 100
    out["y_log"] = np.log(out[OBJETIVO])
    numericas = list(NUMERICAS_MODELO)

    if usar_rezago:
        anterior = df[["nordest", "periodo", OBJETIVO, "PERSOCU", "SALARPER", "EELEC"]].copy()
        anterior["periodo"] = anterior["periodo"] + 1  # el dato de 2022 es el "año anterior" de 2023
        anterior = anterior.rename(columns={OBJETIVO: "prod_ant", "PERSOCU": "persocu_ant",
                                            "SALARPER": "salarper_ant", "EELEC": "eelec_ant"})
        out = out.merge(anterior, on=["nordest", "periodo"], how="inner")
        out["l_prod_ant"] = np.log(out["prod_ant"])
        out["dl_persocu"] = log_con_signo(out["PERSOCU"]) - log_con_signo(out["persocu_ant"])
        out["dl_salarper"] = log_con_signo(out["SALARPER"]) - log_con_signo(out["salarper_ant"])
        out["dl_eelec"] = log_con_signo(out["EELEC"]) - log_con_signo(out["eelec_ant"])
        numericas += REZAGO_MODELO

    return out, numericas, list(CATEGORICAS_MODELO)


def dividir_por_anio(df, anio_prueba):
    """Entrenamiento: años anteriores a anio_prueba. Prueba: anio_prueba."""
    train = df[df["periodo"] < anio_prueba]
    test = df[df["periodo"] == anio_prueba]
    if train.empty or test.empty:
        raise ValueError(f"No hay datos para entrenar antes de {anio_prueba} o para probar en {anio_prueba}")
    return train, test


def linea_base(train, test, usar_rezago):
    """Predicción ingenua en log, para comparar el modelo contra algo simple.

    Con rezago: la producción del año anterior. Sin rezago: la mediana del CIIU en entrenamiento.
    """
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
        "mdape": float(np.median(error_pct)),            # error porcentual mediano
        "pct_error_menor_20": float(np.mean(error_pct < 0.20)),  # % de establecimientos con error < 20 %
    }
