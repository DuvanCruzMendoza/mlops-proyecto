"""Actividad 3: drift del evento (objetivo PRODBIND). Sin MLflow ni Prefect."""
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, ks_2samp

from scripts.load_data import OBJETIVO


def validar_etiquetas(df, nombre="lote"):
    """El lote debe traer PRODBIND real: numérico, sin vacíos, finito y positivo."""
    if OBJETIVO not in df.columns:
        raise ValueError(f"El {nombre} no contiene la columna {OBJETIVO}: las etiquetas reales aún no han llegado")
    if df.empty:
        raise ValueError(f"El {nombre} está vacío")
    y = pd.to_numeric(df[OBJETIVO], errors="coerce")
    if y.isna().any():
        raise ValueError(f"El {nombre} contiene etiquetas {OBJETIVO} vacías o no numéricas")
    if not np.isfinite(y).all():
        raise ValueError(f"El {nombre} contiene etiquetas {OBJETIVO} infinitas")
    if (y <= 0).any():
        raise ValueError(f"El {nombre} contiene etiquetas {OBJETIVO} menores o iguales a cero")
    return y


def drift_de_evento(referencia, lote, umbral_pp):
    """Compara la producción real del lote con la del entrenamiento.

    Evento: establecimiento "grande" (PRODBIND > P75 del entrenamiento; en la referencia es el 25 %).
    Hay drift si la tasa del lote se aleja más de `umbral_pp` puntos porcentuales de la referencia.
    Se usa el tamaño del cambio y no solo el p-valor: con miles de filas, un cambio pequeño y natural
    (2024 frente a 2021-2023) ya es "significativo". El p-valor chi-cuadrado y KS quedan como apoyo.
    """
    ref = validar_etiquetas(referencia, "referencia")
    act = validar_etiquetas(lote, "lote")
    p75 = float(ref.quantile(0.75))
    alto_ref, alto_act = int((ref > p75).sum()), int((act > p75).sum())
    _, p_valor, _, _ = chi2_contingency([[alto_ref, len(ref) - alto_ref], [alto_act, len(act) - alto_act]])
    ks = ks_2samp(ref, act)
    return {
        "filas_referencia": len(ref), "filas_lote": len(act), "p75_referencia": p75,
        "tasa_referencia": alto_ref / len(ref), "tasa_lote": alto_act / len(act),
        "diferencia_pp": 100 * (alto_act / len(act) - alto_ref / len(ref)),
        "p_valor_chi2": float(p_valor), "umbral_pp": umbral_pp,
        "drift": bool(abs(100 * (alto_act / len(act) - alto_ref / len(ref))) > umbral_pp),
        "ks_estadistico": float(ks.statistic),
        "mediana_referencia": float(ref.median()), "mediana_lote": float(act.median()),
        "p90_referencia": float(ref.quantile(0.9)), "p90_lote": float(act.quantile(0.9)),
    }
