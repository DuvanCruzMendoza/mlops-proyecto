"""Actividad 3: drift de features con Evidently. Sin MLflow ni Prefect."""
from pathlib import Path

from evidently import DataDefinition, Dataset, Report
from evidently.presets import DataDriftPreset


def validar_columnas(df, columnas, nombre):
    faltan = [c for c in columnas if c not in df.columns]
    if faltan:
        raise ValueError(f"Al {nombre} le faltan las columnas: {faltan}")


def hay_alerta(fraccion, umbral):
    """Alerta solo si la fracción SUPERA el umbral (un empate no alerta)."""
    return fraccion > umbral


def resumen_evidently(resultado):
    """Fracción y lista de columnas con drift a partir de `snapshot.dict()`."""
    fraccion, con_drift = None, []
    for m in resultado["metrics"]:
        nombre = m["metric_name"]
        if nombre.startswith("DriftedColumnsCount"):
            fraccion = float(m["value"]["share"])
        elif nombre.startswith("ValueDrift"):
            columna = nombre.split("column=")[1].split(",")[0]
            umbral = float(nombre.split("threshold=")[1].rstrip(")"))
            es_p_valor = "p_value" in nombre or "test" in nombre.lower()
            if (m["value"] < umbral) if es_p_valor else (m["value"] >= umbral):
                con_drift.append(columna)
    if fraccion is None:
        raise ValueError("El resultado de Evidently no incluye DriftedColumnsCount")
    return fraccion, con_drift


def calcular_drift_features(referencia, lote, numericas, categoricas, umbral, ruta_html=None):
    """Compara el lote con la referencia. Guarda el HTML si se da ruta_html."""
    columnas = numericas + categoricas
    validar_columnas(referencia, columnas, "referencia")
    validar_columnas(lote, columnas, "lote")
    esquema = DataDefinition(numerical_columns=numericas, categorical_columns=categoricas)
    snapshot = Report([DataDriftPreset(columns=columnas)]).run(
        reference_data=Dataset.from_pandas(referencia[columnas], esquema),
        current_data=Dataset.from_pandas(lote[columnas], esquema),
    )
    if ruta_html:
        Path(ruta_html).parent.mkdir(parents=True, exist_ok=True)
        snapshot.save_html(str(ruta_html))
    fraccion, con_drift = resumen_evidently(snapshot.dict())
    return {
        "fraccion_drift": fraccion,
        "umbral": umbral,
        "alerta": hay_alerta(fraccion, umbral),
        "columnas_evaluadas": len(columnas),
        "columnas_con_drift": con_drift,
    }
