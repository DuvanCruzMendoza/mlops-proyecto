"""Paso 4: flow de Prefect que entrena el modelo de imputación y lo registra en MLflow.

Adaptado de flows/experimento_flow.py del repositorio del curso (actividad 1).

Caso de uso: imputar la producción bruta (PRODBIND) de los establecimientos de la EAM.
Pasos: cargar datos → entrenar y registrar en MLflow → resumen (artefacto de Prefect)

Requisitos: servicios levantados (ver services/ del repo del curso) y las variables de entorno:
    PREFECT_API_URL=http://localhost:4200/api
    MLFLOW_TRACKING_URI=http://localhost:5000

Ejecución (desde la raíz del repositorio):
    poetry run python flows/experimento_flow.py
"""
import os
import sys
from pathlib import Path

# El proyecto usa package-mode = false, así que `scripts` no está instalado como paquete.
# Se agrega la raíz del repositorio a la ruta para poder importar scripts/ desde flows/.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import mlflow
import yaml
from mlflow.models import infer_signature
from prefect import flow, task
from prefect.artifacts import create_markdown_artifact
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from scripts.features import calcular_metricas, construir_variables, dividir_por_anio, linea_base
from scripts.load_data import cargar_procesados, resumen

CONFIG = "configs/config.yaml"
TIPOS_CONFIABLES = ["sklearn.ensemble._hist_gradient_boosting.predictor.TreePredictor"]


@task
def cargar_datos(cfg):
    """Carga la EAM con scripts/load_data.py (CSV unificado o, si no existe, los Excel)."""
    df = cargar_procesados(cfg["datos"])
    r = resumen(df)
    print(f"Datos cargados: {r['filas']} filas, {r['establecimientos']} establecimientos, por año {r['filas_por_anio']}")
    return df


def crear_modelo(tipo, parametros, numericas, categoricas):
    """Preprocesado (escalar numéricas y one-hot a categóricas) + estimador de regresión."""
    if tipo == "ridge":
        estimador = Ridge(**parametros)
    elif tipo == "hist_gradient_boosting":
        estimador = HistGradientBoostingRegressor(**parametros)
    else:
        raise ValueError(f"modelo '{tipo}' no soportado")

    preprocesado = ColumnTransformer(
        [
            ("num", StandardScaler(), numericas),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categoricas),
        ]
    )
    return Pipeline([("preprocesado", preprocesado), ("modelo", estimador)])


@task
def entrenar_y_registrar(df, cfg):
    """Entrena el modelo, lo compara con la línea base y registra todo en MLflow. Devuelve (run_id, métricas)."""
    # 1. Variables y división temporal: entrenar con años anteriores, probar con anio_prueba.
    usar_rezago = cfg["variables"]["usar_rezago"]
    anio_prueba = cfg["division"]["anio_prueba"]
    datos, numericas, categoricas = construir_variables(df, usar_rezago)
    train, test = dividir_por_anio(datos, anio_prueba)
    X_train, y_train = train[numericas + categoricas], train["y_log"]
    X_test, y_test = test[numericas + categoricas], test["y_log"]

    # 2. Crear y entrenar el modelo.
    tipo = cfg["modelo"]["tipo"]
    parametros = cfg["modelo"].get("parametros") or {}
    modelo = crear_modelo(tipo, parametros, numericas, categoricas)
    modelo.fit(X_train, y_train)

    # 3. Métricas sobre el año de prueba, del modelo y de la línea base.
    pred = modelo.predict(X_test)
    metricas = calcular_metricas(y_test, pred)
    base = calcular_metricas(y_test, linea_base(train, test, usar_rezago))
    metricas.update({f"linea_base_{k}": v for k, v in base.items()})
    metricas["mejora_mae_vs_linea_base"] = 1 - metricas["mae_log"] / metricas["linea_base_mae_log"]

    # 4. Registrar el run en MLflow. MLFLOW_TRACKING_URI, si existe, tiene prioridad sobre el YAML.
    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", cfg["mlflow"]["tracking_uri"]))
    mlflow.set_experiment(cfg["mlflow"]["experimento"])
    with mlflow.start_run(run_name=cfg["mlflow"]["nombre_run"]) as run:
        mlflow.log_params({
            "modelo": tipo,
            **parametros,
            "usar_rezago": usar_rezago,
            "anio_prueba": anio_prueba,
            "linea_base": "produccion_anio_anterior" if usar_rezago else "mediana_ciiu4",
            # Tamaño del dataset (entregable recomendado del paso 2)
            "filas_datos": len(df),
            "establecimientos": int(df["nordest"].nunique()),
            "filas_entrenamiento": len(train),
            "filas_prueba": len(test),
            "anios_entrenamiento": sorted(int(a) for a in train["periodo"].unique()),
        })
        mlflow.log_metrics(metricas)
        mlflow.log_dict(resumen(df), "datos/resumen_datos.json")
        mlflow.sklearn.log_model(
            modelo,
            name="model",
            signature=infer_signature(X_test, pred),
            input_example=X_test.head(3),
            # MLflow 3.x guarda los modelos con skops y pide declarar los tipos internos que
            # no conoce. Es seguro porque el modelo lo entrenamos nosotros en este mismo run.
            skops_trusted_types=TIPOS_CONFIABLES,
        )
        run_id = run.info.run_id

    return run_id, metricas


@flow(name="eam-imputacion-v1", log_prints=True)
def eam_imputacion():
    with open(CONFIG, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    df = cargar_datos(cfg)
    run_id, metricas = entrenar_y_registrar(df, cfg)

    create_markdown_artifact(
        key="eam-imputacion-v1",
        markdown=(
            "## Imputación de producción bruta (EAM) V1\n\n"
            f"- Modelo: `{cfg['modelo']['tipo']}` | rezago: `{cfg['variables']['usar_rezago']}`\n"
            f"- Filas: {len(df)} | año de prueba: {cfg['division']['anio_prueba']}\n"
            f"- MAE (log): **{metricas['mae_log']:.4f}** (línea base: {metricas['linea_base_mae_log']:.4f})\n"
            f"- Error porcentual mediano: {metricas['mdape']:.1%}\n"
            f"- Establecimientos con error < 20 %: {metricas['pct_error_menor_20']:.1%}\n"
            f"- Run de MLflow: `{run_id}`"
        ),
    )
    print(f"Run de MLflow: {run_id} | MAE log = {metricas['mae_log']:.4f} "
          f"(línea base {metricas['linea_base_mae_log']:.4f})")
    return run_id


if __name__ == "__main__":
    eam_imputacion()
