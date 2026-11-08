"""Actividad 2: pipeline de inferencia (se ejecuta cada día, ver flows/desplegar.py).

Pasos: cargar modelo champion → ingesta del lote → predicción → guardar CSV → registrar en MLflow

Caso de uso: imputar la producción bruta de un lote de establecimientos. El lote se simula con
una muestra de la EAM del año `inferencia.anio`, como si fueran establecimientos que no
reportaron su producción. Como en la muestra sí se conoce el valor reportado, también se
marcan los registros cuyo valor reportado se aleja mucho del estimado (validación).

El modelo se busca por el alias `champion`, no por número de versión: cuando el entrenamiento
asigna un nuevo champion, la inferencia lo usa sin cambiar código. El run de inferencia
(tag fase = inferencia) guarda la versión y el run de entrenamiento del modelo usado.

Requisitos: servicios levantados, un champion registrado (ver flows/entrenamiento_flow.py)
y las variables:
    PREFECT_API_URL=http://localhost:4200/api
    MLFLOW_TRACKING_URI=http://localhost:5000

Ejecución de prueba (desde la raíz del repositorio):
    poetry run python flows/inferencia_flow.py
"""
import os
import sys
from datetime import date
from pathlib import Path

# El proyecto usa package-mode = false: se agrega la raíz del repositorio a la ruta.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import mlflow
import numpy as np
import yaml
from mlflow.tracking import MlflowClient
from prefect import flow, task
from prefect.artifacts import create_markdown_artifact

from scripts.features import agregar_rezago, calcular_variables
from scripts.load_data import OBJETIVO, cargar_procesados

CONFIG = "configs/config.yaml"


@task
def cargar_champion(cfg):
    """Carga el modelo con el alias champion. Devuelve el modelo, su versión y su run de entrenamiento."""
    nombre = cfg["registro"]["nombre_modelo"]
    version = MlflowClient().get_model_version_by_alias(nombre, "champion")
    modelo = mlflow.sklearn.load_model(f"models:/{nombre}@champion")
    print(f"Modelo cargado: {nombre} versión {version.version} (run {version.run_id})")
    return modelo, str(version.version), version.run_id


@task
def ingesta(cfg):
    """Toma un lote de establecimientos del año de inferencia, con los datos de su año anterior."""
    inf = cfg["inferencia"]
    df = cargar_procesados(cfg["datos"])
    candidatos = agregar_rezago(df)  # el rezago necesita el año anterior, aunque el modelo no lo use
    candidatos = candidatos[candidatos["periodo"] == inf["anio"]]
    lote = candidatos.sample(n=min(inf["tamano_lote"], len(candidatos)), random_state=inf.get("semilla"))
    print(f"Lote de {len(lote)} establecimientos de {inf['anio']} (de {len(candidatos)} disponibles)")
    return lote


@task
def predecir(modelo, lote, cfg):
    """Imputa la producción y la compara con la reportada. Guarda un CSV con la fecha de hoy."""
    # El modelo solo ve sus variables de entrada (feature_names_in_): nunca la producción reportada.
    variables = calcular_variables(lote.drop(columns=[OBJETIVO]))
    pred_log = modelo.predict(variables[list(modelo.feature_names_in_)])

    resultado = lote[["nordest", "periodo", "dpto", "ciiu4"]].copy()
    resultado["produccion_estimada"] = np.exp(pred_log).round()
    resultado["produccion_reportada"] = lote[OBJETIVO]
    resultado["desviacion_log"] = np.log(lote[OBJETIVO]) - pred_log
    resultado["alerta"] = resultado["desviacion_log"].abs() > cfg["inferencia"]["umbral_alerta"]

    carpeta = Path(cfg["inferencia"]["carpeta_salida"])
    carpeta.mkdir(parents=True, exist_ok=True)
    ruta = carpeta / f"predicciones_{date.today().isoformat()}.csv"
    resultado.to_csv(ruta, index=False)
    print(f"Predicciones guardadas en {ruta}")

    # Se devuelven valores simples para que Prefect los pase entre tareas.
    return str(ruta), {
        "filas_predichas": len(resultado),
        "mae_log_lote": float(resultado["desviacion_log"].abs().mean()),
        "pct_alertas": float(resultado["alerta"].mean()),
        "produccion_estimada_mediana": float(resultado["produccion_estimada"].median()),
    }


@task
def registrar_inferencia(version, run_id_entrenamiento, ruta, metricas, cfg):
    """Crea un run de inferencia enlazado a la versión del modelo y a su run de entrenamiento."""
    with mlflow.start_run(run_name=f"inferencia-{date.today().isoformat()}") as run:
        mlflow.set_tag("fase", "inferencia")
        mlflow.set_tag("modelo_version", version)
        mlflow.set_tag("modelo_run_id", run_id_entrenamiento)
        mlflow.log_params({"anio": cfg["inferencia"]["anio"], "tamano_lote": cfg["inferencia"]["tamano_lote"],
                           "umbral_alerta": cfg["inferencia"]["umbral_alerta"]})
        mlflow.log_metrics(metricas)
        mlflow.log_artifact(ruta)
    print(f"Run de inferencia registrado: {run.info.run_id}")
    return run.info.run_id


@flow(name="inferencia-eam", log_prints=True)
def inferencia_flow(config: str = CONFIG):
    with open(config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", cfg["mlflow"]["tracking_uri"]))
    mlflow.set_experiment(cfg["mlflow"]["experimento"])

    modelo, version, run_id_entrenamiento = cargar_champion(cfg)
    lote = ingesta(cfg)
    ruta, metricas = predecir(modelo, lote, cfg)
    run_id = registrar_inferencia(version, run_id_entrenamiento, ruta, metricas, cfg)

    create_markdown_artifact(
        key="inferencia-eam",
        markdown=(
            f"## Inferencia diaria: {date.today().isoformat()}\n\n"
            f"- Modelo: `{cfg['registro']['nombre_modelo']}` versión **{version}** (champion)\n"
            f"- Establecimientos imputados: {metricas['filas_predichas']}\n"
            f"- MAE (log) frente a lo reportado: {metricas['mae_log_lote']:.4f}\n"
            f"- Registros con alerta (desviación > {cfg['inferencia']['umbral_alerta']} en log): "
            f"{metricas['pct_alertas']:.1%}\n"
            f"- Archivo: `{ruta}`\n- Run de MLflow: `{run_id}`"
        ),
    )
    return {"ruta": ruta, "version": version, "run_id": run_id, **metricas}


if __name__ == "__main__":
    inferencia_flow()
