"""Actividad 2: pipeline de entrenamiento del modelo de imputación (EAM).

Pasos: ingesta → entrenamiento → evaluación → registro
  - Cada candidato de config.yaml es un run de MLflow con el tag fase = entrenamiento.
  - El mejor candidato (menor MAE) pasa a champion solo si:
      1. su MAE (log) es menor o igual a registro.max_mae_log,
      2. su MAE es menor que el de la línea base (si registro.exigir_superar_linea_base), y
      3. su MAE es menor que el del champion vigente.
    Si no cumple, el run queda en MLflow pero el alias no cambia.

El modelo es de regresión, así que el criterio usa el MAE en lugar del AUC del ejemplo del curso.

Requisitos: servicios levantados (ver services/ del repo del curso) y las variables de entorno:
    PREFECT_API_URL=http://localhost:4200/api
    MLFLOW_TRACKING_URI=http://localhost:5000

Ejecución (desde la raíz del repositorio):
    poetry run python flows/entrenamiento_flow.py
"""
import os
import sys
from pathlib import Path

# El proyecto usa package-mode = false: se agrega la raíz del repositorio a la ruta
# para poder importar scripts/ al ejecutar este archivo directamente.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import mlflow
import yaml
from mlflow.exceptions import MlflowException
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient
from prefect import flow, task
from prefect.artifacts import create_markdown_artifact
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from scripts.features import (RUTA_FEATURES, calcular_metricas, construir_variables, dividir_por_anio,
                              leer_features, linea_base, resolver_features)
from scripts.load_data import cargar_procesados, resumen

CONFIG = "configs/config.yaml"
# MLflow 3.x guarda los modelos con skops y pide declarar los tipos internos que no conoce.
# Es seguro porque los modelos los entrenamos nosotros en este mismo flow.
TIPOS_CONFIABLES = ["sklearn.ensemble._hist_gradient_boosting.predictor.TreePredictor"]


def leer_yaml(ruta):
    with open(ruta, encoding="utf-8") as f:
        return yaml.safe_load(f)


def conectar_mlflow(cfg):
    """MLFLOW_TRACKING_URI, si existe, tiene prioridad sobre el YAML."""
    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", cfg["mlflow"]["tracking_uri"]))
    mlflow.set_experiment(cfg["mlflow"]["experimento"])


def crear_modelo(tipo, parametros, numericas, categoricas):
    """Preprocesado (escalar numéricas y one-hot a categóricas) + estimador de regresión."""
    if tipo == "ridge":
        estimador = Ridge(**parametros)
    elif tipo == "hist_gradient_boosting":
        estimador = HistGradientBoostingRegressor(**parametros)
    else:
        raise ValueError(f"modelo '{tipo}' no soportado")
    transformadores = [("num", StandardScaler(), numericas)]
    if categoricas:
        transformadores.append(("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categoricas))
    return Pipeline([("preprocesado", ColumnTransformer(transformadores)), ("modelo", estimador)])


@task
def ingesta(cfg):
    """Carga la EAM (CSV unificado o Excel) y la divide por año en entrenamiento y prueba."""
    df = cargar_procesados(cfg["datos"])
    r = resumen(df)
    print(f"Datos cargados: {r['filas']} filas, {r['establecimientos']} establecimientos, por año {r['filas_por_anio']}")
    datos = construir_variables(df, cfg["variables"]["usar_rezago"])
    train, test = dividir_por_anio(datos, cfg["division"]["anio_prueba"])
    print(f"Entrenamiento: {len(train)} filas ({sorted(train['periodo'].unique())}) | prueba: {len(test)} filas")
    return train, test, r


@task
def entrenar(train, candidato, numericas, categoricas):
    """Entrena un candidato con las variables seleccionadas."""
    print(f"Entrenando {candidato['nombre_run']} ({candidato['tipo']}) con {len(numericas) + len(categoricas)} variables")
    modelo = crear_modelo(candidato["tipo"], candidato.get("parametros") or {}, numericas, categoricas)
    modelo.fit(train[numericas + categoricas], train["y_log"])
    return modelo


@task
def evaluar(modelo, train, test, numericas, categoricas, usar_rezago):
    """Métricas sobre el año de prueba, del modelo y de la línea base."""
    pred = modelo.predict(test[numericas + categoricas])
    metricas = calcular_metricas(test["y_log"], pred)
    base = calcular_metricas(test["y_log"], linea_base(train, test, usar_rezago))
    metricas.update({f"linea_base_{k}": v for k, v in base.items()})
    metricas["mejora_mae_vs_linea_base"] = 1 - metricas["mae_log"] / metricas["linea_base_mae_log"]
    print(f"MAE log = {metricas['mae_log']:.4f} (línea base {metricas['linea_base_mae_log']:.4f})")
    return metricas


@task
def registrar_run(modelo, metricas, candidato, cfg, cfg_features, n_variables, resumen_datos, test, numericas, categoricas):
    """Crea el run de MLflow (tag fase = entrenamiento) con parámetros, métricas y modelo."""
    conectar_mlflow(cfg)
    X_ejemplo = test[numericas + categoricas].head(50)
    nombre_run = f"{candidato['nombre_run']}-{cfg_features['seleccion'].lower()}"
    with mlflow.start_run(run_name=nombre_run) as run:
        mlflow.set_tag("fase", "entrenamiento")
        mlflow.log_params({
            "modelo": candidato["tipo"],
            **(candidato.get("parametros") or {}),
            "features": cfg_features["seleccion"],
            "n_features": n_variables,
            "usar_rezago": cfg["variables"]["usar_rezago"],
            "anio_prueba": cfg["division"]["anio_prueba"],
            "filas_datos": resumen_datos["filas"],
            "filas_prueba": len(test),
        })
        mlflow.log_metrics(metricas)
        mlflow.log_dict(resumen_datos, "datos/resumen_datos.json")
        info = mlflow.sklearn.log_model(
            modelo,
            name="model",
            signature=infer_signature(X_ejemplo, modelo.predict(X_ejemplo)),
            input_example=X_ejemplo.head(3),
            skops_trusted_types=TIPOS_CONFIABLES,
        )
    print(f"Run {nombre_run} registrado: {run.info.run_id}")
    return run.info.run_id, info.model_uri


@task
def promover_si_mejora(model_uri, metricas, cfg):
    """Registra una versión nueva y le asigna el alias champion solo si cumple los criterios.

    Devuelve (versión o None, explicación).
    """
    conectar_mlflow(cfg)
    reg = cfg["registro"]
    client = MlflowClient()
    nombre = reg["nombre_modelo"]
    mae = metricas["mae_log"]

    if mae > reg["max_mae_log"]:
        return None, f"MAE {mae:.4f} supera el máximo {reg['max_mae_log']}: el alias no cambia"
    if reg.get("exigir_superar_linea_base", True) and mae >= metricas["linea_base_mae_log"]:
        return None, f"MAE {mae:.4f} no mejora la línea base ({metricas['linea_base_mae_log']:.4f}): el alias no cambia"

    try:
        champion = client.get_model_version_by_alias(nombre, "champion")
    except MlflowException:
        champion = None  # todavía no hay champion
    if champion is not None:
        mae_champion = client.get_run(champion.run_id).data.metrics["mae_log"]
        if mae >= mae_champion:
            return None, (f"MAE {mae:.4f} no mejora al champion v{champion.version} "
                          f"({mae_champion:.4f}): el alias no cambia")

    version = str(mlflow.register_model(model_uri, nombre).version)
    client.set_registered_model_alias(nombre, "champion", version)
    client.set_model_version_tag(nombre, version, "estado", "aprobado")
    anterior = f"reemplaza a v{champion.version}" if champion is not None else "primer champion"
    return version, f"Versión {version} es el nuevo champion (MAE {mae:.4f}, {anterior})"


@flow(name="entrenamiento-eam", log_prints=True)
def entrenamiento_flow(config: str = CONFIG, features: str = RUTA_FEATURES):
    cfg, cfg_features = leer_yaml(config), leer_features(features)
    usar_rezago = cfg["variables"]["usar_rezago"]
    numericas, categoricas = resolver_features(cfg_features, usar_rezago)
    n_variables = len(numericas) + len(categoricas)
    print(f"Features: {cfg_features['seleccion']} ({n_variables} variables)")

    # 1. Ingesta
    train, test, resumen_datos = ingesta(cfg)

    resultados = []
    for candidato in cfg["candidatos"]:
        # 2. Entrenamiento  3. Evaluación  4a. Registro del run
        modelo = entrenar(train, candidato, numericas, categoricas)
        metricas = evaluar(modelo, train, test, numericas, categoricas, usar_rezago)
        run_id, model_uri = registrar_run(modelo, metricas, candidato, cfg, cfg_features, n_variables,
                                          resumen_datos, test, numericas, categoricas)
        resultados.append({"candidato": candidato["nombre_run"], "run_id": run_id,
                           "model_uri": model_uri, "metricas": metricas})

    # 4b. Registro en el Model Registry: solo el mejor candidato (menor MAE) puede ser champion.
    mejor = min(resultados, key=lambda r: r["metricas"]["mae_log"])
    version, decision = promover_si_mejora(mejor["model_uri"], mejor["metricas"], cfg)
    print(decision)

    filas = "\n".join(
        f"| {r['candidato']} | {r['metricas']['mae_log']:.4f} | {r['metricas']['linea_base_mae_log']:.4f} "
        f"| {r['metricas']['mdape']:.1%} | `{r['run_id']}` |" for r in resultados
    )
    create_markdown_artifact(
        key="entrenamiento-eam",
        markdown=(
            "## Entrenamiento: imputación de producción bruta (EAM)\n\n"
            f"Features: **{cfg_features['seleccion']}** ({n_variables} variables) | "
            f"año de prueba: {cfg['division']['anio_prueba']}\n\n"
            "| Candidato | MAE (log) | Línea base | Error mediano | Run de MLflow |\n|---|---|---|---|---|\n"
            f"{filas}\n\n**Decisión:** {decision}"
        ),
    )
    return {"mejor_run_id": mejor["run_id"], "mae_log": mejor["metricas"]["mae_log"],
            "version_champion": version, "decision": decision}


if __name__ == "__main__":
    entrenamiento_flow()
