"""Actividad 2: API de inferencia. Sirve el modelo con el alias `champion` de MLflow.

Imputa la producción bruta de UN establecimiento a partir de sus datos del año y, si el
modelo usa rezago, de los de su año anterior.

Ejecución (desde la raíz del repositorio, con MLFLOW_TRACKING_URI definida):
    poetry run uvicorn api.main:app --port 8000
Documentación interactiva: http://localhost:8000/docs

El modelo se carga al arrancar la API. Si el pipeline de entrenamiento asigna un nuevo
champion, hay que reiniciar la API para que lo use (no hace falta cambiar código).
"""
import os
from contextlib import asynccontextmanager

import mlflow
import numpy as np
import pandas as pd
import yaml
from fastapi import FastAPI, HTTPException
from mlflow.tracking import MlflowClient
from pydantic import BaseModel, Field

from scripts.features import calcular_variables

CONFIG = "configs/config.yaml"
estado = {}  # modelo cargado, nombre y versión


@asynccontextmanager
async def lifespan(app):
    """Al arrancar: carga el champion desde MLflow."""
    with open(CONFIG, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    nombre = cfg["registro"]["nombre_modelo"]
    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", cfg["mlflow"]["tracking_uri"]))
    version = MlflowClient().get_model_version_by_alias(nombre, "champion")
    estado.update(
        modelo=mlflow.sklearn.load_model(f"models:/{nombre}@champion"),
        nombre=nombre,
        version=str(version.version),
        run_id=version.run_id,
    )
    print(f"API lista con {nombre} versión {estado['version']}")
    yield
    estado.clear()


app = FastAPI(
    title="API de imputación de producción bruta (EAM)",
    description="Estima la producción bruta de un establecimiento industrial con el modelo champion de MLflow.",
    lifespan=lifespan,
)

# Valores de ejemplo: medianas de la EAM 2024 (no corresponden a un establecimiento real).
EJEMPLO = {
    "dpto": 11, "ciiu4": 1410,
    "PERSOCU": 24, "PERTOTAL": 44, "SALARPER": 696000, "PRESSPER": 161000, "REMUTEMP": 0,
    "PERAPREN": 1, "EELEC": 164000, "ACTIVFI": 3262000, "DEPRECIA": 157000, "INVEBRTA": 61000,
    "prod_ant": 7569000, "persocu_ant": 23, "salarper_ant": 654000, "eelec_ant": 163000,
}


class Establecimiento(BaseModel):
    """Datos de un establecimiento. Los nombres son los de las columnas de la EAM."""

    dpto: int = Field(description="Código del departamento")
    ciiu4: int = Field(description="Actividad económica (CIIU a 4 dígitos)")
    PERSOCU: float = Field(ge=0, description="Personal ocupado")
    PERTOTAL: float = Field(ge=0, description="Personal total")
    SALARPER: float = Field(ge=0, description="Salarios del personal permanente")
    PRESSPER: float = Field(ge=0, description="Prestaciones del personal permanente")
    REMUTEMP: float = Field(ge=0, description="Remuneración del personal temporal")
    PERAPREN: float = Field(ge=0, description="Aprendices")
    EELEC: float = Field(ge=0, description="Consumo de energía eléctrica")
    ACTIVFI: float = Field(description="Activos fijos")
    DEPRECIA: float = Field(description="Depreciación")
    INVEBRTA: float = Field(description="Inversión bruta (puede ser negativa)")
    # Año anterior: obligatorios si el champion se entrenó con rezago.
    prod_ant: float | None = Field(default=None, gt=0, description="Producción bruta del año anterior")
    persocu_ant: float | None = Field(default=None, ge=0, description="Personal ocupado del año anterior")
    salarper_ant: float | None = Field(default=None, ge=0, description="Salarios del año anterior")
    eelec_ant: float | None = Field(default=None, ge=0, description="Energía eléctrica del año anterior")

    model_config = {"json_schema_extra": {"examples": [EJEMPLO]}}


@app.get("/health")
def health():
    return {"estado": "ok", "modelo": estado["nombre"], "version": estado["version"]}


@app.post("/predict")
def predict(establecimiento: Establecimiento):
    modelo = estado["modelo"]
    datos = {k: v for k, v in establecimiento.model_dump().items() if v is not None}
    variables = calcular_variables(pd.DataFrame([datos]))
    faltan = [c for c in modelo.feature_names_in_ if c not in variables.columns]
    if faltan:
        raise HTTPException(
            status_code=422,
            detail=f"El modelo champion usa datos del año anterior: envía prod_ant, persocu_ant, "
                   f"salarper_ant y eelec_ant (faltan variables {faltan})",
        )
    pred_log = float(modelo.predict(variables[list(modelo.feature_names_in_)])[0])
    return {
        "produccion_estimada": round(float(np.exp(pred_log))),
        "log_produccion_estimada": round(pred_log, 4),
        "modelo": estado["nombre"],
        "version_modelo": estado["version"],
    }
