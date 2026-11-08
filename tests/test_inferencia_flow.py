"""Pruebas del pipeline de inferencia: usa el champion, guarda el CSV y registra el run."""
from datetime import date
from pathlib import Path

import mlflow
import pandas as pd
import pytest
from mlflow.exceptions import MlflowException

from flows.entrenamiento_flow import entrenamiento_flow
from flows.inferencia_flow import inferencia_flow
from tests.conftest import editar_yaml


def test_inferencia_usa_el_champion_y_guarda_el_csv_del_dia(entorno):
    entrenamiento = entrenamiento_flow(entorno["config"], entorno["features"])
    r = inferencia_flow(entorno["config"])

    assert r["version"] == entrenamiento["version_champion"]
    assert Path(r["ruta"]).name == f"predicciones_{date.today().isoformat()}.csv"
    pred = pd.read_csv(r["ruta"])
    assert len(pred) == 50
    assert {"nordest", "produccion_estimada", "produccion_reportada", "alerta"} <= set(pred.columns)
    assert (pred["produccion_estimada"] > 0).all()

    run = mlflow.get_run(r["run_id"])
    assert run.data.tags["fase"] == "inferencia"
    assert run.data.tags["modelo_run_id"] == entrenamiento["mejor_run_id"]


def test_inferencia_sigue_al_nuevo_champion_sin_cambiar_codigo(entorno):
    editar_yaml(entorno["features"], {"seleccion": "CUSTOM"})
    editar_yaml(entorno["config"], {"variables": {"usar_rezago": False}})
    entrenamiento_flow(entorno["config"], entorno["features"])
    assert inferencia_flow(entorno["config"])["version"] == "1"

    editar_yaml(entorno["features"], {"seleccion": "ALL"})
    editar_yaml(entorno["config"], {"variables": {"usar_rezago": True}})
    entrenamiento_flow(entorno["config"], entorno["features"])
    assert inferencia_flow(entorno["config"])["version"] == "2"


def test_inferencia_sin_champion_falla(entorno):
    with pytest.raises(MlflowException):
        inferencia_flow(entorno["config"])
