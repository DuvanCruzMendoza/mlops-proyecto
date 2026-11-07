"""Pruebas del pipeline de entrenamiento: runs en MLflow y reglas para ser champion."""
import mlflow
import pytest
from mlflow.tracking import MlflowClient

from flows.entrenamiento_flow import crear_modelo, entrenamiento_flow, leer_yaml
from tests.conftest import editar_yaml


def _champion(cfg_path):
    nombre = leer_yaml(cfg_path)["registro"]["nombre_modelo"]
    return MlflowClient().get_model_version_by_alias(nombre, "champion")


def test_primer_entrenamiento_crea_champion_y_runs_con_tag(entorno):
    r = entrenamiento_flow(entorno["config"], entorno["features"])
    assert r["version_champion"] == "1"
    assert _champion(entorno["config"]).run_id == r["mejor_run_id"]

    runs = mlflow.search_runs(experiment_names=[leer_yaml(entorno["config"])["mlflow"]["experimento"]])
    assert len(runs) == 2  # un run por candidato
    assert set(runs["tags.fase"]) == {"entrenamiento"}
    assert set(runs["params.features"]) == {"ALL"}
    assert set(runs["params.n_features"]) == {"17"}


def test_repetir_el_mismo_entrenamiento_no_cambia_el_champion(entorno):
    entrenamiento_flow(entorno["config"], entorno["features"])
    r = entrenamiento_flow(entorno["config"], entorno["features"])  # mismo MAE: no mejora
    assert r["version_champion"] is None
    assert str(_champion(entorno["config"]).version) == "1"


def test_un_modelo_mejor_reemplaza_al_champion(entorno):
    # Primero un modelo débil (pocas variables y sin rezago), después el completo.
    editar_yaml(entorno["features"], {"seleccion": "CUSTOM"})
    editar_yaml(entorno["config"], {"variables": {"usar_rezago": False}})
    r1 = entrenamiento_flow(entorno["config"], entorno["features"])
    editar_yaml(entorno["features"], {"seleccion": "ALL"})
    editar_yaml(entorno["config"], {"variables": {"usar_rezago": True}})
    r2 = entrenamiento_flow(entorno["config"], entorno["features"])

    assert r2["mae_log"] < r1["mae_log"]
    assert (r1["version_champion"], r2["version_champion"]) == ("1", "2")
    assert str(_champion(entorno["config"]).version) == "2"


def test_mae_sobre_el_maximo_no_registra_champion(entorno):
    editar_yaml(entorno["config"], {"registro": {"max_mae_log": 0.0001}})
    r = entrenamiento_flow(entorno["config"], entorno["features"])
    assert r["version_champion"] is None and "supera el máximo" in r["decision"]


def test_modelo_no_soportado_falla():
    with pytest.raises(ValueError, match="no soportado"):
        crear_modelo("xgboost", {}, ["a"], [])
