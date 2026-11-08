"""Pruebas de la API: carga el champion, predice y cambia de versión al reiniciarse."""
import pytest
from fastapi.testclient import TestClient

from api.main import EJEMPLO, app
from flows.entrenamiento_flow import entrenamiento_flow
from tests.conftest import editar_yaml


@pytest.fixture
def config_api(entorno, monkeypatch):
    """La API lee configs/config.yaml: se le apunta al YAML temporal de la prueba."""
    monkeypatch.setattr("api.main.CONFIG", entorno["config"])
    return entorno


def test_health_y_prediccion_con_el_champion(config_api):
    r = entrenamiento_flow(config_api["config"], config_api["features"])
    with TestClient(app) as cliente:  # el `with` ejecuta el arranque (carga del champion)
        assert cliente.get("/health").json()["version"] == r["version_champion"]
        resp = cliente.post("/predict", json=EJEMPLO)
    assert resp.status_code == 200
    cuerpo = resp.json()
    assert cuerpo["produccion_estimada"] > 0
    assert cuerpo["version_modelo"] == r["version_champion"]


def test_sin_datos_del_anio_anterior_responde_422(config_api):
    entrenamiento_flow(config_api["config"], config_api["features"])  # champion con rezago
    sin_rezago = {k: v for k, v in EJEMPLO.items() if not k.endswith("_ant")}
    with TestClient(app) as cliente:
        resp = cliente.post("/predict", json=sin_rezago)
    assert resp.status_code == 422
    assert "año anterior" in resp.json()["detail"]


def test_datos_invalidos_responden_422(config_api):
    entrenamiento_flow(config_api["config"], config_api["features"])
    with TestClient(app) as cliente:
        assert cliente.post("/predict", json={**EJEMPLO, "PERSOCU": -5}).status_code == 422


def test_al_reiniciar_la_api_usa_el_nuevo_champion(config_api):
    """Criterio de éxito: nueva versión champion → reiniciar la API → responde con ella."""
    editar_yaml(config_api["features"], {"seleccion": "CUSTOM"})
    editar_yaml(config_api["config"], {"variables": {"usar_rezago": False}})
    entrenamiento_flow(config_api["config"], config_api["features"])
    with TestClient(app) as cliente:
        assert cliente.post("/predict", json=EJEMPLO).json()["version_modelo"] == "1"

    editar_yaml(config_api["features"], {"seleccion": "ALL"})
    editar_yaml(config_api["config"], {"variables": {"usar_rezago": True}})
    entrenamiento_flow(config_api["config"], config_api["features"])
    with TestClient(app) as cliente:  # "reinicio": nuevo arranque de la API
        assert cliente.post("/predict", json=EJEMPLO).json()["version_modelo"] == "2"
