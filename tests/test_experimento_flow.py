"""Pruebas del paso 4: variables, entrenamiento, registro en MLflow y flow completo.

Usan una base temporal de MLflow, así que no escriben en el servidor del curso.
Se ejecutan desde la raíz del repositorio:  poetry run pytest
"""
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import pytest
import yaml

from flows.experimento_flow import eam_imputacion, entrenar_y_registrar
from scripts.features import calcular_metricas, construir_variables
from scripts.load_data import COLUMNAS, NUMERICAS, OBJETIVO

CONFIG = "configs/config.yaml"


@pytest.fixture
def cfg(tmp_path, monkeypatch):
    """Configuración del proyecto con MLflow en una base temporal."""
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}")
    with open(CONFIG, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg["division"]["anio_prueba"] = 2022
    cfg["modelo"]["parametros"]["max_iter"] = 20  # rápido para las pruebas
    return cfg


@pytest.fixture
def datos():
    """Panel falso: 200 establecimientos durante 3 años, con la misma estructura que la EAM."""
    rng = np.random.default_rng(42)
    n = 200
    base = pd.DataFrame({
        "nordemp": np.arange(n), "nordest": np.arange(n),
        "dpto": rng.choice([5, 11, 76], n), "ciiu4": rng.choice([1011, 2395, 3290], n),
    })
    tamano = rng.lognormal(3, 1, n)
    anios = []
    for i, anio in enumerate([2020, 2021, 2022]):
        a = base.copy()
        a["periodo"] = anio
        for c in NUMERICAS:
            a[c] = (tamano * rng.lognormal(10, 0.3, n)).round()
        a[OBJETIVO] = (a["SALARPER"] * 5 * (1.1 ** i) * rng.lognormal(0, 0.2, n)).round() + 1
        anios.append(a)
    return pd.concat(anios, ignore_index=True)[COLUMNAS]


def test_rezago_usa_el_anio_anterior_del_mismo_establecimiento(datos):
    out, numericas, _ = construir_variables(datos, usar_rezago=True)
    assert set(out["periodo"]) == {2021, 2022}  # 2020 no tiene año anterior
    fila = out[(out["nordest"] == 7) & (out["periodo"] == 2022)].iloc[0]
    prod_2021 = datos[(datos["nordest"] == 7) & (datos["periodo"] == 2021)][OBJETIVO].iloc[0]
    assert fila["l_prod_ant"] == pytest.approx(np.log(prod_2021))
    assert "l_prod_ant" in numericas


def test_sin_rezago_conserva_todas_las_filas(datos):
    out, numericas, _ = construir_variables(datos, usar_rezago=False)
    assert len(out) == len(datos)
    assert "l_prod_ant" not in numericas


def test_metricas_perfectas_cuando_la_prediccion_es_exacta():
    m = calcular_metricas([1.0, 2.0, 3.0], [1.0, 2.0, 3.0])
    assert m["mae_log"] == 0 and m["mdape"] == 0 and m["pct_error_menor_20"] == 1


@pytest.mark.parametrize("usar_rezago", [True, False])
def test_entrenar_y_registrar_crea_un_run(cfg, datos, usar_rezago):
    cfg["variables"]["usar_rezago"] = usar_rezago
    run_id, metricas = entrenar_y_registrar(datos, cfg)

    assert {"mae_log", "rmse_log", "mdape", "pct_error_menor_20", "linea_base_mae_log"} <= set(metricas)
    run = mlflow.get_run(run_id)
    assert run.data.params["modelo"] == cfg["modelo"]["tipo"]
    assert run.data.params["filas_datos"] == str(len(datos))
    assert run.data.metrics["mae_log"] == pytest.approx(metricas["mae_log"])


def test_modelo_no_soportado_falla(cfg, datos):
    cfg["modelo"]["tipo"] = "xgboost"
    with pytest.raises(ValueError, match="no soportado"):
        entrenar_y_registrar(datos, cfg)


@pytest.mark.skipif(not Path("data/processed/eam_2020_2024.csv").exists(),
                    reason="Falta data/processed/eam_2020_2024.csv (ejecuta scripts/load_data.py)")
def test_el_flow_completo_termina_y_registra_el_run(cfg):
    run_id = eam_imputacion()
    assert mlflow.get_run(run_id).info.status == "FINISHED"
