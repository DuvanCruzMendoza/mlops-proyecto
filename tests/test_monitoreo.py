"""Pruebas de las utilidades compartidas de monitoreo (sin MLflow ni Prefect)."""
import yaml

from scripts.monitoreo import carpeta_reportes, columnas_modelo, construir_referencia, ruta_lote
from tests.conftest import datos_falsos


def _cfg(entorno):
    with open(entorno["config"], encoding="utf-8") as f:
        return yaml.safe_load(f)


def test_referencia_solo_tiene_entrenamiento_y_columnas_del_modelo(entorno):
    cfg = _cfg(entorno)
    df = datos_falsos()
    ref = construir_referencia(df, cfg, entorno["features"])
    numericas, categoricas = columnas_modelo(cfg, entorno["features"])

    assert list(ref.columns) == numericas + categoricas + ["PRODBIND"]
    assert len(numericas) + len(categoricas) == 17
    # con rezago, 2021 no tiene año anterior en los datos falsos: quedan 2022 y 2023 (200 filas cada uno)
    assert len(ref) == 400
    assert ref["PRODBIND"].gt(0).all()


def test_rutas_por_nivel(entorno):
    cfg = _cfg(entorno)
    assert ruta_lote(cfg, 0.6).name == "lote_nivel_0.6.csv"
    assert carpeta_reportes(cfg, 0).name == "nivel_0.0"
