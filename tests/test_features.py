"""Pruebas de scripts/features.py: variables, rezago, selección de features y métricas."""
import numpy as np
import pytest

from scripts.features import REZAGO, calcular_metricas, construir_variables, resolver_features
from scripts.load_data import OBJETIVO
from tests.conftest import datos_falsos


def test_rezago_usa_el_anio_anterior_del_mismo_establecimiento():
    datos = datos_falsos()
    out = construir_variables(datos, usar_rezago=True)
    assert set(out["periodo"]) == {2022, 2023, 2024}  # el primer año no tiene año anterior
    fila = out[(out["nordest"] == 7) & (out["periodo"] == 2023)].iloc[0]
    prod_2022 = datos[(datos["nordest"] == 7) & (datos["periodo"] == 2022)][OBJETIVO].iloc[0]
    assert fila["l_prod_ant"] == pytest.approx(np.log(prod_2022))


def test_sin_rezago_conserva_todas_las_filas():
    datos = datos_falsos()
    assert len(construir_variables(datos, usar_rezago=False)) == len(datos)


@pytest.mark.parametrize("seleccion, n_num, n_cat", [("ALL", 14, 3), ("NUMERIC", 14, 0), ("CUSTOM", 5, 2)])
def test_selecciones_de_features(seleccion, n_num, n_cat):
    numericas, categoricas = resolver_features({"seleccion": seleccion, "custom": ["PERSOCU", "ciiu4"]})
    assert (len(numericas), len(categoricas)) == (n_num, n_cat)
    assert set(REZAGO) <= set(numericas)


def test_sin_rezago_no_agrega_variables_del_anio_anterior():
    numericas, _ = resolver_features({"seleccion": "NUMERIC"}, usar_rezago=False)
    assert not set(REZAGO) & set(numericas)


@pytest.mark.parametrize("cfg", [{"seleccion": "OTRA"}, {"seleccion": "CUSTOM", "custom": ["NO_EXISTE"]},
                                 {"seleccion": "CUSTOM", "custom": []}])
def test_selecciones_invalidas_fallan(cfg):
    with pytest.raises(ValueError):
        resolver_features(cfg)


def test_metricas_perfectas_cuando_la_prediccion_es_exacta():
    m = calcular_metricas([1.0, 2.0, 3.0], [1.0, 2.0, 3.0])
    assert m["mae_log"] == 0 and m["mdape"] == 0 and m["pct_error_menor_20"] == 1
