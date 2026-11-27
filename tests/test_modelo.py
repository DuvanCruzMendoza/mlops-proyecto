"""Pruebas del modelo (sin MLflow): se entrena, predice valores válidos y supera la línea base."""
import numpy as np
import pytest

from flows.entrenamiento_flow import crear_modelo
from scripts.features import calcular_metricas, construir_variables, dividir_por_anio, linea_base, resolver_features
from tests.conftest import datos_falsos


@pytest.fixture
def particion():
    train, test = dividir_por_anio(construir_variables(datos_falsos(), usar_rezago=True), 2024)
    numericas, categoricas = resolver_features({"seleccion": "ALL"}, usar_rezago=True)
    return train, test, numericas, categoricas


@pytest.mark.parametrize("tipo, parametros", [("ridge", {"alpha": 1.0}),
                                              ("hist_gradient_boosting", {"max_iter": 50, "random_state": 42})])
def test_modelo_predice_valores_finitos_con_las_columnas_esperadas(particion, tipo, parametros):
    train, test, numericas, categoricas = particion
    modelo = crear_modelo(tipo, parametros, numericas, categoricas).fit(train[numericas + categoricas], train["y_log"])
    pred = modelo.predict(test[numericas + categoricas])
    assert list(modelo.feature_names_in_) == numericas + categoricas
    assert len(pred) == len(test) and np.isfinite(pred).all()


def test_ridge_supera_a_la_linea_base_en_datos_con_senal(particion):
    train, test, numericas, categoricas = particion
    modelo = crear_modelo("ridge", {"alpha": 1.0}, numericas, categoricas).fit(train[numericas + categoricas], train["y_log"])
    mae_modelo = calcular_metricas(test["y_log"], modelo.predict(test[numericas + categoricas]))["mae_log"]
    mae_base = calcular_metricas(test["y_log"], linea_base(train, test, True))["mae_log"]
    assert mae_modelo < mae_base


def test_metricas_de_un_modelo_perfecto():
    m = calcular_metricas([10.0, 12.0], [10.0, 12.0])
    assert m["mae_log"] == 0 and m["mdape"] == 0 and m["pct_error_menor_20"] == 1
