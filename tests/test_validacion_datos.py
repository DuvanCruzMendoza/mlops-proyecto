"""Pruebas de datos: esquema, rangos y nulos (con datos falsos y, si existen, con los reales)."""
import numpy as np
import pytest

from scripts.validacion_datos import validar_datos
from tests.conftest import datos_falsos
from tests.test_data import df_real, requiere_datos  # noqa: F401 (fixture con los Excel reales)


@pytest.fixture
def validos():
    df = datos_falsos()
    df["PERTOTAL"] = df["PERSOCU"] + 1
    return df


def test_datos_validos_no_tienen_problemas(validos):
    assert validar_datos(validos, anios=[2021, 2022, 2023, 2024]) == []


@pytest.mark.parametrize("cambio, mensaje", [
    (lambda d: d.drop(columns=["EELEC"]), "Faltan columnas"),
    (lambda d: d.assign(SALARPER=np.where(d.index == 0, np.nan, d["SALARPER"])), "nulos"),
    (lambda d: d.assign(PERSOCU=np.where(d.index == 0, -1, d["PERSOCU"])), "negativos"),
    (lambda d: d.assign(PRODBIND=np.where(d.index == 0, 0, d["PRODBIND"])), "positivo"),
    (lambda d: d.assign(PERSOCU=d["PERTOTAL"] + 5), "PERTOTAL"),
    (lambda d: d.assign(periodo=1999), "Años no esperados"),
])
def test_detecta_problemas(validos, cambio, mensaje):
    problemas = validar_datos(cambio(validos.copy()), anios=[2021, 2022, 2023, 2024])
    assert any(mensaje in p for p in problemas)


@requiere_datos
def test_la_eam_real_pasa_las_validaciones(df_real):
    assert validar_datos(df_real, anios=[2020, 2021, 2022, 2023, 2024]) == []
