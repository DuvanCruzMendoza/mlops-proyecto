"""Pruebas del drift de features con Evidently (sin MLflow ni Prefect)."""
import numpy as np
import pandas as pd
import pytest

from scripts.drift_features import calcular_drift_features, hay_alerta

NUM = ["x1", "x2", "x3", "x4"]
CAT = ["dpto"]


def _datos(n, desplazar=0.0, semilla=0):
    rng = np.random.default_rng(semilla)
    df = pd.DataFrame({c: rng.normal(0, 1, n) for c in NUM})
    df[["x1", "x2", "x3"]] += desplazar
    df["dpto"] = rng.choice([5, 11, 76], n)
    return df


def test_lote_estable_no_alerta_y_desplazado_si(tmp_path):
    ref = _datos(2000)
    estable = calcular_drift_features(ref, _datos(500, semilla=1), NUM, CAT, 0.5, tmp_path / "a.html")
    movido = calcular_drift_features(ref, _datos(500, desplazar=2, semilla=1), NUM, CAT, 0.5, tmp_path / "b.html")

    assert estable["fraccion_drift"] < movido["fraccion_drift"]
    assert not estable["alerta"] and movido["alerta"]
    assert set(movido["columnas_con_drift"]) >= {"x1", "x2", "x3"}
    assert (tmp_path / "a.html").exists() and (tmp_path / "b.html").exists()


def test_columna_faltante_se_nombra():
    with pytest.raises(ValueError, match="x4"):
        calcular_drift_features(_datos(100), _datos(100).drop(columns=["x4"]), NUM, CAT, 0.5)


@pytest.mark.parametrize("fraccion, esperado", [(0.49, False), (0.50, False), (0.51, True)])
def test_alerta_solo_si_supera_el_umbral(fraccion, esperado):
    assert hay_alerta(fraccion, 0.50) is esperado
