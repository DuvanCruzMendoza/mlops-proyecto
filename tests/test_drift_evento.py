"""Pruebas del drift del objetivo PRODBIND (sin MLflow ni Prefect)."""
import numpy as np
import pandas as pd
import pytest

from scripts.drift_evento import drift_de_evento, validar_etiquetas


def _y(valores):
    return pd.DataFrame({"PRODBIND": valores})


def test_misma_distribucion_no_tiene_drift():
    ref = _y(np.arange(1, 1001))
    r = drift_de_evento(ref, ref, umbral_pp=10)
    assert r["ks_estadistico"] == 0
    assert r["diferencia_pp"] == 0
    assert r["tasa_referencia"] == pytest.approx(0.25, abs=0.01)
    assert not r["drift"]


def test_lote_con_mas_establecimientos_grandes_tiene_drift():
    ref = _y(np.arange(1, 1001))               # P75 = 750.25
    lote = _y(np.r_[np.arange(1, 51), np.arange(800, 850)])  # 50 % sobre el P75
    r = drift_de_evento(ref, lote, umbral_pp=10)
    assert r["diferencia_pp"] == pytest.approx(25, abs=1)
    assert r["ks_estadistico"] > 0.2
    assert r["drift"]


@pytest.mark.parametrize("lote, mensaje", [
    (pd.DataFrame({"otra": [1, 2]}), "no contiene la columna PRODBIND"),
    (_y([1.0, None]), "vacías"),
    (_y([1.0, np.inf]), "infinitas"),
    (_y([1.0, 0.0]), "menores o iguales a cero"),
    (_y([]), "vacío"),
])
def test_rechaza_lotes_sin_etiquetas_validas(lote, mensaje):
    with pytest.raises(ValueError, match=mensaje):
        validar_etiquetas(lote)
