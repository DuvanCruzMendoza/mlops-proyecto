"""Pruebas del drift emulado (solo pandas): nivel, reproducibilidad, tamaño y etiquetas reales."""
import numpy as np
import pandas as pd
import pytest

from scripts.emulacion_drift import COLUMNAS_ESCALADAS, emular_drift


@pytest.fixture
def datos():
    rng = np.random.default_rng(0)
    pool = pd.DataFrame({
        "nordest": np.arange(400),
        "SALARPER": rng.uniform(1, 10, 400), "EELEC": rng.uniform(1, 10, 400),
        "PRODBIND": np.r_[rng.uniform(1, 100, 300), rng.uniform(200, 300, 100)],  # 100 "grandes"
    })
    return pool.iloc[:100].copy(), pool, 150.0  # lote base, pool, P75


def test_nivel_cero_devuelve_el_lote_intacto(datos):
    lote, pool, p75 = datos
    pd.testing.assert_frame_equal(emular_drift(lote, pool, 0.0, 42, p75), lote)


def test_no_modifica_los_argumentos(datos):
    lote, pool, p75 = datos
    antes_lote, antes_pool = lote.copy(), pool.copy()
    emular_drift(lote, pool, 0.6, 42, p75)
    pd.testing.assert_frame_equal(lote, antes_lote)
    pd.testing.assert_frame_equal(pool, antes_pool)


def test_misma_semilla_mismo_resultado(datos):
    lote, pool, p75 = datos
    pd.testing.assert_frame_equal(emular_drift(lote, pool, 0.3, 7, p75), emular_drift(lote, pool, 0.3, 7, p75))


def test_sube_la_tasa_del_objetivo_con_etiquetas_reales(datos):
    lote, pool, p75 = datos
    out = emular_drift(lote, pool, 0.6, 42, p75)
    assert len(out) == len(lote)
    assert out["nordest"].is_unique
    assert (out["PRODBIND"] > p75).mean() >= 0.6
    # cada etiqueta es la observada para ese establecimiento en el pool
    reales = pool.set_index("nordest")["PRODBIND"]
    assert np.allclose(out["PRODBIND"], reales.loc[out["nordest"]].to_numpy())


def test_escala_las_entradas_elegidas(datos):
    lote, pool, p75 = datos
    out = emular_drift(lote, pool, 0.5, 42, p75)
    reales = pool.set_index("nordest")
    for c in COLUMNAS_ESCALADAS:
        assert np.allclose(out[c], reales.loc[out["nordest"], c].to_numpy() * 1.5)


def test_nivel_alto_incluye_los_reemplazos_del_moderado(datos):
    lote, pool, p75 = datos
    nuevos = lambda nivel: set(emular_drift(lote, pool, nivel, 42, p75)["nordest"]) - set(lote["nordest"])
    assert nuevos(0.3) <= nuevos(0.6)


@pytest.mark.parametrize("nivel", [-0.1, 1.5, float("nan"), "0.3"])
def test_nivel_invalido_falla(datos, nivel):
    lote, pool, p75 = datos
    with pytest.raises(ValueError, match="nivel"):
        emular_drift(lote, pool, nivel, 42, p75)


def test_candidatos_insuficientes_falla(datos):
    lote, pool, p75 = datos
    with pytest.raises(ValueError, match="candidatos"):
        emular_drift(lote, pool, 1.0, 42, 1e9)
