"""Pruebas de scripts/load_data.py. Se ejecutan desde la raíz del repositorio:

    poetry run pytest

Las pruebas con los datos reales se saltan si los Excel no están en data/raw.
"""
from pathlib import Path

import pandas as pd
import pytest

from scripts.load_data import COLUMNAS, OBJETIVO, cargar_datos, leer_config

CFG = leer_config()["datos"]
HAY_DATOS = all(
    (Path(CFG["carpeta"]) / CFG["patron_archivo"].format(anio=a)).exists() for a in CFG["anios"]
)
requiere_datos = pytest.mark.skipif(not HAY_DATOS, reason="Faltan los Excel de la EAM en data/raw")


@pytest.fixture(scope="module")
def df_real():
    return cargar_datos()


@requiere_datos
def test_carga_todos_los_anios_con_las_columnas_esperadas(df_real):
    assert list(df_real.columns) == COLUMNAS
    assert sorted(df_real["periodo"].unique()) == CFG["anios"]
    assert len(df_real) > 30000


@requiere_datos
def test_objetivo_positivo_y_un_registro_por_establecimiento_y_anio(df_real):
    assert (df_real[OBJETIVO] > 0).all()
    assert not df_real.duplicated(["nordest", "periodo"]).any()


# ---- Pruebas con un Excel pequeño de mentira (no necesitan los datos reales) ----

def _excel_falso(carpeta, anio, columnas=COLUMNAS, prod=(10, 20)):
    df = pd.DataFrame({c: [1, 2] for c in columnas})
    df["periodo"] = anio
    df["nordest"] = [101, 102]
    if OBJETIVO in columnas:
        df[OBJETIVO] = list(prod)
    df["otra_columna"] = 0  # columnas extra se ignoran
    df.to_excel(carpeta / f"EAM_ANONIMIZADA_{anio}.xlsx", index=False)


def test_une_varios_anios(tmp_path):
    _excel_falso(tmp_path, 2020)
    _excel_falso(tmp_path, 2021)
    df = cargar_datos(tmp_path, [2020, 2021], "EAM_ANONIMIZADA_{anio}.xlsx")
    assert list(df.columns) == COLUMNAS
    assert len(df) == 4


def test_columna_faltante_falla(tmp_path):
    _excel_falso(tmp_path, 2020, columnas=[c for c in COLUMNAS if c != "EELEC"])
    with pytest.raises(ValueError, match="EELEC"):
        cargar_datos(tmp_path, [2020], "EAM_ANONIMIZADA_{anio}.xlsx")


def test_archivo_faltante_falla(tmp_path):
    with pytest.raises(FileNotFoundError):
        cargar_datos(tmp_path, [2020], "EAM_ANONIMIZADA_{anio}.xlsx")


def test_descarta_produccion_no_positiva(tmp_path):
    _excel_falso(tmp_path, 2020, prod=(0, 50))
    df = cargar_datos(tmp_path, [2020], "EAM_ANONIMIZADA_{anio}.xlsx")
    assert len(df) == 1 and df[OBJETIVO].iloc[0] == 50
