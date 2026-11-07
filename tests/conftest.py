"""Fixtures compartidas: datos falsos con la estructura de la EAM y configuración temporal.

Las pruebas de entrenamiento, inferencia y API usan un MLflow temporal (SQLite en una
carpeta de pytest), así que no escriben en el servidor del curso ni necesitan los Excel.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from scripts.load_data import COLUMNAS, NUMERICAS, OBJETIVO


def datos_falsos(n=200, anios=(2021, 2022, 2023, 2024), semilla=42):
    """Panel falso: n establecimientos durante varios años, con las columnas de load_data.py."""
    rng = np.random.default_rng(semilla)
    base = pd.DataFrame({
        "nordemp": np.arange(n), "nordest": np.arange(n),
        "dpto": rng.choice([5, 11, 76], n), "ciiu4": rng.choice([1011, 2395, 3290], n),
    })
    tamano = rng.lognormal(3, 1, n)
    partes = []
    for i, anio in enumerate(anios):
        a = base.copy()
        a["periodo"] = anio
        for c in NUMERICAS:
            a[c] = (tamano * rng.lognormal(10, 0.3, n)).round()
        a[OBJETIVO] = (a["SALARPER"] * 5 * (1.05 ** i) * rng.lognormal(0, 0.1, n)).round() + 1
        partes.append(a)
    return pd.concat(partes, ignore_index=True)[COLUMNAS]


@pytest.fixture
def entorno(tmp_path, monkeypatch):
    """Escribe datos falsos y YAML temporales. Devuelve las rutas de config y features."""
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}")
    csv = tmp_path / "eam.csv"
    datos_falsos().to_csv(csv, index=False)

    with open("configs/config.yaml", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg["datos"]["salida"] = csv.as_posix()
    cfg["candidatos"] = [
        {"nombre_run": "ridge", "tipo": "ridge", "parametros": {"alpha": 1.0}},
        {"nombre_run": "hgb", "tipo": "hist_gradient_boosting",
         "parametros": {"max_iter": 30, "random_state": 42}},
    ]
    cfg["registro"]["max_mae_log"] = 1.0
    cfg["registro"]["exigir_superar_linea_base"] = False
    cfg["inferencia"].update({"tamano_lote": 50, "semilla": 1, "carpeta_salida": (tmp_path / "pred").as_posix()})

    rutas = {"config": tmp_path / "config.yaml", "features": tmp_path / "features.yaml", "dir": tmp_path}
    rutas["config"].write_text(yaml.safe_dump(cfg, allow_unicode=True), encoding="utf-8")
    rutas["features"].write_text(yaml.safe_dump({"seleccion": "ALL", "custom": ["PERSOCU", "ciiu4"]}), encoding="utf-8")
    return {k: (v.as_posix() if isinstance(v, Path) else v) for k, v in rutas.items()}


def editar_yaml(ruta, cambios):
    """Actualiza claves de primer nivel (o anidadas con un dict) de un YAML temporal."""
    with open(ruta, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    for clave, valor in cambios.items():
        if isinstance(valor, dict) and isinstance(cfg.get(clave), dict):
            cfg[clave].update(valor)
        else:
            cfg[clave] = valor
    with open(ruta, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, allow_unicode=True)
