"""Actividad 3: pipeline de drift de features (Evidently).

Compara el lote que dejó la inferencia (data/monitoreo/lotes/lote_nivel_X.csv) con los datos de
entrenamiento. Guarda features.html y features.json en data/monitoreo/reportes/nivel_X/ y emite una
ALERTA en los logs si la fracción de variables con drift supera monitoreo.umbral_features.

Ejecución (desde la raíz; sin --nivel usa inferencia.nivel_drift de config.yaml):
    poetry run python flows/drift_features_flow.py --nivel 0.6
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import yaml
from prefect import flow, get_run_logger

from scripts.drift_features import calcular_drift_features
from scripts.load_data import cargar_procesados
from scripts.monitoreo import carpeta_reportes, columnas_modelo, construir_referencia, ruta_lote

CONFIG = "configs/config.yaml"


@flow(name="drift-features-eam", log_prints=True)
def drift_features_flow(nivel: float | None = None, config: str = CONFIG):
    logger = get_run_logger()
    with open(config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    nivel = cfg["inferencia"]["nivel_drift"] if nivel is None else nivel
    lote = pd.read_csv(ruta_lote(cfg, nivel))
    referencia = construir_referencia(cargar_procesados(cfg["datos"]), cfg)
    numericas, categoricas = columnas_modelo(cfg)
    carpeta = carpeta_reportes(cfg, nivel)

    r = calcular_drift_features(referencia, lote, numericas, categoricas,
                                cfg["monitoreo"]["umbral_features"], carpeta / "features.html")
    r.update(nivel=nivel, filas_lote=len(lote), filas_referencia=len(referencia))
    (carpeta / "features.json").write_text(json.dumps(r, indent=2, ensure_ascii=False), encoding="utf-8")

    msg = (f"Nivel {nivel}: {len(r['columnas_con_drift'])} de {r['columnas_evaluadas']} variables con drift "
           f"(fracción {r['fraccion_drift']:.2f}, umbral {r['umbral']})")
    if r["alerta"]:
        logger.warning(f"ALERTA DE DRIFT DE FEATURES. {msg}. Columnas: {r['columnas_con_drift']}")
    else:
        logger.info(f"Sin alerta. {msg}")
    return r


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--nivel", type=float, default=None)
    drift_features_flow(nivel=parser.parse_args().nivel)
