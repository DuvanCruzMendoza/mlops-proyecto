"""Actividad 3: pipeline de drift del evento (objetivo PRODBIND).

Compara la producción REAL del lote (las etiquetas pueden llegar con retraso) con la del
entrenamiento. Si el lote no trae etiquetas, falla con un mensaje claro. Guarda target.json en
data/monitoreo/reportes/nivel_X/.

Ejecución (desde la raíz; sin --nivel usa inferencia.nivel_drift de config.yaml):
    poetry run python flows/drift_evento_flow.py --nivel 0.6
    poetry run python flows/drift_evento_flow.py --lote ruta/a/un_lote.csv
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import yaml
from prefect import flow, get_run_logger

from scripts.drift_evento import drift_de_evento
from scripts.load_data import cargar_procesados
from scripts.monitoreo import carpeta_reportes, construir_referencia, ruta_lote

CONFIG = "configs/config.yaml"


@flow(name="drift-evento-eam", log_prints=True)
def drift_evento_flow(nivel: float | None = None, lote: str | None = None, config: str = CONFIG):
    logger = get_run_logger()
    with open(config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    nivel = cfg["inferencia"]["nivel_drift"] if nivel is None else nivel
    ruta = Path(lote) if lote else ruta_lote(cfg, nivel)
    datos_lote = pd.read_csv(ruta)
    referencia = construir_referencia(cargar_procesados(cfg["datos"]), cfg)

    r = drift_de_evento(referencia, datos_lote, cfg["monitoreo"]["umbral_evento_pp"])  # falla si no hay etiquetas
    r.update(nivel=nivel, lote=str(ruta))
    carpeta = carpeta_reportes(cfg, nivel) if not lote else ruta.parent
    carpeta.mkdir(parents=True, exist_ok=True)
    (carpeta / "target.json").write_text(json.dumps(r, indent=2, ensure_ascii=False), encoding="utf-8")

    msg = (f"Nivel {nivel}: establecimientos sobre el P75 {r['tasa_referencia']:.1%} en entrenamiento y "
           f"{r['tasa_lote']:.1%} en el lote ({r['diferencia_pp']:+.1f} pp, KS {r['ks_estadistico']:.2f})")
    if r["drift"]:
        logger.warning(f"ALERTA DE DRIFT DEL OBJETIVO. {msg}")
    else:
        logger.info(f"Sin drift del objetivo. {msg}")
    return r


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--nivel", type=float, default=None)
    parser.add_argument("--lote", default=None, help="CSV de un lote concreto (p. ej. uno sin etiquetas)")
    args = parser.parse_args()
    drift_evento_flow(nivel=args.nivel, lote=args.lote)
