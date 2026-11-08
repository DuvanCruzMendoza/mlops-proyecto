"""Actividad 2: despliegue de las dos pipelines en Prefect.

- entrenamiento-eam/entrenamiento-manual: sin horario. Se lanza a mano desde la UI de Prefect.
- inferencia-eam/inferencia-diaria: todos los días a las 06:00, hora de Colombia.

Ejecución (desde la raíz del repositorio, con PREFECT_API_URL y MLFLOW_TRACKING_URI definidas):
    poetry run python flows/desplegar.py

Deja esta terminal abierta: mientras esté abierta, Prefect puede lanzar las ejecuciones
(las programadas y las que se lancen desde la UI). Para detenerla: Ctrl + C.
"""
from prefect import serve
from prefect.schedules import Cron

from entrenamiento_flow import entrenamiento_flow
from inferencia_flow import inferencia_flow

if __name__ == "__main__":
    despliegue_entrenamiento = entrenamiento_flow.to_deployment(
        name="entrenamiento-manual",
        description="Entrena los candidatos y promueve a champion el mejor si mejora. Se lanza a mano.",
    )
    despliegue_inferencia = inferencia_flow.to_deployment(
        name="inferencia-diaria",
        description="Imputa un lote de establecimientos con el modelo champion. Todos los días a las 06:00.",
        schedule=Cron("0 6 * * *", timezone="America/Bogota"),
    )
    serve(despliegue_entrenamiento, despliegue_inferencia)
