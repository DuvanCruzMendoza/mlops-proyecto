
# Proyecto MLOps

Proyecto académico desarrollado en la asignatura **MLOps** de la Maestría en Ciencia de Datos.

## Integrantes

- Duvan Cruz
- Natalia Mesa
- Paula Laiton

## Objetivo

Construir una base reproducible para el desarrollo de un proyecto de machine learning, utilizando Git y GitHub para el control de versiones, Poetry para la gestión de dependencias, Prefect para la orquestación de flujos y MLflow para el registro de experimentos.

## Caso de uso

**Imputación de la producción bruta de la industria manufacturera.** A partir de la Encuesta Anual Manufacturera (EAM) del DANE, se entrena un modelo de regresión que estima la producción bruta (`PRODBIND`) de un establecimiento con variables que suelen estar disponibles: personal ocupado, salarios, energía eléctrica, activos fijos, actividad económica (CIIU), departamento y, cuando existe, la producción del año anterior.

El modelo sirve para dos tareas propias de una encuesta económica:

- **Imputar** la producción cuando un establecimiento no la reporta.
- **Validar** los valores reportados, marcando los que se alejan mucho del valor esperado.

Se entrena con 2020-2023 y se evalúa con 2024, para simular la llegada de un año nuevo de datos.

**Actividad 2.** El sistema tiene una pipeline de entrenamiento que promueve el mejor modelo a `champion` en el Model Registry de MLflow, una pipeline de inferencia diaria que usa ese champion y una API que lo sirve.

## Requisitos

- Python 3.12 a 3.14
- Poetry 2.x
- Git
- Docker Desktop (si se utiliza la alternativa de servicios con Docker)

## Estructura del proyecto

```text
mlops-proyecto/
├── configs/
│   ├── config.yaml              # datos, división, candidatos, registro, inferencia y MLflow
│   └── features.yaml            # selección de variables: ALL, NUMERIC o CUSTOM
├── data/
│   ├── README.md                # fuente de los datos y cómo descargarlos
│   ├── raw/                     # Excel de la EAM 2020-2024 (no se versiona)
│   ├── processed/               # CSV unificado que genera load_data.py (no se versiona)
│   └── predicciones/            # CSV diarios de la inferencia (no se versiona)
├── docs/
├── api/
│   └── main.py                  # API de FastAPI que sirve el modelo champion
├── flows/
│   ├── prueba_flow.py           # flow mínimo para comprobar Prefect y MLflow (actividad 1)
│   ├── entrenamiento_flow.py    # ingesta → entrenamiento → evaluación → registro (champion)
│   ├── inferencia_flow.py       # imputa un lote diario con el modelo champion
│   └── desplegar.py             # despliegues en Prefect: entrenamiento manual e inferencia diaria
├── notebooks/
├── scripts/
│   ├── load_data.py             # carga, valida y unifica la EAM
│   └── features.py              # variables, selección de features, línea base y métricas
├── tests/
│   ├── conftest.py              # datos falsos y configuración temporal para las pruebas
│   ├── test_data.py             # carga de datos
│   ├── test_features.py         # variables y selección de features
│   ├── test_entrenamiento_flow.py
│   ├── test_inferencia_flow.py
│   └── test_api.py
├── .gitignore
├── README.md
├── pyproject.toml
└── poetry.lock
```

Las carpetas contienen archivos `__init__.py` para permitir su versionamiento en Git, de acuerdo con las instrucciones de la actividad.

## Instalación del entorno

### 1. Clonar el repositorio

Desde una terminal, ejecutar:

```bash
git clone https://github.com/DuvanCruzMendoza/mlops-proyecto.git
cd mlops-proyecto
```

### 2. Instalar las dependencias

Con Python y Poetry instalados, ejecutar desde la raíz del repositorio:

```bash
poetry install
```

Poetry instalará las dependencias definidas en `pyproject.toml`, utilizando las versiones resueltas registradas en `poetry.lock`.

### 3. Verificar la instalación

Ejecutar:

```bash
poetry run python -c "import prefect, mlflow, sklearn; print('ok')"
```

Si las librerías se importan correctamente, la terminal mostrará:

```text
ok
```

## Datos

**Fuente:** Encuesta Anual Manufacturera (EAM) 2020-2024, microdatos anonimizados del DANE, publicados en el Archivo Nacional de Datos (ANDA): https://microdatos.dane.gov.co

Los archivos Excel no se versionan en Git. Cada integrante los descarga del ANDA y los copia en `data/raw/` con los nombres `EAM_ANONIMIZADA_2020.xlsx` a `EAM_ANONIMIZADA_2024.xlsx`. Las instrucciones completas están en [`data/README.md`](data/README.md).

Para validar los archivos y generar el CSV unificado, ejecutar desde la raíz del repositorio:

```powershell
poetry run python scripts/load_data.py
```

Resultado esperado:

```text
Se descartan 1 registros con PRODBIND <= 0
Filas: 34552  columnas: 16  nulos: 0  establecimientos: 7659
Filas por año: {2020: 7342, 2021: 7138, 2022: 6775, 2023: 6714, 2024: 6583}
CSV unificado guardado en data/processed/eam_2020_2024.csv
```

El script comprueba que cada archivo tenga las columnas esperadas, que el periodo coincida con el año del archivo y que no haya establecimientos repetidos.

## Validación de Prefect y MLflow

La Actividad 1 requiere comprobar el funcionamiento de los servicios de Prefect y MLflow.

### 1. Iniciar los servicios

Seguir las instrucciones de la carpeta `services/` del repositorio del curso.

Se puede utilizar Docker Desktop o la alternativa sin Docker indicada en la guía. No deben ejecutarse ambas alternativas simultáneamente.

Una vez iniciados los servicios, verificar que las interfaces estén disponibles:

- Prefect: http://localhost:4200
- MLflow: http://localhost:5000

### 2. Preparar el flujo de prueba

Copiar el archivo `flows/prueba_flow.py` desde `actividades/actividad-1/flows/prueba_flow.py` del repositorio del curso a la carpeta `flows/` de este proyecto.

### 3. Configurar las variables de entorno

Desde PowerShell, ubicado en la raíz del repositorio:

```powershell
$env:PREFECT_API_URL = "http://localhost:4200/api"
$env:MLFLOW_TRACKING_URI = "http://localhost:5000"
```

### 4. Ejecutar el flujo de prueba

```powershell
poetry run python flows/prueba_flow.py
```

### 5. Verificar los resultados

En Prefect (http://localhost:4200/runs):

- El run `prueba` debe aparecer con estado `Completed`.
- Los logs deben mostrar `Suma de 1 a 5 = 15`.

En MLflow (http://localhost:5000):

- Debe aparecer el experimento `prueba-validacion`.
- El experimento debe contener el run `prueba-mlflow`, con el parámetro `n` y la métrica `suma`.

Estas verificaciones forman parte del paso 3 de la Actividad 1.

## Actividad 2

En cada terminal donde se ejecute un flow o la API, con los servicios levantados:

```powershell
$env:PREFECT_API_URL = "http://localhost:4200/api"
$env:MLFLOW_TRACKING_URI = "http://localhost:5000"
```

### Pipeline de entrenamiento

`flows/entrenamiento_flow.py` (flow `entrenamiento-eam`) tiene los pasos **ingesta → entrenamiento → evaluación → registro**:

1. **Ingesta:** carga la EAM con `scripts/load_data.py`, construye las variables y divide por año (entrena con los años anteriores a `division.anio_prueba` y evalúa con ese año).
2. **Entrenamiento:** entrena cada modelo de `candidatos` en `configs/config.yaml` con las variables de `configs/features.yaml`.
3. **Evaluación:** calcula las métricas del modelo y de la línea base (producción del año anterior).
4. **Registro:** cada candidato es un run de MLflow con el tag `fase = entrenamiento`, los parámetros, `features`, `n_features`, las métricas y el modelo. El mejor candidato (menor MAE) se registra en el Model Registry como `eam-imputacion-prodbind` con el alias **`champion`** solo si cumple el criterio.

```powershell
poetry run python flows/entrenamiento_flow.py
```

**Criterio para ser champion.** El modelo es de regresión, así que en lugar del AUC se usa el **MAE en logaritmo** (menor es mejor). El modelo nuevo pasa a champion solo si:

1. su MAE es menor o igual a `registro.max_mae_log` (0,30),
2. su MAE es menor que el de la línea base, si `registro.exigir_superar_linea_base` es `true`, y
3. su MAE es menor que el del champion vigente.

Si no cumple, el run queda en MLflow pero el alias no cambia. Así siempre hay un único champion, y solo se reemplaza por un modelo que mejora.

**Selección de variables** (`configs/features.yaml`, clave `seleccion`):

| Alias | Variables |
|---|---|
| `ALL` | Las 10 numéricas, el departamento y la actividad CIIU |
| `NUMERIC` | Solo las 10 numéricas |
| `CUSTOM` | Las columnas listadas en `custom` (nombres del CSV) |

Con `variables.usar_rezago: true` se agregan, en los tres casos, las variables del año anterior del establecimiento.

**Resultados** (año de prueba 2024, 6.420 establecimientos, línea base MAE 0,262):

| Ejecución | Features | Mejor candidato | MAE (log) | Resultado |
|---|---|---|---|---|
| 1 | CUSTOM (9 variables) | hgb | 0,2131 | Versión 1, primer champion |
| 2 | ALL (17 variables) | hgb | 0,2041 | Versión 2, reemplaza a la versión 1 |
| 3 | ALL (17 variables) | hgb | 0,2041 | No mejora: el champion sigue siendo la versión 2 |

**Otros parámetros de `configs/config.yaml`:**

| Clave | Descripción |
|---|---|
| `division.anio_prueba` | Año de evaluación. Se entrena con los años anteriores |
| `variables.usar_rezago` | Usa los datos del año anterior del mismo establecimiento |
| `candidatos` | Modelos que se comparan (`ridge` o `hist_gradient_boosting`) y sus hiperparámetros |
| `registro.nombre_modelo` | Nombre en el Model Registry de MLflow |
| `registro.max_mae_log` / `registro.exigir_superar_linea_base` | Criterio para ser champion |
| `mlflow.experimento` | Experimento donde quedan los runs de entrenamiento e inferencia |

### Pipeline de inferencia

`flows/inferencia_flow.py` (flow `inferencia-eam`) carga el modelo por su alias **`champion`** (nunca por número de versión) e imputa la producción de un lote de establecimientos:

1. **Cargar champion:** `models:/eam-imputacion-prodbind@champion`.
2. **Ingesta:** toma `inferencia.tamano_lote` establecimientos del año `inferencia.anio`, con los datos de su año anterior. Simula establecimientos que no reportaron su producción.
3. **Predicción:** calcula la producción estimada. Como en la muestra se conoce el valor reportado, marca con `alerta` los registros cuya diferencia supera `inferencia.umbral_alerta` en logaritmo, lo que sirve para la validación.
4. **Guardar y registrar:** escribe `data/predicciones/predicciones_AAAA-MM-DD.csv` y crea un run en MLflow con el tag `fase = inferencia`, la versión del modelo (`modelo_version`), su run de entrenamiento (`modelo_run_id`) y el CSV como artefacto.

```powershell
poetry run python flows/inferencia_flow.py
```

Columnas del CSV: `nordest`, `periodo`, `dpto`, `ciiu4`, `produccion_estimada`, `produccion_reportada`, `desviacion_log` y `alerta`.

### Despliegues en Prefect

`flows/desplegar.py` registra dos despliegues separados:

| Despliegue | Horario |
|---|---|
| `entrenamiento-eam/entrenamiento-manual` | Sin horario: se lanza a mano desde la UI (**Run → Quick run**) |
| `inferencia-eam/inferencia-diaria` | Todos los días a las 06:00, hora de Colombia (`0 6 * * *`) |

```powershell
poetry run python flows/desplegar.py
```

La terminal debe quedar abierta: mientras esté abierta, Prefect ejecuta las corridas programadas y las que se lancen desde http://localhost:4200/deployments. Para detenerla: `Ctrl + C`.

### API de inferencia

`api/main.py` sirve el modelo **`champion`** con FastAPI. El modelo se carga al arrancar: si el entrenamiento asigna un nuevo champion, hay que **reiniciar la API** para que lo use, sin cambiar código.

```powershell
$env:MLFLOW_TRACKING_URI = "http://localhost:5000"
poetry run uvicorn api.main:app --port 8000
```

Documentación interactiva: http://localhost:8000/docs

| Endpoint | Descripción |
|---|---|
| `GET /health` | Estado de la API, nombre y versión del modelo champion |
| `POST /predict` | Producción estimada de un establecimiento |

**Predicción de ejemplo** (valores medianos de la EAM 2024, no corresponden a un establecimiento real). En PowerShell:

```powershell
$cuerpo = @{
  dpto = 11; ciiu4 = 1410
  PERSOCU = 24; PERTOTAL = 44; SALARPER = 696000; PRESSPER = 161000; REMUTEMP = 0
  PERAPREN = 1; EELEC = 164000; ACTIVFI = 3262000; DEPRECIA = 157000; INVEBRTA = 61000
  prod_ant = 7569000; persocu_ant = 23; salarper_ant = 654000; eelec_ant = 163000
} | ConvertTo-Json
Invoke-RestMethod -Uri http://localhost:8000/predict -Method Post -ContentType "application/json" -Body $cuerpo
```

Respuesta con el champion versión 2:

```json
{
  "produccion_estimada": 7900082,
  "log_produccion_estimada": 15.8824,
  "modelo": "eam-imputacion-prodbind",
  "version_modelo": "2"
}
```

Los campos `prod_ant`, `persocu_ant`, `salarper_ant` y `eelec_ant` son los datos del año anterior. Si el champion se entrenó con rezago y no se envían, la API responde `422` con un mensaje que lo explica.

## Pruebas

Desde la raíz del repositorio:

```powershell
poetry run pytest
```

Las pruebas que necesitan los datos reales se omiten si los Excel no están en `data/raw/`. Las pruebas de los flows y de la API usan datos falsos y un MLflow temporal, por lo que no escriben en el servidor ni necesitan los servicios levantados.

## Reproducibilidad

Cada integrante del equipo debe clonar el repositorio, ejecutar `poetry install`, copiar los datos en `data/raw/` y ejecutar `scripts/load_data.py`.

Los archivos `pyproject.toml` y `poetry.lock` se mantienen versionados en GitHub. El primero declara las dependencias y el segundo registra las versiones resueltas, facilitando la instalación consistente del entorno entre los integrantes del equipo.

Los datos, las predicciones, `mlruns/`, `mlflow.db` y `mlflow_store/` no se versionan (ver `.gitignore`).