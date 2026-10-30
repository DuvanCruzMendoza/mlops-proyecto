
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

## Requisitos

- Python 3.12 a 3.14
- Poetry 2.x
- Git
- Docker Desktop (si se utiliza la alternativa de servicios con Docker)

## Estructura del proyecto

```text
mlops-proyecto/
├── configs/
│   └── config.yaml              # datos, división, modelo y MLflow (sin valores en el código)
├── data/
│   ├── README.md                # fuente de los datos y cómo descargarlos
│   ├── raw/                     # Excel de la EAM 2020-2024 (no se versiona)
│   └── processed/               # CSV unificado que genera load_data.py (no se versiona)
├── docs/
├── flows/
│   ├── prueba_flow.py           # flow mínimo para comprobar Prefect y MLflow
│   └── experimento_flow.py      # flow del modelo: cargar datos → entrenar y registrar → resumen
├── notebooks/
├── scripts/
│   ├── load_data.py             # carga, valida y unifica la EAM
│   └── features.py              # variables del modelo, división por año, línea base y métricas
├── tests/
│   ├── test_data.py             # pruebas de la carga de datos
│   └── test_experimento_flow.py # pruebas del entrenamiento, del registro en MLflow y del flow
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

## Experimento: modelo de imputación

El flow `flows/experimento_flow.py` tiene tres tareas:

1. **Cargar datos:** usa `scripts/load_data.py` y registra en los logs de Prefect el número de filas, de establecimientos y de filas por año.
2. **Entrenar y registrar:** entrena con los años anteriores a `division.anio_prueba`, evalúa con ese año, compara el modelo con una línea base y registra en MLflow los parámetros, el tamaño del dataset, las métricas y el modelo.
3. **Resumen:** crea en Prefect el artefacto `eam-imputacion-v1` con el modelo, las métricas y el ID del run de MLflow.

### Ejecución

Con los servicios iniciados y el CSV de datos generado, desde PowerShell en la raíz del repositorio:

```powershell
$env:PREFECT_API_URL = "http://localhost:4200/api"
$env:MLFLOW_TRACKING_URI = "http://localhost:5000"
poetry run python flows/experimento_flow.py
```

El run del flow `eam-imputacion-v1` debe quedar en estado `Completed` en http://localhost:4200/runs, y el experimento `eam-imputacion-produccion` debe aparecer en http://localhost:5000 con el run `imputacion-prodbind`.

### Configuración (`configs/config.yaml`)

| Clave | Descripción |
|---|---|
| `division.anio_prueba` | Año de evaluación. Se entrena con los años anteriores |
| `variables.usar_rezago` | `true`: usa la producción del año anterior del mismo establecimiento. `false`: solo variables del mismo año |
| `modelo.tipo` | `hist_gradient_boosting` o `ridge` |
| `modelo.parametros` | Argumentos del estimador de scikit-learn |
| `mlflow.experimento` / `mlflow.nombre_run` | Nombres del experimento y del run en MLflow |

### Métricas

El objetivo se modela en logaritmo. Se registran, para el modelo y para la línea base:

- `mae_log` y `rmse_log`: error en escala logarítmica.
- `mdape`: error porcentual mediano.
- `pct_error_menor_20`: proporción de establecimientos con error menor al 20 %.

La línea base es la producción del año anterior (con rezago) o la mediana del CIIU (sin rezago).

### Resultados (prueba con 2024)

| Modelo | Rezago | MAE (log) | Línea base | Error mediano | Error < 20 % |
|---|---|---|---|---|---|
| hist_gradient_boosting | sí | **0,204** | 0,262 | 13,8 % | 64,8 % |
| ridge | sí | 0,239 | 0,262 | 16,0 % | 58,6 % |
| hist_gradient_boosting | no | 0,432 | 1,409 | 31,6 % | 33,2 % |

## Pruebas

Desde la raíz del repositorio:

```powershell
poetry run pytest
```

Las pruebas que necesitan los datos reales se omiten si los Excel no están en `data/raw/`. Las pruebas del flow usan una base temporal de MLflow, por lo que no escriben en el servidor.

## Reproducibilidad

Cada integrante del equipo debe clonar el repositorio, ejecutar `poetry install`, copiar los datos en `data/raw/` y ejecutar `scripts/load_data.py`.

Los archivos `pyproject.toml` y `poetry.lock` se mantienen versionados en GitHub. El primero declara las dependencias y el segundo registra las versiones resueltas, facilitando la instalación consistente del entorno entre los integrantes del equipo.

Los datos, `mlruns/`, `mlflow.db` y `mlflow_store/` no se versionan (ver `.gitignore`).