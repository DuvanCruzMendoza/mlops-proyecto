
# Proyecto MLOps

Proyecto académico desarrollado en la asignatura **MLOps** de la Maestría en Ciencia de Datos.

## Integrantes

- Duván Cruz
- Natalia Mesa
- Paula Laiton

## Objetivo

Construir una base reproducible para el desarrollo de un proyecto de machine learning, utilizando Git y GitHub para el control de versiones, Poetry para la gestión de dependencias, Prefect para la orquestación de flujos y MLflow para el registro de experimentos.

## Requisitos

- Python 3.12 a 3.14
- Poetry 2.x
- Git
- Docker Desktop (si se utiliza la alternativa de servicios con Docker)

## Estructura del proyecto

```text
mlops-proyecto/
├── configs/
├── docs/
├── flows/
├── notebooks/
├── scripts/
├── tests/
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

## Reproducibilidad

Cada integrante del equipo debe clonar el repositorio y ejecutar `poetry install` para reproducir el entorno.

Los archivos `pyproject.toml` y `poetry.lock` se mantienen versionados en GitHub. El primero declara las dependencias y el segundo registra las versiones resueltas, facilitando la instalación consistente del entorno entre los integrantes del equipo.