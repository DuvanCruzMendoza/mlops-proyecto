# Datos

**Fuente:** Encuesta Anual Manufacturera (EAM), microdatos anonimizados, DANE.
Archivo Nacional de Datos (ANDA): https://microdatos.dane.gov.co

Los Excel originales **no se versionan en Git** (pesan unos 50 MB en total y se descargan del ANDA).
Cada integrante los descarga una vez y los copia en `data/raw/`.

## Cómo descargarlos

1. Entrar a https://microdatos.dane.gov.co y buscar "Encuesta Anual Manufacturera".
2. Abrir el estudio de cada año (2020 a 2024) → pestaña **Obtener microdatos** → descargar la base anonimizada.
3. Copiar los archivos en `data/raw/` con estos nombres exactos:

```text
data/raw/
├── EAM_ANONIMIZADA_2020.xlsx
├── EAM_ANONIMIZADA_2021.xlsx
├── EAM_ANONIMIZADA_2022.xlsx
├── EAM_ANONIMIZADA_2023.xlsx
└── EAM_ANONIMIZADA_2024.xlsx
```

4. Desde la raíz del repositorio, ejecutar `poetry run python scripts/load_data.py`.
   Valida los archivos y genera `data/processed/eam_2020_2024.csv` (tampoco se versiona).

## Contenido usado

Una fila por establecimiento y año. Se usan 16 de las 386 columnas (lista en `scripts/load_data.py`):

| Tipo | Columnas |
|---|---|
| Identificadores | `nordemp`, `nordest` (anonimizados), `periodo` |
| Categóricas | `dpto`, `ciiu4` |
| Numéricas | `PERSOCU`, `PERTOTAL`, `SALARPER`, `PRESSPER`, `REMUTEMP`, `PERAPREN`, `EELEC`, `ACTIVFI`, `DEPRECIA`, `INVEBRTA` |
| Objetivo | `PRODBIND` (producción bruta) |

Valores monetarios en pesos corrientes (nominales); las unidades exactas están en el diccionario de datos del ANDA.
