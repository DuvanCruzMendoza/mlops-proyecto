"""Actividad 3: validaciones de datos (esquema, rangos y nulos) de la base procesada de la EAM."""
from scripts.load_data import COLUMNAS, OBJETIVO

# Variables que no pueden ser negativas (conteos, pagos y consumos). INVEBRTA, ACTIVFI y
# DEPRECIA pueden serlo (desinversiones y ajustes contables), por eso no están aquí.
NO_NEGATIVAS = ["PERSOCU", "PERTOTAL", "SALARPER", "PRESSPER", "REMUTEMP", "PERAPREN", "EELEC"]


def validar_datos(df, anios=None):
    """Devuelve la lista de problemas encontrados (vacía si los datos son válidos)."""
    problemas = []
    faltan = [c for c in COLUMNAS if c not in df.columns]
    if faltan:
        return [f"Faltan columnas: {faltan}"]
    nulos = df[COLUMNAS].isna().sum()
    if nulos.any():
        problemas.append(f"Hay nulos en: {nulos[nulos > 0].to_dict()}")
    negativos = (df[NO_NEGATIVAS] < 0).sum()
    if negativos.any():
        problemas.append(f"Valores negativos en: {negativos[negativos > 0].to_dict()}")
    if (df[OBJETIVO] <= 0).any():
        problemas.append(f"{OBJETIVO} debe ser positivo")
    if (df["PERSOCU"] > df["PERTOTAL"]).any():
        problemas.append("PERSOCU no puede superar a PERTOTAL")
    if df.duplicated(["nordest", "periodo"]).any():
        problemas.append("Hay establecimientos repetidos en un mismo año")
    if anios is not None and set(df["periodo"]) - set(anios):
        problemas.append(f"Años no esperados: {sorted(set(df['periodo']) - set(anios))}")
    return problemas
