"""
Módulo de limpieza e ingeniería de datos académicos (Fase 1).
Homogeneiza tipos, ordena cronológicamente y realiza imputación inteligente
de promedios nulos antes de pasar los datos al motor del autómata.
"""
import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Columnas mínimas requeridas en el archivo fuente
COLUMNAS_REQUERIDAS = frozenset({
    'ID', 'PERIODO', 'PROGRAMA', 'PROMEDIO', 'PROMEDIO_ACUMULADO', 'ESTADO'
})

# Mapeo de columnas del archivo original a nombres internos
COLUMNAS_MAPPING = {
    'ID': 'ID',
    'PERIODO': 'PERIODO',
    'PROGRAMA': 'PROGRAMA',
    'PROMEDIO': 'PPP',
    'PROMEDIO_ACUMULADO': 'PPA',
    'ESTADO': 'ESTADO_ORIGINAL',
}


def _validar_archivo(file_path: str) -> None:
    """Valida que el archivo exista, no esté vacío y contenga las columnas mínimas."""
    ruta = Path(file_path)

    if not ruta.exists():
        raise FileNotFoundError(f"El archivo no existe: {ruta.resolve()}")
    if ruta.stat().st_size == 0:
        raise ValueError(f"El archivo está vacío: {ruta.resolve()}")

    # Lectura rápida solo de encabezados para validar columnas
    df_head = pd.read_excel(ruta, nrows=0)
    columnas_faltantes = COLUMNAS_REQUERIDAS - set(df_head.columns)
    if columnas_faltantes:
        raise ValueError(
            f"El archivo no contiene las columnas requeridas: {columnas_faltantes}. "
            f"Columnas encontradas: {list(df_head.columns)}"
        )


def _imputar_promedios(df: pd.DataFrame) -> pd.DataFrame:
    """
    Imputación académica inteligente de promedios nulos.

    Estrategia:
    - Si un estudiante tiene PPP/PPA NaN en un periodo y ya tiene un PPA válido
      en un periodo anterior, se arrastra el último PPA válido (forward fill por ID).
    - Si es el primer periodo absoluto del estudiante y ambos son NaN, se asigna 0.0
      (sin actividad evaluable registrada).
    """
    # Asegurar orden cronológico para que forward fill sea correcto
    df = df.sort_values(by=['ID', 'PERIODO']).copy()

    # Identificar el primer registro de cada estudiante
    es_primer_periodo = df['PERIODO'] == df.groupby('ID')['PERIODO'].transform('min')

    for columna in ['PPP', 'PPA']:
        # Forward fill por estudiante: arrastra el último valor válido dentro del grupo
        df[columna] = df.groupby('ID')[columna].ffill()
        # Para el primer periodo de cada estudiante que sigue siendo NaN, asignar 0.0
        mascara_nan_primer = es_primer_periodo & df[columna].isna()
        df.loc[mascara_nan_primer, columna] = 0.0
        # Si después del ffill aún quedan NaN (caso edge), asignar 0.0
        df[columna] = df[columna].fillna(0.0)

    df['PPP'] = df['PPP'].astype(np.float64)
    df['PPA'] = df['PPA'].astype(np.float64)
    return df


def clean_academic_data(file_path: str) -> pd.DataFrame:
    """
    Fase 1: Pipeline de limpieza e ingeniería de datos.

    Lee el archivo Excel crudo, valida su integridad, homogeneiza tipos,
    ordena cronológicamente e imputa promedios nulos con lógica académica.

    Args:
        file_path: Ruta al archivo Excel con los datos crudos.

    Returns:
        DataFrame limpio y ordenado listo para el motor del autómata.

    Raises:
        FileNotFoundError: Si el archivo no existe.
        ValueError: Si el archivo está vacío o no contiene las columnas requeridas.
    """
    logger.info("Fase 1 — Leyendo archivo crudo: %s", file_path)

    _validar_archivo(file_path)

    df = pd.read_excel(file_path)
    logger.info("Registros originales leídos: %d", len(df))

    # 1. Seleccionar y renombrar solo las columnas esenciales
    columnas_disponibles = {k: v for k, v in COLUMNAS_MAPPING.items() if k in df.columns}
    df = df[list(columnas_disponibles.keys())].rename(columns=columnas_disponibles)

    # 2. Asegurar tipos de datos correctos
    df['ID'] = df['ID'].astype(str).str.strip()
    df['PERIODO'] = df['PERIODO'].astype(np.int64)
    df['PROGRAMA'] = df['PROGRAMA'].astype(str).str.strip()

    # 3. Ordenamiento cronológico estricto por estudiante
    df = df.sort_values(by=['ID', 'PERIODO']).reset_index(drop=True)

    # 4. Imputación inteligente de promedios nulos
    df = _imputar_promedios(df)

    # 5. Limpieza de texto en variables categóricas
    if 'ESTADO_ORIGINAL' in df.columns:
        df['ESTADO_ORIGINAL'] = (
            df['ESTADO_ORIGINAL']
            .fillna('DESCONOCIDO')
            .astype(str)
            .str.upper()
            .str.strip()
        )

    logger.info("Fase 1 completada — Registros procesados: %d", len(df))
    return df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    ruta_prueba = "data/12_only_undergraduate_with_automaton.xlsx"
    if os.path.exists(ruta_prueba):
        df_prueba = clean_academic_data(ruta_prueba)
        print(df_prueba.head(10))
