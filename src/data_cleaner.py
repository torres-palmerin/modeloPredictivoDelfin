"""
Módulo de limpieza e ingeniería de datos académicos (Fase 1).
Homogeneiza tipos, ordena cronológicamente y realiza imputación inteligente
de promedios nulos antes de pasar los datos al motor del autómata.

Diseñado para ser agnóstico a variaciones de nombres de columnas
entre múltiples版本es del dataset institucional.
"""
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ============================================================
# MAPEO FLEXIBLE DE COLUMNAS
# ============================================================
# Cada columna lógica acepta múltiples nombres candidatos (en orden de prioridad).
# El primer candidato encontrado en el Excel se utiliza.
CANDIDATOS_COLUMNAS: Dict[str, List[str]] = {
    'ID': ['ID', 'id', 'Id', 'ID_ESTUDIANTE', 'MATRICULA'],
    'PERIODO': ['PERIODO', 'periodo', 'Periodo', 'SEMESTRE', 'ANIO_PERIODO'],
    'PROGRAMA': ['PROGRAMA', 'programa', 'Programa', 'CARRERA', 'PROGRAMA_ACADEMICO'],
    'PROMEDIO': ['PROMEDIO', 'promedio', 'Promedio', 'PPP', 'PROMEDIO_PERIODICO'],
    'PROMEDIO_ACUMULADO': [
        'PROMEDIO_ACUMULADO', 'promedio_acumulado', 'Promedio_acumulado',
        'PPA', 'PROMEDIO_ACUM',
    ],
    'ESTADO': [
        'ESTADO', 'estado', 'Estado',
        'ESTADO_AUTOMATA', 'estado_automata',
        'ESTADO_ACADEMICO',
    ],
}

# Columnas verdaderamente obligatorias (sin estas no se puede continuar)
COLUMNAS_OBLIGATORIAS = frozenset({'ID', 'PERIODO', 'PROGRAMA'})

# Columnas deseadas que pueden faltar con fallback
COLUMNAS_OPCIONALES_CON_FALLBACK = {'PROMEDIO', 'PROMEDIO_ACUMULADO', 'ESTADO'}


def _resolver_nombre_columna(df: pd.DataFrame, nombre_logico: str) -> Optional[str]:
    """
    Busca la primera columna candidata que exista en el DataFrame.

    Args:
        df: DataFrame fuente.
        nombre_logico: Nombre lógico deseado (ej. 'PROMEDIO').

    Returns:
        Nombre real de la columna encontrada, o None si ninguna existe.
    """
    candidatos = CANDIDATOS_COLUMNAS.get(nombre_logico, [])
    for candidato in candidatos:
        if candidato in df.columns:
            return candidato
    return None


def _construir_mapeo(df: pd.DataFrame) -> Dict[str, str]:
    """
    Construye el diccionario de mapeo {nombre_real -> nombre_interno}
    resolviendo cada columna lógica contra las columnas disponibles.

    Returns:
        Diccionario de mapeo para rename().
    """
    mapeo: Dict[str, str] = {}
    nombres_internos = {
        'ID': 'ID',
        'PERIODO': 'PERIODO',
        'PROGRAMA': 'PROGRAMA',
        'PROMEDIO': 'PPP',
        'PROMEDIO_ACUMULADO': 'PPA',
        'ESTADO': 'ESTADO_ORIGINAL',
    }

    for nombre_logico, nombre_interno in nombres_internos.items():
        real = _resolver_nombre_columna(df, nombre_logico)
        if real is not None:
            mapeo[real] = nombre_interno
            if real != nombre_interno:
                logger.info(
                    "  Columna '%s' mapeada como '%s' → '%s'",
                    nombre_logico, real, nombre_interno,
                )
            else:
                logger.info("  Columna '%s' detectada correctamente", real)
        else:
            logger.warning(
                "  Columna '%s' no encontrada (candidatos: %s)",
                nombre_logico, CANDIDATOS_COLUMNAS.get(nombre_logico, []),
            )

    return mapeo


def _validar_archivo(file_path: str) -> None:
    """Valida que el archivo exista y no esté vacío."""
    ruta = Path(file_path)

    if not ruta.exists():
        raise FileNotFoundError(f"El archivo no existe: {ruta.resolve()}")
    if ruta.stat().st_size == 0:
        raise ValueError(f"El archivo está vacío: {ruta.resolve()}")


def _validar_columnas_minimas(
    df: pd.DataFrame, columnas_requeridas: set, archivo: str
) -> None:
    """Valida que el DataFrame contenga las columnas mínimas obligatorias."""
    disponibles = set(df.columns)
    faltantes = columnas_requeridas - disponibles
    if faltantes:
        raise ValueError(
            f"El archivo '{archivo}' no contiene las columnas obligatorias: {faltantes}. "
            f"Columnas encontradas: {list(disponibles)}"
        )


def _imputar_promedios(df: pd.DataFrame) -> pd.DataFrame:
    """
    Imputación académica inteligente de promedios nulos.

    Estrategia:
    - Forward fill por ID: arrastra el último PPA/PPP válido del estudiante.
    - Primer periodo sin datos: asigna 0.0 (sin actividad evaluable registrada).
    """
    df = df.sort_values(by=['ID', 'PERIODO']).copy()

    es_primer_periodo = df['PERIODO'] == df.groupby('ID')['PERIODO'].transform('min')

    for columna in ['PPP', 'PPA']:
        if columna not in df.columns:
            continue
        df[columna] = df.groupby('ID')[columna].ffill()
        mascara_nan_primer = es_primer_periodo & df[columna].isna()
        df.loc[mascara_nan_primer, columna] = 0.0
        df[columna] = df[columna].fillna(0.0)

    if 'PPP' in df.columns:
        df['PPP'] = df['PPP'].astype(np.float64)
    if 'PPA' in df.columns:
        df['PPA'] = df['PPA'].astype(np.float64)

    return df


def _aplicar_fallback_ppp(df: pd.DataFrame) -> pd.DataFrame:
    """
    Si PPP no está disponible en el dataset, genera una versión
    a partir de PPA como proxy razonable del promedio del periodo.
    Registra un warning claro para trazabilidad.
    """
    if 'PPP' not in df.columns:
        if 'PPA' in df.columns:
            logger.warning(
                "Columna 'PROMEDIO' (PPP) no encontrada en el dataset. "
                "Usando PPA como proxy del promedio del periodo."
            )
            df['PPP'] = df['PPA'].copy()
        else:
            logger.warning(
                "Ni PPP ni PPA disponibles. Asignando 0.0 a ambos promedios."
            )
            df['PPP'] = 0.0
            df['PPA'] = 0.0

    return df


def clean_academic_data(file_path: str) -> pd.DataFrame:
    """
    Fase 1: Pipeline de limpieza e ingeniería de datos.

    Lee el archivo Excel crudo, valida su integridad, homogeneiza tipos,
    ordena cronológicamente e imputa promedios nulos con lógica académica.

    Compatible con múltiples formatos de dataset institucional:
    - Datasets con columna PROMEDIO (PPP) → la utiliza directamente.
    - Datasets sin PROMEDIO → usa PPA como proxy con warning.
    - Nombres de columna variables → resolución por candidatos.

    Args:
        file_path: Ruta al archivo Excel con los datos crudos.

    Returns:
        DataFrame limpio y ordenado listo para el motor del autómata.

    Raises:
        FileNotFoundError: Si el archivo no existe.
        ValueError: Si el archivo está vacío o falta alguna columna obligatoria.
    """
    logger.info("Fase 1 — Leyendo archivo crudo: %s", file_path)

    _validar_archivo(file_path)

    df = pd.read_excel(file_path)
    logger.info("Registros originales leídos: %d", len(df))
    logger.info("Columnas detectadas: %s", list(df.columns))

    # 1. Validar columnas mínimas (solo ID, PERIODO, PROGRAMA)
    nombre_archivo = Path(file_path).name
    _validar_columnas_minimas(df, COLUMNAS_OBLIGATORIAS, nombre_archivo)

    # 2. Resolver mapeo flexible de columnas
    logger.info("Resolviendo mapeo de columnas...")
    mapeo = _construir_mapeo(df)

    # Seleccionar solo las columnas resueltas y renombrar
    columnas_reales = list(mapeo.keys())
    df = df[columnas_reales].rename(columns=mapeo)

    # 3. Fallback de PPP si no se encontró
    df = _aplicar_fallback_ppp(df)

    # 4. Asegurar tipos de datos correctos
    df['ID'] = df['ID'].astype(str).str.strip()
    df['PERIODO'] = df['PERIODO'].astype(np.int64)
    df['PROGRAMA'] = df['PROGRAMA'].astype(str).str.strip()

    # 5. Ordenamiento cronológico estricto por estudiante
    df = df.sort_values(by=['ID', 'PERIODO']).reset_index(drop=True)

    # 6. Imputación inteligente de promedios nulos
    df = _imputar_promedios(df)

    # 7. Limpieza de texto en variables categóricas
    if 'ESTADO_ORIGINAL' in df.columns:
        df['ESTADO_ORIGINAL'] = (
            df['ESTADO_ORIGINAL']
            .fillna('DESCONOCIDO')
            .astype(str)
            .str.upper()
            .str.strip()
        )

    logger.info("Fase 1 completada — Registros procesados: %d", len(df))
    logger.info("Columnas de salida: %s", list(df.columns))
    return df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    ruta_prueba = "data/12_only_undergraduate_with_automaton.xlsx"
    if os.path.exists(ruta_prueba):
        df_prueba = clean_academic_data(ruta_prueba)
        print(df_prueba.head(10))
