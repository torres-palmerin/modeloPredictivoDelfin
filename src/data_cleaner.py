"""
Módulo de limpieza e ingeniería de datos académicos (Fase 1).
Homogeneiza tipos, ordena cronológicamente y realiza imputación inteligente
de promedios nulos antes de pasar los datos al motor del autómata.

Diseñado para ser agnóstico a variaciones de nombres de columnas
entre múltiples versiones del dataset institucional.

Compatible con ejecución local (rutas de archivo) y en la nube (BytesIO).
"""
import io
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ============================================================
# MAPEO FLEXIBLE DE COLUMNAS
# ============================================================
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

COLUMNAS_OBLIGATORIAS = frozenset({'ID', 'PERIODO', 'PROGRAMA'})


def _resolver_nombre_columna(df: pd.DataFrame, nombre_logico: str) -> Optional[str]:
    """Busca la primera columna candidata que exista en el DataFrame."""
    candidatos = CANDIDATOS_COLUMNAS.get(nombre_logico, [])
    for candidato in candidatos:
        if candidato in df.columns:
            return candidato
    return None


def _construir_mapeo(df: pd.DataFrame) -> Dict[str, str]:
    """Construye el diccionario de mapeo {nombre_real -> nombre_interno}."""
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
                    "  Columna '%s' mapeada como '%s' -> '%s'",
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


def _es_streams_bytes(source: Union[str, Path, io.BytesIO]) -> bool:
    """Determina si la fuente es un objeto de flujo de bytes (BytesIO)."""
    return isinstance(source, io.BytesIO)


def _validar_fuente(source: Union[str, Path, io.BytesIO]) -> None:
    """
    Valida que la fuente de datos sea legible y no esté vacía.

    Acepta rutas de archivo (str/Path) u objetos BytesIO.
    """
    if _es_streams_bytes(source):
        posicion_actual = source.tell()
        source.seek(0, io.SEEK_END)
        tamaño = source.tell()
        source.seek(posicion_actual)
        if tamaño == 0:
            raise ValueError("El flujo de bytes está vacío (0 bytes).")
        logger.info("Fuente tipo BytesIO validada — Tamaño: %d bytes", tamaño)
    else:
        ruta = Path(source)
        if not ruta.exists():
            raise FileNotFoundError(f"El archivo no existe: {ruta.resolve()}")
        if ruta.stat().st_size == 0:
            raise ValueError(f"El archivo está vacío: {ruta.resolve()}")


def _obtener_nombre_archivo(source: Union[str, Path, io.BytesIO]) -> str:
    """Extrae un nombre legible de la fuente de datos."""
    if _es_streams_bytes(source):
        return "stream_bytesio"
    return Path(source).name


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


def clean_academic_data(
    file_path: Union[str, Path, io.BytesIO],
) -> pd.DataFrame:
    """
    Fase 1: Pipeline de limpieza e ingeniería de datos.

    Lee un archivo Excel (desde ruta local o flujo de bytes en memoria),
    valida su integridad, homogeneiza tipos, ordena cronológicamente
    e imputa promedios nulos con lógica académica.

    Compatible con:
    - Ejecución local: file_path es un str o Path con la ruta al .xlsx.
    - AWS Lambda: file_path es un io.BytesIO con el contenido del archivo.

    Args:
        file_path: Ruta al archivo Excel o objeto BytesIO con el contenido.

    Returns:
        DataFrame limpio y ordenado listo para el motor del autómata.

    Raises:
        FileNotFoundError: Si la ruta local no existe.
        ValueError: Si la fuente está vacía o falta alguna columna obligatoria.
    """
    nombre_archivo = _obtener_nombre_archivo(file_path)
    logger.info("Fase 1 — Leyendo archivo crudo: %s", nombre_archivo)

    _validar_fuente(file_path)

    df = pd.read_excel(file_path)
    logger.info("Registros originales leídos: %d", len(df))
    logger.info("Columnas detectadas: %s", list(df.columns))

    # 1. Validar columnas mínimas
    _validar_columnas_minimas(df, COLUMNAS_OBLIGATORIAS, nombre_archivo)

    # 2. Resolver mapeo flexible de columnas
    logger.info("Resolviendo mapeo de columnas...")
    mapeo = _construir_mapeo(df)

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
