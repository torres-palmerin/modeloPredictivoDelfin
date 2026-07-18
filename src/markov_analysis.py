"""
Módulo de análisis de cadenas de Markov (Fase 2b).
Calcula la matriz de transición de probabilidades de Markov a partir
de las trayectorias generadas por el autómata finito.
"""
import logging

import pandas as pd

logger = logging.getLogger(__name__)


def compute_markov_transition_matrix(
    df: pd.DataFrame,
    state_col: str = 'AUTOMATA_ESTADO_MATH',
) -> pd.DataFrame:
    """
    Calcula la matriz de transición de probabilidades de Markov.

    Analiza la transición de un estado t a un estado t+1 para cada estudiante
    y construye una matriz de probabilidades de primer orden.

    Args:
        df: DataFrame con las trayectorias del autómata.
        state_col: Nombre de la columna con los estados académicos.

    Returns:
        DataFrame (matriz de transición) donde las filas son el estado actual
        y las columnas el estado siguiente, con probabilidades normalizadas.

    Raises:
        ValueError: Si la columna de estados no existe en el DataFrame.
    """
    logger.info("Fase 2b — Calculando matriz de transición de Markov...")

    if state_col not in df.columns:
        raise ValueError(
            f"Columna '{state_col}' no encontrada. "
            f"Columnas disponibles: {list(df.columns)}"
        )

    # Diagnóstico: verificar que la columna no esté toda en NaN
    nulos_estado = df[state_col].isna().sum()
    total_registros = len(df)
    logger.info(
        "  Columna '%s': %d nulos de %d registros (%.1f%%)",
        state_col, nulos_estado, total_registros,
        (nulos_estado / total_registros * 100) if total_registros > 0 else 0,
    )

    df_sorted = df.sort_values(by=['ID', 'PERIODO']).copy()

    # Crear columnas de estado actual y siguiente dentro de cada estudiante
    df_sorted['ESTADO_ACTUAL'] = df_sorted[state_col]
    df_sorted['ESTADO_SIGUIENTE'] = df_sorted.groupby('ID')[state_col].shift(-1)

    # Filtrar transiciones válidas (excluir el último periodo de cada estudiante)
    df_transiciones = df_sorted.dropna(subset=['ESTADO_SIGUIENTE'])

    if df_transiciones.empty:
        raise ValueError("No se encontraron transiciones válidas en los datos.")

    # Tabla de contingencia (frecuencias absolutas)
    crosstab = pd.crosstab(
        df_transiciones['ESTADO_ACTUAL'],
        df_transiciones['ESTADO_SIGUIENTE'],
    )

    # Normalizar a probabilidades de transición (suma de cada fila = 1.0)
    matriz_markov = crosstab.div(crosstab.sum(axis=1), axis=0)

    logger.info(
        "Fase 2b completada — Matriz %dx%d con %d transiciones analizadas",
        matriz_markov.shape[0],
        matriz_markov.shape[1],
        len(df_transiciones),
    )
    return matriz_markov
