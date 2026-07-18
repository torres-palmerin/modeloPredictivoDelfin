"""
Módulo de modelado predictivo (Fases 3 y 4).
Prepara el conjunto de datos para machine learning y entrena un clasificador
Random Forest para predecir el siguiente estado académico del estudiante.
"""
import logging
from typing import Any, Tuple

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)

# Columnas utilizadas como features para el modelo
FEATURES = ['PPP', 'PPA', 'ESTADO_ACTUAL_ENCODED', 'PROGRAMA_ENCODED']


def prepare_ml_dataset(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, Any]:
    """
    Fase 3: Construcción del conjunto de entrenamiento.

    Crea la variable target desfacando el estado del siguiente periodo por alumno
    y codifica las variables categóricas para el modelo.

    Args:
        df: DataFrame con las columnas AUTOMATA_ESTADO_MATH, PROGRAMA, PPP, PPA.

    Returns:
        Tupla (X, y, categorias_estado) donde:
            - X: DataFrame de features numéricas.
            - y: Serie con el target (estado del siguiente periodo).
            - categorias_estado: Categorías del estado académico codificado.

    Raises:
        ValueError: Si faltan columnas requeridas en el DataFrame.
    """
    logger.info("Fase 3 — Preparando conjunto de datos para ML...")

    columnas_necesarias = {'AUTOMATA_ESTADO_MATH', 'PROGRAMA', 'PPP', 'PPA', 'ID', 'PERIODO'}
    faltantes = columnas_necesarias - set(df.columns)
    if faltantes:
        raise ValueError(f"Columnas faltantes para preparar dataset ML: {faltantes}")

    df = df.sort_values(by=['ID', 'PERIODO']).copy()

    # El target es el estado del SIGUIENTE periodo
    df['TARGET_ESTADO'] = df.groupby('ID')['AUTOMATA_ESTADO_MATH'].shift(-1)

    # Eliminar filas donde el target sea nulo (último semestre de cada alumno)
    df_ml = df.dropna(subset=['TARGET_ESTADO']).copy()

    if df_ml.empty:
        raise ValueError("El dataset ML quedó vacío tras eliminar últimos periodos.")

    # Codificar variables categóricas de forma limpia
    estado_cat = df_ml['AUTOMATA_ESTADO_MATH'].astype('category')
    df_ml['ESTADO_ACTUAL_ENCODED'] = estado_cat.cat.codes
    categorias_estado = estado_cat.cat.categories

    programa_cat = df_ml['PROGRAMA'].astype('category')
    df_ml['PROGRAMA_ENCODED'] = programa_cat.cat.codes

    X = df_ml[FEATURES].copy()
    y = df_ml['TARGET_ESTADO'].copy()

    logger.info(
        "Fase 3 completada — %d muestras, %d features, %d clases target",
        len(X),
        len(FEATURES),
        y.nunique(),
    )
    return X, y, categorias_estado


def train_and_evaluate_model(
    X: pd.DataFrame,
    y: pd.Series,
    class_names: Any,
    test_size: float = 0.2,
    n_estimators: int = 100,
    max_depth: int = 10,
) -> RandomForestClassifier:
    """
    Fase 4: Entrenamiento y evaluación del modelo predictivo.

    Utiliza Random Forest con class_weight='balanced' para manejar el
    fuerte desbalance de clases (ej. Exclusión vs Continuo regular).

    Args:
        X: Features de entrenamiento.
        y: Variable target.
        class_names: Nombres de las clases para el reporte.
        test_size: Proporción del conjunto de prueba (default: 0.2).
        n_estimators: Número de árboles en el bosque (default: 100).
        max_depth: Profundidad máxima de cada árbol (default: 10).

    Returns:
        Modelo RandomForestClassifier entrenado.
    """
    logger.info("Fase 4 — Dividiendo datos (entrenamiento: %.0f%%, prueba: %.0f%%)...",
                (1 - test_size) * 100, test_size * 100)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=42,
        stratify=y,
    )

    logger.info("Fase 4 — Entrenando Random Forest (%d árboles, profundidad máxima: %d)...",
                n_estimators, max_depth)

    modelo = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        class_weight='balanced',
        random_state=42,
    )
    modelo.fit(X_train, y_train)

    # Predicciones y evaluación
    y_pred = modelo.predict(X_test)

    reporte = classification_report(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)

    logger.info("=== REPORTE DE CLASIFICACIÓN ===\n%s", reporte)
    logger.info("Accuracy General: %.4f", accuracy)

    # Importancia de features
    importancias = modelo.feature_importances_
    for nombre, importancia in zip(X.columns, importancias):
        logger.info("  Importancia [%s]: %.4f", nombre, importancia)

    logger.info("Fase 4 completada — Modelo entrenado exitosamente.")
    return modelo
