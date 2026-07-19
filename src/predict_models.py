"""
Módulo de modelado predictivo (Fases 3 y 4).
Prepara el conjunto de datos para machine learning y entrena un clasificador
Random Forest para predecir el siguiente estado académico del estudiante.
"""
import logging
from dataclasses import dataclass, field
from typing import Any, List, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)

# Columnas utilizadas como features para el modelo
FEATURES_FULL = ['PPP', 'PPA', 'ESTADO_ACTUAL_ENCODED', 'PROGRAMA_ENCODED']
FEATURES_NUMERICAS = ['PPP', 'PPA']

FEATURES = FEATURES_FULL


@dataclass
class ResultadoModelo:
    """
    Contenedor de resultados del pipeline de modelado predictivo.
    Centraliza el modelo entrenado, las métricas de evaluación y los
    artefactos necesarios para generación de gráficos.
    """
    modelo: RandomForestClassifier
    y_test: pd.Series
    y_pred: np.ndarray
    feature_names: List[str]
    importancias: dict = field(default_factory=dict)
    accuracy: float = 0.0
    reporte: str = ''


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

    # 1. Crear target ANTES de cualquier encoding — shift en columna cruda
    df['TARGET_ESTADO'] = df.groupby('ID')['AUTOMATA_ESTADO_MATH'].shift(-1)

    # 2. Encode sobre el DataFrame COMPLETO (captura todas las categorías,
    #    incluidas las que solo aparecen en el último periodo del alumno)
    estado_cat = df['AUTOMATA_ESTADO_MATH'].astype('category')
    df['ESTADO_ACTUAL_ENCODED'] = estado_cat.cat.codes
    categorias_estado = estado_cat.cat.categories

    programa_cat = df['PROGRAMA'].astype('category')
    df['PROGRAMA_ENCODED'] = programa_cat.cat.codes

    # 3. Eliminar filas donde el target sea nulo (último semestre de cada alumno)
    df_ml = df.dropna(subset=['TARGET_ESTADO']).copy()

    if df_ml.empty:
        raise ValueError("El dataset ML quedó vacío tras eliminar últimos periodos.")

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
    n_estimators: int = 150,
    max_depth: int = 5,
    max_features: str = 'sqrt',
    min_samples_leaf: int = 5,
) -> ResultadoModelo:
    """
    Fase 4: Entrenamiento y evaluación del modelo predictivo.

    Utiliza Random Forest con class_weight='balanced' para manejar el
    fuerte desbalance de clases y regularización agresiva contra sobreajuste
    (max_depth bajo, max_features='sqrt', min_samples_leaf=5).

    Args:
        X: Features de entrenamiento.
        y: Variable target.
        class_names: Nombres de las clases para el reporte.
        test_size: Proporción del conjunto de prueba (default: 0.2).
        n_estimators: Número de árboles en el bosque (default: 150).
        max_depth: Profundidad máxima de cada árbol (default: 5).
        max_features: Nº de features por split (default: 'sqrt').
        min_samples_leaf: Mínimo de muestras por hoja (default: 5).

    Returns:
        ResultadoModelo con el modelo, predicciones, importancias y métricas.
    """
    logger.info("Fase 4 — Dividiendo datos (entrenamiento: %.0f%%, prueba: %.0f%%)...",
                (1 - test_size) * 100, test_size * 100)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=42,
        stratify=y,
    )

    logger.info(
        "Fase 4 — Entrenando RF (%d árboles, depth=%d, max_features=%s, min_leaf=%d)...",
        n_estimators, max_depth, max_features, min_samples_leaf,
    )

    modelo = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        max_features=max_features,
        min_samples_leaf=min_samples_leaf,
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

    # Importancia de features como diccionario limpio
    importancias = dict(zip(X.columns, modelo.feature_importances_))
    for nombre, importancia in importancias.items():
        logger.info("  Importancia [%s]: %.4f", nombre, importancia)

    logger.info("Fase 4 completada — Modelo entrenado exitosamente.")

    return ResultadoModelo(
        modelo=modelo,
        y_test=y_test,
        y_pred=y_pred,
        feature_names=list(X.columns),
        importancias=importancias,
        accuracy=accuracy,
        reporte=reporte,
    )
