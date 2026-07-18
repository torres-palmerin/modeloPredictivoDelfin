"""
Entrenamiento local — Pipeline de Modelado Predictivo.

Lee el archivo procesado (checkpoint) del pipeline de autómata,
ejecuta las Fases 3 y 4 (ML + evaluación) y guarda:
    - modelo_trayectorias.joblib  (Random Forest entrenado)
    - metricas_modelo.json        (accuracy, importancias, reporte)

Uso:
    python run_local_training.py

Opcional — especificar ruta del dataset:
    python run_local_training.py --input data/mi_archivo.csv
"""
import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

import joblib
import pandas as pd

# ============================================================
# CONFIGURACIÓN
# ============================================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    stream=sys.stdout,
)
logger = logging.getLogger('local_training')

# Hiperparámetros por defecto
N_ESTIMATORS = 100
MAX_DEPTH = 10
TEST_SIZE = 0.2

# Archivos de salida
RUTA_MODELO_SALIDA = 'modelo_trayectorias.joblib'
RUTA_METRICAS_SALIDA = 'metricas_modelo.json'


def _buscar_checkpoint() -> str:
    """
    Busca automáticamente el archivo de checkpoint más reciente.

    Prioridad de búsqueda:
        1. Raíz del proyecto: procesado_*.csv
        2. Raíz del proyecto: checkpoint_dataset_procesado.csv
        3. Carpeta data/: checkpoint_dataset_procesado.* (CSV o XLSX)
        4. Cualquier archivo *.csv en data/
    """
    raiz = Path('.')

    # 1. Buscar en la raíz: procesado_*.csv
    procesados_raiz = sorted(raiz.glob('procesado_*.csv'))
    if procesados_raiz:
        logger.info("Checkpoint encontrado en raíz: %s", procesados_raiz[-1].name)
        return str(procesados_raiz[-1])

    # 2. Buscar en la raíz: checkpoint_dataset_procesado.csv
    csv_raiz = raiz / 'checkpoint_dataset_procesado.csv'
    if csv_raiz.exists():
        return str(csv_raiz)

    # 3. Buscar en data/
    directorio_data = Path('data')
    if directorio_data.exists():
        csv_checkpoint = directorio_data / 'checkpoint_dataset_procesado.csv'
        if csv_checkpoint.exists():
            return str(csv_checkpoint)

        xlsx_checkpoint = directorio_data / 'checkpoint_dataset_procesado.xlsx'
        if xlsx_checkpoint.exists():
            return str(xlsx_checkpoint)

        csvs = sorted(directorio_data.glob('*.csv'))
        if csvs:
            logger.info("Usando checkpoint de data/: %s", csvs[-1].name)
            return str(csvs[-1])

    raise FileNotFoundError(
        "No se encontró ningún archivo de checkpoint. "
        "Buscado: procesado_*.csv en raíz, checkpoint en data/. "
        "Ejecuta primero 'python main.py' para generar el dataset procesado."
    )


def _cargar_datos(ruta: str) -> pd.DataFrame:
    """Carga el dataset desde CSV o XLSX según la extensión."""
    extension = Path(ruta).suffix.lower()
    tamaño_mb = os.path.getsize(ruta) / (1024 * 1024)

    logger.info("Cargando dataset: %s (%.2f MB)", Path(ruta).name, tamaño_mb)
    logger.info("  Formato detectado: %s", extension.upper())

    inicio = time.time()
    if extension == '.csv':
        df = pd.read_csv(ruta)
    elif extension in ('.xlsx', '.xls'):
        df = pd.read_excel(ruta)
    else:
        raise ValueError(f"Formato no soportado: {extension}")

    duracion = time.time() - inicio
    logger.info("  Carga completada en %.2f segundos", duracion)
    logger.info("  Registros: %d | Columnas: %d", len(df), len(df.columns))
    logger.info("  Columnas: %s", list(df.columns))

    return df


def _validar_modelo(resultado, X, y, duracion: float) -> dict:
    """
    Ejecuta validación científica del modelo:
        1. Accuracy de entrenamiento (para detectar overfitting)
        2. Validación cruzada 5-Fold (robustez generalizable)
    Retorna un diccionario con todas las métricas para el JSON.
    """
    from sklearn.metrics import confusion_matrix, precision_score, recall_score
    from sklearn.model_selection import cross_val_score, train_test_split

    logger.info("Ejecutando validación científica del modelo...")

    # ---- 1. Accuracy de Entrenamiento ----
    logger.info("  [1/2] Calculando accuracy de entrenamiento...")
    X_train_recon, X_test_recon, y_train_recon, y_test_recon = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=42,
        stratify=y,
    )
    y_pred_train = resultado.modelo.predict(X_train_recon)
    accuracy_train = round(float(
        (y_pred_train == y_train_recon).mean()
    ), 4)
    brecha_overfitting = round(abs(resultado.accuracy - accuracy_train), 4)

    logger.info("    Accuracy entrenamiento: %.4f", accuracy_train)
    logger.info("    Accuracy prueba:        %.4f", resultado.accuracy)
    logger.info("    Brecha (overfitting):   %.4f", brecha_overfitting)

    # ---- 2. Validación Cruzada 5-Fold ----
    logger.info("  [2/2] Ejecutando validación cruzada 5-Fold (n_jobs=-1)...")
    from sklearn.ensemble import RandomForestClassifier

    modelo_cv = RandomForestClassifier(
        n_estimators=N_ESTIMATORS,
        max_depth=MAX_DEPTH,
        class_weight='balanced',
        random_state=42,
    )

    cv_scores = cross_val_score(
        modelo_cv, X, y,
        cv=5,
        scoring='accuracy',
        n_jobs=-1,
    )

    cv_mean = round(float(cv_scores.mean()), 4)
    cv_std = round(float(cv_scores.std()), 4)

    logger.info("    Scores por pliegue: %s", [round(float(s), 4) for s in cv_scores])
    logger.info("    Media: %.4f | Desviación estándar: %.4f", cv_mean, cv_std)

    # ---- 3. Métricas detalladas de prueba ----
    cm = confusion_matrix(resultado.y_test, resultado.y_pred)

    precision_macro = round(float(precision_score(
        resultado.y_test, resultado.y_pred, average='macro', zero_division=0
    )), 4)
    recall_macro = round(float(recall_score(
        resultado.y_test, resultado.y_pred, average='macro', zero_division=0
    )), 4)

    return {
        'accuracy_train': accuracy_train,
        'accuracy_test': round(float(resultado.accuracy), 4),
        'brecha_overfitting': brecha_overfitting,
        'cross_val_scores': [round(float(s), 4) for s in cv_scores],
        'cross_val_mean': cv_mean,
        'cross_val_std': cv_std,
        'precision_macro': precision_macro,
        'recall_macro': recall_macro,
        'n_muestras_test': int(len(resultado.y_test)),
        'n_muestras_train': int(len(y_train_recon)),
        'n_muestras_totales': int(len(y)),
        'n_features': len(resultado.feature_names),
        'n_clases_target': len(set(resultado.y_test)),
        'matriz_confusion': {
            'valores_absolutos': cm.tolist(),
            'etiquetas': [str(c) for c in sorted(set(resultado.y_test))],
        },
        'importancias_features': {
            k: round(float(v), 4) for k, v in resultado.importancias.items()
        },
        'reporte_clasificacion': resultado.reporte,
    }


def _guardar_metricas(validacion: dict, duracion: float) -> None:
    """Guarda las métricas completas del modelo como JSON legible."""
    metricas = {
        'modelo': 'Random Forest Classifier',
        'hiperparametros': {
            'n_estimators': N_ESTIMATORS,
            'max_depth': MAX_DEPTH,
            'test_size': TEST_SIZE,
            'class_weight': 'balanced',
            'random_state': 42,
        },
        'validacion': {
            'accuracy_train': validacion['accuracy_train'],
            'accuracy_test': validacion['accuracy_test'],
            'brecha_overfitting': validacion['brecha_overfitting'],
            'cross_val_scores': validacion['cross_val_scores'],
            'cross_val_mean': validacion['cross_val_mean'],
            'cross_val_std': validacion['cross_val_std'],
        },
        'resultado': {
            'precision_macro': validacion['precision_macro'],
            'recall_macro': validacion['recall_macro'],
            'n_muestras_totales': validacion['n_muestras_totales'],
            'n_muestras_train': validacion['n_muestras_train'],
            'n_muestras_test': validacion['n_muestras_test'],
            'n_features': validacion['n_features'],
            'n_clases_target': validacion['n_clases_target'],
        },
        'matriz_confusion': validacion['matriz_confusion'],
        'importancias_features': validacion['importancias_features'],
        'reporte_clasificacion': validacion['reporte_clasificacion'],
        'duracion_segundos': round(duracion, 2),
    }

    with open(RUTA_METRICAS_SALIDA, 'w', encoding='utf-8') as f:
        json.dump(metricas, f, indent=2, ensure_ascii=False)

    logger.info("Métricas guardadas: %s", RUTA_METRICAS_SALIDA)


def main() -> None:
    """Punto de entrada del entrenamiento local."""
    parser = argparse.ArgumentParser(
        description='Entrenamiento local del modelo predictivo de trayectorias académicas.'
    )
    parser.add_argument(
        '--input', '-i',
        type=str,
        default=None,
        help='Ruta al dataset CSV/XLSX. Si no se especifica, busca automáticamente en data/.',
    )
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("ENTRENAMIENTO LOCAL — Pipeline de Modelado Predictivo")
    logger.info("=" * 60)
    logger.info("Hiperparámetros:")
    logger.info("  n_estimators: %d", N_ESTIMATORS)
    logger.info("  max_depth:    %d", MAX_DEPTH)
    logger.info("  test_size:    %.0f%%", TEST_SIZE * 100)
    logger.info("  class_weight: balanced")
    logger.info("  random_state: 42")
    logger.info("=" * 60)

    try:
        # ---- Importar módulos del pipeline ----
        from src.predict_models import prepare_ml_dataset, train_and_evaluate_model
        from src.visualizer import plot_feature_importances, plot_confusion_matrix

        # ---- FASE 3: Preparación del dataset ----
        logger.info("-" * 40)
        logger.info("FASE 3: Preparación del dataset predictivo")
        logger.info("-" * 40)

        if args.input:
            ruta_dataset = args.input
        else:
            ruta_dataset = _buscar_checkpoint()

        df = _cargar_datos(ruta_dataset)
        X, y, clases = prepare_ml_dataset(df)

        logger.info("  Muestras totales: %d", len(X))
        logger.info("  Features: %s", list(X.columns))
        logger.info("  Clases target: %s", list(clases))

        # ---- FASE 4: Entrenamiento y evaluación ----
        logger.info("-" * 40)
        logger.info("FASE 4: Modelo predictivo Random Forest")
        logger.info("-" * 40)

        inicio = time.time()
        resultado = train_and_evaluate_model(
            X, y, clases,
            test_size=TEST_SIZE,
            n_estimators=N_ESTIMATORS,
            max_depth=MAX_DEPTH,
        )
        duracion = time.time() - inicio

        # ---- FASE 4.1: Validación científica ----
        logger.info("-" * 40)
        logger.info("FASE 4.1: Validación científica del modelo")
        logger.info("-" * 40)

        validacion = _validar_modelo(resultado, X, y, duracion)

        # ---- FASE 5: Visualizaciones ----
        logger.info("-" * 40)
        logger.info("FASE 5: Generación de visualizaciones")
        logger.info("-" * 40)

        plot_feature_importances(
            importancias=resultado.importancias,
            output_dir='reports/figures',
        )
        plot_confusion_matrix(
            y_true=resultado.y_test,
            y_pred=resultado.y_pred,
            class_names=list(clases),
            output_dir='reports/figures',
        )

        # ---- GUARDADO FINAL ----
        logger.info("-" * 40)
        logger.info("GUARDADO DE ARTEFACTOS")
        logger.info("-" * 40)

        joblib.dump(resultado.modelo, RUTA_MODELO_SALIDA)
        tamaño_modelo = os.path.getsize(RUTA_MODELO_SALIDA) / (1024 * 1024)
        logger.info("Modelo guardado: %s (%.2f MB)", RUTA_MODELO_SALIDA, tamaño_modelo)

        _guardar_metricas(validacion, duracion)

        # ---- RESUMEN FINAL ----
        logger.info("")
        logger.info("=" * 60)
        logger.info("ENTRENAMIENTO LOCAL COMPLETADO")
        logger.info("=" * 60)
        logger.info("  Archivo de entrada:  %s", Path(ruta_dataset).name)
        logger.info("  Muestras totales:    %d", len(X))
        logger.info("  Muestras entrenamiento: %d", validacion['n_muestras_train'])
        logger.info("  Muestras prueba:     %d", validacion['n_muestras_test'])
        logger.info("")
        logger.info("  --- Métricas de rendimiento ---")
        logger.info("  Accuracy entreno:    %.4f", validacion['accuracy_train'])
        logger.info("  Accuracy prueba:     %.4f", validacion['accuracy_test'])
        logger.info("  Brecha overfitting:  %.4f", validacion['brecha_overfitting'])
        logger.info("  Validación cruzada:  %.4f ± %.4f", validacion['cross_val_mean'], validacion['cross_val_std'])
        logger.info("  Scores por pliegue:  %s", validacion['cross_val_scores'])
        logger.info("  Precision macro:     %.4f", validacion['precision_macro'])
        logger.info("  Recall macro:        %.4f", validacion['recall_macro'])
        logger.info("")
        logger.info("  Duración total:      %.2f segundos", duracion)
        logger.info("")
        logger.info("  Artefactos generados:")
        logger.info("    - %s  (modelo serializado)", RUTA_MODELO_SALIDA)
        logger.info("    - %s  (métricas e importancias)", RUTA_METRICAS_SALIDA)
        logger.info("    - reports/figures/   (gráficos científicos)")
        logger.info("=" * 60)

    except FileNotFoundError as e:
        logger.error("Archivo no encontrado: %s", e)
        sys.exit(1)
    except ValueError as e:
        logger.error("Error de validación: %s", e)
        sys.exit(1)
    except Exception as e:
        logger.exception("Error inesperado: %s", e)
        sys.exit(1)


if __name__ == '__main__':
    main()
