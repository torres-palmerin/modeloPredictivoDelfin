"""
Entry Point de SageMaker — Pipeline de Modelado Predictivo.

Ejecuta las Fases 3, 4 y 5 del pipeline dentro del contenedor
de SageMaker, leyendo datos del canal de entrada estándar y
depositando el modelo entrenado y gráficos en /opt/ml/model/.

Rutas estándar de SageMaker:
    /opt/ml/input/data/{canal}/   — Datos de entrada (CSV desde S3)
    /opt/ml/model/                — Modelo y artefactos de salida
    /opt/ml/output/               — Logs destdout (CloudWatch)
"""
import glob
import json
import logging
import os
import sys
import tarfile
import time
from pathlib import Path

import pandas as pd

# matplotlib se configura en modo Agg antes de cualquier import de gráficos
import matplotlib
matplotlib.use('Agg')

# ============================================================
# CONFIGURACIÓN DE LOGGING PARA CLOUDWATCH
# ============================================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(name)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    stream=sys.stdout,
)
logger = logging.getLogger('sagemaker_entry')

# ============================================================
# RUTAS ESTÁNDAR DE SAGEMAKER
# ============================================================
RUTA_INPUT = Path('/opt/ml/input/data')
RUTA_MODEL_OUTPUT = Path('/opt/ml/model')
RUTA_TRAIN = RUTA_INPUT / 'train'

# Parámetros del modelo (con defaults razonables)
N_ESTIMATORS = int(os.environ.get('N_ESTIMATORS', '100'))
MAX_DEPTH = int(os.environ.get('MAX_DEPTH', '10'))
TEST_SIZE = float(os.environ.get('TEST_SIZE', '0.2'))


def _cargar_datos() -> pd.DataFrame:
    """
    Carga el DataFrame procesado desde los archivos CSV en el canal de entrada.

    SageMaker deposita los archivos del canal 'train' en:
        /opt/ml/input/data/train/

    Returns:
        DataFrame con las trayectorias del autómata.

    Raises:
        FileNotFoundError: Si no hay archivos CSV en el directorio de entrada.
        ValueError: Si los archivos están vacíos o son inválidos.
    """
    logger.info("Cargando datos desde: %s", RUTA_TRAIN)

    if not RUTA_TRAIN.exists():
        raise FileNotFoundError(
            f"Directorio de entrada no encontrado: {RUTA_TRAIN}. "
            f"Contenido de /opt/ml/input/data/: {list(RUTA_INPUT.iterdir()) if RUTA_INPUT.exists() else 'no existe'}"
        )

    archivos_csv = sorted(glob.glob(str(RUTA_TRAIN / '*.csv')))

    if not archivos_csv:
        # Intentar con .xlsx como fallback
        archivos_xlsx = sorted(glob.glob(str(RUTA_TRAIN / '*.xlsx')))
        if archivos_xlsx:
            logger.info("No se encontraron CSV, intentando con %d archivos Excel...", len(archivos_xlsx))
            dataframes = []
            for archivo in archivos_xlsx:
                df_chunk = pd.read_excel(archivo)
                dataframes.append(df_chunk)
                logger.info("  Cargado: %s (%d registros)", Path(archivo).name, len(df_chunk))
            df = pd.concat(dataframes, ignore_index=True)
            logger.info("Total de registros cargados: %d", len(df))
            return df
        raise FileNotFoundError(
            f"No se encontraron archivos CSV ni Excel en: {RUTA_TRAIN}. "
            f"Archivos encontrados: {list(RUTA_TRAIN.iterdir()) if RUTA_TRAIN.exists() else 'directorio vacío'}"
        )

    logger.info("Encontrados %d archivos CSV para cargar", len(archivos_csv))

    dataframes = []
    for archivo in archivos_csv:
        df_chunk = pd.read_csv(archivo)
        dataframes.append(df_chunk)
        logger.info("  Cargado: %s (%d registros)", Path(archivo).name, len(df_chunk))

    df = pd.concat(dataframes, ignore_index=True)
    logger.info("Total de registros cargados: %d", len(df))

    if df.empty:
        raise ValueError("El DataFrame cargado está vacío.")

    return df


def _guardar_modelo(modelo, directorio: Path) -> None:
    """
    Guarda el modelo serializado en formato .tar.gz compatible con
    SageMaker Model Registry.

    SageMaker espera que todo artefacto de salida esté en /opt/ml/model/
    empaquetado como .tar.gz.
    """
    logger.info("Guardando modelo en: %s", directorio)
    directorio.mkdir(parents=True, exist_ok=True)

    # Serializar modelo con joblib
    import joblib
    ruta_modelo = directorio / 'model.joblib'
    joblib.dump(modelo, ruta_modelo)
    logger.info("Modelo serializado: %s (%.2f MB)", ruta_modelo, ruta_modelo.stat().st_size / 1e6)


def _empaquetar_artefactos(directorio_modelo: Path) -> None:
    """
    Empaqueta todos los archivos del directorio de modelo en un único
    archivo .tar.gz en /opt/ml/model/ para que SageMaker lo suba
    automáticamente a S3.
    """
    ruta_tar = directorio_modelo / 'model.tar.gz'

    logger.info("Empaquetando artefactos en: %s", ruta_tar)
    with tarfile.open(ruta_tar, 'w:gz') as tar:
        for archivo in directorio_modelo.iterdir():
            if archivo.name == 'model.tar.gz':
                continue
            tar.add(archivo, arcname=archivo.name)

    logger.info(
        "Artefactos empaquetados: %s (%.2f MB)",
        ruta_tar,
        ruta_tar.stat().st_size / 1e6,
    )


def _guardar_metricas(resultado, directorio: Path) -> None:
    """
    Guarda las métricas del modelo como un archivo JSON legible.
    SageMaker captura este archivo para el Model Registry.
    """
    metricas = {
        'accuracy': float(resultado.accuracy),
        'n_muestras_test': int(len(resultado.y_test)),
        'n_clases': int(len(set(resultado.y_test))),
        'importancias': {k: float(v) for k, v in resultado.importancias.items()},
        'reporte_clasificacion': resultado.reporte,
    }

    ruta_metricas = directorio / 'metricas_modelo.json'
    with open(ruta_metricas, 'w', encoding='utf-8') as f:
        json.dump(metricas, f, indent=2, ensure_ascii=False)

    logger.info("Métricas guardadas: %s", ruta_metricas)


def ejecutar_pipeline_ml() -> None:
    """
    Función principal que ejecuta las Fases 3, 4 y 5 del pipeline
    dentro del entorno de SageMaker.
    """
    inicio = time.time()

    # ---- Importar módulos del proyecto ----
    # Se importan aquí porque SageMaker inyecta el código vía source_dir
    from src.predict_models import prepare_ml_dataset, train_and_evaluate_model
    from src.visualizer import plot_feature_importances, plot_confusion_matrix

    # ==============================================================
    # FASE 3: Preparación del dataset para ML
    # ==============================================================
    logger.info("=" * 60)
    logger.info("FASE 3: Preparación del dataset predictivo")
    logger.info("=" * 60)

    df = _cargar_datos()

    X, y, clases = prepare_ml_dataset(df)

    # ==============================================================
    # FASE 4: Entrenamiento y evaluación del modelo
    # ==============================================================
    logger.info("=" * 60)
    logger.info("FASE 4: Modelo predictivo Random Forest")
    logger.info("=" * 60)

    resultado = train_and_evaluate_model(
        X, y, clases,
        test_size=TEST_SIZE,
        n_estimators=N_ESTIMATORS,
        max_depth=MAX_DEPTH,
    )

    # ==============================================================
    # FASE 5: Generación de visualizaciones
    # ==============================================================
    logger.info("=" * 60)
    logger.info("FASE 5: Generación de visualizaciones")
    logger.info("=" * 60)

    plot_feature_importances(
        importancias=resultado.importancias,
        output_dir=str(RUTA_MODEL_OUTPUT),
    )
    plot_confusion_matrix(
        y_true=resultado.y_test,
        y_pred=resultado.y_pred,
        class_names=list(clases),
        output_dir=str(RUTA_MODEL_OUTPUT),
    )

    # ==============================================================
    # GUARDADO FINAL: Modelo + Métricas + Empaquetado
    # ==============================================================
    logger.info("=" * 60)
    logger.info("GUARDADO DE ARTEFACTOS")
    logger.info("=" * 60)

    _guardar_modelo(resultado.modelo, RUTA_MODEL_OUTPUT)
    _guardar_metricas(resultado, RUTA_MODEL_OUTPUT)
    _empaquetar_artefactos(RUTA_MODEL_OUTPUT)

    duracion = time.time() - inicio
    logger.info("=" * 60)
    logger.info("PIPELINE SAGEMAKER COMPLETADO")
    logger.info("  Duración: %.2f segundos", duracion)
    logger.info("  Muestras entrenadas: %d", X.shape[0])
    logger.info("  Accuracy: %.4f", resultado.accuracy)
    logger.info("  Artefactos en: %s", RUTA_MODEL_OUTPUT)
    logger.info("=" * 60)


if __name__ == '__main__':
    try:
        ejecutar_pipeline_ml()
    except Exception as e:
        logger.exception("Error fatal en el pipeline SageMaker: %s", e)
        sys.exit(1)
