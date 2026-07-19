"""
Pipeline principal — Proyecto Modelo Predictivo (Estancia Delfín).
Orquesta las fases del pipeline:
    Fase 1: Limpieza de datos
    Fase 2: Construcción de trayectorias con autómata finito
    Fase 2b: Matriz de transición de Markov
    Fase 3: Preparación del dataset para machine learning
    Fase 4: Entrenamiento y evaluación del modelo predictivo
"""
import logging
import os
import sys
from pathlib import Path

# ============================================================
# CONFIGURACIÓN CENTRALIZADA
# ============================================================
RUTA_DATOS_CRUDOS = "data/12_only_undergraduate_with_automaton.xlsx"
RUTA_CHECKPOINT = "data/checkpoint_dataset_procesado.xlsx"
RUTA_FIGURAS = "reports/figures"

# Umbrales académicos configurables
PPP_THRESHOLD = 3.2
PPA_THRESHOLD = 3.2

# Parámetros del modelo (anti-overfitting)
TEST_SIZE = 0.2
N_ESTIMATORS = 150
MAX_DEPTH = 5
MAX_FEATURES = 'sqrt'
MIN_SAMPLES_LEAF = 5

# ============================================================
# CONFIGURACIÓN DE LOGGING
# ============================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger("pipeline")


def main() -> None:
    """Ejecuta el pipeline completo de principio a fin."""
    logger.info("=" * 60)
    logger.info("INICIANDO PIPELINE ACADÉMICO (ESTANCIA DELFÍN)")
    logger.info("=" * 60)

    try:
        # --- Validación inicial del archivo de entrada ---
        if not os.path.exists(RUTA_DATOS_CRUDOS):
            raise FileNotFoundError(
                f"No se encontró el archivo de datos crudos: {RUTA_DATOS_CRUDOS}"
            )

        # Importar módulos internos (evita problemas si faltan dependencias)
        from src.data_cleaner import clean_academic_data
        from src.automaton_motor import AcademicAutomaton
        from src.markov_analysis import compute_markov_transition_matrix
        from src.predict_models import prepare_ml_dataset, train_and_evaluate_model
        from src.visualizer import plot_feature_importances, plot_confusion_matrix

        # =======================================================
        # FASE 1: Limpieza de datos
        # =======================================================
        logger.info("-" * 40)
        logger.info("FASE 1: Limpieza de datos")
        logger.info("-" * 40)
        df_limpio = clean_academic_data(RUTA_DATOS_CRUDOS)

        # =======================================================
        # FASE 2: Construcción del autómata y trayectorias
        # =======================================================
        logger.info("-" * 40)
        logger.info("FASE 2: Motor del autómata finito")
        logger.info("-" * 40)
        automaton = AcademicAutomaton(
            ppp_threshold=PPP_THRESHOLD,
            ppa_threshold=PPA_THRESHOLD,
        )
        df_con_autómata = automaton.build_trajectories(df_limpio)

        # =======================================================
        # FASE 2b: Matriz de transición de Markov
        # =======================================================
        logger.info("-" * 40)
        logger.info("FASE 2b: Matriz de transición de Markov")
        logger.info("-" * 40)
        matriz_markov = compute_markov_transition_matrix(
            df_con_autómata,
            state_col='AUTOMATA_ESTADO_MATH',
        )
        logger.info("Matriz de Markov resultante:\n%s", matriz_markov.round(3).to_string())

        # =======================================================
        # Guardar checkpoint
        # =======================================================
        # Asegurar que el directorio de salida exista
        Path(RUTA_CHECKPOINT).parent.mkdir(parents=True, exist_ok=True)
        df_con_autómata.to_excel(RUTA_CHECKPOINT, index=False)
        logger.info("Checkpoint guardado: %s", RUTA_CHECKPOINT)

        # =======================================================
        # FASE 3: Preparación del dataset para ML
        # =======================================================
        logger.info("-" * 40)
        logger.info("FASE 3: Preparación del dataset predictivo")
        logger.info("-" * 40)
        X, y, clases = prepare_ml_dataset(df_con_autómata)

        # =======================================================
        # FASE 4: Entrenamiento y evaluación del modelo
        # =======================================================
        logger.info("-" * 40)
        logger.info("FASE 4: Modelo predictivo Random Forest")
        logger.info("-" * 40)
        resultado = train_and_evaluate_model(
            X, y, clases,
            test_size=TEST_SIZE,
            n_estimators=N_ESTIMATORS,
            max_depth=MAX_DEPTH,
            max_features=MAX_FEATURES,
            min_samples_leaf=MIN_SAMPLES_LEAF,
        )

        # =======================================================
        # FASE 5: Generación de gráficos científicos
        # =======================================================
        logger.info("-" * 40)
        logger.info("FASE 5: Generación de visualizaciones")
        logger.info("-" * 40)
        Path(RUTA_FIGURAS).mkdir(parents=True, exist_ok=True)

        plot_feature_importances(
            importancias=resultado.importancias,
            output_dir=RUTA_FIGURAS,
        )
        plot_confusion_matrix(
            y_true=resultado.y_test,
            y_pred=resultado.y_pred,
            class_names=list(clases),
            output_dir=RUTA_FIGURAS,
        )

        # =======================================================
        # RESUMEN FINAL
        # =======================================================
        logger.info("=" * 60)
        logger.info("PIPELINE COMPLETADO EXITOSAMENTE")
        logger.info("  Registros procesados: %d", len(df_con_autómata))
        logger.info("  Estudiantes: %d", df_con_autómata['ID'].nunique())
        logger.info("  Features ML: %d", X.shape[1])
        logger.info("  Muestras ML: %d", X.shape[0])
        logger.info("  Clases target: %d", y.nunique())
        logger.info("  Accuracy: %.4f", resultado.accuracy)
        logger.info("  Figuras generadas en: %s", RUTA_FIGURAS)
        logger.info("=" * 60)

    except FileNotFoundError as e:
        logger.error("Archivo no encontrado: %s", e)
        sys.exit(1)
    except ValueError as e:
        logger.error("Error de validación de datos: %s", e)
        sys.exit(1)
    except ImportError as e:
        logger.error("Error de importación — verifica dependencias: %s", e)
        sys.exit(1)
    except Exception as e:
        logger.exception("Error inesperado en el pipeline: %s", e)
        sys.exit(1)


if __name__ == "__main__":
    main()
