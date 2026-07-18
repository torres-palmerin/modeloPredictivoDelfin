"""
Módulo de visualización científica (Fase 5).
Genera gráficos de nivel científico para informe técnico y presentación:
    - Importancia de variables (Feature Importances)
    - Matriz de confusión normalizada (Confusion Matrix)
"""
import logging
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix

logger = logging.getLogger(__name__)

# ============================================================
# PALETA DE COLORES INSTITUCIONALES (azules y grises)
# ============================================================
COLOR_PRIMARIO = '#1B3A5C'      # Azul institucional oscuro
COLOR_SECUNDARIO = '#3A7CA5'    # Azul medio
COLOR_ACCENTO = '#5DADE2'       # Azul claro
COLOR_FONDO = '#F4F6F8'         # Gris muy claro para fondo
PALETA_BARRAS = sns.color_palette("Blues_r", n_colors=6)
PALETA_HEATMAP = sns.color_palette("Blues_r", as_cmap=False)

# Nombres amigables para las features (mapeo técnico -> español)
NOMBRES_FEATURES = {
    'PPP': 'Promedio del Periodo (PPP)',
    'PPA': 'Promedio Acumulado (PPA)',
    'ESTADO_ACTUAL_ENCODED': 'Estado Académico Actual',
    'PROGRAMA_ENCODED': 'Programa Académico',
}


def _configurar_estilo() -> None:
    """Configura el estilo visual global para todos los gráficos."""
    sns.set_theme(style="whitegrid", palette="Blues_r")
    plt.rcParams.update({
        'figure.facecolor': 'white',
        'axes.facecolor': 'white',
        'axes.edgecolor': '#CCCCCC',
        'axes.labelcolor': '#333333',
        'text.color': '#333333',
        'xtick.color': '#555555',
        'ytick.color': '#555555',
        'font.size': 11,
        'axes.titlesize': 13,
        'axes.labelsize': 11,
    })


def plot_feature_importances(
    importancias: dict,
    output_dir: str,
    nombre_archivo: str = 'feature_importances.png',
    dpi: int = 300,
) -> None:
    """
    Genera un gráfico de barras horizontal con la importancia de las features.

    Args:
        importancias: Diccionario {nombre_feature: valor_importancia}.
        output_dir: Directorio donde se guardará el gráfico.
        nombre_archivo: Nombre del archivo de salida.
        dpi: Resolución del gráfico en puntos por pulgada.
    """
    logger.info("Generando gráfico de importancia de variables...")
    _configurar_estilo()

    try:
        # Preparar datos ordenados de mayor a menor
        serie = pd.Series(importancias).sort_values(ascending=True)
        nombres_limpios = [NOMBRES_FEATURES.get(n, n) for n in serie.index]

        # Crear figura
        fig, ax = plt.subplots(figsize=(9, 4.5))

        barras = ax.barh(
            range(len(serie)),
            serie.values,
            color=PALETA_BARRAS[:len(serie)],
            edgecolor='white',
            linewidth=0.8,
            height=0.6,
        )

        # Etiquetas de eje Y
        ax.set_yticks(range(len(serie)))
        ax.set_yticklabels(nombres_limpios, fontsize=10)
        ax.set_xlabel('Importancia Relativa', fontsize=11, labelpad=10)

        # Valores al final de cada barra
        for i, (valor, barra) in enumerate(zip(serie.values, barras)):
            ax.text(
                valor + 0.005,
                barra.get_y() + barra.get_height() / 2,
                f'{valor:.4f}',
                va='center',
                ha='left',
                fontsize=9,
                color='#444444',
            )

        # Título y diseño
        ax.set_title(
            'Importancia de Variables — Random Forest',
            fontsize=13,
            fontweight='bold',
            pad=15,
            color=COLOR_PRIMARIO,
        )
        ax.set_xlim(0, max(serie.values) * 1.25)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        plt.tight_layout()

        # Guardar
        ruta_salida = Path(output_dir)
        ruta_salida.mkdir(parents=True, exist_ok=True)
        ruta_completa = ruta_salida / nombre_archivo
        fig.savefig(ruta_completa, dpi=dpi, bbox_inches='tight', facecolor='white')
        plt.close(fig)

        logger.info("Gráfico guardado: %s", ruta_completa)

    except Exception as e:
        logger.error("Error al generar gráfico de importancias: %s", e)
        plt.close('all')
        raise


def plot_confusion_matrix(
    y_true: pd.Series,
    y_pred: np.ndarray,
    class_names: list,
    output_dir: str,
    nombre_archivo: str = 'confusion_matrix.png',
    dpi: int = 300,
) -> None:
    """
    Genera una matriz de confusión normalizada como heatmap con porcentajes.

    Args:
        y_true: Valores reales del target.
        y_pred: Predicciones del modelo.
        class_names: Nombres de las clases para los ejes.
        output_dir: Directorio donde se guardará el gráfico.
        nombre_archivo: Nombre del archivo de salida.
        dpi: Resolución del gráfico en puntos por pulgada.
    """
    logger.info("Generando matriz de confusión...")
    _configurar_estilo()

    try:
        # Calcular matriz de confusión normalizada (por fila)
        cm = confusion_matrix(y_true, y_pred, labels=class_names)
        cm_normalizada = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        cm_normalizada = np.nan_to_num(cm_normalizada)

        # Crear figure con tamaño proporcional al número de clases
        n_clases = len(class_names)
        tamanho = max(7, n_clases * 1.1)
        fig, ax = plt.subplots(figsize=(tamanho, tamanho * 0.85))

        # Heatmap con porcentajes
        sns.heatmap(
            cm_normalizada,
            annot=True,
            fmt='.1%',
            cmap='Blues',
            xticklabels=class_names,
            yticklabels=class_names,
            linewidths=0.8,
            linecolor='white',
            square=True,
            ax=ax,
            cbar_kws={'label': 'Proporción', 'shrink': 0.8},
            annot_kws={'size': 9, 'weight': 'bold'},
            vmin=0,
            vmax=1,
        )

        # Etiquetas de ejes
        ax.set_xlabel('Predicción del Modelo', fontsize=11, labelpad=10)
        ax.set_ylabel('Estado Real', fontsize=11, labelpad=10)
        ax.set_title(
            'Matriz de Confusión Normalizada — Random Forest',
            fontsize=13,
            fontweight='bold',
            pad=15,
            color=COLOR_PRIMARIO,
        )

        # Rotar etiquetas si hay muchas clases
        if n_clases > 5:
            plt.setp(ax.get_xticklabels(), rotation=45, ha='right', fontsize=9)
            plt.setp(ax.get_yticklabels(), rotation=0, fontsize=9)

        plt.tight_layout()

        # Guardar
        ruta_salida = Path(output_dir)
        ruta_salida.mkdir(parents=True, exist_ok=True)
        ruta_completa = ruta_salida / nombre_archivo
        fig.savefig(ruta_completa, dpi=dpi, bbox_inches='tight', facecolor='white')
        plt.close(fig)

        logger.info("Gráfico guardado: %s", ruta_completa)

    except Exception as e:
        logger.error("Error al generar matriz de confusión: %s", e)
        plt.close('all')
        raise
