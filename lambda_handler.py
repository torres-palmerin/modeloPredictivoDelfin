"""
Handler de AWS Lambda — API de Predicción de Trayectorias Académicas.

Punto de entrada para la función Lambda detrás de API Gateway.
Recibe un JSON con las características académicas de un estudiante y
retorna la predicción del estado futuro junto con la certeza del modelo.

Flujo:
    API Gateway (POST JSON) → Lambda → Carga modelo (S3 o /tmp/) →
    → predict + predict_proba → Respuesta JSON con CORS

Runtime: Python 3.10
Memoria recomendada: 512 MB+
Timeout recomendado: 30 segundos
"""
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict

import boto3
import joblib
import numpy as np

# ============================================================
# CONFIGURACIÓN
# ============================================================
BUCKET_MODELOS = os.environ.get('BUCKET_MODELOS', 'delfin-modelos-sagemaker')
CLAVE_MODELO = os.environ.get('CLAVE_MODELO', 'modelo_trayectorias.joblib')
RUTA_LOCAL_MODELO = '/tmp/modelo_trayectorias.joblib'

CAMPOS_REQUERIDOS = ('PPP', 'PPA', 'ESTADO_ACTUAL_ENCODED', 'PROGRAMA_ENCODED')

# Mapeo de clases target (debe coincidir con el orden del modelo entrenado)
CLASES_TARGET = [
    'Continuo regular',
    'Exclusión',
    'PAP',
    'PAT',
    'Primera vez en una carrera',
]

# Clientes y modelo (reutilizados entre invocaciones en Lambda warm start)
_s3_client = None
_modelo_cargado = None


def _obtener_s3_client():
    """Obtiene o reutiliza el cliente S3 (patrón singleton para Lambda)."""
    global _s3_client
    if _s3_client is None:
        _s3_client = boto3.client('s3')
    return _s3_client


def _cargar_modelo():
    """
    Carga el modelo desde /tmp/ si ya existe, o lo descarga de S3.

    En Lambda warm starts, el modelo previamente descargado persiste
    en /tmp/, evitando la descarga repetida del bucket S3.
    """
    global _modelo_cargado

    if _modelo_cargado is not None:
        logger.info("Modelo ya cargado en memoria (warm start)")
        return _modelo_cargado

    # Verificar si ya existe en /tmp/ (descarga previa)
    if Path(RUTA_LOCAL_MODELO).exists():
        logger.info("Cargando modelo desde /tmp/ (descarga previa)")
        _modelo_cargado = joblib.load(RUTA_LOCAL_MODELO)
        return _modelo_cargado

    # Descargar de S3
    logger.info("Descargando modelo de s3://%s/%s...", BUCKET_MODELOS, CLAVE_MODELO)
    s3 = _obtener_s3_client()
    s3.download_file(BUCKET_MODELOS, CLAVE_MODELO, RUTA_LOCAL_MODELO)
    logger.info("Modelo descargado a /tmp/ (%.2f MB)", Path(RUTA_LOCAL_MODELO).stat().st_size / (1024 * 1024))

    _modelo_cargado = joblib.load(RUTA_LOCAL_MODELO)
    logger.info("Modelo cargado exitosamente")
    return _modelo_cargado


def _a_entero_seguro(valor: Any, nombre_campo: str) -> int:
    """
    Convierte un valor a entero de forma segura.

    Si ya es int/float numérico, lo convierte directamente.
    Si es string numérico (ej. '0', '3.0'), lo convierte.
    Si es un label de clase (ej. 'Continuo regular'), busca su índice en CLASES_TARGET.
    En caso contrario, lanza ValueError con mensaje claro.
    """
    if isinstance(valor, (int, float)):
        if isinstance(valor, float) and not valor == int(valor):
            raise ValueError(
                f"El campo '{nombre_campo}' debe ser un entero, "
                f"recibido float no entero: {valor}"
            )
        return int(valor)

    if isinstance(valor, str):
        # Intentar conversión numérica directa
        try:
            return int(valor)
        except ValueError:
            pass
        try:
            return int(float(valor))
        except ValueError:
            pass

        # Buscar como label de clase
        valor_limpio = valor.strip()
        for idx, clase in enumerate(CLASES_TARGET):
            if clase.lower() == valor_limpio.lower():
                logger.warning(
                    "Campo '%s' contenía el label '%s' en vez del encoded; resuelto a índice %d",
                    nombre_campo, valor, idx,
                )
                return idx

        raise ValueError(
            f"El campo '{nombre_campo}' no se pudo convertir a entero. "
            f"Valor recibido: {repr(valor)} (tipo {type(valor).__name__})"
        )

    raise ValueError(
        f"El campo '{nombre_campo}' tiene un tipo inesperado: "
        f"{type(valor).__name__} = {repr(valor)}"
    )


def _validar_entrada(body: Dict[str, Any]) -> None:
    """Valida que el JSON de entrada contenga todos los campos requeridos."""
    if not isinstance(body, dict):
        raise ValueError("El cuerpo de la solicitud debe ser un objeto JSON")

    campos_faltantes = [c for c in CAMPOS_REQUERIDOS if c not in body]
    if campos_faltantes:
        raise ValueError(
            f"Campos faltantes: {', '.join(campos_faltantes)}. "
            f"Campos requeridos: {', '.join(CAMPOS_REQUERIDOS)}"
        )

    for campo in CAMPOS_REQUERIDOS:
        valor = body[campo]
        if not isinstance(valor, (int, float)):
            raise ValueError(f"El campo '{campo}' debe ser numérico, recibido: {type(valor).__name__}")
        if np.isnan(valor):
            raise ValueError(f"El campo '{campo}' no puede ser NaN")


def _construir_respuesta(
    prediccion: str,
    certeza: float,
    probabilidades: Dict[str, float],
    duracion: float,
) -> Dict[str, Any]:
    """Construye la respuesta formateada con headers CORS."""
    return {
        'statusCode': 200,
        'headers': {
            'Access-Control-Allow-Origin': '*',
            'Access-Control-Allow-Headers': 'Content-Type,Authorization',
            'Access-Control-Allow-Methods': 'POST,OPTIONS',
            'Content-Type': 'application/json',
        },
        'body': json.dumps({
            'prediccion': prediccion,
            'certeza': round(certeza, 4),
            'probabilidades': probabilidades,
            'modelo': 'Random Forest Classifier',
            'duracion_segundos': round(duracion, 4),
        }, ensure_ascii=False, indent=2),
    }


def _construir_error(status_code: int, mensaje: str) -> Dict[str, Any]:
    """Construye una respuesta de error con headers CORS."""
    return {
        'statusCode': status_code,
        'headers': {
            'Access-Control-Allow-Origin': '*',
            'Access-Control-Allow-Headers': 'Content-Type,Authorization',
            'Access-Control-Allow-Methods': 'POST,OPTIONS',
            'Content-Type': 'application/json',
        },
        'body': json.dumps({
            'error': mensaje,
        }, ensure_ascii=False, indent=2),
    }


# ============================================================
# HANDLER PRINCIPAL
# ============================================================
logger = logging.getLogger('lambda_prediction')
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(name)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    stream=sys.stdout,
)


def lambda_handler(event: Dict[str, Any], context: Any) -> Dict[str, Any]:
    """
    Punto de entrada de la Lambda de predicción.

    Espera un evento JSON con:
        - PPP: Promedio Ponderado Permanente (float)
        - PPA: Promedio Ponderado Acumulado (float)
        - ESTADO_ACTUAL_ENCODED: Estado codificado del estudiante (int)
        - PROGRAMA_ENCODED: Programa académico codificado (int)

    Retorna:
        - prediccion: Etiqueta del estado futuro predicho
        - certeza: Porcentaje de certeza del modelo (0-1)
        - probabilidades: Diccionario con la probabilidad de cada clase
    """
    inicio = time.time()

    logger.info("Lambda de predicción invocada — request_id=%s",
                getattr(context, 'aws_request_id', 'local'))

    try:
        # 1. Manejar preflight CORS
        http_method = event.get('httpMethod', '')
        if http_method == 'OPTIONS':
            logger.info("Solicitud OPTIONS (preflight CORS)")
            return {
                'statusCode': 200,
                'headers': {
                    'Access-Control-Allow-Origin': '*',
                    'Access-Control-Allow-Headers': 'Content-Type,Authorization',
                    'Access-Control-Allow-Methods': 'POST,OPTIONS',
                },
                'body': '',
            }

        # 2. Extraer y parsear el body
        body_raw = event.get('body', '{}')

        if isinstance(body_raw, str):
            body = json.loads(body_raw)
        else:
            body = body_raw

        logger.info("Entrada recibida: %s", json.dumps(body, default=str)[:500])

        # 3. Validar entrada
        _validar_entrada(body)

        # 4. Cargar modelo
        modelo = _cargar_modelo()

        # 5. Preparar features (el orden debe coincidir con el entrenamiento)
        estado_encoded = _a_entero_seguro(body['ESTADO_ACTUAL_ENCODED'], 'ESTADO_ACTUAL_ENCODED')
        programa_encoded = _a_entero_seguro(body['PROGRAMA_ENCODED'], 'PROGRAMA_ENCODED')

        features = np.array([[
            float(body['PPP']),
            float(body['PPA']),
            estado_encoded,
            programa_encoded,
        ]])

        logger.info("Features: PPP=%.2f, PPA=%.2f, Estado=%s, Programa=%s",
                     features[0][0], features[0][1],
                     features[0][2], features[0][3])

        # 6. Ejecutar predicción
        prediccion_cruda = modelo.predict(features)[0]

        if isinstance(prediccion_cruda, (int, np.integer, float, np.floating)):
            prediccion_idx = int(prediccion_cruda)
        elif isinstance(prediccion_cruda, str):
            prediccion_limpia = prediccion_cruda.strip()
            prediccion_idx = None
            for idx, clase in enumerate(CLASES_TARGET):
                if clase.lower() == prediccion_limpia.lower():
                    prediccion_idx = idx
                    break
            if prediccion_idx is None:
                raise ValueError(
                    f"El modelo retornó un label no reconocido: {repr(prediccion_cruda)}. "
                    f"Labels válidos: {CLASES_TARGET}"
                )
            logger.warning(
                "Modelo retornó label string '%s', resuelto a índice %d",
                prediccion_cruda, prediccion_idx,
            )
        else:
            raise ValueError(
                f"Tipo inesperado en predicción: {type(prediccion_cruda).__name__} = {repr(prediccion_cruda)}"
            )

        probabilidades_raw = modelo.predict_proba(features)[0]

        prediccion_label = CLASES_TARGET[prediccion_idx]
        certeza = float(probabilidades_raw[prediccion_idx])

        probabilidades = {
            CLASES_TARGET[i]: round(float(probabilidades_raw[i]), 4)
            for i in range(len(CLASES_TARGET))
        }

        # Ordenar por probabilidad descendente
        probabilidades = dict(sorted(probabilidades.items(), key=lambda x: x[1], reverse=True))

        duracion = time.time() - inicio

        logger.info("Predicción exitosa: %s (%.2f%% certeza) en %.4fs",
                     prediccion_label, certeza * 100, duracion)

        # 7. Respuesta exitosa
        return _construir_respuesta(prediccion_label, certeza, probabilidades, duracion)

    except json.JSONDecodeError as e:
        logger.error("JSON inválido: %s", e)
        return _construir_error(400, f'JSON inválido: {e}')

    except ValueError as e:
        logger.error("Error de validación: %s", e)
        return _construir_error(400, str(e))

    except FileNotFoundError as e:
        logger.error("Modelo no encontrado: %s", e)
        return _construir_error(500, f'Modelo no disponible: {e}')

    except Exception as e:
        duracion = time.time() - inicio
        logger.exception("Error inesperado: %s", e)
        return _construir_error(500, f'Error interno del servidor: {e}')
