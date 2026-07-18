"""
Handler de AWS Lambda — Pipeline de Trayectorias Académicas.

Punto de entrada para la función Lambda que se activa automáticamente
cuando se sube un archivo .xlsx al bucket 'delfin-datos-entrada'.

Flujo:
    S3 Event → Lambda → Descarga a memoria → Limpieza → Autómata →
    → Subida de resultado a 'delfin-datos-procesados'

Runtime: Python 3.10
Memoria recomendada: 1024 MB+ (procesamiento de DataFrames grandes)
Timeout recomendado: 300 segundos
"""
import io
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict

import boto3
import pandas as pd

# ============================================================
# CONFIGURACIÓN
# ============================================================
BUCKET_ENTRADA = os.environ.get('BUCKET_ENTRADA', 'delfin-datos-entrada')
BUCKET_SALIDA = os.environ.get('BUCKET_SALIDA', 'delfin-datos-procesados')
PREFIJO_SALIDA = os.environ.get('PREFIJO_SALIDA', 'procesado_')
PPP_THRESHOLD = float(os.environ.get('PPP_THRESHOLD', '3.2'))
PPA_THRESHOLD = float(os.environ.get('PPA_THRESHOLD', '3.2'))

# Extensiones de archivo aceptadas
EXTENSIONES_ACEPTADAS = {'.xlsx', '.xls'}

# Clientes S3 (reutilizados entre invocaciones en Lambda warm start)
_s3_client = None


def _obtener_s3_client():
    """Obtiene o reutiliza el cliente S3 (patrón singleton para Lambda)."""
    global _s3_client
    if _s3_client is None:
        _s3_client = boto3.client('s3')
    return _s3_client


def _configurar_logging() -> None:
    """Configura logging compatible con AWS CloudWatch."""
    log_level = os.environ.get('LOG_LEVEL', 'INFO').upper()
    logging.basicConfig(
        level=getattr(logging, log_level, logging.INFO),
        format='%(asctime)s | %(levelname)-8s | %(name)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        stream=sys.stdout,
    )


def _extraer_info_evento(event: Dict[str, Any]) -> tuple:
    """
    Extrae bucket y key del evento S3 de forma robusta.

    Soporta tanto el formato estándar de S3 Event Notification
    como test events simplificados.

    Returns:
        Tupla (bucket_name, object_key).

    Raises:
        ValueError: Si el evento no contiene información válida de S3.
    """
    # Intento 1: Formato estándar de S3 Event Notification
    try:
        record = event['Records'][0]
        bucket = record['s3']['bucket']['name']
        key = record['s3']['object']['key']
        return bucket, key
    except (KeyError, IndexError):
        pass

    # Intento 2: Formato simplificado para testing directo
    if 'bucket' in event and 'key' in event:
        return event['bucket'], event['key']

    raise ValueError(
        f"No se pudo extraer bucket/key del evento. "
        f"Estructura recibida: {json.dumps(event, default=str)[:500]}"
    )


def _validar_archivo_entrante(bucket: str, key: str) -> None:
    """Valida que el archivo de entrada sea procesable."""
    extension = Path(key).suffix.lower()
    if extension not in EXTENSIONES_ACEPTADAS:
        raise ValueError(
            f"Extensión '{extension}' no soportada. "
            f"Extensiones aceptadas: {EXTENSIONES_ACEPTADAS}"
        )

    # Verificar que el objeto exista en S3
    s3 = _obtener_s3_client()
    try:
        respuesta = s3.head_object(Bucket=bucket, Key=key)
        tamaño = respuesta.get('ContentLength', 0)
        logger.info(
            "Archivo encontrado en S3: s3://%s/%s (%d bytes)",
            bucket, key, tamaño,
        )
        if tamaño == 0:
            raise ValueError(f"El archivo en S3 está vacío: s3://{bucket}/{key}")
    except s3.exceptions.ClientError as e:
        error_code = e.response['Error']['Code']
        if error_code == '404':
            raise FileNotFoundError(f"El archivo no existe en S3: s3://{bucket}/{key}")
        raise


def _descargar_a_memoria(bucket: str, key: str) -> io.BytesIO:
    """
    Descarga un archivo de S3 directamente a un BytesIO en memoria.

    Evita escribir en el disco efímero de Lambda, optimizando
    rendimiento y limpieza de recursos.

    Args:
        bucket: Nombre del bucket S3.
        key: Clave (ruta) del objeto en S3.

    Returns:
        Objeto BytesIO con el contenido del archivo.
    """
    logger.info("Descargando s3://%s/%s a memoria...", bucket, key)
    s3 = _obtener_s3_client()

    buffer = io.BytesIO()
    s3.download_fileobj(Bucket=bucket, Key=key, Fileobj=buffer)
    buffer.seek(0)

    logger.info("Descarga completada — %d bytes en memoria", buffer.getbuffer().nbytes)
    return buffer


def _subir_resultado(df: pd.DataFrame, bucket: str, key_origen: str) -> str:
    """
    Convierte el DataFrame a CSV y lo sube al bucket de salida.

    Args:
        df: DataFrame procesado con las trayectorias del autómata.
        bucket: Bucket de salida.
        key_origen: Key del archivo original (para generar la key de salida).

    Returns:
        Key del archivo subido al bucket de salida.
    """
    # Generar nombre de archivo de salida
    nombre_base = Path(key_origen).stem
    key_salida = f"{PREFIJO_SALIDA}{nombre_base}.csv"

    logger.info("Convirtiendo DataFrame a CSV...")
    csv_buffer = io.BytesIO()
    df.to_csv(csv_buffer, index=False, encoding='utf-8')
    csv_buffer.seek(0)

    logger.info("Subiendo resultado a s3://%s/%s...", bucket, key_salida)
    s3 = _obtener_s3_client()
    s3.upload_fileobj(
        Fileobj=csv_buffer,
        Bucket=bucket,
        Key=key_salida,
        ExtraArgs={
            'ContentType': 'text/csv; charset=utf-8',
            'Metadata': {
                'registros_procesados': str(len(df)),
                'estudiantes_unicos': str(df['ID'].nunique()),
                'fuente_original': key_origen,
            },
        },
    )

    logger.info("Resultado subido exitosamente: s3://%s/%s", bucket, key_salida)
    return key_salida


def _ejecutar_pipeline(buffer_bytes: io.BytesIO, nombre_archivo: str) -> pd.DataFrame:
    """
    Ejecuta las Fases 1 y 2 del pipeline: limpieza + autómata.

    Args:
        buffer_bytes: Contenido del Excel en memoria.
        nombre_archivo: Nombre legible del archivo para logging.

    Returns:
        DataFrame con las trayectorias del autómata calculadas.
    """
    # Importar módulos del pipeline (dentro de la función para manejar
    # correctamente el path de Lambda Layers si se usan en el futuro)
    from src.data_cleaner import clean_academic_data
    from src.automaton_motor import AcademicAutomaton

    # Fase 1: Limpieza
    logger.info("=== FASE 1: Limpieza de datos ===")
    df_limpio = clean_academic_data(buffer_bytes)

    # Fase 2: Autómata
    logger.info("=== FASE 2: Motor del autómata finito ===")
    automaton = AcademicAutomaton(
        ppp_threshold=PPP_THRESHOLD,
        ppa_threshold=PPA_THRESHOLD,
    )
    df_procesado = automaton.build_trajectories(df_limpio)

    return df_procesado


# ============================================================
# HANDLER PRINCIPAL
# ============================================================
logger = logging.getLogger('lambda_handler')


def lambda_handler(event: Dict[str, Any], context: Any) -> Dict[str, Any]:
    """
    Punto de entrada de la función Lambda.

    Se activa automáticamente cuando se sube un archivo .xlsx al
    bucket 'delfin-datos-entrada'. Ejecuta el pipeline de limpieza
    y autómata, y sube el resultado a 'delfin-datos-procesados'.

    Args:
        event: Evento S3 de Amazon (S3 Event Notification).
        context: Contexto de ejecución de Lambda (AWS).

    Returns:
        Diccionario con el resultado de la ejecución para CloudWatch.
    """
    _configurar_logging()
    inicio = time.time()

    logger.info("=" * 60)
    logger.info("LAMBDA INICIADA — Pipeline de Trayectorias Académicas")
    logger.info("ID de ejecución: %s", getattr(context, 'aws_request_id', 'local'))
    logger.info("Memory limit (MB): %s", getattr(context, 'memory_limit_in_mb', 'N/A'))
    logger.info("Time limit (s): %s", getattr(context, 'get_remaining_time_in_millis', lambda: 'N/A')())
    logger.info("=" * 60)

    try:
        # 1. Extraer información del evento S3
        bucket_origen, key_origen = _extraer_info_evento(event)
        logger.info("Evento S3 recibido — Bucket: %s, Key: %s", bucket_origen, key_origen)

        # 2. Validar archivo
        _validar_archivo_entrante(bucket_origen, key_origen)

        # 3. Descargar a memoria
        buffer = _descargar_a_memoria(bucket_origen, key_origen)

        # 4. Ejecutar pipeline
        df_procesado = _ejecutar_pipeline(buffer, Path(key_origen).name)

        # 5. Subir resultado
        key_salida = _subir_resultado(df_procesado, BUCKET_SALIDA, key_origen)

        # 6. Limpiar memoria explícitamente
        del buffer

        # 7. Calcular métricas
        duracion = time.time() - inicio
        resultado = {
            'statusCode': 200,
            'body': {
                'mensaje': 'Pipeline ejecutado exitosamente',
                'archivo_entrada': f's3://{bucket_origen}/{key_origen}',
                'archivo_salida': f's3://{BUCKET_SALIDA}/{key_salida}',
                'registros_procesados': len(df_procesado),
                'estudiantes_unicos': int(df_procesado['ID'].nunique()),
                'duracion_segundos': round(duracion, 2),
                'estados_generados': list(df_procesado['AUTOMATA_ESTADO_MATH'].unique()),
            },
        }

        logger.info("=" * 60)
        logger.info("LAMBDA COMPLETADA EXITOSAMENTE")
        logger.info("  Duración: %.2f segundos", duracion)
        logger.info("  Registros: %d", len(df_procesado))
        logger.info("  Estudiantes: %d", df_procesado['ID'].nunique())
        logger.info("  Salida: s3://%s/%s", BUCKET_SALIDA, key_salida)
        logger.info("=" * 60)

        return resultado

    except FileNotFoundError as e:
        logger.error("Archivo no encontrado: %s", e)
        return {
            'statusCode': 404,
            'body': {'error': f'Archivo no encontrado: {e}'},
        }

    except ValueError as e:
        logger.error("Error de validación: %s", e)
        return {
            'statusCode': 400,
            'body': {'error': f'Error de validación: {e}'},
        }

    except ImportError as e:
        logger.error("Error de importación — verifica dependencias en Lambda Layer: %s", e)
        return {
            'statusCode': 500,
            'body': {'error': f'Error de dependencias: {e}'},
        }

    except Exception as e:
        duracion = time.time() - inicio
        logger.exception("Error inesperado en Lambda: %s", e)
        return {
            'statusCode': 500,
            'body': {
                'error': f'Error interno del servidor: {e}',
                'duracion_segundos': round(duracion, 2),
            },
        }
