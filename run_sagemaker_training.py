# ============================================================
# Script de orquestación — SageMaker Training Job (Boto3 API)
# Proyecto: Modelo Predictivo (Estancia Delfín)
# ============================================================
"""
Lanza un trabajo de entrenamiento en SageMaker que ejecuta las Fases 3-5
del pipeline de trayectorias académicas usando la API nativa de Boto3.

Evita al 100% los problemas de compilación de gevent/sagemaker en Windows
y lee el ARN del rol de forma dinámica mediante variables de entorno.
"""
import logging
import os
import sys
import time
import boto3

# CONFIGURACIÓN DE LOGS
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    stream=sys.stdout,
)
logger = logging.getLogger('sagemaker_boto3_launcher')

REGION = "us-east-2"  # Ajusta si tus buckets están en otra región
BUCKET_DATOS = os.environ.get('BUCKET_DATOS', 'delfin-datos-procesados')
BUCKET_MODELOS = os.environ.get('BUCKET_MODELOS', 'delfin-modelos-sagemaker')

INSTANCIA_TIPO = os.environ.get('INSTANCIA_TIPO', 'ml.m5.large')
N_ESTIMATORS = os.environ.get('N_ESTIMATORS', '100')
MAX_DEPTH = os.environ.get('MAX_DEPTH', '10')

job_name = f"delfin-trayectorias-{int(time.time())}"

def main():
    logger.info("=" * 60)
    logger.info("SAGEMAKER TRAINING VIA BOTO3 — Inicialización Local")
    logger.info("=" * 60)

    # 1. Recuperar el Rol dinámicamente (Best Practice)
    role_arn = os.environ.get('SAGEMAKER_ROLE_ARN')
    if not role_arn:
        logger.error("[ERROR] Falta la variable de entorno 'SAGEMAKER_ROLE_ARN'")
        logger.info("Por favor ejecuta en tu terminal antes de correr el script:")
        logger.info('  $env:SAGEMAKER_ROLE_ARN="arn:aws:iam::102726256690:role/DelfinLambdaS3ExecutionRole"')
        sys.exit(1)

    try:
        # 2. Verificar identidad con STS
        identity = boto3.client('sts').get_caller_identity()
        logger.info("Conexión con AWS Validada. Cuenta: %s", identity['Account'])

        # 3. Inicializar cliente de SageMaker
        sm_client = boto3.client("sagemaker", region_name=REGION)

        # 4. Configurar parámetros del Job nativo
        training_params = {
            "TrainingJobName": job_name,
            "AlgorithmSpecification": {
                # Contenedor oficial optimizado de Scikit-Learn provisto por AWS
                "TrainingImage": f"257758044811.dkr.ecr.{REGION}.amazonaws.com/sagemaker-scikit-learn:1.2-1-cpu-py3",
                "TrainingInputMode": "File"
            },
            "RoleArn": role_arn,
            "InputDataConfig": [
                {
                    "ChannelName": "train",
                    "DataSource": {
                        "S3DataSource": {
                            "S3DataType": "S3Prefix",
                            "S3Uri": f"s3://{BUCKET_DATOS}/",
                            "S3DataDistributionType": "FullyReplicated"
                        }
                    },
                    "ContentType": "text/csv"
                }
            ],
            "OutputDataConfig": {
                "S3OutputPath": f"s3://{BUCKET_MODELOS}/modelos/"
            },
            "ResourceConfig": {
                "InstanceType": INSTANCIA_TIPO,
                "InstanceCount": 1,
                "VolumeSizeInGB": 30
            },
            "HyperParameters": {
                "n-estimators": str(N_ESTIMATORS),
                "max-depth": str(MAX_DEPTH),
                "sagemaker_program": "sagemaker_entry.py",
                # SageMaker empaqueta el contenido de src de forma automática aquí
                "sagemaker_submit_directory": f"s3://{BUCKET_MODELOS}/source/{job_name}/sourcedir.tar.gz"
            },
            "StoppingCondition": {
                "MaxRuntimeInSeconds": 3600
            }
        }

        logger.info("Enviando orden de creación de Training Job: %s", job_name)
        sm_client.create_training_job(**training_params)
        
        logger.info("=" * 60)
        logger.info("💥 ¡JOB CREADO EXITOSAMENTE EN LA NUBE!")
        logger.info("Instancia solicitada: %s", INSTANCIA_TIPO)
        logger.info("Monitorea el progreso aquí:")
        logger.info("https://%s.console.aws.amazon.com/sagemaker/home?region=%s#/jobs/%s", REGION, REGION, job_name)
        logger.info("=" * 60)

    except Exception as e:
        logger.exception("Error al lanzar el entrenamiento en SageMaker: %s", e)

if __name__ == '__main__':
    main()