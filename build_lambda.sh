#!/usr/bin/env bash
# =============================================================
# build_lambda.sh — Script de empaquetado para AWS Lambda
# Proyecto: Modelo Predictivo (Estancia Delfín)
# =============================================================
#
# Propósito:
#   Construye una imagen Docker con las dependencias compiladas
#   para Linux x86_64 y extrae el deployment_package.zip final.
#
# Uso:
#   chmod +x build_lambda.sh
#   ./build_lambda.sh
#
# =============================================================

set -euo pipefail

# ---- Configuración ----
IMAGEN_NOMBRE="delfin-lambda-builder"
CONTENEDOR_NOMBRE="delfin-lambda-extractor"
ARCHIVO_ZIP="deployment_package.zip"
RUTA_PROYECTO="$(cd "$(dirname "$0")" && pwd)"
RUTA_SALIDA="${RUTA_PROYECTO}/${ARCHIVO_ZIP}"

# Colores para logs
ROJO='\033[0;31m'
VERDE='\033[0;32m'
AMARILLO='\033[1;33m'
CYAN='\033[0;36m'
SIN_COLOR='\033[0m'

log_info()    { echo -e "${CYAN}[INFO]${SIN_COLOR}    $*"; }
log_exito()   { echo -e "${VERDE}[ÉXITO]${SIN_COLOR}  $*"; }
log_advertencia() { echo -e "${AMARILLO}[AVISO]${SIN_COLOR}  $*"; }
log_error()   { echo -e "${ROJO}[ERROR]${SIN_COLOR}  $*"; }

# ============================================================
# PASO 0: Validar prerrequisitos
# ============================================================
log_info "Validando prerrequisitos..."

if ! command -v docker &> /dev/null; then
    log_error "Docker no está instalado o no está en el PATH."
    log_error "Instala Docker Desktop: https://docs.docker.com/get-docker/"
    exit 1
fi

if ! docker info &> /dev/null 2>&1; then
    log_error "El daemon de Docker no está ejecutándose."
    log_error "Inicia Docker Desktop y vuelve a intentar."
    exit 1
fi

log_exito "Docker detectado: $(docker --version)"

# ============================================================
# PASO 1: Limpiar empaquetados anteriores
# ============================================================
log_info "Limpiando artefactos anteriores..."

if [ -f "${RUTA_SALIDA}" ]; then
    rm -f "${RUTA_SALIDA}"
    log_info "Archivo anterior eliminado: ${ARCHIVO_ZIP}"
fi

# Eliminar imagen y contenedor si existen
docker rmi "${IMAGEN_NOMBRE}" 2>/dev/null || true
docker rm "${CONTENEDOR_NOMBRE}" 2>/dev/null || true

# ============================================================
# PASO 2: Verificar archivos fuente requeridos
# ============================================================
log_info "Verificando archivos fuente..."

ARCHIVOS_REQUERIDOS=(
    "Dockerfile.lambda"
    "requirements_lambda.txt"
    "lambda_handler.py"
    "src/__init__.py"
    "src/data_cleaner.py"
    "src/automaton_motor.py"
)

FALTAN=0
for archivo in "${ARCHIVOS_REQUERIDOS[@]}"; do
    if [ ! -f "${RUTA_PROYECTO}/${archivo}" ]; then
        log_error "Archivo faltante: ${archivo}"
        FALTAN=1
    fi
done

if [ "${FALTAN}" -eq 1 ]; then
    log_error "Faltan archivos requeridos. Abortando."
    exit 1
fi

log_exito "Todos los archivos fuente verificados."

# ============================================================
# PASO 3: Construir imagen Docker
# ============================================================
log_info "Construyendo imagen Docker (esto puede tardar 2-5 minutos)..."
log_info "Compilando dependencias para Linux x86_64..."

docker build \
    -f "${RUTA_PROYECTO}/Dockerfile.lambda" \
    -t "${IMAGEN_NOMBRE}" \
    "${RUTA_PROYECTO}"

if [ $? -ne 0 ]; then
    log_error "Error al construir la imagen Docker."
    exit 1
fi

log_exito "Imagen Docker construida exitosamente."

# ============================================================
# PASO 4: Extraer deployment_package.zip
# ============================================================
log_info "Extrayendo ${ARCHIVO_ZIP}..."

# Crear directorio temporal dentro del contenedor y copiar el ZIP
docker create --name "${CONTENEDOR_NOMBRE}" "${IMAGEN_NOMBRE}" true
docker cp "${CONTENEDOR_NOMBRE}:/output/${ARCHIVO_ZIP}" "${RUTA_SALIDA}"
docker rm "${CONTENEDOR_NOMBRE}" > /dev/null

if [ ! -f "${RUTA_SALIDA}" ]; then
    log_error "No se pudo extraer ${ARCHIVO_ZIP}."
    exit 1
fi

TAMANO=$(du -h "${RUTA_SALIDA}" | cut -f1)
log_exito "Archivo generado: ${ARCHIVO_ZIP} (${TAMANO})"

# ============================================================
# PASO 5: Limpiar imágenes Docker temporales
# ============================================================
log_info "Limpiando imágenes Docker temporales..."
docker rmi "${IMAGEN_NOMBRE}" 2>/dev/null || true
log_info "Limpieza completada."

# ============================================================
# INSTRUCCIONES DE DESPLIEGUE
# ============================================================
echo ""
echo -e "${CYAN}============================================================${SIN_COLOR}"
echo -e "${CYAN}  DESPLIEGUE EN AWS LAMBDA — INSTRUCCIONES${SIN_COLOR}"
echo -e "${CYAN}============================================================${SIN_COLOR}"
echo ""
echo -e "  ${VERDE}1.${SIN_COLOR} Ve a la consola de AWS Lambda:"
echo -e "     ${AMARILLO}https://console.aws.amazon.com/lambda/home${SIN_COLOR}"
echo ""
echo -e "  ${VERDE}2.${SIN_COLOR} Crea o selecciona tu función Lambda:"
echo -e "     - Nombre sugerido: ${AMARILLO}delfin-pipeline-trayectorias${SIN_COLOR}"
echo -e "     - Runtime: ${AMARILLO}Python 3.10${SIN_COLOR}"
echo -e "     - Arquitectura: ${AMARILLO}x86_64${SIN_COLOR}"
echo ""
echo -e "  ${VERDE}3.${SIN_COLOR} En la pestaña 'Code', selecciona 'Upload from':"
echo -e "     - Elige ${AMARILLO}'.zip file'${SIN_COLOR}"
echo -e "     - Sube: ${AMARILLO}${ARCHIVO_ZIP}${SIN_COLOR}"
echo ""
echo -e "  ${VERDE}4.${SIN_COLOR} Configura el Handler:"
echo -e "     - Handler: ${AMARILLO}lambda_handler.lambda_handler${SIN_COLOR}"
echo ""
echo -e "  ${VERDE}5.${SIN_COLOR} Configura las variables de entorno:"
echo -e "     - ${AMARILLO}BUCKET_ENTRADA=delfin-datos-entrada${SIN_COLOR}"
echo -e "     - ${AMARILLO}BUCKET_SALIDA=delfin-datos-procesados${SIN_COLOR}"
echo -e "     - ${AMARILLO}PPP_THRESHOLD=3.2${SIN_COLOR}"
echo -e "     - ${AMARILLO}PPA_THRESHOLD=3.2${SIN_COLOR}"
echo ""
echo -e "  ${VERDE}6.${SIN_COLOR} Configura el trigger S3:"
echo -e "     - Bucket: ${AMARILLO}delfin-datos-entrada${SIN_COLOR}"
echo -e "     - Eventos: ${AMARILLO}PUT (s3:ObjectCreated:*)${SIN_COLOR}"
echo -e "     - Sufijo: ${AMARILLO}.xlsx${SIN_COLOR}"
echo ""
echo -e "  ${VERDE}7.${SIN_COLOR} Configuración de memoria y timeout:"
echo -e "     - Memoria: ${AMARILLO}1024 MB${SIN_COLOR} (mínimo recomendado)"
echo -e "     - Timeout: ${AMARILLO}300 segundos${SIN_COLOR}"
echo ""
echo -e "${CYAN}============================================================${SIN_COLOR}"
echo -e "${VERDE}  Empaquetado completado. Archivo listo para desplegar.${SIN_COLOR}"
echo -e "${CYAN}============================================================${SIN_COLOR}"
echo ""
