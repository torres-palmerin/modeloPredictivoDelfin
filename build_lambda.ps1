<#
.SYNOPSIS
    Empaquetado de AWS Lambda — Proyecto Modelo Predictivo (Estancia Delfín).

.DESCRIPTION
    Construye una imagen Docker con dependencias compiladas para Linux x86_64
    y extrae el deployment_package.zip final para subir a AWS Lambda.

.USO
    .\build_lambda.ps1

.NOTAS
    Requiere: Docker Desktop en ejecución
    Runtime:  Python 3.10 (AWS Lambda)
#>

# ============================================================
# Configuración
# ============================================================
$ErrorActionPreference = "Stop"

$IMAGEN_NOMBRE   = "delfin-lambda-builder"
$CONTENEDOR_NOMBRE = "delfin-lambda-extractor"
$ARCHIVO_ZIP     = "deployment_package.zip"
$RUTA_PROYECTO   = $PSScriptRoot
$RUTA_SALIDA     = Join-Path $RUTA_PROYECTO $ARCHIVO_ZIP

# Funciones de logging
function Log-Info    { param([string]$Msg) Write-Host "[INFO]    $Msg" -ForegroundColor Cyan }
function Log-Exito   { param([string]$Msg) Write-Host "[ÉXITO]   $Msg" -ForegroundColor Green }
function Log-Aviso   { param([string]$Msg) Write-Host "[AVISO]   $Msg" -ForegroundColor Yellow }
function Log-Error   { param([string]$Msg) Write-Host "[ERROR]   $Msg" -ForegroundColor Red }

# ============================================================
# PASO 0: Validar prerrequisitos
# ============================================================
Log-Info "Validando prerrequisitos..."

try {
    $dockerVersion = docker --version 2>$null
    if (-not $dockerVersion) {
        throw "Docker no encontrado"
    }
    Log-Exito "Docker detectado: $dockerVersion"
}
catch {
    Log-Error "Docker no está instalado o no está en el PATH."
    Log-Error "Instala Docker Desktop: https://docs.docker.com/get-docker/"
    exit 1
}

# Verificar que Docker esté ejecutándose
try {
    $dockerInfo = docker info 2>$null
    if (-not $dockerInfo) {
        throw "Daemon no activo"
    }
}
catch {
    Log-Error "El daemon de Docker no está ejecutándose."
    Log-Error "Inicia Docker Desktop y vuelve a intentar."
    exit 1
}

# ============================================================
# PASO 1: Limpiar empaquetados anteriores
# ============================================================
Log-Info "Limpiando artefactos anteriores..."

if (Test-Path $RUTA_SALIDA) {
    Remove-Item $RUTA_SALIDA -Force
    Log-Info "Archivo anterior eliminado: $ARCHIVO_ZIP"
}

# Eliminar imagen y contenedor si existen
docker rmi $IMAGEN_NOMBRE 2>$null | Out-Null
docker rm $CONTENEDOR_NOMBRE 2>$null | Out-Null

# ============================================================
# PASO 2: Verificar archivos fuente requeridos
# ============================================================
Log-Info "Verificando archivos fuente..."

$ARCHIVOS_REQUERIDOS = @(
    "Dockerfile.lambda",
    "requirements_lambda.txt",
    "lambda_handler.py",
    "src\__init__.py",
    "src\data_cleaner.py",
    "src\automaton_motor.py"
)

$FALTAN = $false
foreach ($archivo in $ARCHIVOS_REQUERIDOS) {
    $rutaCompleta = Join-Path $RUTA_PROYECTO $archivo
    if (-not (Test-Path $rutaCompleta)) {
        Log-Error "Archivo faltante: $archivo"
        $FALTAN = $true
    }
}

if ($FALTAN) {
    Log-Error "Faltan archivos requeridos. Abortando."
    exit 1
}

Log-Exito "Todos los archivos fuente verificados."

# ============================================================
# PASO 3: Construir imagen Docker
# ============================================================
Log-Info "Construyendo imagen Docker (esto puede tardar 2-5 minutos)..."
Log-Info "Compilando dependencias para Linux x86_64..."

$dockerfileRuta = Join-Path $RUTA_PROYECTO "Dockerfile.lambda"

docker build `
    -f $dockerfileRuta `
    -t $IMAGEN_NOMBRE `
    $RUTA_PROYECTO

if ($LASTEXITCODE -ne 0) {
    Log-Error "Error al construir la imagen Docker."
    exit 1
}

Log-Exito "Imagen Docker construida exitosamente."

# ============================================================
# PASO 4: Extraer deployment_package.zip
# ============================================================
Log-Info "Extrayendo artefactos del contenedor..."

$TEMP_DIR = Join-Path $RUTA_PROYECTO ".lambda_build_tmp"

# Limpiar directorio temporal si existe
if (Test-Path $TEMP_DIR) {
    Remove-Item $TEMP_DIR -Recurse -Force
}

docker create --name $CONTENEDOR_NOMBRE $IMAGEN_NOMBRE true 2>$null | Out-Null

# Copiar todo el contenido de LAMBDA_TASK_ROOT (/var/task) a directorio temporal
docker cp "${CONTENEDOR_NOMBRE}:/var/task/." $TEMP_DIR
docker rm $CONTENEDOR_NOMBRE 2>$null | Out-Null

if (-not (Test-Path $TEMP_DIR)) {
    Log-Error "No se pudieron extraer los artefactos del contenedor."
    exit 1
}

# Comprimir en ZIP
Log-Info "Comprimiendo artefactos en $ARCHIVO_ZIP..."
Compress-Archive -Path "$TEMP_DIR\*" -DestinationPath $RUTA_SALIDA -Force

# Limpiar directorio temporal
Remove-Item $TEMP_DIR -Recurse -Force

if (-not (Test-Path $RUTA_SALIDA)) {
    Log-Error "No se pudo extraer $ARCHIVO_ZIP."
    exit 1
}

$tamanoBytes = (Get-Item $RUTA_SALIDA).Length
$tamanoMB = [math]::Round($tamanoBytes / 1MB, 2)
Log-Exito "Archivo generado: $ARCHIVO_ZIP (${tamanoMB} MB)"

# ============================================================
# PASO 5: Limpiar imágenes Docker temporales
# ============================================================
Log-Info "Limpiando imágenes Docker temporales..."
docker rmi $IMAGEN_NOMBRE 2>$null | Out-Null
Log-Info "Limpieza completada."

# ============================================================
# INSTRUCCIONES DE DESPLIEGUE
# ============================================================
Write-Host ""
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "  DESPLIEGUE EN AWS LAMBDA — INSTRUCCIONES" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "  1. Ve a la consola de AWS Lambda:" -ForegroundColor White
Write-Host "     https://console.aws.amazon.com/lambda/home" -ForegroundColor Yellow
Write-Host ""
Write-Host "  2. Crea o selecciona tu función Lambda:" -ForegroundColor White
Write-Host "     - Nombre sugerido: delfin-pipeline-trayectorias" -ForegroundColor Yellow
Write-Host "     - Runtime: Python 3.10" -ForegroundColor Yellow
Write-Host "     - Arquitectura: x86_64" -ForegroundColor Yellow
Write-Host ""
Write-Host "  3. En la pestaña 'Code', selecciona 'Upload from':" -ForegroundColor White
Write-Host "     - Elige '.zip file'" -ForegroundColor Yellow
Write-Host "     - Sube: $ARCHIVO_ZIP" -ForegroundColor Yellow
Write-Host ""
Write-Host "  4. Configura el Handler:" -ForegroundColor White
Write-Host "     - Handler: lambda_handler.lambda_handler" -ForegroundColor Yellow
Write-Host ""
Write-Host "  5. Configura las variables de entorno:" -ForegroundColor White
Write-Host "     - BUCKET_ENTRADA=delfin-datos-entrada" -ForegroundColor Yellow
Write-Host "     - BUCKET_SALIDA=delfin-datos-procesados" -ForegroundColor Yellow
Write-Host "     - PPP_THRESHOLD=3.2" -ForegroundColor Yellow
Write-Host "     - PPA_THRESHOLD=3.2" -ForegroundColor Yellow
Write-Host ""
Write-Host "  6. Configura el trigger S3:" -ForegroundColor White
Write-Host "     - Bucket: delfin-datos-entrada" -ForegroundColor Yellow
Write-Host "     - Eventos: PUT (s3:ObjectCreated:*)" -ForegroundColor Yellow
Write-Host "     - Sufijo: .xlsx" -ForegroundColor Yellow
Write-Host ""
Write-Host "  7. Configuración de memoria y timeout:" -ForegroundColor White
Write-Host "     - Memoria: 1024 MB (mínimo recomendado)" -ForegroundColor Yellow
Write-Host "     - Timeout: 300 segundos" -ForegroundColor Yellow
Write-Host ""
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "  Empaquetado completado. Archivo listo para desplegar." -ForegroundColor Green
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""
