# ============================================================================
# Script de Despliegue Automatizado: Docker a AWS ECR y Lambda
# Proyecto: Modelo Predictivo (Estancia Delfín)
# ============================================================================

$AWS_ACCOUNT_ID = "102726256690"
$AWS_REGION     = "us-east-2"   # Ajusta si tu infraestructura está en otra región
$REPO_NAME      = "delfin-lambda-inference"
$LAMBDA_NAME    = "DelfinInferenceHandler"

Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "🚀 INICIANDO DESPLIEGUE DEL CONTENEDOR SERVERLESS" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan

# 1. Autenticar Docker con Amazon ECR
Write-Host "`n🔐 1. Autenticando Docker con Amazon ECR..." -ForegroundColor Yellow
aws ecr get-login-password --region $AWS_REGION | docker login --username AWS --password-stdin "$AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com"

if ($LASTEXITCODE -ne 0) {
    Write-Error "❌ Error al autenticar con AWS ECR. Verifica que Docker Desktop esté abierto."
    exit $LASTEXITCODE
}

# 2. Crear el repositorio en ECR si no existe
Write-Host "`n📦 2. Verificando existencia del repositorio ECR..." -ForegroundColor Yellow
aws ecr describe-repositories --repository-names $REPO_NAME --region $AWS_REGION 2>$null
if ($LASTEXITCODE -ne 0) {
    Write-Host "   El repositorio no existe. Creando '$REPO_NAME'..." -ForegroundColor Cyan
    aws ecr create-repository --repository-name $REPO_NAME --region $AWS_REGION
}

# 3. Construir la imagen Docker localmente asegurando compatibilidad estricta con AWS Lambda
Write-Host "`n🏗️ 3. Construyendo imagen Docker local desde Dockerfile.lambda..." -ForegroundColor Yellow
$env:BUILDX_NO_DEFAULT_ATTESTATIONS=1
docker build --no-cache --provenance=false --platform=linux/amd64 --output type=docker -t ($REPO_NAME + ":latest") -f Dockerfile.lambda .

if ($LASTEXITCODE -ne 0) {
    Write-Error "❌ Error durante la compilación del Dockerfile."
    exit $LASTEXITCODE
}

# 4. Taggear la imagen con la ruta del registro de AWS
Write-Host "`n🏷️ 4. Etiquetando imagen para producción..." -ForegroundColor Yellow
$ECR_IMAGE_URI = $AWS_ACCOUNT_ID + ".dkr.ecr." + $AWS_REGION + ".amazonaws.com/" + $REPO_NAME + ":latest"
docker tag ($REPO_NAME + ":latest") $ECR_IMAGE_URI

# 5. Subir la imagen a AWS ECR (Push)
Write-Host "`n⬆️ 5. Subiendo imagen de Docker a AWS ECR..." -ForegroundColor Yellow
docker push $ECR_IMAGE_URI

# 6. Actualizar el código de la función AWS Lambda con la nueva imagen
Write-Host "`n⚡ 6. Actualizando AWS Lambda con el nuevo contenedor..." -ForegroundColor Yellow
aws lambda update-function-code --function-name $LAMBDA_NAME --image-uri $ECR_IMAGE_URI --region $AWS_REGION

if ($LASTEXITCODE -ne 0) {
    Write-Host "`n⚠️  No se pudo actualizar la Lambda directamente." -ForegroundColor DarkYellow
    Write-Host "   Si aún no has creado la función Lambda en la consola web, créala seleccionando" -ForegroundColor Gray
    Write-Host "   la opción 'Container Image' y pega esta URI de imagen:" -ForegroundColor Gray
    Write-Host "   👉 $ECR_IMAGE_URI" -ForegroundColor Cyan
} else {
    Write-Host "`n🎉 ¡DESPLIEGUE COMPLETADO EXITOSAMENTE!" -ForegroundColor Green
    Write-Host "   Tu modelo e inferencia están sincronizados en caliente en la Lambda." -ForegroundColor Green
}
Write-Host "============================================================" -ForegroundColor Cyan