# scripts/start-config-management.ps1
# Script para iniciar el sistema de gestión de configuración dinámica

param(
    [switch]$Build,
    [switch]$Clean,
    [switch]$Logs,
    [string]$Service = "all"
)

$ErrorActionPreference = "Stop"

Write-Host "🚀 Iniciando Sistema de Gestión de Configuración Dinámica" -ForegroundColor Green

# Verificar que Docker esté ejecutándose
try {
    docker version | Out-Null
} catch {
    Write-Error "Docker no está ejecutándose. Por favor, inicia Docker Desktop."
    exit 1
}

# Limpiar contenedores y volúmenes si se solicita
if ($Clean) {
    Write-Host "🧹 Limpiando contenedores y volúmenes..." -ForegroundColor Yellow
    docker-compose down -v --remove-orphans
    docker system prune -f
}

# Construir imágenes si se solicita
if ($Build) {
    Write-Host "🔨 Construyendo imágenes Docker..." -ForegroundColor Yellow
    docker-compose build --no-cache
}

# Crear directorios necesarios
$directories = @(
    "logs",
    "config/grafana/dashboards",
    "config/grafana/datasources",
    "data/postgres",
    "data/redis",
    "data/minio"
)

foreach ($dir in $directories) {
    if (!(Test-Path $dir)) {
        New-Item -ItemType Directory -Path $dir -Force | Out-Null
        Write-Host "📁 Creado directorio: $dir" -ForegroundColor Cyan
    }
}

# Iniciar servicios
if ($Service -eq "all") {
    Write-Host "🐳 Iniciando todos los servicios..." -ForegroundColor Yellow
    docker-compose up -d
} else {
    Write-Host "🐳 Iniciando servicio: $Service" -ForegroundColor Yellow
    docker-compose up -d $Service
}

# Esperar a que los servicios estén listos
Write-Host "⏳ Esperando a que los servicios estén listos..." -ForegroundColor Yellow
Start-Sleep -Seconds 10

# Verificar estado de los servicios
Write-Host "📊 Estado de los servicios:" -ForegroundColor Green
docker-compose ps

# Verificar conectividad de servicios clave
$services = @{
    "PostgreSQL" = "http://localhost:5432"
    "Redis" = "http://localhost:6379"
    "Config Management API" = "http://localhost:8080/api/config/health"
    "Grafana" = "http://localhost:3000"
    "Prometheus" = "http://localhost:9090"
    "MinIO Console" = "http://localhost:9001"
}

Write-Host "`n🔍 Verificando conectividad de servicios:" -ForegroundColor Green

foreach ($service in $services.GetEnumerator()) {
    try {
        if ($service.Key -eq "PostgreSQL" -or $service.Key -eq "Redis") {
            # Para PostgreSQL y Redis, solo verificamos que el contenedor esté ejecutándose
            $containerName = if ($service.Key -eq "PostgreSQL") { "trading_postgres" } else { "trading_redis" }
            $status = docker inspect --format='{{.State.Status}}' $containerName 2>$null
            if ($status -eq "running") {
                Write-Host "  ✅ $($service.Key): Ejecutándose" -ForegroundColor Green
            } else {
                Write-Host "  ❌ $($service.Key): No disponible" -ForegroundColor Red
            }
        } else {
            # Para servicios HTTP, hacer una petición
            $response = Invoke-WebRequest -Uri $service.Value -Method GET -TimeoutSec 5 -UseBasicParsing -ErrorAction SilentlyContinue
            if ($response.StatusCode -eq 200) {
                Write-Host "  ✅ $($service.Key): Disponible" -ForegroundColor Green
            } else {
                Write-Host "  ⚠️  $($service.Key): Respuesta inesperada" -ForegroundColor Yellow
            }
        }
    } catch {
        Write-Host "  ❌ $($service.Key): No disponible" -ForegroundColor Red
    }
}

# Mostrar URLs útiles
Write-Host "`n🌐 URLs de acceso:" -ForegroundColor Cyan
Write-Host "  • Config Management API: http://localhost:8080/docs" -ForegroundColor White
Write-Host "  • Web Frontend: http://localhost:3001" -ForegroundColor White
Write-Host "  • Grafana: http://localhost:3000 (admin/admin_pass)" -ForegroundColor White
Write-Host "  • Prometheus: http://localhost:9090" -ForegroundColor White
Write-Host "  • MinIO Console: http://localhost:9001 (trading_admin/trading_admin_pass)" -ForegroundColor White

# Mostrar logs si se solicita
if ($Logs) {
    Write-Host "`n📋 Mostrando logs..." -ForegroundColor Yellow
    docker-compose logs -f
}

Write-Host "`n✅ Sistema de Gestión de Configuración iniciado correctamente!" -ForegroundColor Green
Write-Host "💡 Usa 'docker-compose logs -f [servicio]' para ver logs específicos" -ForegroundColor Cyan
Write-Host "💡 Usa 'docker-compose down' para detener todos los servicios" -ForegroundColor Cyan
