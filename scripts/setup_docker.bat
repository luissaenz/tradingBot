@echo off
ECHO Configurando entorno Docker para tradingBot...

:: Verificar si Docker Desktop esta corriendo
docker info >nul 2>&1
IF %ERRORLEVEL% NEQ 0 (
    ECHO Error: Docker Desktop no esta corriendo o no esta configurado correctamente.
    ECHO Asegúrate de que Docker Desktop esta instalado y usa WSL2 como backend.
    ECHO Inicia Docker Desktop y verifica con 'docker --version' y 'docker run hello-world'.
    pause
    exit /b 1
)

:: Crear directorios para volúmenes
ECHO Creando directorios para volúmenes...
mkdir D:\Develop\Personal\tradingBot\data\timescaledb 2>nul
mkdir D:\Develop\Personal\tradingBot\data\redis 2>nul
mkdir D:\Develop\Personal\tradingBot\data\minio 2>nul
mkdir D:\Develop\Personal\tradingBot\data\grafana 2>nul
mkdir D:\Develop\Personal\tradingBot\config\prometheus 2>nul

:: Iniciar contenedores Docker
ECHO Iniciando contenedores Docker...
cd /d D:\Develop\Personal\tradingBot
docker-compose up -d
IF %ERRORLEVEL% NEQ 0 (
    ECHO Error: No se pudieron iniciar los contenedores. Revisa docker-compose.yml y Docker Desktop.
    pause
    exit /b 1
)

:: Verificar si el contenedor tradingbot-python-1 existe
docker ps -q -f name=tradingbot-python-1 >nul
IF %ERRORLEVEL% NEQ 0 (
    ECHO Error: El contenedor tradingbot-python-1 no esta corriendo.
    ECHO Verifica 'docker ps -a' para ver si el contenedor existe o revisa los logs con 'docker logs tradingbot-python-1'.
    pause
    exit /b 1
)

:: Instalar dependencias en contenedor Python
ECHO Instalando dependencias en contenedor Python...
docker exec tradingbot-python-1 pip install -r /app/requirements.txt
IF %ERRORLEVEL% NEQ 0 (
    ECHO Error: No se pudieron instalar dependencias en el contenedor Python.
    ECHO Verifica que requirements.txt existe en D:\Develop\Personal\tradingBot y revisa los logs.
    pause
    exit /b 1
)

ECHO Configuración Docker completada.
pause