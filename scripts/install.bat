@echo off
ECHO Iniciando instalacion de dependencias para tradingBot...

:: Verificar Python 3.10
python --version | findstr /C:"3.10" >nul
IF %ERRORLEVEL% NEQ 0 (
    ECHO Error: Python 3.10 no esta instalado. Descargalo desde https://python.org/downloads.
    ECHO Asegurate de marcar "Add Python to PATH".
    pause
    exit /b 1
)

:: Verificar pip global
python -m pip --version >nul
IF %ERRORLEVEL% NEQ 0 (
    ECHO Error: pip no esta instalado. Instalando pip...
    curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
    python get-pip.py
    del get-pip.py
    python -m pip --version >nul
    IF %ERRORLEVEL% NEQ 0 (
        ECHO Error: No se pudo instalar pip. Revisa tu instalacion de Python.
        pause
        exit /b 1
    )
)

:: Cambiar al directorio del proyecto
cd /d D:\Develop\Personal\tradingBot

:: Crear entorno virtual
IF NOT EXIST "venv" (
    ECHO Creando entorno virtual...
    python -m venv venv
)

:: Activar entorno virtual
ECHO Activando entorno virtual...
call venv\Scripts\activate.bat

:: Asegurar pip en el entorno virtual
python -m ensurepip --upgrade
python -m pip install --upgrade pip

:: Verificar requirements.txt
IF NOT EXIST "requirements.txt" (
    ECHO Creando requirements.txt con dependencias basicas...
    (
        echo ccxt
        echo python-binance
        echo transformers
        echo lightgbm
        echo pandas
        echo numpy
        echo psycopg2-binary
        echo backtrader
        echo redis
        echo minio
        echo prometheus-client
    ) > requirements.txt
)

:: Instalar dependencias
ECHO Instalando dependencias...
pip install -r requirements.txt

:: Verificar instalacion
IF %ERRORLEVEL% EQU 0 (
    ECHO Instalacion completada exitosamente.
) ELSE (
    ECHO Error durante la instalacion de dependencias. Revisa los errores arriba.
)

ECHO Entorno virtual activado. Usa 'deactivate' para salir.
pause