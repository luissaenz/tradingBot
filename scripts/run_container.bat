@echo off
ECHO Ejecutando scripts en contenedor Docker...

:: Menú de opciones
ECHO Seleccione el script a ejecutar:
ECHO 1 - Inicializar base de datos (init_db.py)
ECHO 2 - Ejecutar bot de trading
ECHO 3 - Ejecutar backtesting
set /p choice="Ingrese opción (1-3): "

:: Ejecutar script según selección
IF "%choice%"=="1" (
    docker exec tradingbot-python python /app/scripts/init_db.py
) ELSE IF "%choice%"=="2" (
    docker exec tradingbot-python python /app/src/main.py
) ELSE IF "%choice%"=="3" (
    docker exec tradingbot-python python /app/src/backtesting/main.py
) ELSE (
    ECHO Opción inválida
)

pause
