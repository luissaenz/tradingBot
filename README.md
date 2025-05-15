# tradingBot - Bot Institucional de Trading BTC/USD

Bot de trading cuantitativo para futuros perpetuos BTC/USD con estrategia trend-following basada en:

- Order book imbalances
- Delta volume
- Sentimiento en X

## Requisitos del Sistema

- Windows 11
- Docker Desktop con WSL2
- Python 3.10
- 8 núcleos CPU
- 16 GB RAM
- 500 GB SSD

## Estructura del Proyecto

```
tradingBot/
├── src/                # Código fuente
│   ├── ingestion/      # Binance WebSocket y API de X
│   ├── microstructure/ # Cálculo de imbalances y delta volume
│   ├── sentiment/      # Procesamiento con FinBERT
│   ├── signals/        # Generación de señales con LightGBM
│   ├── execution/      # Ejecución en Binance
│   └── ...             # Otros módulos
├── config/             # Archivos de configuración
├── scripts/            # Scripts de instalación y ejecución
├── data/               # Volúmenes Docker
└── docs/               # Documentación
```

## Instalación

1. **Pre-requisitos**:

   - Instalar [Docker Desktop](https://docker.com/products/docker-desktop)
   - Habilitar WSL2
   - Instalar [Python 3.10](https://python.org/downloads)

2. **Configuración inicial**:

```bash
git clone https://github.com/tu_usuario/tradingBot.git
cd tradingBot
```

3. **Instalar dependencias**:

```bash
scripts\install.bat
```

4. **Configurar Docker**:

```bash
scripts\setup_docker.bat
```

5. **Inicializar base de datos**:

```bash
scripts\run_container.bat
# Seleccionar opción 1
```

6. **Configurar variables de entorno**:
   - Editar `.env` con tus credenciales de API
   - Configurar parámetros en `config/config.yaml`

## Uso

Para ejecutar el bot:

```bash
scripts\run_container.bat
# Seleccionar opción 2
```

Para backtesting:

```bash
scripts\run_container.bat
# Seleccionar opción 3
```

## Servicios Docker

- **TimescaleDB**: `localhost:5432`
- **Redis**: `localhost:6379`
- **MinIO**: `localhost:9000` (Consola: `localhost:9001`)
- **Prometheus**: `localhost:9090`
- **Grafana**: `localhost:3000`

## Notas

- Configura los archivos `.env` y `config/config.yaml` antes de ejecutar
- Para desarrollo local, activa el entorno virtual con `venv\Scripts\activate.bat`
