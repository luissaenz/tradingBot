# docker/config-management.dockerfile
FROM python:3.11-slim

WORKDIR /app

# Instalar dependencias del sistema
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copiar requirements
COPY requirements/config-management.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copiar código fuente
COPY src/ src/
COPY config/ config/

# Variables de entorno
ENV PYTHONPATH=/app
ENV ENVIRONMENT=production

# Puerto para API
EXPOSE 8080

# Comando por defecto
CMD ["python", "-m", "src.api.config_management_api"]
