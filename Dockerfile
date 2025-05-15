# Use Python 3.10 slim image
FROM python:3.10-slim

# Install system dependencies required for LightGBM and other packages
RUN apt-get update && apt-get install -y \
    build-essential \
    libgomp1 \
    cmake \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements file
COPY requirements.txt ./

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# If requirements.txt is missing, install necessary packages directly
RUN apt-get update && apt-get install -y libgomp1

RUN pip list | grep -q "ccxt" || pip install --no-cache-dir \
    ccxt \
    scikit-learn \
    lightgbm \
    pandas \
    numpy \
    sqlalchemy \
    psycopg2-binary \
    python-dotenv

RUN pip install scikit-learn lightgbm pandas numpy psycopg2-binary python-dotenv ccxt


# Copy source code
COPY src/ /app/src/

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH

# Default command
CMD ["python"]
