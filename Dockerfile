# Multi-stage build for MozhiGPT Tamil ChatGPT
FROM python:3.10-slim as base

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd --create-home --shell /bin/bash app

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY . .

# Change ownership to app user
RUN chown -R app:app /app

# Switch to non-root user
USER app

# Production stage
FROM base as production

# Set environment variables for production
ENV PYTHONPATH=/app \
    MODEL_PATH=/app/models/tamil-gpt \
    TOKENIZER_VOCAB_PATH=/app/tamil_tokenizers/tamil_vocab.json \
    USE_CUSTOM_TOKENIZER=true \
    MAX_NEW_TOKENS=256 \
    TEMPERATURE=0.7 \
    TOP_K=50 \
    TOP_P=0.9 \
    DEVICE=auto \
    HOST=0.0.0.0 \
    PORT=8000 \
    WORKERS=1 \
    RELOAD=false

# Create necessary directories
RUN mkdir -p /app/models/tamil-gpt /app/tamil_tokenizers /app/data

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Default command
CMD ["python", "-m", "uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]

# Development stage
FROM base as development

# Set environment variables for development
ENV PYTHONPATH=/app \
    MODEL_PATH=/app/models/tamil-gpt \
    TOKENIZER_VOCAB_PATH=/app/tamil_tokenizers/tamil_vocab.json \
    USE_CUSTOM_TOKENIZER=true \
    MAX_NEW_TOKENS=256 \
    TEMPERATURE=0.7 \
    DEVICE=auto \
    HOST=0.0.0.0 \
    PORT=8000 \
    WORKERS=1 \
    RELOAD=true

# Install development dependencies
RUN pip install --no-cache-dir \
    jupyter \
    notebook \
    ipython \
    black \
    flake8 \
    pytest

# Expose ports (API + Jupyter)
EXPOSE 8000 8888

# Default command for development
CMD ["python", "-m", "uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
