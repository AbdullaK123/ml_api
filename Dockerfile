# syntax=docker/dockerfile:1.4

# ===== Build Stage =====
FROM ubuntu:22.04 AS builder

# Build arguments
ARG PYTHON_VERSION=3.10
ARG POETRY_VERSION=1.4.2
ARG DEBIAN_FRONTEND=noninteractive

# Set build-time environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=off \
    PIP_DISABLE_PIP_VERSION_CHECK=on \
    POETRY_VERSION=$POETRY_VERSION \
    POETRY_HOME="/opt/poetry" \
    POETRY_VIRTUALENVS_IN_PROJECT=true \
    POETRY_NO_INTERACTION=1 \
    POETRY_INSTALLER_MAX_WORKERS=10 \
    PATH="/opt/poetry/bin:$PATH" \
    PYTHON_VERSION=$PYTHON_VERSION

WORKDIR /app

# Install build dependencies efficiently
RUN apt-get update && apt-get install -y --no-install-recommends \
    python${PYTHON_VERSION} \
    python${PYTHON_VERSION}-dev \
    python${PYTHON_VERSION}-distutils \
    python3-pip \
    curl \
    build-essential \
    libpq-dev \
    git \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.10 as default
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python${PYTHON_VERSION} 1 \
    && update-alternatives --install /usr/bin/python python /usr/bin/python${PYTHON_VERSION} 1

# Install latest pip
RUN curl -sSL https://bootstrap.pypa.io/get-pip.py | python3 \
    && python3 -m pip install --upgrade pip

# Install Poetry
RUN curl -sSL https://install.python-poetry.org | python3 - \
    && ln -s /opt/poetry/bin/poetry /usr/local/bin/poetry

# Copy dependency files
COPY pyproject.toml poetry.lock README.rst ./

# Create empty README.md for dependencies that expect it
RUN touch README.md

# Install dependencies efficiently
RUN poetry config installer.max-workers 10 && \
    poetry config installer.parallel false && \
    poetry install --only main --no-interaction --no-ansi --no-root --no-dev

# Copy application files
COPY ml_api/ ./ml_api/
COPY static/ ./static/
COPY templates/ ./templates/
COPY startup.sh ./

# ===== Runtime Stage =====
FROM ubuntu:22.04 AS runtime

# Runtime arguments
ARG PYTHON_VERSION=3.10
ARG DEBIAN_FRONTEND=noninteractive

# Set runtime environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=8000 \
    HOST=0.0.0.0 \
    WORKERS=2 \
    TIMEOUT=120 \
    MAX_REQUESTS=250 \
    MAX_REQUESTS_JITTER=25 \
    KEEP_ALIVE=30 \
    GRACEFUL_TIMEOUT=30 \
    MODEL_CACHE_DIR=/app/model_cache \
    MEMORY_MONITOR_THRESHOLD=3500 \
    MODEL_TIMEOUT=300 \
    PYTHON_VERSION=$PYTHON_VERSION \
    PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512 \
    MALLOC_TRIM_THRESHOLD_=100000 \
    OMP_NUM_THREADS=2 \
    MKL_NUM_THREADS=2 \
    NUMEXPR_NUM_THREADS=2 \
    PYTHONWARNINGS="ignore" \
    PATH="/app/.venv/bin:$PATH" \
    PYTHONPATH="/app:$PYTHONPATH" \
    PADDLEOCR_HOME=/app/model_cache/paddleocr \
    HF_HOME=/app/model_cache/huggingface

WORKDIR /app

# Install runtime dependencies efficiently
RUN apt-get update && apt-get install -y --no-install-recommends \
    python${PYTHON_VERSION} \
    python${PYTHON_VERSION}-distutils \
    libpq-dev \
    libopencv-dev \
    python3-opencv \
    ffmpeg \
    libsm6 \
    libxext6 \
    curl \
    linux-tools-generic \
    libjemalloc2 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* \
    && ln -s /usr/lib/x86_64-linux-gnu/libjemalloc.so.2 /usr/local/lib/libjemalloc.so

# Set Python 3.10 as default
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python${PYTHON_VERSION} 1 \
    && update-alternatives --install /usr/bin/python python /usr/bin/python${PYTHON_VERSION} 1

# Copy Python dependencies and application from builder
COPY --from=builder /app/.venv /app/.venv
COPY --from=builder /app/ml_api /app/ml_api
COPY --from=builder /app/static /app/static
COPY --from=builder /app/templates /app/templates
COPY --from=builder /app/startup.sh /app/startup.sh

# Create necessary directories and set permissions
RUN mkdir -p /app/model_cache/paddleocr/whl/det/en \
    && mkdir -p /app/model_cache/paddleocr/whl/rec/en \
    && mkdir -p /app/model_cache/huggingface \
    && chmod -R 777 /app/model_cache \
    && chmod -R 755 /app/static /app/templates \
    && chmod +x /app/startup.sh

# Pre-download models
RUN python3 -c "from paddleocr import PaddleOCR; ocr = PaddleOCR(use_angle_cls=True, lang='en', use_gpu=False, home_path='/app/model_cache/paddleocr')" || true

# Create optimized startup script with proper model paths
RUN echo '#!/bin/bash\n\
# Enable jemalloc for better memory management\n\
export LD_PRELOAD=/usr/local/lib/libjemalloc.so\n\
\n\
# Set Python garbage collection thresholds\n\
export PYTHONGC="1000,100,100"\n\
\n\
# Set model cache directories\n\
export PADDLEOCR_HOME=/app/model_cache/paddleocr\n\
export HF_HOME=/app/model_cache/huggingface\n\
\n\
# Enable memory profiling\n\
export MALLOC_CONF="background_thread:true,metadata_thp:auto,dirty_decay_ms:30000,muzzy_decay_ms:30000"\n\
\n\
# Start the application with optimized settings\n\
exec python -m uvicorn ml_api.inference_api:app \\\n\
    --host $HOST \\\n\
    --port $PORT \\\n\
    --workers $WORKERS \\\n\
    --timeout-keep-alive 75 \\\n\
    --limit-concurrency 20 \\\n\
    --backlog 2048 \\\n\
    --proxy-headers \\\n\
    --forwarded-allow-ips "*" \\\n\
    --no-access-log \\\n\
    --log-level warning' > /app/entrypoint.sh \
    && chmod +x /app/entrypoint.sh

# Set up model cache volume
VOLUME ["/app/model_cache"]

# Expose port
EXPOSE 8000

# Run the optimized startup script
CMD ["/app/entrypoint.sh"]

# Add healthcheck
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1