# Build stage
FROM ubuntu:22.04 AS builder

# Prevent timezone prompts
ENV DEBIAN_FRONTEND=noninteractive

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=off \
    PIP_DISABLE_PIP_VERSION_CHECK=on \
    POETRY_VERSION=1.4.2 \
    POETRY_HOME="/opt/poetry" \
    POETRY_VIRTUALENVS_IN_PROJECT=true \
    POETRY_NO_INTERACTION=1 \
    POETRY_INSTALLER_MAX_WORKERS=10 \
    PYTHON_VERSION=3.10

WORKDIR /app

# Install Python and build dependencies
RUN apt-get update && apt-get install -y \
    python${PYTHON_VERSION} \
    python${PYTHON_VERSION}-dev \
    python${PYTHON_VERSION}-distutils \
    python3-pip \
    curl \
    build-essential \
    libpq-dev \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.10 as default
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python${PYTHON_VERSION} 1 \
    && update-alternatives --install /usr/bin/python python /usr/bin/python${PYTHON_VERSION} 1

# Upgrade pip
RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3 \
    && python3 -m pip install --upgrade pip

# Install Poetry
RUN curl -sSL https://install.python-poetry.org | python3 - \
    && ln -s /opt/poetry/bin/poetry /usr/local/bin/poetry

# Copy dependency files
COPY pyproject.toml poetry.lock README.rst ./

# Install dependencies
RUN poetry config installer.max-workers 10 && \
    poetry config installer.parallel false && \
    poetry install --only main --no-interaction --no-ansi

# Copy application files
COPY ml_api/ ./ml_api/
COPY static/ ./static/
COPY templates/ ./templates/
COPY startup.sh ./

# Runtime stage
FROM ubuntu:22.04

# Prevent timezone prompts
ENV DEBIAN_FRONTEND=noninteractive

# Set runtime environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=8000 \
    HOST=0.0.0.0 \
    WORKERS=1 \
    TIMEOUT=120 \
    MAX_REQUESTS=100 \
    KEEP_ALIVE=5 \
    MODEL_CACHE_DIR=/app/model_cache \
    PYTHON_VERSION=3.10

WORKDIR /app

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    python${PYTHON_VERSION} \
    python${PYTHON_VERSION}-distutils \
    libpq-dev \
    libopencv-dev \
    python3-opencv \
    ffmpeg \
    libsm6 \
    libxext6 \
    curl \
    --no-install-recommends \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.10 as default
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python${PYTHON_VERSION} 1 \
    && update-alternatives --install /usr/bin/python python /usr/bin/python${PYTHON_VERSION} 1

# Copy Python dependencies and application from builder
COPY --from=builder /app/.venv /app/.venv
COPY --from=builder /app/ml_api /app/ml_api
COPY --from=builder /app/static /app/static
COPY --from=builder /app/templates /app/templates
COPY --from=builder /app/startup.sh /app/startup.sh

ENV PATH="/app/.venv/bin:$PATH"

# Create and set permissions for directories
RUN mkdir -p /app/model_cache && \
    chmod -R 755 /app/static /app/templates && \
    chmod 777 /app/model_cache && \
    chmod +x /app/startup.sh

# Create .dockerignore
RUN echo "**/__pycache__\n*.pyc\n*.pyo\n*.pyd\n.Python\n*.log\n.git\n.pytest_cache" > .dockerignore

# Expose port
EXPOSE 8000

# Run the application
CMD ["/app/startup.sh"]

# Add healthcheck
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1