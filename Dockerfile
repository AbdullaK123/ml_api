# Use Ubuntu as base image
FROM ubuntu:22.04

# Prevent timezone prompts during package installation
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
    PATH="$POETRY_HOME/bin:$PATH" \
    APP_HOME="/app"

# Set working directory
WORKDIR $APP_HOME

# Install system dependencies and Python
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3.10-dev \
    python3.10-distutils \
    python3-pip \
    build-essential \
    curl \
    git \
    libpq-dev \
    libopencv-dev \
    python3-opencv \
    ffmpeg \
    libsm6 \
    libxext6 \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Ensure python3 and pip3 point to Python 3.10
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1 \
    && update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1

# Upgrade pip
RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.10 \
    && python3.10 -m pip install --upgrade pip

# Install Poetry
RUN curl -sSL https://install.python-poetry.org | python3 - \
    && ln -s /opt/poetry/bin/poetry /usr/local/bin/poetry

# Copy poetry files
COPY pyproject.toml poetry.lock ./

# Install dependencies
RUN poetry install --no-root --no-dev

# Create cache directory for models
RUN mkdir -p /app/model_cache \
    && chmod 777 /app/model_cache

# Copy the rest of the application
COPY . .

# Create necessary directories with proper permissions
RUN mkdir -p /app/static /app/templates \
    && chmod -R 755 /app/static /app/templates

# Set up model cache directory
ENV MODEL_CACHE_DIR=/app/model_cache

# Expose port
EXPOSE 8000

# Run the application
CMD ["poetry", "run", "uvicorn", "ml_api.inference_api:app", "--host", "0.0.0.0", "--port", "8000"]

# Add healthcheck
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1