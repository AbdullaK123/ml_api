#!/bin/bash
set -e

# Create necessary directories
mkdir -p /app/model_cache

# Set permissions
chmod -R 755 /app/static /app/templates
chmod 777 /app/model_cache

# Start the application
exec uvicorn ml_api.inference_api:app \
    --host 0.0.0.0 \
    --port 8000 \
    --workers 1 \
    --timeout-keep-alive 5 \
    --limit-concurrency 20 \
    --limit-max-requests 1000 \
    --backlog 100