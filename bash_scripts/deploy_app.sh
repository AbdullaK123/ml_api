#!/bin/bash

# Variables
APP_NAME="ocr-extractor-app"
DOCKER_IMAGE="abdullak123/ml-api:latest"

# Check flyctl
if ! command -v flyctl &> /dev/null; then
    echo "Installing flyctl..."
    curl -L https://fly.io/install.sh | sh
    export PATH="$HOME/.fly/bin:$PATH"
fi

# Check login
if ! flyctl auth whoami &> /dev/null; then
    echo "Please run 'flyctl auth login' first"
    exit 1
fi

# First destroy existing app if it exists
if flyctl apps list | grep -q "$APP_NAME"; then
    echo "Destroying existing app: $APP_NAME"
    flyctl apps destroy $APP_NAME --yes
fi

# Create new app
echo "Creating new app: $APP_NAME"
flyctl apps create $APP_NAME

# Create fly.toml configuration
cat > fly.toml << EOL
app = "$APP_NAME"
primary_region = "lax"

[build]
  image = "$DOCKER_IMAGE"

[env]
  PORT = "8000"
  HOST = "0.0.0.0"
  PYTHONUNBUFFERED = "1"
  WORKERS = "4"
  TIMEOUT = "300"
  MAX_REQUESTS = "1000"
  KEEP_ALIVE = "75"

[http_service]
  internal_port = 8000
  force_https = true
  auto_stop_machines = true
  auto_start_machines = true
  min_machines_running = 1
  processes = ["app"]
  
  [http_service.concurrency]
    type = "connections"
    hard_limit = 50
    soft_limit = 40

[[vm]]
  cpu_kind = "shared"
  cpus = 4
  memory_mb = 4096
EOL

# Deploy with more verbose output
echo "Deploying app..."
flyctl deploy --image $DOCKER_IMAGE --strategy immediate --verbose

# Allocate IPv4 (using shared address to avoid charges)
echo "Allocating shared IPv4 address..."
flyctl ips allocate-v4 --shared

# Scale the app
echo "Scaling app..."
flyctl scale memory 4096
flyctl scale cpu 4

# Show deployment info
echo "Checking app status..."
flyctl status
flyctl logs

echo "Deployment complete! Your app should be available at: https://$APP_NAME.fly.dev"