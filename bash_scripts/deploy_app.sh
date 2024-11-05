#!/bin/bash

# Configuration Variables
APP_NAME="ocr-extractor-app"
DOCKER_IMAGE="abdullak123/ml-api:latest"
PRIMARY_REGION="lax"  # Los Angeles
BACKUP_REGION="sfo"   # San Francisco (closer to LAX for better failover)
VOLUME_NAME="model_cache"
VOLUME_SIZE="20"      # 20GB for model storage
CLEANUP_VOLUMES=true  # Aggressive cleanup
MIN_MACHINES=1
MAX_MACHINES=3
MEMORY_SIZE="8192"    # 8GB RAM
CPU_COUNT="4"         # 4 CPUs for better processing
SWAP_SIZE="4096"      # 4GB swap

# Error handling
set -euo pipefail
trap 'handle_error $? $LINENO' ERR

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Utility functions
print_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
print_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
print_warning() { echo -e "${YELLOW}[WARNING]${NC} $1"; }
print_error() { echo -e "${RED}[ERROR]${NC} $1"; }

handle_error() {
    local exit_code=$1
    local line_no=$2
    print_error "Error occurred in script at line: $line_no"
    print_error "Exit code: $exit_code"
    aggressive_cleanup
    exit $exit_code
}

aggressive_cleanup() {
    print_warning "Performing aggressive cleanup..."

    # Stop all machines
    print_info "Stopping all machines..."
    flyctl scale count 0 --yes || true

    # Remove volumes
    print_info "Removing volumes..."
    flyctl volumes list | grep "$VOLUME_NAME" | awk '{print $1}' | xargs -I{} flyctl volumes destroy {} -y || true

    # Cleanup Docker resources
    print_info "Cleaning up Docker resources..."
    docker system prune -af --volumes || true

    # Remove app if it exists
    print_info "Removing application..."
    flyctl apps destroy "$APP_NAME" --yes || true

    print_info "Cleanup completed"
}

verify_prerequisites() {
    print_info "Verifying prerequisites..."

    # Check for flyctl
    if ! command -v flyctl &> /dev/null; then
        print_info "Installing flyctl..."
        curl -L https://fly.io/install.sh | sh
        export PATH="$HOME/.fly/bin:$PATH"
    fi

    # Check for Docker
    if ! command -v docker &> /dev/null; then
        print_error "Docker is required but not installed"
        exit 1
    fi

    # Check fly.io authentication
    if ! flyctl auth whoami &> /dev/null; then
        print_error "Please run 'flyctl auth login' first"
        exit 1
    fi

    # Verify Docker image
    print_info "Verifying Docker image..."
    if ! docker image inspect "$DOCKER_IMAGE" >/dev/null 2>&1; then
        print_warning "Docker image not found locally. Attempting to pull..."
        if ! docker pull "$DOCKER_IMAGE"; then
            print_error "Failed to pull Docker image"
            exit 1
        fi
    fi

    print_success "Prerequisites verified"
}

create_fly_config() {
    print_info "Creating optimized fly.toml configuration..."
    cat > fly.toml << EOL
app = "$APP_NAME"
primary_region = "$PRIMARY_REGION"
kill_signal = "SIGINT"
kill_timeout = "30s"

[experimental]
  auto_rollback = true
  enable_consul = true

[build]
  image = "$DOCKER_IMAGE"

[deploy]
  strategy = "rolling"

[env]
  PORT = "8000"
  HOST = "0.0.0.0"
  PYTHONUNBUFFERED = "1"
  PYTHONDONTWRITEBYTECODE = "1"
  PYTORCH_CUDA_ALLOC_CONF = "max_split_size_mb:1024"
  MALLOC_TRIM_THRESHOLD_ = "100000"
  OMP_NUM_THREADS = "4"
  MKL_NUM_THREADS = "4"
  NUMEXPR_NUM_THREADS = "4"
  WORKERS = "4"
  TIMEOUT = "300"
  MAX_REQUESTS = "1000"
  MAX_REQUESTS_JITTER = "50"
  KEEP_ALIVE = "75"
  GRACEFUL_TIMEOUT = "60"
  MODEL_CACHE_DIR = "/app/model_cache"
  MEMORY_MONITOR_THRESHOLD = "7000"
  MODEL_TIMEOUT = "600"
  PADDLEOCR_HOME = "/app/model_cache/paddleocr"
  HF_HOME = "/app/model_cache/huggingface"

[http_service]
  internal_port = 8000
  force_https = true
  auto_stop_machines = false
  auto_start_machines = true
  min_machines_running = $MIN_MACHINES
  processes = ["app"]

  [http_service.concurrency]
    type = "connections"
    hard_limit = 50
    soft_limit = 35

  [[http_service.checks]]
    interval = "30s"
    timeout = "10s"
    grace_period = "30s"
    method = "GET"
    path = "/health"
    protocol = "http"
    tls_skip_verify = false

    [http_service.checks.headers]
      User-Agent = "fly/healthcheck"

[[metrics]]
  port = 9091
  path = "/metrics"

[vm]
  memory = "${MEMORY_SIZE}"
  cpu_kind = "performance"
  cpus = ${CPU_COUNT}
  swap_size_mb = ${SWAP_SIZE}

  [vm.gc]
    enabled = true
    total_memory_ratio = 0.8
    free_memory_ratio = 0.3
    gc_period = "45s"
  
  [vm.agent]
    enabled = true
    statsd_address = "localhost:8125"

[mounts]
  source="$VOLUME_NAME"
  destination="/app/model_cache"
EOL
    print_success "fly.toml created with optimized settings"
}

setup_volume() {
    print_info "Setting up volume in $PRIMARY_REGION..."
    
    # Create volume if it doesn't exist
    if ! flyctl volumes list | grep -q "$VOLUME_NAME"; then
        print_info "Creating volume: $VOLUME_NAME"
        if ! flyctl volumes create "$VOLUME_NAME" \
            --size "$VOLUME_SIZE" \
            --region "$PRIMARY_REGION" \
            --no-encryption; then
            print_error "Failed to create volume"
            exit 1
        fi
    else
        print_info "Using existing volume"
    fi
}

deploy_application() {
    print_info "Starting deployment process..."

    # Load environment variables
    ENV_VARS=""
    if [ -f .env ]; then
        print_info "Loading environment variables..."
        while IFS='=' read -r key value; do
            [[ -z "$key" || "$key" == \#* ]] && continue
            ENV_VARS="$ENV_VARS -e ${key}=${value}"
        done < .env
    fi

    # Deploy with optimized settings
    print_info "Deploying application..."
    if ! flyctl deploy \
        --remote-only \
        --wait-timeout 600 \
        --strategy rolling \
        $ENV_VARS \
        --build-arg BUILDKIT_INLINE_CACHE=1 \
        --build-arg DOCKER_BUILDKIT=1 \
        --vm-memory "${MEMORY_SIZE}" \
        --vm-cpu-kind "performance" \
        --vm-cpus "${CPU_COUNT}" \
        --ha \
        --verbose; then
        
        print_error "Deployment failed"
        print_info "Fetching logs..."
        flyctl logs
        return 1
    fi

    # Wait for deployment to stabilize
    print_info "Waiting for deployment to stabilize..."
    sleep 30

    print_success "Deployment successful!"
}

scale_application() {
    print_info "Configuring application scaling..."

    # Set VM size
    print_info "Setting VM size..."
    flyctl scale vm performance-2x || print_warning "Failed to set VM size"

    # Set memory
    print_info "Setting memory limit..."
    flyctl scale memory $MEMORY_SIZE || print_warning "Failed to set memory"

    # Set instance count
    print_info "Setting instance count..."
    flyctl scale count $MIN_MACHINES || print_warning "Failed to set instance count"

    print_success "Scaling configured"
}

main() {
    print_info "Starting optimized deployment for $APP_NAME"

    # Verify prerequisites
    verify_prerequisites

    # Clean up existing resources
    aggressive_cleanup

    # Create new app
    print_info "Creating new app: $APP_NAME"
    flyctl apps create "$APP_NAME"

    # Setup resources
    create_fly_config
    setup_volume
    
    # Deploy and scale
    if deploy_application; then
        scale_application
        print_success "Deployment and scaling completed successfully!"
        
        # Show final status
        print_info "Final application status:"
        flyctl status
    else
        print_error "Deployment failed"
        aggressive_cleanup
        exit 1
    fi

    # Print useful information
    cat << EOL

${BLUE}Application Information:${NC}
- URL: https://$APP_NAME.fly.dev
- Region: $PRIMARY_REGION
- Memory: ${MEMORY_SIZE}MB
- CPUs: $CPU_COUNT
- Volume Size: ${VOLUME_SIZE}GB

${BLUE}Monitoring Commands:${NC}
- View logs: ${GREEN}flyctl logs${NC}
- Check status: ${GREEN}flyctl status${NC}
- View scaling: ${GREEN}flyctl scale show${NC}

${BLUE}Performance Monitoring:${NC}
1. Status: ${GREEN}flyctl status${NC}
2. Logs: ${GREEN}flyctl logs${NC}
3. Resources: ${GREEN}flyctl scale show${NC}

EOL
}

# Run main function
main