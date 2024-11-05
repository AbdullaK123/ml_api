#!/bin/bash

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
IMAGE_NAME="ml-api"
CONTAINER_NAME="ml-api-container"
PORT="8000"
MEMORY_LIMIT="4g"
MEMORY_SWAP="6g"
CPU_LIMIT="2"
HEALTHCHECK_INTERVAL="30s"
HEALTHCHECK_TIMEOUT="30s"
HEALTHCHECK_RETRIES="3"
VOLUME_NAME="ml_api_model_cache"

# Print functions
print_info() {
    echo -e "${BLUE}[INFO] $1${NC}"
}

print_success() {
    echo -e "${GREEN}[SUCCESS] $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}[WARNING] $1${NC}"
}

print_error() {
    echo -e "${RED}[ERROR] $1${NC}"
}

# Function to check if Docker is running
check_docker() {
    if ! docker info >/dev/null 2>&1; then
        print_error "Docker is not running. Please start Docker first."
        exit 1
    fi
}

# Enhanced system resource check
check_system_resources() {
    print_info "Checking system resources..."
    
    # Check available memory
    total_mem=$(free -g | awk '/^Mem:/{print $2}')
    available_mem=$(free -g | awk '/^Mem:/{print $7}')
    if [ $total_mem -lt 5 ]; then
        print_warning "System has less than 5GB total RAM. This might affect performance."
    fi
    if [ $available_mem -lt 2 ]; then
        print_warning "Less than 2GB RAM available. Consider freeing up memory."
    fi
    
    # Check available CPU cores
    cpu_cores=$(nproc)
    if [ $cpu_cores -lt 2 ]; then
        print_warning "System has less than 2 CPU cores. This might affect performance."
    fi
    
    # Check available disk space
    available_space=$(df -BG / | awk 'NR==2 {print $4}' | sed 's/G//')
    if [ $available_space -lt 10 ]; then
        print_warning "Less than 10GB disk space available. This might affect container performance."
    fi
}

# Enhanced cleanup function
cleanup() {
    print_info "Starting cleanup process..."

    # Stop the existing container if running
    if [ "$(docker ps -q -f name=$CONTAINER_NAME)" ]; then
        print_info "Stopping existing container..."
        docker stop $CONTAINER_NAME
    fi

    # Remove the existing container
    if [ "$(docker ps -aq -f name=$CONTAINER_NAME)" ]; then
        print_info "Removing existing container..."
        docker rm -f $CONTAINER_NAME
    fi

    # Clean up dangling images
    if [ "$(docker images -f "dangling=true" -q)" ]; then
        print_info "Removing dangling images..."
        docker rmi $(docker images -f "dangling=true" -q)
    fi

    # Prune unused resources but preserve volumes
    print_info "Removing unused Docker resources..."
    docker system prune -f --volumes=false

    print_success "Cleanup completed!"
}

# Optimized build function
build_image() {
    print_info "Building Docker image..."
    
    # Enable BuildKit for improved build performance
    export DOCKER_BUILDKIT=1
    
    if docker build \
        --progress=plain \
        --tag $IMAGE_NAME:latest \
        --build-arg BUILDKIT_INLINE_CACHE=1 \
        --build-arg PYTHON_VERSION=3.10 \
        --build-arg POETRY_VERSION=1.4.2 \
        .; then
        print_success "Docker image built successfully!"
    else
        print_error "Failed to build Docker image"
        exit 1
    fi
}

# Enhanced volume setup
setup_volume() {
    print_info "Setting up Docker volume..."
    
    # Create volume if it doesn't exist
    if ! docker volume inspect $VOLUME_NAME >/dev/null 2>&1; then
        docker volume create $VOLUME_NAME
        print_success "Created new volume: $VOLUME_NAME"
    else
        print_info "Using existing volume: $VOLUME_NAME"
    fi
}

# Enhanced container run function
run_container() {
    print_info "Starting container..."
    
    # Check if port is already in use
    if lsof -Pi :$PORT -sTCP:LISTEN -t >/dev/null ; then
        print_warning "Port $PORT is already in use. Attempting to free it..."
        sudo fuser -k $PORT/tcp
    fi

    # Run the container with optimized settings
    if docker run -d \
        --name $CONTAINER_NAME \
        --memory=$MEMORY_LIMIT \
        --memory-swap=$MEMORY_SWAP \
        --cpus=$CPU_LIMIT \
        --health-cmd="curl -f http://localhost:$PORT/health || exit 1" \
        --health-interval=$HEALTHCHECK_INTERVAL \
        --health-timeout=$HEALTHCHECK_TIMEOUT \
        --health-retries=$HEALTHCHECK_RETRIES \
        --volume $VOLUME_NAME:/app/model_cache \
        --restart unless-stopped \
        -p $PORT:$PORT \
        --env-file .env \
        --security-opt=no-new-privileges \
        --cap-drop=ALL \
        --cap-add=NET_BIND_SERVICE \
        --shm-size=2g \
        --ulimit nofile=65535:65535 \
        --ulimit memlock=-1:-1 \
        $IMAGE_NAME:latest; then
        
        print_success "Container started successfully!"
        print_info "Container details:"
        docker ps -f name=$CONTAINER_NAME
        
        # Wait for container to initialize with progress indicator
        print_info "Waiting for container to initialize..."
        for i in {1..10}; do
            echo -n "."
            sleep 1
        done
        echo
        check_container_health
    else
        print_error "Failed to start container"
        exit 1
    fi
}

# Enhanced environment file creation
create_env_file() {
    if [ ! -f .env ]; then
        print_info "Creating default .env file..."
        cat > .env << EOL
PORT=8000
HOST=0.0.0.0
WORKERS=2
TIMEOUT=120
MAX_REQUESTS=250
MAX_REQUESTS_JITTER=25
KEEP_ALIVE=30
GRACEFUL_TIMEOUT=30
MEMORY_MONITOR_THRESHOLD=3500
MODEL_TIMEOUT=300
PYTHONUNBUFFERED=1
PYTHONDONTWRITEBYTECODE=1
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
MALLOC_TRIM_THRESHOLD_=100000
OMP_NUM_THREADS=2
MKL_NUM_THREADS=2
NUMEXPR_NUM_THREADS=2
PYTHONPATH=/app
LD_PRELOAD=/usr/local/lib/libjemalloc.so
MALLOC_CONF=background_thread:true,metadata_thp:auto,dirty_decay_ms:30000,muzzy_decay_ms:30000
EOL
        print_success "Created .env file with optimized settings"
    else
        print_info "Using existing .env file"
    fi
}

# Enhanced container health check
check_container_health() {
    print_info "Checking container health..."
    
    local max_retries=30
    local retry_count=0
    local status
    local health
    
    while [ $retry_count -lt $max_retries ]; do
        status=$(docker inspect --format='{{.State.Status}}' $CONTAINER_NAME 2>/dev/null)
        health=$(docker inspect --format='{{.State.Health.Status}}' $CONTAINER_NAME 2>/dev/null)
        
        if [ "$status" = "running" ] && [ "$health" = "healthy" ]; then
            print_success "Container is running and healthy"
            print_info "API is available at http://localhost:$PORT"
            
            # Show resource usage
            print_info "Container resource usage:"
            docker stats $CONTAINER_NAME --no-stream
            
            # Show logs
            print_info "Recent logs:"
            docker logs --tail 10 $CONTAINER_NAME
            return 0
        fi
        
        retry_count=$((retry_count + 1))
        echo -n "."
        sleep 1
    done
    
    print_error "Container health check failed after $max_retries attempts"
    print_info "Container logs:"
    docker logs $CONTAINER_NAME
    exit 1
}

# Main function with error handling
main() {
    print_info "Starting Docker deployment process..."
    
    # Run all setup functions with error handling
    trap 'print_error "An error occurred during deployment"; exit 1' ERR
    
    check_docker
    check_system_resources
    create_env_file
    cleanup
    setup_volume
    build_image
    run_container
    
    print_success "Deployment completed successfully!"
    trap - ERR
}

# Enhanced helper function display
show_helpers() {
    echo -e "\n${BLUE}Helpful commands:${NC}"
    echo -e "View logs:          ${GREEN}docker logs -f $CONTAINER_NAME${NC}"
    echo -e "View stats:         ${GREEN}docker stats $CONTAINER_NAME${NC}"
    echo -e "Stop container:     ${GREEN}docker stop $CONTAINER_NAME${NC}"
    echo -e "Start container:    ${GREEN}docker start $CONTAINER_NAME${NC}"
    echo -e "Restart container:  ${GREEN}docker restart $CONTAINER_NAME${NC}"
    echo -e "Remove container:   ${GREEN}docker rm -f $CONTAINER_NAME${NC}"
    echo -e "Check health:       ${GREEN}docker inspect --format='{{.State.Health.Status}}' $CONTAINER_NAME${NC}"
    echo -e "Container status:   ${GREEN}docker ps -f name=$CONTAINER_NAME${NC}"
    echo -e "Inspect volume:     ${GREEN}docker volume inspect $VOLUME_NAME${NC}"
}

# Trap for cleanup on script exit
trap cleanup EXIT

# Run main function and show helpers
main
show_helpers