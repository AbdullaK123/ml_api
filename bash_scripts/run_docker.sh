#!/bin/bash

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
IMAGE_NAME="ml-api"
CONTAINER_NAME="ml-api"
PORT="8000"

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

# Function to clean up Docker resources
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
        docker rm $CONTAINER_NAME
    fi

    # Remove unused images
    print_info "Removing unused images..."
    docker image prune -f

    # Remove unused containers
    print_info "Removing unused containers..."
    docker container prune -f

    # Remove unused volumes
    print_info "Removing unused volumes..."
    docker volume prune -f

    # Remove unused networks
    print_info "Removing unused networks..."
    docker network prune -f

    print_success "Cleanup completed!"
}

# Function to build Docker image
build_image() {
    print_info "Building Docker image..."
    
    if docker build -t $IMAGE_NAME:latest .; then
        print_success "Docker image built successfully!"
    else
        print_error "Failed to build Docker image"
        exit 1
    fi
}

# Function to run container
run_container() {
    print_info "Starting container..."
    
    # Check if port is already in use
    if lsof -Pi :$PORT -sTCP:LISTEN -t >/dev/null ; then
        print_warning "Port $PORT is already in use. Attempting to free it..."
        sudo fuser -k $PORT/tcp
    fi

    # Run the container
    if docker run -d \
        --name $CONTAINER_NAME \
        -p $PORT:$PORT \
        -v $(pwd)/model_cache:/app/model_cache \
        --restart unless-stopped \
        $IMAGE_NAME:latest; then
        
        print_success "Container started successfully!"
        print_info "Container details:"
        docker ps -f name=$CONTAINER_NAME
        
        # Wait a bit and check health
        sleep 5
        check_container_health
    else
        print_error "Failed to start container"
        exit 1
    fi
}

# Function to check container health
check_container_health() {
    print_info "Checking container health..."
    
    # Get container status
    local status=$(docker inspect --format='{{.State.Status}}' $CONTAINER_NAME 2>/dev/null)
    
    if [ "$status" = "running" ]; then
        print_success "Container is running"
        print_info "API is available at http://localhost:$PORT"
        
        # Show logs
        print_info "Recent logs:"
        docker logs --tail 10 $CONTAINER_NAME
    else
        print_error "Container is not running properly. Status: $status"
        print_info "Container logs:"
        docker logs $CONTAINER_NAME
        exit 1
    fi
}

# Main function
main() {
    print_info "Starting Docker deployment process..."
    
    # Check if Docker is running
    check_docker
    
    # Perform cleanup
    cleanup
    
    # Build image
    build_image
    
    # Run container
    run_container
    
    print_success "Deployment completed successfully!"
}

# Helper functions message
show_helpers() {
    echo -e "\n${BLUE}Helpful commands:${NC}"
    echo -e "View logs:          ${GREEN}docker logs -f $CONTAINER_NAME${NC}"
    echo -e "Stop container:     ${GREEN}docker stop $CONTAINER_NAME${NC}"
    echo -e "Start container:    ${GREEN}docker start $CONTAINER_NAME${NC}"
    echo -e "Restart container:  ${GREEN}docker restart $CONTAINER_NAME${NC}"
    echo -e "Remove container:   ${GREEN}docker rm $CONTAINER_NAME${NC}"
    echo -e "Container status:   ${GREEN}docker ps -f name=$CONTAINER_NAME${NC}"
}

# Run main function and show helpers
main
show_helpers