#!/bin/bash

# Variables
DOCKER_USERNAME="abdullak123"
IMAGE_NAME="ml-api"
IMAGE_TAG="latest"

# Build the image
echo "Building Docker image..."
docker build -t $IMAGE_NAME:$IMAGE_TAG .

# Tag with Docker Hub username
echo "Tagging image for Docker Hub..."
docker tag $IMAGE_NAME:$IMAGE_TAG $DOCKER_USERNAME/$IMAGE_NAME:$IMAGE_TAG

# Push to Docker Hub
echo "Pushing to Docker Hub..."
docker push $DOCKER_USERNAME/$IMAGE_NAME:$IMAGE_TAG

echo "Done! Your image is now on Docker Hub at $DOCKER_USERNAME/$IMAGE_NAME:$IMAGE_TAG"