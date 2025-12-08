#!/bin/bash

# Navigate to the directory containing the docker-compose.yml file
cd "$(dirname "$0")"

# Start Docker Compose services in the background
docker-compose up -d

# Wait for a moment to ensure services start
sleep 2

# Show logs in real-time
docker-compose logs -f
