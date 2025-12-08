#!/bin/bash

# Navigate to the directory containing the docker-compose.yml file
cd "$(dirname "$0")"

case "$1" in
    start)
        echo "Starting Docker Compose services..."
        docker-compose up -d
        sleep 2
        docker-compose logs -f
        ;;
    
    stop)
        echo "Stopping Docker Compose services..."
        docker-compose down
        ;;
    
    delete)
        echo "Stopping and removing all containers, volumes, and networks..."
        docker-compose down -v
        ;;
    
    *)
        echo "Usage: $0 {start|stop|delete}"
        exit 1
        ;;
esac
