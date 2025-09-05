#!/bin/bash

# Script to start Airflow standalone setup

echo "Starting Airflow standalone setup..."

# Set environment variables
export AIRFLOW_UID=50000
export AIRFLOW_GID=0
export AIRFLOW_PROJ_DIR=.

# Create necessary directories
mkdir -p logs dags plugins config

# Set proper permissions
chmod 755 logs dags plugins config

echo "Building and starting Airflow services..."
docker-compose up --build -d

echo "Waiting for services to be ready..."
sleep 30

echo "Airflow is starting up..."
echo "Web UI will be available at: http://localhost:8089"
echo "Username: airflow"
echo "Password: airflow"
echo ""
echo "Other services:"
echo "- Kafka UI: http://localhost:8085 (admin/admin1234)"
echo "- Spark UI: http://localhost:8080"
echo "- MLflow: http://localhost:5003"
echo "- Grafana: http://localhost:3000 (admin/admin)"
echo "- RabbitMQ: http://localhost:15672 (admin/admin1234)"

echo ""
echo "To stop all services: docker-compose down"
echo "To view logs: docker-compose logs -f"
