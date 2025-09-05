@echo off
REM Script to start Airflow standalone setup on Windows

echo Starting Airflow standalone setup...

REM Set environment variables
set AIRFLOW_UID=50000
set AIRFLOW_GID=0
set AIRFLOW_PROJ_DIR=.

REM Create necessary directories
if not exist logs mkdir logs
if not exist dags mkdir dags
if not exist plugins mkdir plugins
if not exist config mkdir config

echo Building and starting Airflow services...
docker-compose up --build -d

echo Waiting for services to be ready...
timeout /t 30 /nobreak > nul

echo Airflow is starting up...
echo Web UI will be available at: http://localhost:8089
echo Username: airflow
echo Password: airflow
echo.
echo Other services:
echo - Kafka UI: http://localhost:8085 (admin/admin1234)
echo - Spark UI: http://localhost:8080
echo - MLflow: http://localhost:5003
echo - Grafana: http://localhost:3000 (admin/admin)
echo - RabbitMQ: http://localhost:15672 (admin/admin1234)

echo.
echo To stop all services: docker-compose down
echo To view logs: docker-compose logs -f

pause
