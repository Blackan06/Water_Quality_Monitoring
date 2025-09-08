@echo off
REM Script để restart Airflow với cấu hình tối ưu cho VPS 4-core
REM Tắt các service không cần thiết và giới hạn resources

echo 🔄 Restarting Airflow with optimized configuration for 4-core VPS...

REM Dừng tất cả containers
echo ⏹️  Stopping all containers...
docker-compose down

REM Xóa containers cũ để đảm bảo cấu hình mới được áp dụng
echo 🧹 Cleaning up old containers...
docker-compose rm -f

REM Tắt các service không cần thiết để tiết kiệm resources
echo ⚙️  Starting only essential services...

REM Chỉ start các service cần thiết
docker-compose up -d airflow-init airflow-scheduler airflow-dag-processor airflow-triggerer airflow-apiserver

REM Đợi các service khởi động
echo ⏳ Waiting for services to start...
timeout /t 30 /nobreak >nul

REM Kiểm tra trạng thái
echo 📊 Checking service status...
docker-compose ps

REM Kiểm tra logs
echo 📋 Checking logs for errors...
docker-compose logs --tail=20 airflow-scheduler

echo ✅ Airflow restarted with optimized configuration!
echo 🌐 Web UI: http://localhost:8089
echo 📊 Monitor resources: python scripts/monitor_resources.py

REM Hiển thị resource usage
echo 📈 Current resource usage:
python scripts/monitor_resources.py --once

pause
