#!/bin/bash

# Script để restart Airflow với cấu hình tối ưu cho VPS 4-core
# Tắt các service không cần thiết và giới hạn resources

echo "🔄 Restarting Airflow with optimized configuration for 4-core VPS..."

# Dừng tất cả containers
echo "⏹️  Stopping all containers..."
docker-compose down

# Xóa containers cũ để đảm bảo cấu hình mới được áp dụng
echo "🧹 Cleaning up old containers..."
docker-compose rm -f

# Tắt các service không cần thiết để tiết kiệm resources
echo "⚙️  Starting only essential services..."

# Chỉ start các service cần thiết
docker-compose up -d \
    airflow-init \
    airflow-scheduler \
    airflow-dag-processor \
    airflow-triggerer \
    airflow-apiserver

# Đợi các service khởi động
echo "⏳ Waiting for services to start..."
sleep 30

# Kiểm tra trạng thái
echo "📊 Checking service status..."
docker-compose ps

# Kiểm tra logs
echo "📋 Checking logs for errors..."
docker-compose logs --tail=20 airflow-scheduler

echo "✅ Airflow restarted with optimized configuration!"
echo "🌐 Web UI: http://localhost:8089"
echo "📊 Monitor resources: python scripts/monitor_resources.py"

# Hiển thị resource usage
echo "📈 Current resource usage:"
python3 scripts/monitor_resources.py --once
