# Airflow Standalone Setup Guide

## Tổng quan
Dự án đã được chuyển từ Astronomer sang Airflow standalone để dễ dàng quản lý và triển khai.

## Cấu trúc thay đổi

### Files đã thay đổi:
- `Dockerfile`: Chuyển từ Astronomer runtime sang Apache Airflow official image
- `docker-compose.yml`: Cấu hình mới cho Airflow standalone với external PostgreSQL database
- `requirements.txt`: Cập nhật dependencies cho Airflow
- `airflow.cfg`: File cấu hình Airflow
- `start_airflow.sh` / `start_airflow.bat`: Script khởi động

### Files đã xóa:
- `airflow_settings.yaml`: Không cần thiết với Airflow standalone
- `docker-compose.override.yml`: Đã merge vào docker-compose.yml chính

## Cách sử dụng

### 1. Khởi động Airflow

**Trên Linux/Mac:**
```bash
./start_airflow.sh
```

**Trên Windows:**
```cmd
start_airflow.bat
```

**Hoặc thủ công:**
```bash
# Set environment variables
export AIRFLOW_UID=50000
export AIRFLOW_GID=0

# Start services
docker-compose up --build -d
```

### 2. Truy cập các services

- **Airflow Web UI**: http://localhost:8089
  - Username: `airflow`
  - Password: `airflow`

- **Kafka UI**: http://localhost:8085
  - Username: `admin`
  - Password: `admin1234`

- **Spark UI**: http://localhost:8080

- **MLflow**: http://localhost:5003

- **Grafana**: http://localhost:3000
  - Username: `admin`
  - Password: `admin`

- **RabbitMQ Management**: http://localhost:15672
  - Username: `admin`
  - Password: `admin1234`

### 3. Quản lý services

**Dừng tất cả services:**
```bash
docker-compose down
```

**Xem logs:**
```bash
docker-compose logs -f
```

**Xem logs của service cụ thể:**
```bash
docker-compose logs -f airflow-webserver
docker-compose logs -f airflow-scheduler
```

**Restart service:**
```bash
docker-compose restart airflow-webserver
```

### 4. Cấu trúc thư mục

```
├── dags/                    # Airflow DAGs
├── logs/                    # Airflow logs
├── plugins/                 # Airflow plugins
├── config/                  # Airflow config
├── models/                  # ML models
├── data/                    # Data files
├── include/                 # Custom modules
└── monitoring/              # Monitoring configs
```

### 5. Troubleshooting

**Nếu gặp lỗi permission:**
```bash
sudo chown -R 50000:0 logs dags plugins config
```

**Nếu database connection failed:**
```bash
# Kiểm tra kết nối đến external database
# Database: 194.238.16.14:5432/wqi_db
# Username: postgres, Password: postgres1234
```

**Nếu Airflow không start:**
```bash
docker-compose down
docker-compose up --build -d
```

### 6. Development

**Chạy DAG test:**
```bash
docker-compose exec airflow-webserver airflow dags test <dag_id>
```

**List DAGs:**
```bash
docker-compose exec airflow-webserver airflow dags list
```

**Check DAG syntax:**
```bash
docker-compose exec airflow-webserver python /opt/airflow/dags/<dag_file>.py
```

## Lưu ý quan trọng

1. **Port conflicts**: Đảm bảo các port 8080, 8081, 8085, 5003, 3000, 15672 không bị sử dụng
2. **Memory**: Cần ít nhất 4GB RAM để chạy tất cả services
3. **Docker**: Cần Docker và Docker Compose được cài đặt
4. **Permissions**: Trên Linux, có thể cần sudo để set permissions

## Migration từ Astronomer

Nếu bạn đang migrate từ Astronomer:

1. Backup dữ liệu quan trọng
2. Stop Astronomer services
3. Chạy setup mới theo hướng dẫn trên
4. Import lại DAGs và configs nếu cần
