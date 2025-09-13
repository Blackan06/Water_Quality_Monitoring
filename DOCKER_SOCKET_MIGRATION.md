# Migration từ Docker TCP (port 2375) sang Unix Socket

## Tổng quan
Đã thực hiện migration từ việc sử dụng Docker TCP port 2375 sang Unix socket để tăng cường bảo mật và hiệu suất.

## Những thay đổi đã thực hiện

### 1. Cập nhật DockerOperator trong DAGs
- **File**: `dags/load_historical_data_dag.py`
- **File**: `dags/water_quality_dag.py`
- **Thay đổi**: 
  ```python
  # Trước
  docker_url='tcp://docker-proxy:2375'
  
  # Sau
  docker_url='unix://var/run/docker.sock'
  ```

### 2. Cập nhật docker-compose.yaml
- **Thêm mount Docker socket**:
  ```yaml
  volumes:
    - /var/run/docker.sock:/var/run/docker.sock:ro
  ```
- **Loại bỏ docker-proxy service** (đã comment out)

## Lợi ích

### Bảo mật
- ✅ Không expose Docker daemon qua network
- ✅ Giảm attack surface
- ✅ Chỉ containers trong cùng host mới có thể truy cập

### Hiệu suất
- ✅ Giao tiếp trực tiếp với Docker daemon
- ✅ Không cần proxy layer
- ✅ Giảm latency

### Đơn giản hóa
- ✅ Loại bỏ docker-proxy service
- ✅ Ít components để maintain
- ✅ Cấu hình đơn giản hơn

## Yêu cầu hệ thống

### Linux/macOS
- Docker socket phải có quyền truy cập từ Airflow containers
- User trong container phải có quyền đọc `/var/run/docker.sock`

### Windows
- Cần Docker Desktop với WSL2 backend
- Docker socket sẽ được mount tự động

## Kiểm tra hoạt động

### 1. Restart services
```bash
docker-compose down
docker-compose up -d
```

### 2. Kiểm tra Docker socket
```bash
# Trong Airflow container
docker exec -it <airflow-container> ls -la /var/run/docker.sock
```

### 3. Test DockerOperator
- Chạy DAG và kiểm tra logs
- Đảm bảo containers được tạo thành công

## Troubleshooting

### Lỗi permission denied
```bash
# Kiểm tra quyền của Docker socket
ls -la /var/run/docker.sock

# Nếu cần, thay đổi quyền (tạm thời)
sudo chmod 666 /var/run/docker.sock
```

### Container không thể tạo containers
- Kiểm tra Airflow container có quyền truy cập Docker socket
- Đảm bảo Docker daemon đang chạy
- Kiểm tra logs của Airflow scheduler

## Rollback (nếu cần)

Nếu cần quay lại sử dụng TCP:

1. **Uncomment docker-proxy service** trong `docker-compose.yaml`
2. **Thay đổi docker_url** về `tcp://docker-proxy:2375`
3. **Restart services**

```bash
docker-compose down
docker-compose up -d
```
