# Tối Ưu Hóa Airflow cho VPS 4-Core

## 🎯 Mục Tiêu
Giảm CPU usage từ 45%+ xuống dưới 20% trên VPS 4-core Ubuntu

## ⚙️ Các Thay Đổi Đã Thực Hiện

### 1. Airflow Configuration (`config/airflow.cfg`)
- **parallelism**: 32 → 4 (giảm 87.5%)
- **max_active_tasks_per_dag**: 16 → 2 (giảm 87.5%)
- **max_active_runs_per_dag**: 16 → 2 (giảm 87.5%)
- **worker_concurrency**: 16 → 2 (giảm 87.5%)
- **scheduler_heartbeat_sec**: 5 → 30 (giảm tần suất heartbeat)
- **scheduler_health_check_threshold**: 30 → 60 (giảm tần suất health check)
- **parsing_processes**: 2 → 1 (giảm 50%)
- **auto_refresh_interval**: 3 → 0 (tắt auto refresh)
- **refresh_interval**: 300 → 0 (tắt refresh)

### 2. Docker Compose Resource Limits
- **airflow-apiserver**: CPU 0.5, Memory 512M
- **airflow-scheduler**: CPU 1.0, Memory 1G
- **airflow-dag-processor**: CPU 0.5, Memory 512M
- **airflow-triggerer**: CPU 0.5, Memory 512M
- **spark-worker**: CPU 1.0, Memory 1G

### 3. DAG Concurrency Settings
- **load_historical_data_dag**: max_active_tasks=1, max_active_runs=1, concurrency=1
- **streaming_process_dag**: max_active_tasks=1, max_active_runs=1, concurrency=1

### 4. Resource Configuration (`config/resource_config.yaml`)
- **XGBoost**: max_estimators 300→150, optuna_trials 30→10
- **LSTM**: max_units 256→128, max_epochs 300→50, optuna_trials 25→8
- **Parallel processing**: max_workers 4→1, use_multiprocessing=false
- **CPU threads**: max_threads 2

## 🚀 Cách Sử Dụng

### Restart Airflow với cấu hình tối ưu:
```bash
# Linux/Mac
./scripts/restart_airflow_optimized.sh

# Windows
scripts\restart_airflow_optimized.bat
```

### Monitor Resources:
```bash
# Chạy một lần
python scripts/monitor_resources.py --once

# Monitor liên tục (30s interval)
python scripts/monitor_resources.py

# Monitor với custom interval
python scripts/monitor_resources.py --interval 60 --duration 3600
```

## 📊 Kết Quả Mong Đợi

### Trước Tối Ưu:
- CPU Usage: 45%+ (nhiều process Airflow)
- Memory Usage: Cao do nhiều parallel tasks
- Response Time: Chậm do resource contention

### Sau Tối Ưu:
- CPU Usage: <20% (giới hạn strict)
- Memory Usage: Giảm 50-70%
- Response Time: Cải thiện đáng kể
- Stability: Tăng do ít resource contention

## ⚠️ Lưu Ý

1. **Performance Trade-off**: Giảm concurrency sẽ làm chậm việc xử lý nhưng tăng stability
2. **Monitoring**: Cần monitor thường xuyên để đảm bảo không bị bottleneck
3. **Scaling**: Khi cần tăng performance, có thể tăng dần các giá trị
4. **DAG Design**: Nên thiết kế DAGs để chạy tuần tự thay vì parallel

## 🔧 Troubleshooting

### Nếu CPU vẫn cao:
1. Kiểm tra logs: `docker-compose logs airflow-scheduler`
2. Monitor processes: `python scripts/monitor_resources.py --once`
3. Giảm thêm concurrency nếu cần

### Nếu DAGs chạy chậm:
1. Tăng dần max_active_tasks (1→2→4)
2. Tăng parallelism (4→8→16)
3. Monitor để đảm bảo không vượt quá 80% CPU

## 📈 Monitoring Commands

```bash
# Xem CPU usage real-time
top -p $(pgrep -f airflow)

# Xem Docker stats
docker stats

# Xem Airflow processes
ps aux | grep airflow

# Monitor resources với script
python scripts/monitor_resources.py --interval 10
```
