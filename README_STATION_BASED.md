# Water Quality Monitoring System - Station-Based Architecture

## Tổng quan

Hệ thống giám sát chất lượng nước được thiết kế lại để phân tách theo từng trạm quan trắc, giúp cải thiện độ chính xác của mô hình dự đoán và quản lý hiệu quả hơn.

## Kiến trúc mới

### 1. Phân tách theo trạm quan trắc
- **5 trạm quan trắc**: ST001, ST002, ST003, ST004, ST005
- **Model riêng biệt**: Mỗi trạm có model XGBoost riêng
- **Dữ liệu tách biệt**: Dữ liệu được lưu trữ và xử lý riêng cho từng trạm

### 2. Tự động retrain
- **Data drift detection**: Phát hiện khi dữ liệu thay đổi đáng kể
- **Performance monitoring**: Theo dõi hiệu suất model
- **Auto-retrain**: Tự động retrain khi cần thiết

### 3. Monitoring với Prometheus & Grafana
- **Real-time metrics**: Theo dõi thời gian thực
- **Alerting**: Cảnh báo khi có vấn đề
- **Dashboard**: Giao diện trực quan

## Các thành phần chính

### DAGs
- `station_based_pipeline_dag.py`: Pipeline chính cho từng trạm
- `pipelineiot_dag.py`: Pipeline cũ (để tham khảo)

### Modules
- `station_processor.py`: Xử lý dữ liệu theo trạm
- `model_manager.py`: Quản lý model và retrain
- `prometheus_exporter.py`: Export metrics cho Prometheus

### Spark Jobs
- `iot_stream.py`: Xử lý streaming với hỗ trợ station-specific

## Cài đặt và triển khai

### 1. Khởi động hệ thống monitoring
```bash
# Khởi động Prometheus, Grafana, Alertmanager
docker-compose -f docker-compose.monitoring.yml up -d

# Truy cập Grafana: http://localhost:3000
# Username: admin, Password: admin123
```

### 2. Khởi động Airflow DAG
```bash
# DAG sẽ tự động chạy mỗi 15 phút
# Có thể trigger thủ công từ Airflow UI
```

### 3. Kiểm tra metrics
```bash
# Prometheus: http://localhost:9090
# Water Quality Exporter: http://localhost:8000/metrics
```

## Cấu hình

### 1. Trạm quan trắc
Các trạm được định nghĩa trong `kafka_producer_streaming.py`:
```python
STATIONS = [
    {"station_id": "ST001", "station_name": "Trạm Sông Sài Gòn - Quận 1", "location": "10.7769,106.7009"},
    {"station_id": "ST002", "station_name": "Trạm Kênh Nhiêu Lộc", "location": "10.7833,106.6833"},
    # ...
]
```

### 2. Ngưỡng retrain
Cấu hình trong `model_manager.py`:
```python
RETRAIN_THRESHOLDS = {
    'data_drift_threshold': 0.15,  # 15% drift
    'performance_threshold': 0.8,  # R² score
    'data_age_threshold_days': 30,  # 30 ngày
    'min_data_points': 100  # Tối thiểu 100 điểm dữ liệu
}
```

### 3. Alert thresholds
Cấu hình trong `prometheus.yml/water_quality_rules.yml`:
- WQI < 50: Warning
- WQI < 30: Critical
- pH < 6.0 hoặc > 9.0: Warning
- DO < 4.0: Warning
- Temperature > 35°C: Warning

## Workflow

### 1. Thu thập dữ liệu
- Kafka producer tạo dữ liệu mẫu cho 3-5 trạm
- Mỗi record chứa thông tin trạm quan trắc

### 2. Xử lý theo trạm
- Dữ liệu được nhóm theo `station_id`
- Tính WQI cho từng record
- Lưu vào PostgreSQL với thông tin trạm

### 3. Dự đoán
- Load model riêng cho từng trạm
- Dự đoán WQI cho tháng tiếp theo
- Phân tích và gửi thông báo nếu cần

### 4. Monitoring
- Export metrics đến Prometheus
- Hiển thị trên Grafana dashboard
- Gửi alert qua email/Slack

### 5. Auto-retrain
- Kiểm tra data drift
- Kiểm tra model performance
- Tự động retrain khi cần thiết

## Dashboard Grafana

### Panels chính:
1. **WQI by Station**: Hiển thị WQI hiện tại của từng trạm
2. **pH Levels**: Theo dõi pH theo thời gian
3. **Temperature**: Theo dõi nhiệt độ nước
4. **Dissolved Oxygen**: Theo dõi DO
5. **Alerts Generated**: Số lượng alert đã tạo
6. **Model Performance**: Hiệu suất model theo thời gian
7. **Data Drift Score**: Điểm drift dữ liệu
8. **Pipeline Status**: Trạng thái pipeline

### Variables:
- `station`: Lọc theo trạm quan trắc

## Monitoring Metrics

### Water Quality Metrics
- `water_quality_wqi`: WQI hiện tại
- `water_quality_ph`: pH hiện tại
- `water_quality_temperature`: Nhiệt độ hiện tại
- `water_quality_do`: DO hiện tại
- `water_quality_turbidity`: Độ đục hiện tại
- `water_quality_conductivity`: Độ dẫn điện hiện tại

### Model Metrics
- `model_performance_r2_score`: R² score của model
- `model_performance_mae`: Mean Absolute Error
- `data_drift_score`: Điểm drift dữ liệu
- `prediction_duration_seconds`: Thời gian dự đoán

### Pipeline Metrics
- `pipeline_runs_total`: Tổng số lần chạy pipeline
- `pipeline_duration_seconds`: Thời gian chạy pipeline
- `water_quality_records_total`: Tổng số record đã xử lý

### Alert Metrics
- `alerts_generated_total`: Tổng số alert đã tạo

## Troubleshooting

### 1. Model không load được
```bash
# Kiểm tra model path
ls -la /opt/bitnami/spark/models/

# Retrain model thủ công
docker exec -it retrain_ST001_container python /app/retrain_model.py
```

### 2. Metrics không hiển thị
```bash
# Kiểm tra exporter
curl http://localhost:8000/health

# Kiểm tra Prometheus
curl http://localhost:9090/api/v1/targets
```

### 3. Alert không gửi
```bash
# Kiểm tra Alertmanager
curl http://localhost:9093/api/v1/alerts

# Kiểm tra cấu hình email/Slack
```

## Phát triển

### Thêm trạm mới
1. Thêm vào `STATIONS` trong `kafka_producer_streaming.py`
2. Cập nhật Prometheus rules
3. Retrain model cho trạm mới

### Thêm metrics mới
1. Định nghĩa metric trong `prometheus_exporter.py`
2. Cập nhật Spark job để gửi metric
3. Thêm panel vào Grafana dashboard

### Tùy chỉnh alert
1. Cập nhật rules trong `water_quality_rules.yml`
2. Cấu hình receiver trong `alertmanager.yml`
3. Test alert

## Lợi ích của kiến trúc mới

1. **Độ chính xác cao hơn**: Model riêng cho từng trạm
2. **Quản lý hiệu quả**: Tách biệt dữ liệu và xử lý
3. **Monitoring toàn diện**: Real-time metrics và alerting
4. **Tự động hóa**: Auto-retrain và self-healing
5. **Scalability**: Dễ dàng thêm trạm mới
6. **Maintainability**: Code modular và dễ bảo trì 