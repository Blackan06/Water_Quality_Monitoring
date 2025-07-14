# Water Quality Monitoring Pipeline Flow

## Tổng quan hệ thống

Hệ thống giám sát chất lượng nước được thiết kế với 3 DAG chính, hoạt động độc lập và bổ sung cho nhau:

## 1. DAG: `load_historical_data` 
**Mục đích**: Load dữ liệu lịch sử từ file CSV vào database

### Thông tin:
- **Schedule**: Chạy thủ công (không theo lịch)
- **Mô tả**: Load dữ liệu từ `data/WQI_data.csv` vào bảng `historical_wqi_data`
- **Khi nào chạy**: 
  - Lần đầu setup hệ thống
  - Khi có dữ liệu lịch sử mới cần cập nhật

### Tasks:
1. `load_historical_data`: Load dữ liệu từ CSV
2. `verify_data_loaded`: Xác minh dữ liệu đã được load thành công

---

## 2. DAG: `streaming_process_dag` (đã có sẵn)
**Mục đích**: Nhận dữ liệu từ Kafka và lưu vào database

### Thông tin:
- **Schedule**: Chạy liên tục (streaming)
- **Mô tả**: Nhận dữ liệu từ Kafka topic, xử lý và lưu vào bảng `raw_sensor_data`
- **Input**: Dữ liệu streaming từ Kafka
- **Output**: Dữ liệu được lưu vào database

---

## 3. DAG: `streaming_data_processor` ⭐ **DAG CHÍNH**
**Mục đích**: Xử lý dữ liệu streaming với logic thông minh train/predict

### Thông tin:
- **Schedule**: Chạy thủ công (được trigger bởi `streaming_process_dag`)
- **Mô tả**: DAG chính xử lý dữ liệu từ Kafka, tự động quyết định train hay predict
- **Trigger**: Được gọi tự động sau khi `streaming_process_dag` lưu dữ liệu vào database
- **Logic thông minh**:
  - Nếu station có model sẵn → Predict
  - Nếu station chưa có model → Train → Predict

### Tasks:
1. `process_streaming_data`: Phân tích dữ liệu chưa xử lý và phân loại stations
2. `train_new_stations`: Train models cho stations mới
3. `predict_existing_stations`: Predict cho stations có sẵn model
4. `predict_newly_trained_stations`: Predict cho stations vừa train
5. `compare_model_performance`: So sánh performance các model
6. `update_monitoring_metrics`: Cập nhật metrics monitoring
7. `generate_alerts_and_notifications`: Tạo alerts và notifications
8. `summarize_pipeline_execution`: Tóm tắt kết quả

### Luồng xử lý:
```
process_streaming_data
    ↓
    ├── train_new_stations → predict_newly_trained_stations
    └── predict_existing_stations
    ↓
compare_model_performance → update_monitoring_metrics → generate_alerts_and_notifications → summarize_pipeline_execution
```

---

## Luồng hoạt động tổng thể

### Bước 1: Setup ban đầu
```bash
# Chạy DAG load_historical_data để load dữ liệu lịch sử
airflow dags trigger load_historical_data
```

### Bước 2: Vận hành hàng ngày
1. **Bạn truyền dữ liệu vào Kafka topic `water-quality-data`**
2. **`streaming_process_dag`** được kích hoạt tự động:
   - Nhận dữ liệu từ Kafka
   - Lưu vào bảng `raw_sensor_data`
   - **Tự động trigger `streaming_data_processor`**
3. **`streaming_data_processor`** chạy:
   - Lấy tất cả dữ liệu chưa xử lý từ `raw_sensor_data`
   - Phân loại stations: có model vs cần train
   - Train models cho stations mới
   - Predict cho tất cả stations
   - So sánh performance và tạo alerts

### Bước 3: Monitoring và Alerts
- Kết quả dự đoán được lưu vào `prediction_results`
- Alerts được tạo dựa trên ngưỡng WQI
- Metrics được cập nhật cho monitoring

---

## Ưu điểm của thiết kế mới

### ✅ **Tách biệt rõ ràng**
- Load dữ liệu lịch sử vs xử lý streaming
- Mỗi DAG có trách nhiệm riêng biệt

### ✅ **Logic thông minh**
- Tự động quyết định train hay predict
- Không train lại model không cần thiết
- Xử lý stations mới một cách tự động

### ✅ **Hiệu quả và tiết kiệm**
- Chỉ train khi cần thiết
- Tận dụng models đã có
- Giảm thời gian xử lý

### ✅ **Dễ maintain và debug**
- Mỗi DAG độc lập
- Logging chi tiết
- Error handling tốt

---

## Cấu hình và triển khai

### Environment Variables cần thiết:
```bash
DB_HOST=194.238.16.14
DB_PORT=5432
DB_NAME=wqi_db
DB_USER=postgres
DB_PASSWORD=postgres1234
```

### Dependencies:
- PostgreSQL database
- Kafka cluster
- Airflow
- MLflow (cho model management)

### Monitoring:
- Grafana dashboards
- Prometheus metrics
- Airflow UI để theo dõi DAG execution

---

## Troubleshooting

### DAG không chạy:
1. Kiểm tra kết nối database
2. Kiểm tra Kafka connection
3. Xem logs trong Airflow UI

### Không có dữ liệu mới:
1. Kiểm tra `streaming_process_dag` có chạy không
2. Kiểm tra Kafka topic có dữ liệu không
3. Kiểm tra thời gian window (30 phút)

### Model training thất bại:
1. Kiểm tra dữ liệu lịch sử có đủ không
2. Kiểm tra MLflow connection
3. Xem logs chi tiết trong task 