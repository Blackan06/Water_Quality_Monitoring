# Tests Directory

Thư mục này chứa tất cả các script test cho hệ thống Water Quality Monitoring.

## 📁 Cấu trúc thư mục

```
tests/
├── dags/                           # Test cho Airflow DAGs
│   ├── test_dag_example.py        # Test DAG cơ bản (có sẵn)
│   └── test_streaming_dag.py      # Test streaming DAG với AwaitMessageTriggerFunctionSensor
├── kafka/                          # Test cho Kafka
│   ├── test_kafka_connection.py   # Test kết nối Kafka cơ bản (script)
│   ├── test_kafka_integration.py  # Test Kafka integration (pytest)
│   └── test_airflow_kafka_dag.py  # Test Airflow DAG với Kafka (script)
├── monitoring/                     # Test cho monitoring services
│   └── test_monitoring_services.py # Test Kafka UI, Prometheus, Grafana
├── integration/                    # Test tích hợp
│   └── run_all_tests.py           # Script chạy tất cả test
├── test_global_multiseries.py     # Test Global Multi-Series WQI Forecasting
└── README.md                       # File này
```

## 🚀 Cách sử dụng

### 1. Chạy tất cả test (Khuyến nghị)

Từ thư mục gốc của project:
```bash
python tests/integration/run_all_tests.py
```

### 2. Chạy test pytest

#### Test streaming DAG với AwaitMessageTriggerFunctionSensor
```bash
pytest tests/dags/test_streaming_dag.py -v
```

#### Test Kafka integration
```bash
pytest tests/kafka/test_kafka_integration.py -v
```

#### Test Global Multi-Series WQI Forecasting
```bash
python tests/test_global_multiseries.py
```

#### Chạy tất cả test pytest
```bash
pytest tests/ -v
```

### 3. Chạy từng test script riêng lẻ

#### Test kết nối Kafka cơ bản
```bash
python tests/kafka/test_kafka_connection.py
```

#### Test Airflow DAG
```bash
python tests/kafka/test_airflow_kafka_dag.py
```

#### Test monitoring services
```bash
python tests/monitoring/test_monitoring_services.py
```

#### Test Global Multi-Series WQI Forecasting
```bash
python tests/test_global_multiseries.py
```

### 4. Chạy test DAG cơ bản (có sẵn)
```bash
python tests/dags/test_dag_example.py
```

## 📋 Mô tả các test

### `tests/test_global_multiseries.py` (Unittest)
- **Mục đích**: Test Global Multi-Series WQI Forecasting implementation
- **Chức năng**:
  - Test data loading và preparation
  - Test Global Multi-Series data preparation với tất cả stations
  - Test XGBoost Global Multi-Series training
  - Test LSTM Global Multi-Series training
  - Test feature engineering robustness
  - Test station features inclusion (one-hot encoding, embedding)
  - Verify model learns from all stations simultaneously

### `tests/dags/test_streaming_dag.py` (Pytest)
- **Mục đích**: Test streaming DAG với AwaitMessageTriggerFunctionSensor
- **Chức năng**:
  - Test import DAG không có lỗi
  - Test DAG có AwaitMessageTriggerFunctionSensor
  - Test cấu hình Kafka sensor
  - Test task dependencies
  - Test tags và metadata
  - Test Airflow connection configuration

### `tests/kafka/test_kafka_integration.py` (Pytest)
- **Mục đích**: Test Kafka integration functionality
- **Chức năng**:
  - Test kết nối Kafka broker
  - Test topic tồn tại
  - Test producer/consumer functions
  - Test message format
  - Test consumer group
  - Test serialization
  - Test database manager

### `tests/kafka/test_kafka_connection.py` (Script)
- **Mục đích**: Kiểm tra kết nối Kafka cơ bản
- **Chức năng**:
  - Test kết nối đến Kafka broker (194.238.16.14:9092)
  - Test tạo/xóa topic
  - Test producer gửi message
  - Test consumer nhận message
  - Test consumer đọc từ đầu topic

### `tests/kafka/test_airflow_kafka_dag.py` (Script)
- **Mục đích**: Kiểm tra Airflow DAG với Kafka
- **Chức năng**:
  - Test kafka producer task
  - Test kafka consumer task
  - Test database manager
  - Kiểm tra môi trường Airflow

### `tests/monitoring/test_monitoring_services.py` (Script)
- **Mục đích**: Kiểm tra các monitoring services
- **Chức năng**:
  - Test Kafka UI (port 8085)
  - Test Prometheus (port 9090)
  - Test Grafana (port 3000)
  - Test Kafka Exporter (port 9308)
  - Kiểm tra Docker services

### `tests/integration/run_all_tests.py` (Script)
- **Mục đích**: Chạy tất cả test và đưa ra báo cáo tổng hợp
- **Chức năng**:
  - Kiểm tra môi trường và dependencies
  - Chạy tất cả test theo thứ tự
  - Đưa ra báo cáo tổng hợp
  - Hướng dẫn khắc phục sự cố

## 🔧 Yêu cầu hệ thống

### Thư viện Python cần thiết:
```bash
pip install kafka-python requests psycopg2-binary elasticsearch pytest pandas numpy scikit-learn xgboost tensorflow
```

### Services cần chạy:
- Kafka broker tại `194.238.16.14:9092`
- Docker containers cho monitoring services
- Airflow services
- MLflow tracking server

## 📊 Kết quả mong đợi

### Khi tất cả test thành công:
```
🎉 TẤT CẢ TEST ĐỀU THÀNH CÔNG!
✅ Hệ thống Kafka hoạt động bình thường
✅ Global Multi-Series WQI Forecasting hoạt động bình thường
```

### Khi có test thất bại:
```
⚠️ MỘT SỐ TEST THẤT BẠI
🔧 Vui lòng kiểm tra và khắc phục các vấn đề
```

### Kết quả Global Multi-Series test:
```
🧪 Running Global Multi-Series WQI Forecasting Tests
============================================================
✅ Data loaded: 748 records
   Stations: [0, 1, 2]
   Date range: 2003-01-15 00:00:00 to 2023-12-15 00:00:00
   WQI range: 45.20 - 95.80

✅ Global Multi-Series data preparation successful!
   Train samples: 598
   Test samples: 150
   Total features: 156
   WQI range: 45.20 - 95.80

✅ XGBoost Global Multi-Series training successful!
   MAE: 0.1234
   R²: 0.8765
   Records used: 748
   Stations included: [0, 1, 2]

✅ LSTM Global Multi-Series training successful!
   MAE: 0.1345
   R²: 0.8654
   Records used: 748
   Stations included: [0, 1, 2]

🎉 All Global Multi-Series tests passed!
```

### Kết quả pytest:
```
============================= test session starts ==============================
collecting ... collected 15 tests
tests/dags/test_streaming_dag.py::test_streaming_dag_import PASSED
tests/dags/test_streaming_dag.py::test_streaming_dag_exists PASSED
tests/dags/test_streaming_dag.py::test_streaming_dag_has_kafka_sensor PASSED
...
============================== 15 passed in 5.23s ==============================
```

## 🔗 URLs hữu ích

- **Kafka UI**: http://localhost:8085
  - Username: `admin`
  - Password: `admin1234`

- **Airflow UI**: http://localhost:8080
  - Username: `airflow`
  - Password: `airflow`

- **Prometheus**: http://localhost:9090

- **Grafana**: http://localhost:3000
  - Username: `admin`
  - Password: `admin123`

- **MLflow UI**: http://localhost:5000

## 🐛 Khắc phục sự cố

### 1. Không thể kết nối Kafka
- Kiểm tra Kafka broker có đang chạy không
- Kiểm tra network connectivity đến `194.238.16.14:9092`
- Kiểm tra firewall settings

### 2. Không thể kết nối monitoring services
- Kiểm tra Docker containers có đang chạy không
- Kiểm tra ports có bị conflict không
- Chạy lại Docker Compose: `docker-compose up -d`

### 3. Airflow DAG lỗi
- Kiểm tra Airflow connections trong UI
- Kiểm tra cấu hình `airflow_settings.yaml`
- Restart Airflow services

### 4. Global Multi-Series test thất bại
- Kiểm tra file data `data/WQI_data.csv` có tồn tại không
- Kiểm tra MLflow tracking server có đang chạy không
- Kiểm tra đủ memory cho model training
- Kiểm tra dependencies: `pandas`, `numpy`, `scikit-learn`, `xgboost`, `tensorflow`

### 5. Thiếu thư viện
```bash
pip install kafka-python requests psycopg2-binary elasticsearch pytest pandas numpy scikit-learn xgboost tensorflow
```

## 📝 Ghi chú

### Test Types:
- **Pytest tests**: Sử dụng framework pytest, có thể chạy với `pytest` command
- **Script tests**: Chạy trực tiếp với `python` command

### Test Coverage:
- **DAG tests**: Kiểm tra cấu trúc và cấu hình DAG
- **Integration tests**: Kiểm tra tương tác giữa các components
- **Monitoring tests**: Kiểm tra các services monitoring
- **Connection tests**: Kiểm tra kết nối network và database

### Best Practices:
- Chạy pytest tests trước khi deploy
- Sử dụng script tests cho debugging nhanh
- Kiểm tra logs chi tiết khi có lỗi
- Cập nhật tests khi thay đổi cấu hình 