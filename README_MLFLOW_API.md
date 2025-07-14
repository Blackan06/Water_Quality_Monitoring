# MLflow API Integration

## Tổng quan

Project này sử dụng MLflow REST API thay vì import trực tiếp thư viện MLflow. Điều này cho phép:

- **Tách biệt**: MLflow có thể được triển khai trên server riêng
- **Scalability**: Dễ dàng scale MLflow service độc lập
- **Flexibility**: Có thể sử dụng MLflow cloud hoặc on-premise
- **Security**: Authentication và authorization riêng biệt

## Cấu hình MLflow API

### 1. Biến môi trường

Tạo file `.env` hoặc set environment variables:

```bash
# MLflow Configuration
MLFLOW_TRACKING_URI=http://localhost:5000
MLFLOW_REGISTRY_URI=http://localhost:5000
MLFLOW_API_BASE_URL=http://localhost:5000/api/2.0
MLFLOW_USERNAME=your_username
MLFLOW_PASSWORD=your_password
```

### 2. Cấu hình MLflow Server

#### Option 1: Local MLflow Server

```bash
# Install MLflow
pip install mlflow

# Start MLflow server
mlflow server \
    --host 0.0.0.0 \
    --port 5000 \
    --backend-store-uri sqlite:///mlflow.db \
    --default-artifact-root ./mlruns
```

#### Option 2: MLflow với PostgreSQL

```bash
# Start MLflow với PostgreSQL backend
mlflow server \
    --host 0.0.0.0 \
    --port 5000 \
    --backend-store-uri postgresql://user:password@localhost:5432/mlflow \
    --default-artifact-root s3://your-bucket/mlflow-artifacts
```

#### Option 3: MLflow Cloud

Sử dụng MLflow AI Platform hoặc Databricks:

```bash
export MLFLOW_TRACKING_URI=https://your-mlflow-instance.com
export MLFLOW_API_BASE_URL=https://your-mlflow-instance.com/api/2.0
```

### 3. Test kết nối

```bash
# Load environment variables
source config/mlflow.env

# Test MLflow API connection
python include/iot_streaming/test_mlflow_api.py
```

## Cách hoạt động

### 1. Model Registration

Khi train model mới, `ModelManager` sẽ:

1. **Train model locally** và lưu file
2. **Gọi MLflow API** để tạo experiment
3. **Log parameters** và metrics
4. **Register model** trong MLflow Registry

```python
# Ví dụ từ model_manager.py
def register_model_in_mlflow(self, station_id: int, model_type: str, model_data: dict):
    # Tạo experiment
    experiment_data = {'name': f"water-quality-station-{station_id}"}
    experiment_response = self.mlflow_api_call('mlflow/experiments/create', 'POST', experiment_data)
    
    # Tạo run
    run_data = {
        'experiment_id': experiment_id,
        'start_time': int(datetime.now().timestamp() * 1000),
        'tags': [
            {'key': 'station_id', 'value': str(station_id)},
            {'key': 'model_type', 'value': model_type}
        ]
    }
    
    # Log parameters và metrics
    # ...
```

### 2. Model Retrieval

Khi cần load model:

1. **Thử load từ MLflow** trước
2. **Fallback** về local file nếu MLflow không có
3. **Cache model** để tăng performance

```python
def load_model(self, station_id: int, model_type: str):
    # Thử load từ MLflow trước
    mlflow_model = self.get_model_from_mlflow(station_id, model_type)
    
    if mlflow_model:
        # Download model từ MLflow
        return download_model_from_mlflow(mlflow_model['model_uri'])
    
    # Fallback: Load từ local file
    model_path = self.get_model_path(station_id, model_type)
    return joblib.load(model_path)
```

### 3. API Endpoints được sử dụng

- `GET /api/2.0/mlflow/experiments/list` - List experiments
- `POST /api/2.0/mlflow/experiments/create` - Create experiment
- `POST /api/2.0/mlflow/runs/create` - Create run
- `POST /api/2.0/mlflow/runs/log-parameter` - Log parameters
- `POST /api/2.0/mlflow/runs/log-metric` - Log metrics
- `POST /api/2.0/mlflow/runs/update` - End run
- `GET /api/2.0/mlflow/registered-models/get-latest-versions` - Get model versions

## Monitoring và Management

### 1. MLflow UI

Truy cập MLflow UI để xem experiments và models:

```bash
# Nếu chạy local
open http://localhost:5000

# Hoặc truy cập trực tiếp
curl http://localhost:5000
```

### 2. API Monitoring

```python
# Kiểm tra model info
model_info = model_manager.get_model_info(station_id=1, model_type='xgboost')
print(f"Model source: {model_info['source']}")
print(f"Model version: {model_info.get('version', 'N/A')}")
```

### 3. Model Lifecycle

```python
# Register model mới
model_manager.register_model_in_mlflow(station_id=1, model_type='xgboost', model_data=result)

# Get model từ registry
mlflow_model = model_manager.get_model_from_mlflow(station_id=1, model_type='xgboost')

# Delete model
model_manager.delete_model(station_id=1, model_type='xgboost')
```

## Troubleshooting

### 1. Connection Issues

```bash
# Test connectivity
curl -X GET http://localhost:5000/health

# Test API
curl -X GET http://localhost:5000/api/2.0/mlflow/experiments/list
```

### 2. Authentication Issues

```bash
# Set credentials
export MLFLOW_USERNAME=your_username
export MLFLOW_PASSWORD=your_password

# Test với authentication
python include/iot_streaming/test_mlflow_api.py
```

### 3. Model Download Issues

Nếu không thể download model từ MLflow:

```python
# Fallback về local file
model = model_manager.load_model(station_id=1, model_type='xgboost')
if model is None:
    print("Model not available in MLflow, using local file")
```

### 4. Performance Issues

- **Cache models** locally để tránh download lại
- **Batch API calls** khi có thể
- **Use connection pooling** cho HTTP requests

## Best Practices

### 1. Error Handling

```python
def mlflow_api_call(self, endpoint: str, method: str = 'GET', data: dict = None):
    try:
        # API call logic
        if response.status_code == 200:
            return response.json()
        else:
            logger.error(f"MLflow API call failed: {response.status_code}")
            return None
    except Exception as e:
        logger.error(f"Error calling MLflow API: {e}")
        return None
```

### 2. Fallback Strategy

```python
# Always have fallback
def load_model(self, station_id: int, model_type: str):
    # Try MLflow first
    mlflow_model = self.get_model_from_mlflow(station_id, model_type)
    
    # Fallback to local
    if mlflow_model is None:
        return self.load_local_model(station_id, model_type)
```

### 3. Versioning

```python
# Use semantic versioning
model_version = f"{model_type}_v{major}.{minor}.{patch}_station_{station_id}"

# Track model lineage
tags = [
    {'key': 'station_id', 'value': str(station_id)},
    {'key': 'model_type', 'value': model_type},
    {'key': 'version', 'value': model_version}
]
```

## Deployment

### 1. Production MLflow

```bash
# Use production database
mlflow server \
    --host 0.0.0.0 \
    --port 5000 \
    --backend-store-uri postgresql://user:pass@db:5432/mlflow \
    --default-artifact-root s3://bucket/mlflow \
    --workers 4
```

### 2. Docker Compose

```yaml
services:
  mlflow:
    image: python:3.9
    command: mlflow server --host 0.0.0.0 --port 5000
    ports:
      - "5000:5000"
    environment:
      - MLFLOW_TRACKING_URI=http://localhost:5000
    volumes:
      - ./mlruns:/mlruns
```

### 3. Kubernetes

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: mlflow
spec:
  replicas: 3
  selector:
    matchLabels:
      app: mlflow
  template:
    metadata:
      labels:
        app: mlflow
    spec:
      containers:
      - name: mlflow
        image: python:3.9
        command: ["mlflow", "server", "--host", "0.0.0.0", "--port", "5000"]
        ports:
        - containerPort: 5000
```

## Kết luận

Việc sử dụng MLflow REST API thay vì import trực tiếp mang lại nhiều lợi ích:

- **Separation of concerns**: MLflow service độc lập
- **Scalability**: Dễ dàng scale và maintain
- **Flexibility**: Có thể switch giữa local và cloud MLflow
- **Reliability**: Fallback strategy đảm bảo system luôn hoạt động

Đảm bảo test kết nối trước khi deploy và có fallback strategy phù hợp. 