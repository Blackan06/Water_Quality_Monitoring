# Model Training System - Water Quality Monitoring

## Tổng quan

Hệ thống model training được thiết kế để train và so sánh 2 loại model cho dự đoán Water Quality Index (WQI):
- **XGBoost**: Gradient boosting model cho regression
- **LSTM**: Deep learning model cho time series prediction

## Tính năng chính

### 1. **Dataset Management**
- **Train/Test Split**: 80% / 20%
- **Cross-validation**: 5-fold CV cho XGBoost
- **Data preprocessing**: StandardScaler cho features
- **Feature selection**: `ph`, `temperature`, `do` (3 features có sẵn)

### 2. **Hyperparameter Tuning**
#### XGBoost Parameters:
```python
'n_estimators': [50, 100, 200]
'max_depth': [3, 6, 9]
'learning_rate': [0.01, 0.1, 0.2]
'subsample': [0.8, 0.9, 1.0]
'colsample_bytree': [0.8, 0.9, 1.0]
```

#### LSTM Parameters:
```python
'units': [32, 50, 64]
'dropout': [0.1, 0.2, 0.3]
'batch_size': [16, 32, 64]
'sequence_length': [5, 10, 15]
'learning_rate': [0.001, 0.01]
```

### 3. **Model Evaluation Metrics**
- **MAE** (Mean Absolute Error)
- **MSE** (Mean Squared Error)
- **RMSE** (Root Mean Squared Error)
- **R² Score** (Coefficient of determination)
- **MAPE** (Mean Absolute Percentage Error)
- **Cross-validation Score**

### 4. **Model Selection**
Hệ thống tự động chọn model tốt nhất dựa trên weighted score:
```python
score = (
    -0.3 * MAE +      # MAE càng thấp càng tốt
    0.4 * R2_score +  # R2 càng cao càng tốt
    -0.2 * MAPE +     # MAPE càng thấp càng tốt
    0.1 * CV_score    # CV score càng cao càng tốt
)
```

### 5. **Experiment Tracking**
- **Experiment ID**: Unique identifier cho mỗi training run
- **Hyperparameters**: Lưu tất cả parameters đã thử
- **Metrics**: Validation và test metrics
- **Model versions**: Timestamp-based versioning
- **MLflow integration**: Remote experiment tracking

## Workflow

### 1. **Data Preparation**
```python
# Load historical data
historical_data = db_manager.get_historical_data(station_id, limit=1000)

# Prepare training data
X_train, X_test, y_train, y_test = model_manager.prepare_training_data(data, 'xgboost')
```

### 2. **Hyperparameter Tuning**
```python
# Grid search với cross-validation
best_params, best_cv_score = model_manager.hyperparameter_tuning(X_train, y_train, 'xgboost')
```

### 3. **Model Training**
```python
# Train XGBoost
xgb_result = model_manager.train_xgboost_model(station_id, historical_data)

# Train LSTM
lstm_result = model_manager.train_lstm_model(station_id, historical_data)
```

### 4. **Model Comparison**
```python
# So sánh và chọn model tốt nhất
comparison_result = model_manager.compare_and_select_best_model(station_id, xgb_result, lstm_result)
```

### 5. **Model Persistence**
- **Local storage**: Models lưu trong `models/` directory
- **Scalers**: Lưu trong `scalers/` directory
- **Experiments**: Lưu trong `experiments/` directory
- **Database**: Metadata lưu trong PostgreSQL
- **MLflow**: Remote model registry

## File Structure

```
models/
├── xgboost_station_1_v20241201_143022.pkl
├── lstm_station_1_v20241201_143022.h5
└── ...

scalers/
├── xgboost_scaler_station_1_v20241201_143022.pkl
├── lstm_scaler_station_1_v20241201_143022.pkl
└── ...

experiments/
├── xgboost_station_1_20241201_143022_abc12345.json
├── lstm_station_1_20241201_143022_def67890.json
└── ...
```

## Database Tables

### 1. **model_registry**
```sql
CREATE TABLE model_registry (
    id SERIAL PRIMARY KEY,
    station_id INTEGER,
    model_type VARCHAR(50),
    model_version VARCHAR(100),
    model_path TEXT,
    accuracy FLOAT,
    mae FLOAT,
    r2_score FLOAT,
    training_date TIMESTAMP,
    is_active BOOLEAN DEFAULT TRUE
);
```

### 2. **training_history**
```sql
CREATE TABLE training_history (
    id SERIAL PRIMARY KEY,
    station_id INTEGER,
    model_type VARCHAR(50),
    training_start TIMESTAMP,
    training_end TIMESTAMP,
    training_duration INTEGER,
    records_used INTEGER,
    accuracy FLOAT,
    mae FLOAT,
    r2_score FLOAT,
    status VARCHAR(50),
    error_message TEXT
);
```

### 3. **model_comparison**
```sql
CREATE TABLE model_comparison (
    id SERIAL PRIMARY KEY,
    station_id INTEGER,
    comparison_date TIMESTAMP,
    best_model VARCHAR(50),
    reason TEXT,
    xgboost_score FLOAT,
    lstm_score FLOAT,
    xgboost_mae FLOAT,
    lstm_mae FLOAT,
    xgboost_r2 FLOAT,
    lstm_r2 FLOAT
);
```

## Usage Examples

### 1. **Train Models for a Station**
```python
from include.iot_streaming.pipeline_processor import PipelineProcessor

# Train models
training_results = PipelineProcessor.train_models_for_stations([1, 2, 3])

# Check results
for station_id, result in training_results.items():
    if 'error' not in result:
        best_model = result['comparison']['best_model']
        print(f"Station {station_id}: Best model = {best_model}")
```

### 2. **Get Best Model for Prediction**
```python
from include.iot_streaming.model_manager import model_manager

# Get best model type
best_model_type = model_manager.get_best_model_for_station(station_id)

# Make prediction
if best_model_type == 'xgboost':
    prediction = model_manager.predict_xgboost(station_id, data)
else:
    prediction = model_manager.predict_lstm(station_id, data)
```

### 3. **Load Experiment Data**
```python
# Load experiment
experiment_data = model_manager.load_experiment(experiment_id)

# Check metrics
test_metrics = experiment_data['test_metrics']
print(f"Test MAE: {test_metrics['mae']:.4f}")
print(f"Test R2: {test_metrics['r2_score']:.4f}")
```

## Monitoring và Logging

### 1. **Training Logs**
- Hyperparameter search progress
- Cross-validation scores
- Model evaluation metrics
- Model comparison results

### 2. **Performance Metrics**
- Training time per model
- Memory usage
- Model file sizes
- Prediction latency

### 3. **Quality Assurance**
- Data validation checks
- Model performance thresholds
- Automatic retraining triggers
- Model drift detection

## Best Practices

### 1. **Data Quality**
- Minimum 100 samples per station
- Handle missing values appropriately
- Validate data ranges (pH: 0-14, Temperature: -50 to 100°C, etc.)

### 2. **Model Selection**
- Use cross-validation for reliable estimates
- Consider business requirements (speed vs accuracy)
- Monitor model drift over time

### 3. **Reproducibility**
- Set random seeds for all operations
- Version control for hyperparameters
- Document data preprocessing steps

### 4. **Scalability**
- Parallel training for multiple stations
- Efficient hyperparameter search
- Model caching and lazy loading

## Troubleshooting

### 1. **Insufficient Data**
```
Error: Insufficient data for station X: Y records
Solution: Collect more historical data or reduce minimum requirements
```

### 2. **Training Failures**
```
Error: Hyperparameter tuning failed
Solution: Check data quality and adjust parameter ranges
```

### 3. **Memory Issues**
```
Error: Out of memory during LSTM training
Solution: Reduce batch size or sequence length
```

### 4. **Model Performance**
```
Warning: Low R2 score (< 0.5)
Solution: Check feature engineering or collect more relevant features
```

## Future Enhancements

1. **Advanced Hyperparameter Tuning**
   - Bayesian optimization
   - Population-based training
   - Multi-objective optimization

2. **Ensemble Methods**
   - Stacking models
   - Voting mechanisms
   - Dynamic ensemble selection

3. **Online Learning**
   - Incremental model updates
   - Concept drift detection
   - Adaptive hyperparameters

4. **Explainability**
   - SHAP values for XGBoost
   - Attention mechanisms for LSTM
   - Feature importance analysis 