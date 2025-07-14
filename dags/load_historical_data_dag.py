from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import logging
import pandas as pd
import os
import numpy as np
import shutil
import joblib
import json
import tensorflow as tf
import mlflow

# Cấu hình logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Default arguments
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2024, 1, 1),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

def convert_numpy_types(obj):
    """Convert numpy types to Python native types for XCom serialization"""
    if isinstance(obj, dict):
        return {str(k): convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj

def load_historical_data():
    """Load dữ liệu lịch sử từ file CSV"""
    try:
        csv_path = 'data/WQI_data.csv'
        if not os.path.exists(csv_path):
            logger.error(f"Historical data file not found: {csv_path}")
            return None
        
        # Load CSV data
        df = pd.read_csv(csv_path)
        logger.info(f"Loaded historical data: {len(df)} records from {csv_path}")
        
        # Convert Date column to datetime
        df['Date'] = pd.to_datetime(df['Date'])
        df['timestamp'] = df['Date']  # Add timestamp column for consistency
        
        # Rename columns to match expected format
        df = df.rename(columns={
            'PH': 'ph',
            'DO': 'do',
            'Temperature': 'temperature',
            'WQI': 'wqi'
        })
        
        # Verify data quality
        logger.info(f"Data columns: {list(df.columns)}")
        logger.info(f"Date range: {df['Date'].min()} to {df['Date'].max()}")
        logger.info(f"Stations: {sorted(df['station_id'].unique())}")
        logger.info(f"Records per station: {df.groupby('station_id').size().to_dict()}")
        
        # Check for missing values
        missing_data = df.isnull().sum()
        if missing_data.sum() > 0:
            logger.warning(f"Missing values found: {missing_data[missing_data > 0].to_dict()}")
        
        return df
        
    except Exception as e:
        logger.error(f"Error loading historical data: {e}")
        return None

def train_models_with_historical_data(**context):
    """Train Global Multi-Series models cho WQI forecasting sử dụng dữ liệu lịch sử 2003-2023"""
    from include.iot_streaming.model_manager import model_manager
    
    logger.info("Starting Global Multi-Series WQI forecasting training with historical data 2003-2023")
    
    # Load dữ liệu lịch sử từ CSV
    historical_df = load_historical_data()
    if historical_df is None:
        logger.error("Failed to load historical data")
        return "Failed to load historical data"
    
    # Get unique station IDs from data
    unique_stations = sorted(historical_df['station_id'].unique())
    logger.info(f"Found stations in data: {unique_stations}")
    
    # Train models for each station individually
    all_xgb_results = {}
    all_lstm_results = {}
    
    for station_id in unique_stations:
        logger.info(f"Training models for station {station_id}")
        
        # Filter data for this station
        station_data = historical_df[historical_df['station_id'] == station_id].copy()
        
        if len(station_data) < 50:
            logger.warning(f"Insufficient data for station {station_id}: {len(station_data)} records")
            continue
        
        # Train XGBoost model for this station
        xgb_result = model_manager.train_xgboost_model(station_id, station_data)
        all_xgb_results[station_id] = xgb_result
        
        # Train LSTM model for this station
        lstm_result = model_manager.train_lstm_model(station_id, station_data)
        all_lstm_results[station_id] = lstm_result
        
        # Create best model for this station
        if 'error' not in xgb_result and 'error' not in lstm_result:
            model_manager.create_best_model(station_id, xgb_result, lstm_result)
            logger.info(f"✅ Best model created for station {station_id}")
        else:
            logger.warning(f"❌ Failed to create best model for station {station_id}")
            if 'error' in xgb_result:
                logger.error(f"XGBoost error for station {station_id}: {xgb_result['error']}")
            if 'error' in lstm_result:
                logger.error(f"LSTM error for station {station_id}: {lstm_result['error']}")
    
    # Use results from station 0 as the main results (for backward compatibility)
    xgb_result = all_xgb_results.get(0, {'error': 'No models trained'})
    lstm_result = all_lstm_results.get(0, {'error': 'No models trained'})
    
    # Calculate overall metrics from all stations
    successful_stations = []
    total_xgb_r2 = 0.0
    total_lstm_r2 = 0.0
    successful_count = 0
    
    for station_id in unique_stations:
        xgb_result = all_xgb_results.get(station_id, {})
        lstm_result = all_lstm_results.get(station_id, {})
        
        if 'error' not in xgb_result and 'error' not in lstm_result:
            successful_stations.append(station_id)
            total_xgb_r2 += xgb_result.get('r2_score', 0.0)
            total_lstm_r2 += lstm_result.get('r2_score', 0.0)
            successful_count += 1
    
    if successful_count > 0:
        avg_xgb_r2 = total_xgb_r2 / successful_count
        avg_lstm_r2 = total_lstm_r2 / successful_count
        logger.info(f"Average XGBoost R² across {successful_count} stations: {avg_xgb_r2:.4f}")
        logger.info(f"Average LSTM R² across {successful_count} stations: {avg_lstm_r2:.4f}")
    else:
        avg_xgb_r2 = 0.0
        avg_lstm_r2 = 0.0
        logger.warning("No successful model training across all stations")
    
    # Chọn best model dựa vào average R2
    best_model_type = None
    best_r2 = -np.inf
    if avg_xgb_r2 > best_r2:
        best_model_type = 'xgboost'
        best_r2 = avg_xgb_r2
    if avg_lstm_r2 > best_r2:
        best_model_type = 'lstm'
        best_r2 = avg_lstm_r2
    
    logger.info(f"Best model type is {best_model_type} with average R2={best_r2:.4f}")
    
    # Create global best model using station 0 results (if available)
    if 0 in all_xgb_results and 0 in all_lstm_results:
        xgb_result_0 = all_xgb_results[0]
        lstm_result_0 = all_lstm_results[0]
        
        if 'error' not in xgb_result_0 and 'error' not in lstm_result_0:
            logger.info("Creating global best model using station 0 results...")
            best_model_created = model_manager.create_best_model(0, xgb_result_0, lstm_result_0)
            if best_model_created:
                logger.info("✅ Global best model created successfully using station 0")
            else:
                logger.error("❌ Failed to create global best model using station 0")
        else:
            logger.warning("⚠️ Cannot create global best model - station 0 models failed to train")
    else:
        logger.warning("⚠️ Station 0 not available for global best model creation")
    
    # Verify model registration in MLflow Registry
    if successful_count > 0:
        logger.info("Verifying model registration in MLflow Registry...")
        try:
            # Use MLflow client directly to check model registration
            from mlflow.tracking import MlflowClient
            client = MlflowClient()
            
            model_name = "water_quality"
            try:
                # Get the latest versions of the model
                latest_versions = client.get_latest_versions(model_name, stages=["None"])
                if latest_versions:
                    logger.info(f"✅ Models successfully registered in MLflow Registry as '{model_name}'")
                    logger.info(f"Latest version: {latest_versions[0].version}")
                else:
                    logger.warning(f"⚠️ No model versions found in MLflow Registry for '{model_name}'")
            except Exception as e:
                logger.warning(f"⚠️ Could not verify model registration: {e}")
                
        except Exception as e:
            logger.error(f"Error verifying model registration: {e}")
    
    logger.info(f"Multi-station WQI forecasting training completed")
    logger.info(f"Stations trained: {successful_stations}")
    logger.info(f"Average XGBoost R²: {avg_xgb_r2:.4f}")
    logger.info(f"Average LSTM R²: {avg_lstm_r2:.4f}")
    logger.info(f"Best model type: {best_model_type} with average R² = {best_r2:.4f}")
    logger.info(f"Total records used: {len(historical_df)}")
    logger.info(f"All stations included: {unique_stations}")
    
    # Push info to XCom
    context['task_instance'].xcom_push(key='best_model_type', value=best_model_type)
    context['task_instance'].xcom_push(key='best_model_r2', value=convert_numpy_types(best_r2))
    context['task_instance'].xcom_push(key='successful_stations', value=successful_stations)
    
    return f"Best model ({best_model_type}) trained for {len(successful_stations)} stations with average R2={best_r2}"

def generate_training_summary(**context):
    """Tạo summary chi tiết về quá trình training (pipeline chỉ còn 1 best model toàn cục)"""
    from datetime import datetime
    best_model_type = context['task_instance'].xcom_pull(
        task_ids='train_models_with_historical_data', key='best_model_type'
    )
    best_model_r2 = context['task_instance'].xcom_pull(
        task_ids='train_models_with_historical_data', key='best_model_r2'
    )
    summary = {
        'best_model_type': best_model_type,
        'best_model_r2': best_model_r2,
        'execution_time': datetime.now().isoformat()
    }
    logger.info(f"Training summary generated: {summary}")
    context['task_instance'].xcom_push(key='training_summary', value=summary)
    return f"Training summary: {best_model_type} (R2={best_model_r2})"

# Tạo DAG
dag = DAG(
    'load_historical_data_and_train',
    default_args=default_args,
    description='Load historical WQI data 2003-2023 and train Global Multi-Series forecasting models for WQI prediction',
    schedule_interval=None,  # Chạy thủ công
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=['water-quality', 'global-multiseries', 'wqi-forecasting', 'time-series', '2003-2023']
)

# Định nghĩa các tasks
train_models_task = PythonOperator(
    task_id='train_models_with_historical_data',
    python_callable=train_models_with_historical_data,
    dag=dag
)

summary_task = PythonOperator(
    task_id='generate_training_summary',
    python_callable=generate_training_summary,
    dag=dag
)

# Định nghĩa dependencies
train_models_task >> summary_task 