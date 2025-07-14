from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import logging
import math
import pandas as pd
import os
import json # Added for loading model_info.json

# Cấu hình logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Default arguments
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2025, 1, 1),
    'email_on_failure': False,
    'email_on_retry': False,
}

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

def calculate_wqi(ph, temperature, do):
    """Tính WQI từ các thông số pH, temperature, DO"""
    try:
        # Convert Decimal to float để tránh lỗi type mismatch
        ph = float(ph) if ph is not None else 0.0
        temperature = float(temperature) if temperature is not None else 0.0
        do = float(do) if do is not None else 0.0
        
        # Tính sub-indices cho từng thông số
        # pH sub-index (giá trị tối ưu: 7.0)
        if ph <= 7.0:
            ph_subindex = 100 - (7.0 - ph) * 20  # Giảm 20 điểm cho mỗi đơn vị pH dưới 7.0
        else:
            ph_subindex = 100 - (ph - 7.0) * 20  # Giảm 20 điểm cho mỗi đơn vị pH trên 7.0
        
        ph_subindex = max(0, min(100, ph_subindex))
        
        # Temperature sub-index (giá trị tối ưu: 20-25°C)
        if 20 <= temperature <= 25:
            temp_subindex = 100
        elif temperature < 20:
            temp_subindex = 100 - (20 - temperature) * 5  # Giảm 5 điểm cho mỗi độ dưới 20
        else:
            temp_subindex = 100 - (temperature - 25) * 5  # Giảm 5 điểm cho mỗi độ trên 25
        
        temp_subindex = max(0, min(100, temp_subindex))
        
        # DO sub-index (giá trị tối ưu: >8 mg/L)
        if do >= 8:
            do_subindex = 100
        else:
            do_subindex = do * 12.5  # Tỷ lệ thuận với DO, tối đa 100
        
        do_subindex = max(0, min(100, do_subindex))
        
        # Tính WQI tổng hợp (trung bình có trọng số)
        # Trọng số: pH (30%), Temperature (20%), DO (50%)
        wqi = (ph_subindex * 0.3) + (temp_subindex * 0.2) + (do_subindex * 0.5)
        
        return round(wqi, 2)
        
    except Exception as e:
        logger.error(f"Error calculating WQI: {e}")
        logger.error(f"Input values - ph: {ph} (type: {type(ph)}), temperature: {temperature} (type: {type(temperature)}), do: {do} (type: {type(do)})")
        return None

def process_streaming_data(**context):
    """Xử lý dữ liệu streaming từ Kafka và phân loại stations"""
    from include.iot_streaming.database_manager import db_manager
    
    logger.info("Starting streaming data processing")
    
    # Lấy dữ liệu chưa được xử lý từ raw_sensor_data
    conn = db_manager.get_connection()
    if not conn:
        logger.error("Cannot connect to database")
        return "No database connection"
    
    cur = conn.cursor()
    # Lấy tất cả dữ liệu chưa được xử lý (chưa có trong processed_water_quality_data)
    # Chỉ lấy các cột cần thiết, không bao gồm wqi từ raw_sensor_data
    cur.execute("""
        SELECT DISTINCT rs.station_id, rs.measurement_time, rs.ph, rs.temperature, rs."do"
        FROM raw_sensor_data rs
        WHERE rs.is_processed = FALSE  -- Chưa được xử lý
        ORDER BY rs.station_id, rs.measurement_time DESC
    """)
    
    unprocessed_data = cur.fetchall()
    cur.close()
    conn.close()
    
    if not unprocessed_data:
        logger.info("No unprocessed streaming data found")
        return "No unprocessed streaming data found"
    
    logger.info(f"Found {len(unprocessed_data)} unprocessed data records")
    
    # Tính WQI và cập nhật vào raw_sensor_data, sau đó lưu vào processed_water_quality_data
    processed_count = 0
    conn = db_manager.get_connection()
    if not conn:
        logger.error("Cannot connect to database for processing")
        return "No database connection"
    
    for row in unprocessed_data:
        station_id, measurement_time, ph, temperature, do = row
        
        # Tính WQI
        wqi = calculate_wqi(ph, temperature, do)
        
        if wqi is not None:
            try:
                cur = conn.cursor()
                
                # Cập nhật WQI vào raw_sensor_data (nếu cột wqi tồn tại)
                try:
                    cur.execute("""
                        UPDATE raw_sensor_data 
                        SET wqi = %s
                        WHERE station_id = %s AND measurement_time = %s
                    """, (wqi, station_id, measurement_time))
                except Exception as update_error:
                    logger.warning(f"Could not update WQI in raw_sensor_data (column may not exist): {update_error}")
                    # Continue without updating raw_sensor_data
                
                # Lưu vào processed_water_quality_data
                try:
                    cur.execute("""
                        INSERT INTO processed_water_quality_data 
                        (station_id, measurement_time, ph, temperature, "do", wqi)
                        VALUES (%s, %s, %s, %s, %s, %s)
                        ON CONFLICT (station_id, measurement_time) DO NOTHING
                    """, (station_id, measurement_time, ph, temperature, do, wqi))
                except Exception as conflict_error:
                    # Nếu không có unique constraint, thử INSERT thường
                    if "no unique or exclusion constraint" in str(conflict_error):
                        logger.warning("No unique constraint found, using regular INSERT")
                        cur.execute("""
                            INSERT INTO processed_water_quality_data 
                            (station_id, measurement_time, ph, temperature, "do", wqi)
                            VALUES (%s, %s, %s, %s, %s, %s)
                        """, (station_id, measurement_time, ph, temperature, do, wqi))
                    else:
                        raise conflict_error
                
                # Đánh dấu đã xử lý
                cur.execute("""
                    UPDATE raw_sensor_data 
                    SET is_processed = TRUE
                    WHERE station_id = %s AND measurement_time = %s
                """, (station_id, measurement_time))
                
                conn.commit()
                processed_count += 1
                logger.debug(f"Processed data for station {station_id}: WQI = {wqi}")
                
            except Exception as e:
                logger.error(f"Error processing data for station {station_id}: {e}")
                try:
                    conn.rollback()
                except Exception as rollback_error:
                    logger.error(f"Error during rollback: {rollback_error}")
                    # Reset connection if rollback fails
                    conn.close()
                    conn = db_manager.get_connection()
                    if not conn:
                        logger.error("Cannot reconnect to database")
                        break
            finally:
                try:
                    cur.close()
                except Exception as close_error:
                    logger.error(f"Error closing cursor: {close_error}")
    
    conn.close()
    
    # Phân loại stations: CHỈ sử dụng models đã có sẵn, KHÔNG train mới
    stations_with_models = []
    stations_need_training = []  # Sẽ luôn rỗng vì không train mới
    
    # Lấy danh sách stations unique từ dữ liệu đã xử lý
    unique_stations = list(set([row[0] for row in unprocessed_data]))
    
    # Kiểm tra xem có models sẵn cho các stations không
    from include.iot_streaming.model_manager import model_manager
    
    # Kiểm tra tất cả stations để xem có model nào không
    available_stations = []
    # Get unique stations from the data
    data_stations = list(set([row[0] for row in unprocessed_data]))
    logger.info(f"Found stations in streaming data: {data_stations}")
    
    for station_id in data_stations:
        # Kiểm tra xem có best_model cho station này không
        best_model_path = f"models/best_model_station_{station_id}"
        global_best_model_path = "models/best_model"
        
        if os.path.exists(best_model_path) or os.path.exists(global_best_model_path):
            available_stations.append(station_id)
            logger.info(f"✅ Found pre-trained model for station {station_id}")
        else:
            logger.info(f"⚠️ No pre-trained model found for station {station_id}")
    
    if available_stations:
        # Có models available, chỉ xử lý stations đã có model
        logger.info(f"Found pre-trained models for {len(available_stations)} stations: {available_stations}")
        
        # Kiểm tra từng station trong unprocessed data xem có model không
        for station_id in unique_stations:
            if station_id in available_stations:
                stations_with_models.append(station_id)
                logger.info(f"✅ Station {station_id} will be processed (has pre-trained model)")
            else:
                logger.info(f"⚠️ Station {station_id} has no pre-trained model, skipping prediction")
        
        logger.info(f"Processing {len(stations_with_models)} stations with pre-trained models: {stations_with_models}")
        logger.info(f"Skipping {len(unique_stations) - len(stations_with_models)} stations without pre-trained models")
    else:
        # Chưa có model nào, không train station mới
        logger.info(f"No pre-trained models found for any station, skipping all {len(unique_stations)} stations")
        logger.info("To enable predictions, ensure pre-trained models are available in the models/ directory")
    
    # Lưu kết quả phân loại
    context['task_instance'].xcom_push(key='stations_with_models', value=stations_with_models)
    context['task_instance'].xcom_push(key='stations_need_training', value=stations_need_training)
    context['task_instance'].xcom_push(key='total_unprocessed_records', value=len(unprocessed_data))
    context['task_instance'].xcom_push(key='processed_records', value=processed_count)
    
    logger.info(f"Stations with pre-trained models: {stations_with_models}")
    logger.info(f"Stations need training: {stations_need_training}")
    
    return f"Processed {processed_count}/{len(unprocessed_data)} records with WQI calculation: {len(stations_with_models)} with pre-trained models, {len(stations_need_training)} need training"

def predict_existing_stations(**context):
    """Dự đoán WQI tương lai cho các trạm sử dụng pre-trained models cho monthly data"""
    from include.iot_streaming.model_manager import model_manager
    
    stations_with_models = context['task_instance'].xcom_pull(
        task_ids='process_streaming_data', key='stations_with_models'
    )
    
    if not stations_with_models:
        logger.info("No stations to predict")
        return "No stations to predict"
    
    logger.info(f"Starting future WQI predictions using pre-trained models for stations: {stations_with_models}")
    
    # Thực hiện dự đoán cho từng station sử dụng pre-trained models
    prediction_results = {}
    
    # Các khoảng thời gian dự đoán cho monthly data
    prediction_horizons = [1, 3, 12]  # 1 tháng, 3 tháng, 12 tháng (1 năm) nữa
    
    for station_id in stations_with_models:
        try:
            # Kiểm tra xem có best_model cho station này không
            best_model_path = f"models/best_model_station_{station_id}"
            global_best_model_path = "models/best_model"
            
            # Ưu tiên station-specific model, nếu không có thì dùng global model
            if os.path.exists(best_model_path):
                model_path = best_model_path
                logger.info(f"Using station-specific model for station {station_id}")
            elif os.path.exists(global_best_model_path):
                model_path = global_best_model_path
                logger.info(f"Using global model for station {station_id}")
            else:
                logger.warning(f"No pre-trained model found for station {station_id}")
                prediction_results[station_id] = {
                    'success': False,
                    'error': 'No pre-trained model available'
                }
                continue
            
            # Load model info để biết loại model nào tốt nhất
            model_info_path = os.path.join(model_path, "model_info.json")
            if os.path.exists(model_info_path):
                with open(model_info_path, 'r') as f:
                    model_info = json.load(f)
                best_model_type = model_info.get('best_model', 'xgboost')
                logger.info(f"Best model type for station {station_id}: {best_model_type}")
            else:
                best_model_type = 'xgboost'  # Default to XGBoost
                logger.warning(f"No model_info.json found for station {station_id}, using XGBoost as default")
            
            # Lấy dữ liệu mới nhất của station
            from include.iot_streaming.database_manager import db_manager
            conn = db_manager.get_connection()
            cur = conn.cursor()
            
            # Lấy dữ liệu hiện tại và lịch sử gần đây (monthly data)
            cur.execute("""
                SELECT ph, temperature, "do", wqi, measurement_time
                FROM processed_water_quality_data 
                WHERE station_id = %s 
                ORDER BY measurement_time DESC 
                LIMIT 24  -- Lấy 24 tháng gần nhất (2 năm)
            """, (station_id,))
            
            historical_data = cur.fetchall()
            cur.close()
            conn.close()
            
            if historical_data:
                # Lấy dữ liệu mới nhất
                latest_ph, latest_temperature, latest_do, latest_wqi, latest_time = historical_data[0]
                
                # Chuẩn bị dữ liệu cho prediction
                input_data = {
                    'ph': latest_ph,
                    'temperature': latest_temperature,
                    'do': latest_do,
                    'station_id': station_id,
                    'current_wqi': latest_wqi,
                    'current_time': latest_time,
                    'historical_data': historical_data
                }
                
                # Thực hiện prediction cho từng horizon
                future_predictions = {}
                
                for horizon_months in prediction_horizons:
                    try:
                        # Thêm thông tin horizon vào input data
                        input_data['prediction_horizon'] = horizon_months
                        
                        # Thực hiện prediction với pre-trained model
                        if best_model_type == 'xgboost':
                            prediction = model_manager.predict_xgboost(station_id, input_data)
                        elif best_model_type == 'lstm':
                            prediction = model_manager.predict_lstm(station_id, input_data)
                        else:
                            # Fallback to XGBoost for unknown model types
                            logger.warning(f"Unknown model type '{best_model_type}' for station {station_id}, falling back to XGBoost")
                            prediction = model_manager.predict_xgboost(station_id, input_data)
                        
                        if prediction and prediction.get('wqi_prediction'):
                            # Tính thời gian dự đoán (thêm số tháng)
                            from dateutil.relativedelta import relativedelta
                            prediction_time = latest_time + relativedelta(months=horizon_months)
                            
                            future_predictions[f'{horizon_months}month'] = {
                                'wqi_prediction': prediction.get('wqi_prediction'),
                                'confidence_score': prediction.get('confidence_score'),
                                'prediction_time': prediction_time
                            }
                            logger.info(f"✅ Station {station_id}: {horizon_months} month(s) ahead WQI = {prediction.get('wqi_prediction'):.2f}, Confidence = {prediction.get('confidence_score'):.2f}")
                        else:
                            logger.warning(f"⚠️ Station {station_id}: Failed to predict {horizon_months} month(s) ahead")
                            
                    except Exception as e:
                        logger.error(f"Station {station_id}: Error predicting {horizon_months} month(s) ahead - {e}")
                
                if future_predictions:
                    prediction_results[station_id] = {
                        'success': True,
                        'model_type': best_model_type,
                        'model_path': model_path,
                        'current_wqi': latest_wqi,
                        'current_time': latest_time,
                        'future_predictions': future_predictions,
                        'prediction_horizons': prediction_horizons
                    }
                    logger.info(f"✅ Station {station_id}: Successfully predicted WQI for {len(future_predictions)} time horizons")
                else:
                    prediction_results[station_id] = {
                        'success': False,
                        'error': 'All predictions failed'
                    }
                    logger.warning(f"⚠️ Station {station_id}: All predictions failed")
            else:
                prediction_results[station_id] = {
                    'success': False,
                    'error': 'No recent data available'
                }
                logger.warning(f"Station {station_id}: No recent data available")
                
        except Exception as e:
            prediction_results[station_id] = {
                'success': False,
                'error': str(e)
            }
            logger.error(f"Station {station_id}: Error during prediction - {e}")
    
    context['task_instance'].xcom_push(key='prediction_results', value=prediction_results)
    
    successful_predictions = [result for result in prediction_results.values() if result.get('success', False)]
    logger.info(f"Successfully predicted future WQI for {len(successful_predictions)} out of {len(stations_with_models)} stations using pre-trained models")
    
    return f"Future WQI predictions completed for {len(successful_predictions)} out of {len(stations_with_models)} stations using pre-trained models"

def compare_model_performance(**context):
    """So sánh performance của các pre-trained models cho future predictions"""
    from include.iot_streaming.pipeline_processor import PipelineProcessor
    
    # Lấy kết quả dự đoán từ pre-trained models
    existing_predictions = context['task_instance'].xcom_pull(
        task_ids='predict_existing_stations', key='prediction_results'
    ) or {}
    
    all_predictions = existing_predictions
    
    if not all_predictions:
        logger.info("No predictions to compare")
        return "No predictions to compare"
    
    logger.info(f"Comparing performance for {len(all_predictions)} future predictions from pre-trained models")
    
    # So sánh performance của các model cho future predictions
    comparison_results = PipelineProcessor.compare_model_performance(all_predictions)
    context['task_instance'].xcom_push(key='comparison_results', value=comparison_results)
    
    return f"Compared pre-trained models for {len(comparison_results)} stations with future predictions"

def update_monitoring_metrics(**context):
    """Cập nhật metrics cho monitoring"""
    from include.iot_streaming.pipeline_processor import PipelineProcessor
    
    logger.info("Updating monitoring metrics")
    
    # Cập nhật metrics
    metrics_updated = PipelineProcessor.update_monitoring_metrics()
    
    return f"Updated {metrics_updated} monitoring metrics"

def summarize_pipeline_execution(**context):
    """Tóm tắt kết quả thực thi pipeline với pre-trained models cho future predictions"""
    total_unprocessed_records = context['task_instance'].xcom_pull(
        task_ids='process_streaming_data', key='total_unprocessed_records'
    ) or 0
    
    stations_with_models = context['task_instance'].xcom_pull(
        task_ids='process_streaming_data', key='stations_with_models'
    ) or []
    
    stations_need_training = context['task_instance'].xcom_pull(
        task_ids='process_streaming_data', key='stations_need_training'
    ) or []
    
    existing_predictions = context['task_instance'].xcom_pull(
        task_ids='predict_existing_stations', key='prediction_results'
    ) or {}
    
    comparison_results = context['task_instance'].xcom_pull(
        task_ids='compare_model_performance', key='comparison_results'
    ) or {}
    
    # Đếm số predictions thành công từ pre-trained models
    existing_predictions_success = 0
    total_future_predictions = 0
    
    for station_id, result in existing_predictions.items():
        if result.get('success', False):
            existing_predictions_success += 1
            # Đếm số future predictions
            future_predictions = result.get('future_predictions', {})
            total_future_predictions += len(future_predictions)
    
    summary = {
        'total_unprocessed_records': total_unprocessed_records,
        'stations_with_pre_trained_models': len(stations_with_models),
        'stations_need_training': len(stations_need_training),
        'successful_training': 0,  # No training in this pipeline
        'existing_predictions': existing_predictions_success,
        'total_future_predictions': total_future_predictions,
        'new_predictions': 0,  # No newly trained predictions
        'model_comparisons': len(comparison_results),
        'execution_time': datetime.now().isoformat(),
        'pipeline_type': 'pre_trained_models_future_predictions'
    }
    
    context['task_instance'].xcom_push(key='pipeline_summary', value=summary)
    
    logger.info(f"Pipeline summary: {summary}")
    
    return f"Pipeline completed: {summary['total_unprocessed_records']} new records, {summary['existing_predictions']} stations with future predictions ({summary['total_future_predictions']} total predictions)"

def generate_alerts_and_notifications(**context):
    """Tạo alerts và notifications cho kết quả dự đoán WQI tương lai từ pre-trained models"""
    from include.iot_streaming.pipeline_processor import PipelineProcessor
    
    # Lấy kết quả dự đoán từ pre-trained models
    existing_predictions = context['task_instance'].xcom_pull(
        task_ids='predict_existing_stations', key='prediction_results'
    ) or {}
    
    all_predictions = existing_predictions
    
    if not all_predictions:
        logger.info("No predictions to generate alerts for")
        return "No predictions to generate alerts for"
    
    logger.info(f"Generating alerts for {len(all_predictions)} future predictions from pre-trained models")
    
    # Tạo alerts dựa trên kết quả dự đoán tương lai
    alerts_generated = PipelineProcessor.generate_alerts()
    
    return f"Generated {alerts_generated} alerts for {len(all_predictions)} future predictions from pre-trained models"

def load_historical_data_to_db(**context):
    """Load dữ liệu lịch sử từ CSV vào database"""
    from include.iot_streaming.database_manager import db_manager
    
    logger.info("Loading historical data to database")
    
    # Load dữ liệu lịch sử từ CSV
    historical_df = load_historical_data()
    if historical_df is None:
        logger.error("Failed to load historical data")
        return "Failed to load historical data"
    
    # Kết nối database
    conn = db_manager.get_connection()
    if not conn:
        logger.error("Cannot connect to database")
        return "No database connection"
    
    try:
        cur = conn.cursor()
        
        # Insert stations vào monitoring_stations nếu chưa có
        for station_id in historical_df['station_id'].unique():
            station_id = int(station_id)
            cur.execute("""
                INSERT INTO monitoring_stations (station_id, station_name, location, is_active)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT (station_id) DO NOTHING
            """, (station_id, f"Station {station_id}", f"Location {station_id}", True))
        
        # Insert historical data vào processed_water_quality_data
        inserted_count = 0
        for _, row in historical_df.iterrows():
            try:
                # Convert numpy types to Python native types
                station_id = int(row['station_id']) if pd.notna(row['station_id']) else None
                measurement_time = row['Date'] if pd.notna(row['Date']) else None
                ph = float(row['ph']) if pd.notna(row['ph']) else None
                temperature = float(row['temperature']) if pd.notna(row['temperature']) else None
                do = float(row['do']) if pd.notna(row['do']) else None
                wqi = float(row['wqi']) if pd.notna(row['wqi']) else None
                
                try:
                    cur.execute("""
                        INSERT INTO processed_water_quality_data 
                        (station_id, measurement_time, ph, temperature, "do", wqi)
                        VALUES (%s, %s, %s, %s, %s, %s)
                        ON CONFLICT (station_id, measurement_time) DO NOTHING
                    """, (
                        station_id,
                        measurement_time,
                        ph,
                        temperature,
                        do,
                        wqi
                    ))
                except Exception as conflict_error:
                    # Nếu không có unique constraint, thử INSERT thường
                    if "no unique or exclusion constraint" in str(conflict_error):
                        logger.warning("No unique constraint found, using regular INSERT")
                        cur.execute("""
                            INSERT INTO processed_water_quality_data 
                            (station_id, measurement_time, ph, temperature, "do", wqi)
                            VALUES (%s, %s, %s, %s, %s, %s)
                        """, (
                            station_id,
                            measurement_time,
                            ph,
                            temperature,
                            do,
                            wqi
                        ))
                    else:
                        raise conflict_error
                inserted_count += 1
            except Exception as e:
                logger.warning(f"Error inserting row for station {row['station_id']}: {e}")
        
        conn.commit()
        cur.close()
        
        logger.info(f"Successfully loaded {inserted_count} historical records to database")
        context['task_instance'].xcom_push(key='historical_records_loaded', value=inserted_count)
        
        return f"Loaded {inserted_count} historical records to database"
        
    except Exception as e:
        logger.error(f"Error loading historical data to database: {e}")
        conn.rollback()
        return f"Error: {str(e)}"
    finally:
        conn.close()

# Tạo DAG
dag = DAG(
    'streaming_data_processor',
    default_args=default_args,
    description='Main pipeline for processing streaming water quality data from Kafka with intelligent train/predict logic',
    schedule_interval=None,  # Chạy thủ công hoặc được trigger bởi DAG khác
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=['water-quality', 'streaming', 'kafka', 'ml-pipeline', 'main-pipeline']
)

# Định nghĩa các tasks
process_streaming_task = PythonOperator(
    task_id='process_streaming_data',
    python_callable=process_streaming_data,
    dag=dag
)

predict_existing_task = PythonOperator(
    task_id='predict_existing_stations',
    python_callable=predict_existing_stations,
    dag=dag
)

compare_models_task = PythonOperator(
    task_id='compare_model_performance',
    python_callable=compare_model_performance,
    dag=dag
)

update_metrics_task = PythonOperator(
    task_id='update_monitoring_metrics',
    python_callable=update_monitoring_metrics,
    dag=dag
)

alerts_task = PythonOperator(
    task_id='generate_alerts_and_notifications',
    python_callable=generate_alerts_and_notifications,
    dag=dag
)

summarize_task = PythonOperator(
    task_id='summarize_pipeline_execution',
    python_callable=summarize_pipeline_execution,
    dag=dag
)

load_historical_data_to_db_task = PythonOperator(
    task_id='load_historical_data_to_db',
    python_callable=load_historical_data_to_db,
    dag=dag
)

# Định nghĩa dependencies - chỉ sử dụng models đã train sẵn
process_streaming_task >> predict_existing_task
predict_existing_task >> compare_models_task
compare_models_task >> update_metrics_task >> alerts_task >> summarize_task 