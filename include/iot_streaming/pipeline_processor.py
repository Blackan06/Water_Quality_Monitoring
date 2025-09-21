"""
Pipeline Processor Module
Chứa các hàm xử lý cho station-based water quality monitoring pipeline
"""

import logging
import pandas as pd
from datetime import datetime
from .database_manager import db_manager
from .station_processor  import station_processor
from .model_manager import model_manager

# Cấu hình logging
logger = logging.getLogger(__name__)

class PipelineProcessor:
    """Class xử lý pipeline cho water quality monitoring"""
    
    @staticmethod
    def initialize_database():
        """Khởi tạo database và load dữ liệu lịch sử"""
        try:
            logger.info("Initializing database...")
            
            # Khởi tạo database
            db_manager.init_database()
            
            # Load dữ liệu lịch sử từ WQI_data.csv
            success = db_manager.load_wqi_data_to_db()
            
            if success:
                logger.info("Database initialized successfully")
            else:
                logger.warning("Database initialization completed with warnings")
                
            return "Database initialized"
            
        except Exception as e:
            logger.error(f"Error initializing database: {e}")
            raise

    @staticmethod
    def collect_sensor_data():
        """Thu thập dữ liệu từ sensors (đọc từ database)"""
        try:
            logger.info("Collecting sensor data from database...")
            
            # Lấy tất cả stations
            stations = db_manager.get_all_stations()
            
            if not stations:
                logger.warning("No stations found in database")
                return []
            
            sensor_data = []
            
            for station in stations:
                if not station['is_active']:
                    continue
                    
                station_id = station['station_id']
                
                try:
                    # Lấy dữ liệu mới nhất cho mỗi trạm
                    recent_data = db_manager.get_station_data(station_id, limit=1)
                    
                    if recent_data:
                        latest_record = recent_data[0]
                        
                        # Chuyển đổi thành format cần thiết
                        # get_station_data returns: measurement_time, ph, temperature, "do", wqi
                        data = {
                            'station_id': station_id,
                            'station_name': station['station_name'],
                            'measurement_time': latest_record[0],  # measurement_time
                            'ph': latest_record[1],
                            'temperature': latest_record[2],
                            'do': latest_record[3],
                            'wqi': latest_record[4]
                        }
                        sensor_data.append(data)
                        logger.info(f"Collected data for station {station_id}")
                    else:
                        logger.warning(f"No recent data found for station {station_id}")
                        
                except Exception as e:
                    logger.error(f"Error collecting data for station {station_id}: {e}")
            
            logger.info(f"Collected data for {len(sensor_data)} stations")
            return sensor_data
            
        except Exception as e:
            logger.error(f"Error collecting sensor data: {e}")
            raise

    @staticmethod
    def process_station_data(sensor_data):
        """Xử lý dữ liệu theo từng trạm"""
        try:
            logger.info("Processing station data...")
            
            if not sensor_data:
                logger.warning("No sensor data found")
                return []
            
            processed_stations = []
            
            for data in sensor_data:
                try:
                    # Xử lý dữ liệu cho từng trạm
                    success = station_processor.process_station_data(data)
                    
                    if success:
                        processed_stations.append(data['station_id'])
                        logger.info(f"Successfully processed station {data['station_id']}")
                    else:
                        logger.error(f"Failed to process station {data['station_id']}")
                        
                except Exception as e:
                    logger.error(f"Error processing station {data['station_id']}: {e}")
            
            logger.info(f"Processed data for {len(processed_stations)} stations")
            return processed_stations
            
        except Exception as e:
            logger.error(f"Error processing station data: {e}")
            raise

    @staticmethod
    def train_models_for_stations(processed_stations):
        """Train models cho các trạm cần thiết"""
        try:
            logger.info("Training models for stations...")
            
            if not processed_stations:
                logger.warning("No stations to train models for")
                return {}
            
            training_results = {}
            
            for station_id in processed_stations:
                try:
                    logger.info(f"Training models for station {station_id}")
                    
                    # Lấy dữ liệu lịch sử cho training
                    historical_data = db_manager.get_historical_data(station_id, limit=1000)
                    
                    if historical_data.empty:
                        logger.warning(f"No historical data for station {station_id}")
                        continue
                    
                    if len(historical_data) < 100:
                        logger.warning(f"Insufficient data for station {station_id}: {len(historical_data)} records")
                        continue
                    
                    logger.info(f"Training with {len(historical_data)} records for station {station_id}")
                    
                    # Train XGBoost
                    xgb_result = model_manager.train_xgboost_model(station_id, historical_data)
                    
                    # Train LSTM
                    lstm_result = model_manager.train_lstm_model(station_id, historical_data)
                    
                    # So sánh và chọn model tốt nhất
                    comparison_result = model_manager.compare_and_select_best_model(station_id, xgb_result, lstm_result)
                    
                    # Lưu kết quả training vào database
                    if 'error' not in xgb_result:
                        db_manager.insert_model_registry(xgb_result)
                        db_manager.insert_training_history({
                            'station_id': station_id,
                            'model_type': 'xgboost',
                            'training_start': xgb_result.get('training_date'),
                            'training_end': datetime.now(),
                            'training_duration': 0,  # Có thể tính thực tế
                            'records_used': xgb_result.get('records_used', 0),
                            'accuracy': xgb_result.get('accuracy', 0),
                            'mae': xgb_result.get('mae', 0),
                            'r2_score': xgb_result.get('r2_score', 0),
                            'status': 'success'
                        })
                    
                    if 'error' not in lstm_result:
                        db_manager.insert_model_registry(lstm_result)
                        db_manager.insert_training_history({
                            'station_id': station_id,
                            'model_type': 'lstm',
                            'training_start': lstm_result.get('training_date'),
                            'training_end': datetime.now(),
                            'training_duration': 0,
                            'records_used': lstm_result.get('records_used', 0),
                            'accuracy': lstm_result.get('accuracy', 0),
                            'mae': lstm_result.get('mae', 0),
                            'r2_score': lstm_result.get('r2_score', 0),
                            'status': 'success'
                        })
                    
                    # Lưu kết quả so sánh
                    if 'error' not in comparison_result:
                        db_manager.insert_model_comparison({
                            'station_id': station_id,
                            'comparison_date': datetime.now(),
                            'best_model': comparison_result['best_model'],
                            'reason': comparison_result['reason'],
                            'xgboost_score': comparison_result.get('xgboost_score', 0),
                            'lstm_score': comparison_result.get('lstm_score', 0),
                            'xgboost_mae': comparison_result.get('comparison_metrics', {}).get('xgboost', {}).get('mae', 0),
                            'lstm_mae': comparison_result.get('comparison_metrics', {}).get('lstm', {}).get('mae', 0),
                            'xgboost_r2': comparison_result.get('comparison_metrics', {}).get('xgboost', {}).get('r2_score', 0),
                            'lstm_r2': comparison_result.get('comparison_metrics', {}).get('lstm', {}).get('r2_score', 0)
                        })
                    
                    training_results[station_id] = {
                        'xgboost': xgb_result,
                        'lstm': lstm_result,
                        'comparison': comparison_result
                    }
                    
                    logger.info(f"Training completed for station {station_id}")
                    if 'error' not in comparison_result:
                        logger.info(f"Best model for station {station_id}: {comparison_result['best_model']} ({comparison_result['reason']})")
                    
                except Exception as e:
                    logger.error(f"Error training models for station {station_id}: {e}")
                    training_results[station_id] = {'error': str(e)}
            
            logger.info(f"Training completed for {len(training_results)} stations")
            return training_results
            
        except Exception as e:
            logger.error(f"Error training models: {e}")
            raise

    @staticmethod
    def perform_predictions(processed_stations):
        """Thực hiện dự đoán cho các trạm"""
        try:
            logger.info("Performing predictions...")
            
            if not processed_stations:
                logger.warning("No stations to predict for")
                return {}
            
            prediction_results = {}
            
            for station_id in processed_stations:
                try:
                    logger.info(f"Performing predictions for station {station_id}")
                    
                    # Lấy dữ liệu mới nhất
                    recent_data = db_manager.get_station_data(station_id, limit=1)
                    
                    if not recent_data:
                        logger.warning(f"No recent data for station {station_id}")
                        continue
                    
                    # Tạo dữ liệu cho prediction
                    latest_record = recent_data[0]
                    # get_station_data returns: measurement_time, ph, temperature, "do", wqi
                    prediction_data = {
                        'ph': latest_record[1],
                        'temperature': latest_record[2],
                        'do': latest_record[3],
                        'wqi': latest_record[4]
                    }
                    
                    # Thực hiện dự đoán
                    predictions = station_processor.perform_predictions(station_id, prediction_data)
                    
                    prediction_results[station_id] = predictions
                    
                    logger.info(f"Predictions completed for station {station_id}")
                    
                except Exception as e:
                    logger.error(f"Error performing predictions for station {station_id}: {e}")
                    prediction_results[station_id] = {'error': str(e)}
            
            logger.info(f"Predictions completed for {len(prediction_results)} stations")
            return prediction_results
            
        except Exception as e:
            logger.error(f"Error performing predictions: {e}")
            raise

    @staticmethod
    def compare_model_performance(prediction_results):
        """So sánh performance của các model"""
        try:
            logger.info("Comparing model performance...")
            
            if not prediction_results:
                logger.warning("No prediction results to compare")
                return {}
            
            comparison_results = {}
            
            for station_id, predictions in prediction_results.items():
                try:
                    if 'error' in predictions:
                        continue
                    
                    # So sánh models
                    station_processor.compare_models(station_id, predictions)
                    
                    comparison_results[station_id] = {
                        'xgboost_prediction': predictions.get('xgboost', {}).get('wqi_prediction', 0),
                        'lstm_prediction': predictions.get('lstm', {}).get('wqi_prediction', 0),
                        'xgboost_confidence': predictions.get('xgboost', {}).get('confidence_score', 0),
                        'lstm_confidence': predictions.get('lstm', {}).get('confidence_score', 0)
                    }
                    
                    logger.info(f"Model comparison completed for station {station_id}")
                    
                except Exception as e:
                    logger.error(f"Error comparing models for station {station_id}: {e}")
            
            logger.info(f"Model comparison completed for {len(comparison_results)} stations")
            return comparison_results
            
        except Exception as e:
            logger.error(f"Error comparing model performance: {e}")
            raise

    @staticmethod
    def update_monitoring_metrics():
        """Cập nhật metrics cho monitoring"""
        try:
            logger.info("Updating monitoring metrics...")
            
            # Lấy tất cả stations
            stations = db_manager.get_all_stations()
            
            for station in stations:
                if not station['is_active']:
                    continue
                
                station_id = station['station_id']
                station_name = station['station_name']
                
                try:
                    # Lấy dữ liệu mới nhất
                    recent_data = db_manager.get_station_data(station_id, limit=1)
                    
                    if recent_data:
                        latest_record = recent_data[0]
                        
                        # Metrics functionality removed (Prometheus)
                        pass
                    
                except Exception as e:
                    logger.error(f"Error updating metrics for station {station_id}: {e}")
            
            logger.info("Monitoring metrics updated")
            return "Monitoring metrics updated"
            
        except Exception as e:
            logger.error(f"Error updating monitoring metrics: {e}")
            raise

    @staticmethod
    def generate_alerts():
        """Tạo alerts cho các trường hợp cần thiết"""
        try:
            logger.info("Generating alerts...")
            
            # Lấy tất cả stations
            stations = db_manager.get_all_stations()
            
            alert_count = 0
            
            for station in stations:
                if not station['is_active']:
                    continue
                
                station_id = station['station_id']
                
                try:
                    # Lấy dữ liệu mới nhất
                    recent_data = db_manager.get_station_data(station_id, limit=1)
                    
                    if recent_data:
                        latest_record = recent_data[0]
                        
                        # Kiểm tra và tạo alerts
                        # get_station_data returns: measurement_time, ph, temperature, "do", wqi
                        station_processor.check_alerts(station_id, {
                            'wqi': latest_record[4],
                            'ph': latest_record[1],
                            'temperature': latest_record[2],
                            'do': latest_record[3]
                        })
                        
                        alert_count += 1
                    
                except Exception as e:
                    logger.error(f"Error generating alerts for station {station_id}: {e}")
            
            logger.info(f"Generated alerts for {alert_count} stations")
            return f"Generated alerts for {alert_count} stations"
            
        except Exception as e:
            logger.error(f"Error generating alerts: {e}")
            raise

    @staticmethod
    def summarize_pipeline_execution(processed_stations, training_results, prediction_results, comparison_results):
        """Tóm tắt kết quả thực thi pipeline"""
        try:
            logger.info("Summarizing pipeline execution...")
            
            # Tạo summary
            summary = {
                'execution_time': datetime.now().isoformat(),
                'processed_stations': processed_stations or [],
                'training_results': training_results or {},
                'prediction_results': prediction_results or {},
                'comparison_results': comparison_results or {},
                'total_stations_processed': len(processed_stations) if processed_stations else 0,
                'successful_predictions': len([r for r in (prediction_results or {}).values() if 'error' not in r]),
                'successful_training': len([r for r in (training_results or {}).values() if 'error' not in r])
            }
            
            logger.info(f"Pipeline execution summary: {summary}")
            return summary
            
        except Exception as e:
            logger.error(f"Error summarizing pipeline execution: {e}")
            raise

    @staticmethod
    def get_unprocessed_raw_data():
        """Lấy dữ liệu raw sensors chưa được xử lý"""
        try:
            logger.info("Getting unprocessed raw sensor data...")
            
            conn = db_manager.get_connection()
            if not conn:
                logger.error("Cannot connect to database")
                return []
            
            cur = conn.cursor()
            cur.execute("""
                SELECT DISTINCT rs.station_id, rs.measurement_time, rs.ph, rs.temperature, rs."do"
                FROM raw_sensor_data rs
                WHERE rs.is_processed = FALSE
                ORDER BY rs.station_id, rs.measurement_time DESC
                LIMIT 100
            """)
            
            raw_data = cur.fetchall()
            cur.close()
            conn.close()
            
            logger.info(f"Found {len(raw_data)} unprocessed raw sensor records")
            return raw_data
            
        except Exception as e:
            logger.error(f"Error getting unprocessed raw data: {e}")
            return []

    @staticmethod
    def process_raw_data(raw_data):
        """Process raw data thành processed data với WQI"""
        try:
            logger.info("Processing raw sensor data...")
            
            processed_count = 0
            conn = db_manager.get_connection()
            if not conn:
                logger.error("Cannot connect to database for processing")
                return 0
            
            for row in raw_data:
                station_id, measurement_time, ph, temperature, do_val = row
                
                try:
                    cur = conn.cursor()
                    
                    # Tính WQI từ dữ liệu thô
                    wqi = PipelineProcessor._calculate_wqi(ph, temperature, do_val)
                    
                    if wqi is not None:
                        # Lưu vào processed_water_quality_data
                        cur.execute("""
                            INSERT INTO processed_water_quality_data 
                            (station_id, measurement_time, ph, temperature, "do", wqi)
                            VALUES (%s, %s, %s, %s, %s, %s)
                            ON CONFLICT (station_id, measurement_time) DO UPDATE SET
                                ph = EXCLUDED.ph,
                                temperature = EXCLUDED.temperature,
                                "do" = EXCLUDED."do",
                                wqi = EXCLUDED.wqi
                        """, (station_id, measurement_time, ph, temperature, do_val, wqi))
                        
                        # Đánh dấu raw data đã được xử lý
                        cur.execute("""
                            UPDATE raw_sensor_data 
                            SET is_processed = TRUE
                            WHERE station_id = %s AND measurement_time = %s
                        """, (station_id, measurement_time))
                        
                        conn.commit()
                        processed_count += 1
                        logger.debug(f"Processed raw data: station {station_id}, WQI = {wqi}")
                        
                    else:
                        logger.warning(f"Failed to calculate WQI for station {station_id}")
                    
                except Exception as e:
                    logger.error(f"Error processing raw data for station {station_id}: {e}")
                    try:
                        conn.rollback()
                    except Exception as rollback_error:
                        logger.error(f"Error during rollback: {rollback_error}")
                finally:
                    try:
                        cur.close()
                    except Exception as close_error:
                        logger.error(f"Error closing cursor: {close_error}")
            
            conn.close()
            logger.info(f"Processed {processed_count}/{len(raw_data)} raw sensor records")
            return processed_count
            
        except Exception as e:
            logger.error(f"Error processing raw data: {e}")
            return 0

    @staticmethod
    def get_stations_with_models():
        """Lấy danh sách stations có models sẵn"""
        try:
            import os
            
            # Kiểm tra xem có best models không
            ensemble_metadata_path = "models/ensemble_metadata.json"
            rf_model_path = "models/spark_rf_model.pkl"
            xgb_model_path = "models/spark_xgb_model.pkl"
            
            available_stations = []
            
            if (os.path.exists(ensemble_metadata_path) or 
                os.path.exists(rf_model_path) or 
                os.path.exists(xgb_model_path)):
                
                # Lấy tất cả stations từ database
                conn = db_manager.get_connection()
                if conn:
                    cur = conn.cursor()
                    cur.execute("SELECT station_id FROM monitoring_stations WHERE is_active = TRUE")
                    stations = cur.fetchall()
                    cur.close()
                    conn.close()
                    
                    available_stations = [station[0] for station in stations]
                    logger.info(f"Found {len(available_stations)} stations with available models")
                else:
                    logger.warning("Cannot connect to database to get stations")
            else:
                logger.warning("No best models found")
            
            return available_stations
            
        except Exception as e:
            logger.error(f"Error getting stations with models: {e}")
            return []

    @staticmethod
    def get_stations_need_training():
        """Lấy danh sách stations cần training (luôn rỗng vì không train mới)"""
        return []

    @staticmethod
    def _calculate_wqi(ph, temperature, do):
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

    @staticmethod
    def check_database_status():
        """Kiểm tra trạng thái database và dữ liệu"""
        try:
            logger.info("Checking database status...")
            
            conn = db_manager.get_connection()
            if not conn:
                logger.error("❌ Cannot connect to database")
                return "Error: Cannot connect to database"
            
            cur = conn.cursor()
            
            # Kiểm tra dữ liệu thô mới nhất từ sensors
            cur.execute("""
                SELECT station_id, measurement_time, ph, temperature, "do", wqi
                FROM raw_sensor_data
                ORDER BY measurement_time DESC
                LIMIT 50
            """)
            
            recent_raw_data = cur.fetchall()
            
            # Kiểm tra dữ liệu đã xử lý gần đây
            cur.execute("""
                SELECT station_id, measurement_time, ph, temperature, "do", wqi
                FROM processed_water_quality_data
                ORDER BY measurement_time DESC
                LIMIT 20
            """)
            
            recent_processed_data = cur.fetchall()
            
            # Kiểm tra thông tin stations
            cur.execute("SELECT station_id, station_name FROM monitoring_stations WHERE is_active = TRUE")
            stations = {row[0]: row[1] for row in cur.fetchall()}
            
            cur.close()
            conn.close()
            
            if recent_raw_data:
                logger.info(f"✅ Database connection successful - Found {len(recent_raw_data)} recent raw sensor data points")
                logger.info(f"✅ Active stations: {list(stations.keys())}")
                
                # Log thông tin dữ liệu thô mẫu
                sample_raw_data = recent_raw_data[0] if recent_raw_data else None
                if sample_raw_data:
                    station_id, measurement_time, ph, temp, do_val, wqi = sample_raw_data
                    logger.info(f"📊 Latest raw sensor data - Station {station_id}: WQI={wqi}, pH={ph}, Temp={temp}, DO={do_val}")
            else:
                logger.warning("⚠️ No recent raw sensor data found in database")
            
            # Log thông tin về processed data
            if recent_processed_data:
                logger.info(f"📊 Found {len(recent_processed_data)} recent processed data points")
                sample_processed_data = recent_processed_data[0] if recent_processed_data else None
                if sample_processed_data:
                    station_id, measurement_time, ph, temp, do_val, wqi = sample_processed_data
                    logger.info(f"📊 Latest processed data - Station {station_id}: WQI={wqi}, pH={ph}, Temp={temp}, DO={do_val}")
            else:
                logger.info("📊 No recent processed data found")
            
            return f"Database connection initialized - {len(recent_raw_data)} raw sensor data points, {len(recent_processed_data)} processed data points"
            
        except Exception as e:
            logger.error(f"❌ Error checking database status: {e}")
            return f"Error: {e}"

    @staticmethod
    def get_station_historical_data(station_id):
        """Lấy dữ liệu lịch sử cho một station"""
        try:
            conn = db_manager.get_connection()
            if not conn:
                logger.error("Cannot connect to database")
                return []
            
            cur = conn.cursor()
            
            # Lấy dữ liệu lịch sử (48 tháng = 4 năm)
            cur.execute("""
                SELECT ph, temperature, "do", wqi, measurement_time
                FROM processed_water_quality_data 
                WHERE station_id = %s 
                ORDER BY measurement_time DESC 
                LIMIT 48
            """, (station_id,))
            
            historical_data = cur.fetchall()
            cur.close()
            conn.close()
            
            logger.info(f"Retrieved {len(historical_data)} historical records for station {station_id}")
            return historical_data
            
        except Exception as e:
            logger.error(f"Error getting historical data for station {station_id}: {e}")
            return []

    @staticmethod
    def create_summary(data):
        """Create pipeline summary from orchestration data"""
        try:
            unprocessed_count = data.get('unprocessed_count', 0)
            prediction_results = data.get('prediction_results', [])
            alerts_result = data.get('alerts_result', '')
            notifications_sent = data.get('notifications_sent', 0)
            
            # Handle None values
            if prediction_results is None:
                prediction_results = []
            if alerts_result is None:
                alerts_result = ''
            if notifications_sent is None:
                notifications_sent = 0
            
            # Calculate summary statistics
            successful_predictions = len([p for p in prediction_results if p and p.get('success')])
            total_predictions = len(prediction_results)
            
            summary = {
                'total_raw_sensor_records': unprocessed_count,
                'processed_records': 0,  # Not actually processed in this version
                'successful_predictions': successful_predictions,
                'total_predictions': total_predictions,
                'prediction_success_rate': successful_predictions / total_predictions if total_predictions > 0 else 0,
                'alerts_result': alerts_result,
                'notifications_sent': notifications_sent,
                'pipeline_type': 'streaming_data_processor',
                'execution_time': datetime.now().isoformat()
            }
            
            logger.info(f"Pipeline summary created: {summary}")
            return summary
            
        except Exception as e:
            logger.error(f"Error creating pipeline summary: {e}")
            return {
                'error': str(e),
                'pipeline_type': 'streaming_data_processor',
                'execution_time': datetime.now().isoformat()
            } 