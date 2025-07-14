"""
Pipeline Processor Module
Chứa các hàm xử lý cho station-based water quality monitoring pipeline
"""

import logging
import pandas as pd
from datetime import datetime
from .database_manager import db_manager
from .station_processor_v2 import station_processor
from .model_manager import model_manager
from .prometheus_exporter import get_prometheus_exporter

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
                        
                        # Cập nhật metrics
                        # get_station_data returns: measurement_time, ph, temperature, "do", wqi
                        get_prometheus_exporter().update_wqi_metric(station_id, latest_record[4], station_name)
                        get_prometheus_exporter().update_ph_metric(station_id, latest_record[1], station_name)
                        get_prometheus_exporter().update_temperature_metric(station_id, latest_record[2], station_name)
                        get_prometheus_exporter().update_do_metric(station_id, latest_record[3], station_name)
                    
                    # Cập nhật station activity
                    get_prometheus_exporter().update_station_activity(station_id, station_name)
                    
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