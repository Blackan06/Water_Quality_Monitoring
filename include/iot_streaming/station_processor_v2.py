import logging
import json
import time
from datetime import datetime, timedelta
from .database_manager import db_manager
from .model_manager import ModelManager
from .prometheus_exporter import get_prometheus_exporter
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

class StationProcessorV2:
    def __init__(self):
        self.db_manager = db_manager
        self.model_manager = ModelManager()
        self.prometheus_exporter = get_prometheus_exporter()
        
        # Cấu hình ngưỡng
        self.wqi_thresholds = {
            'excellent': 90,
            'good': 70,
            'moderate': 50,
            'poor': 25,
            'very_poor': 0
        }
        
        # Cấu hình retrain
        self.retrain_config = {
            'min_records': 100,
            'accuracy_threshold': 0.7,
            'drift_threshold': 0.1,
            'retrain_interval_hours': 24
        }
        
        # Thông tin các trạm mặc định
        self.default_stations = [
            {
                'station_id': 0,
                'station_name': 'Trạm quan trắc sông Hồng - Hà Nội',
                'location': 'Cầu Long Biên, Hà Nội',
                'latitude': 21.0285,
                'longitude': 105.8542,
                'description': 'Trạm quan trắc chất lượng nước sông Hồng tại Hà Nội'
            },
            {
                'station_id': 1,
                'station_name': 'Trạm quan trắc sông Sài Gòn - TP.HCM',
                'location': 'Cầu Bình Lợi, TP.HCM',
                'latitude': 10.8231,
                'longitude': 106.6297,
                'description': 'Trạm quan trắc chất lượng nước sông Sài Gòn tại TP.HCM'
            },
            {
                'station_id': 2,
                'station_name': 'Trạm quan trắc sông Hương - Huế',
                'location': 'Cầu Tràng Tiền, Huế',
                'latitude': 16.4637,
                'longitude': 107.5909,
                'description': 'Trạm quan trắc chất lượng nước sông Hương tại Huế'
            },
            {
                'station_id': 3,
                'station_name': 'Trạm quan trắc sông Hàn - Đà Nẵng',
                'location': 'Cầu Sông Hàn, Đà Nẵng',
                'latitude': 16.0544,
                'longitude': 108.2022,
                'description': 'Trạm quan trắc chất lượng nước sông Hàn tại Đà Nẵng'
            },
            {
                'station_id': 4,
                'station_name': 'Trạm quan trắc sông Cầu - Bắc Ninh',
                'location': 'Cầu Phả Lại, Bắc Ninh',
                'latitude': 21.1861,
                'longitude': 106.0763,
                'description': 'Trạm quan trắc chất lượng nước sông Cầu tại Bắc Ninh'
            }
        ]
        
        # Khởi tạo các trạm mặc định
        self.initialize_default_stations()

    def initialize_default_stations(self):
        """Khởi tạo các trạm mặc định trong database"""
        try:
            for station in self.default_stations:
                if not self.db_manager.check_station_exists(station['station_id']):
                    success = self.db_manager.insert_station(station)
                    if success:
                        logger.info(f"Initialized default station {station['station_id']}: {station['station_name']}")
                    else:
                        logger.error(f"Failed to initialize station {station['station_id']}")
               
        except Exception as e:
            logger.error(f"Error initializing default stations: {e}")

    def process_station_data(self, message):
        """Xử lý dữ liệu từ một trạm"""
        try:
            station_id = message['station_id']

            # Kiểm tra trạm có tồn tại không
            if not self.db_manager.check_station_exists(station_id):
                logger.info(f"New station detected: {station_id}")
                # Tạo trạm mới nếu chưa có
                station_info = {
                    'station_id': station_id,
                    'station_name': message.get('station_name', f'Trạm {station_id}'),
                    'location': message.get('location', 'Unknown'),
                    'latitude': message.get('latitude', 0),
                    'longitude': message.get('longitude', 0),
                    'description': message.get('description', 'Auto-created station')
                }
                self.db_manager.insert_station(station_info)
            
            # Lưu dữ liệu thô
            raw_data = {
                'station_id': station_id,
                'measurement_time': datetime.fromisoformat(message['measurement_time']),
                'ph': message.get('ph', 0),
                'temperature': message.get('temperature', 0),
                'do': message.get('do', 0)
            }
            self.db_manager.insert_raw_data(raw_data)
            
            # Lưu dữ liệu đã xử lý (có WQI)
            processed_data = {
                'station_id': station_id,
                'measurement_time': datetime.fromisoformat(message['measurement_time']),
                'ph': message.get('ph', 0),
                'temperature': message.get('temperature', 0),
                'do': message.get('do', 0),
                'wqi': message.get('wqi', 0)
            }
            self.db_manager.insert_processed_data(processed_data)
            
            # Thực hiện dự đoán
            predictions = self.perform_predictions(station_id, processed_data)
            
            # So sánh model performance
            if predictions:
                self.compare_models(station_id, predictions)
            
            # Kiểm tra và tạo alerts
            self.check_alerts(station_id, processed_data)
            
            # Cập nhật metrics
            self.update_metrics(station_id, processed_data, predictions)
            
            # Kiểm tra retrain
            self.check_retrain_conditions(station_id)
            
            logger.debug(f"Successfully processed data for station {station_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error processing station data: {e}")
            return False

    def perform_predictions(self, station_id, data):
        """Thực hiện dự đoán với cả XGBoost và LSTM"""
        predictions = {}
        
        try:
            # Dự đoán với XGBoost
            start_time = time.time()
            xgb_prediction = self.model_manager.predict_xgboost(station_id, data)
            xgb_processing_time = time.time() - start_time
            
            if xgb_prediction:
                predictions['xgboost'] = {
                    'wqi_prediction': xgb_prediction['wqi_prediction'],
                    'confidence_score': xgb_prediction.get('confidence_score', 0.8),
                    'processing_time': xgb_processing_time,
                    'model_version': xgb_prediction.get('model_version', 'unknown')
                }
                
                # Lưu kết quả dự đoán
                prediction_data = {
                    'station_id': station_id,
                    'prediction_time': datetime.now(),
                    'model_type': 'xgboost',
                    'wqi_prediction': xgb_prediction['wqi_prediction'],
                    'confidence_score': xgb_prediction.get('confidence_score', 0.8),
                    'processing_time': xgb_processing_time,
                    'model_version': xgb_prediction.get('model_version', 'unknown')
                }
                self.db_manager.insert_prediction_result(prediction_data)
            
            # Dự đoán với LSTM
            start_time = time.time()
            lstm_prediction = self.model_manager.predict_lstm(station_id, data)
            lstm_processing_time = time.time() - start_time
            
            if lstm_prediction:
                predictions['lstm'] = {
                    'wqi_prediction': lstm_prediction['wqi_prediction'],
                    'confidence_score': lstm_prediction.get('confidence_score', 0.8),
                    'processing_time': lstm_processing_time,
                    'model_version': lstm_prediction.get('model_version', 'unknown')
                }
                
                # Lưu kết quả dự đoán
                prediction_data = {
                    'station_id': station_id,
                    'prediction_time': datetime.now(),
                    'model_type': 'lstm',
                    'wqi_prediction': lstm_prediction['wqi_prediction'],
                    'confidence_score': lstm_prediction.get('confidence_score', 0.8),
                    'processing_time': lstm_processing_time,
                    'model_version': lstm_prediction.get('model_version', 'unknown')
                }
                self.db_manager.insert_prediction_result(prediction_data)
            
            logger.info(f"Predictions completed for station {station_id}: XGBoost={predictions.get('xgboost', {}).get('wqi_prediction', 'N/A')}, LSTM={predictions.get('lstm', {}).get('wqi_prediction', 'N/A')}")
            
        except Exception as e:
            logger.error(f"Error performing predictions for station {station_id}: {e}")
        
        return predictions

    def compare_models(self, station_id, predictions):
        """So sánh performance của các model"""
        try:
            if 'xgboost' not in predictions or 'lstm' not in predictions:
                return
            
            xgb_pred = predictions['xgboost']
            lstm_pred = predictions['lstm']
            
            # Tính accuracy improvement (giả sử)
            accuracy_improvement = 0.0
            best_model = 'xgboost'  # Mặc định
            
            # So sánh confidence score
            if lstm_pred['confidence_score'] > xgb_pred['confidence_score']:
                best_model = 'lstm'
                accuracy_improvement = lstm_pred['confidence_score'] - xgb_pred['confidence_score']
            
            # Lưu kết quả so sánh
            comparison_data = {
                'station_id': station_id,
                'comparison_date': datetime.now(),
                'xgboost_accuracy': xgb_pred['confidence_score'],
                'lstm_accuracy': lstm_pred['confidence_score'],
                'xgboost_processing_time': xgb_pred['processing_time'],
                'lstm_processing_time': lstm_pred['processing_time'],
                'xgboost_wqi_prediction': xgb_pred['wqi_prediction'],
                'lstm_wqi_prediction': lstm_pred['wqi_prediction'],
                'best_model': best_model,
                'accuracy_improvement': accuracy_improvement
            }
            
            self.db_manager.insert_model_comparison(comparison_data)
            
            logger.info(f"Model comparison for station {station_id}: Best model = {best_model}")
            
        except Exception as e:
            logger.error(f"Error comparing models for station {station_id}: {e}")

    def check_alerts(self, station_id, data):
        """Kiểm tra và tạo alerts"""
        try:
            wqi = data['wqi']
            alerts = []
            
            # Kiểm tra ngưỡng WQI
            if wqi < self.wqi_thresholds['very_poor']:
                alerts.append({
                    'alert_type': 'WQI_CRITICAL',
                    'severity': 'critical',
                    'message': f'WQI critical level: {wqi}',
                    'wqi_value': wqi,
                    'threshold_value': self.wqi_thresholds['very_poor']
                })
            elif wqi < self.wqi_thresholds['poor']:
                alerts.append({
                    'alert_type': 'WQI_POOR',
                    'severity': 'high',
                    'message': f'WQI poor level: {wqi}',
                    'wqi_value': wqi,
                    'threshold_value': self.wqi_thresholds['poor']
                })
            elif wqi < self.wqi_thresholds['moderate']:
                alerts.append({
                    'alert_type': 'WQI_MODERATE',
                    'severity': 'medium',
                    'message': f'WQI moderate level: {wqi}',
                    'wqi_value': wqi,
                    'threshold_value': self.wqi_thresholds['moderate']
                })
            
            # Kiểm tra các thông số khác
            if data['ph'] < 6.0 or data['ph'] > 9.0:
                alerts.append({
                    'alert_type': 'PH_OUT_OF_RANGE',
                    'severity': 'medium',
                    'message': f'pH out of normal range: {data["ph"]}',
                    'wqi_value': wqi,
                    'threshold_value': 0
                })
            
            if data['do'] < 3.0:
                alerts.append({
                    'alert_type': 'LOW_DO',
                    'severity': 'high',
                    'message': f'Low dissolved oxygen: {data["do"]} mg/L',
                    'wqi_value': wqi,
                    'threshold_value': 3.0
                })
            
            # Lưu alerts
            for alert in alerts:
                alert['station_id'] = station_id
                self.db_manager.insert_alert(alert)
                logger.warning(f"Alert created for station {station_id}: {alert['message']}")
            
        except Exception as e:
            logger.error(f"Error checking alerts for station {station_id}: {e}")

    def update_metrics(self, station_id, data, predictions):
        """Cập nhật metrics cho Prometheus"""
        try:
            # Cập nhật WQI metrics
            self.prometheus_exporter.update_wqi_metric(station_id, data['wqi'])
            
            # Cập nhật các thông số khác
            self.prometheus_exporter.update_ph_metric(station_id, data['ph'])
            self.prometheus_exporter.update_temperature_metric(station_id, data['temperature'])
            self.prometheus_exporter.update_do_metric(station_id, data['do'])
            
            # Cập nhật prediction metrics
            if predictions:
                if 'xgboost' in predictions:
                    self.prometheus_exporter.update_prediction_metric(
                        station_id, 'xgboost', 
                        predictions['xgboost']['wqi_prediction'],
                        predictions['xgboost']['processing_time']
                    )
                
                if 'lstm' in predictions:
                    self.prometheus_exporter.update_prediction_metric(
                        station_id, 'lstm', 
                        predictions['lstm']['wqi_prediction'],
                        predictions['lstm']['processing_time']
                    )
            
            # Cập nhật station activity
            self.prometheus_exporter.update_station_activity(station_id)
            
        except Exception as e:
            logger.error(f"Error updating metrics for station {station_id}: {e}")

    def check_retrain_conditions(self, station_id):
        """Kiểm tra điều kiện retrain"""
        try:
            # Lấy số lượng dữ liệu mới
            recent_data = self.db_manager.get_station_data(station_id, limit=1000)
            
            if len(recent_data) < self.retrain_config['min_records']:
                return
            
            # Kiểm tra performance của model hiện tại
            latest_prediction = self.db_manager.get_latest_prediction(station_id, 'xgboost')
            if latest_prediction:
                accuracy = latest_prediction['confidence_score']
                
                if accuracy < self.retrain_config['accuracy_threshold']:
                    logger.info(f"Low accuracy detected for station {station_id}: {accuracy}")
                    self.trigger_retrain(station_id, 'low_accuracy')
            
            # Kiểm tra data drift (đơn giản)
            if len(recent_data) >= 100:
                recent_wqi = [row[4] for row in recent_data[:100]]  # WQI column (index 4)
                historical_wqi = [row[4] for row in recent_data[100:200]]  # Older data
                
                if len(historical_wqi) >= 50:
                    recent_avg = sum(recent_wqi) / len(recent_wqi)
                    historical_avg = sum(historical_wqi) / len(historical_wqi)
                    
                    drift = abs(recent_avg - historical_avg) / historical_avg
                    
                    if drift > self.retrain_config['drift_threshold']:
                        logger.info(f"Data drift detected for station {station_id}: {drift}")
                        self.trigger_retrain(station_id, 'data_drift')
            
        except Exception as e:
            logger.error(f"Error checking retrain conditions for station {station_id}: {e}")

    def trigger_retrain(self, station_id, reason):
        """Kích hoạt retrain model"""
        try:
            logger.info(f"Triggering retrain for station {station_id} - Reason: {reason}")
            
            # Lưu lịch sử training
            training_data = {
                'station_id': station_id,
                'model_type': 'both',  # Retrain cả XGBoost và LSTM
                'training_start': datetime.now(),
                'status': 'in_progress',
                'error_message': f'Retrain triggered: {reason}'
            }
            self.db_manager.insert_training_history(training_data)
            
            # Thực hiện retrain (có thể gọi qua Airflow hoặc async)
            # self.model_manager.retrain_models(station_id)
            
            logger.info(f"Retrain triggered for station {station_id}")
            
        except Exception as e:
            logger.error(f"Error triggering retrain for station {station_id}: {e}")

    def get_station_summary(self, station_id):
        """Lấy tổng quan về một trạm"""
        try:
            station_info = self.db_manager.get_station_info(station_id)
            if not station_info:
                return None
            
            # Lấy dữ liệu gần đây
            recent_data = self.db_manager.get_station_data(station_id, limit=100)
            
            # Lấy dự đoán gần đây
            latest_xgb = self.db_manager.get_latest_prediction(station_id, 'xgboost')
            latest_lstm = self.db_manager.get_latest_prediction(station_id, 'lstm')
            
            # Tính toán thống kê
            if recent_data:
                wqi_values = [row[4] for row in recent_data]  # WQI column (index 4)
                avg_wqi = sum(wqi_values) / len(wqi_values)
                min_wqi = min(wqi_values)
                max_wqi = max(wqi_values)
            else:
                avg_wqi = min_wqi = max_wqi = 0
            
            summary = {
                'station_info': station_info,
                'data_count': len(recent_data),
                'avg_wqi': round(avg_wqi, 2),
                'min_wqi': round(min_wqi, 2),
                'max_wqi': round(max_wqi, 2),
                'latest_xgb_prediction': latest_xgb,
                'latest_lstm_prediction': latest_lstm,
                'last_updated': datetime.now().isoformat()
            }
            
            return summary
            
        except Exception as e:
            logger.error(f"Error getting station summary for {station_id}: {e}")
            return None

    def get_all_stations_summary(self):
        """Lấy tổng quan tất cả các trạm"""
        try:
            stations = self.db_manager.get_all_stations()
            summaries = []
            
            for station in stations:
                if station['is_active']:
                    summary = self.get_station_summary(station['station_id'])
                    if summary:
                        summaries.append(summary)
            
            return summaries
            
        except Exception as e:
            logger.error(f"Error getting all stations summary: {e}")
            return []

# Global instance
station_processor = StationProcessorV2() 