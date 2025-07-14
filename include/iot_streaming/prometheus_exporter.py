import logging
import time
import socket
from prometheus_client import Counter, Gauge, Histogram, Summary, start_http_server, CollectorRegistry
import os

logger = logging.getLogger(__name__)

class PrometheusExporter:
    _instance = None
    _initialized = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(PrometheusExporter, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not self._initialized:
            self.port = int(os.getenv('PROMETHEUS_PORT', '8000'))
            
            # Metrics cho WQI
            self.wqi_gauge = Gauge(
                'water_quality_index',
                'Water Quality Index by station',
                ['station_id', 'station_name']
            )
            
            # Metrics cho các thông số khác
            self.ph_gauge = Gauge(
                'water_ph',
                'Water pH level by station',
                ['station_id', 'station_name']
            )
            
            self.temperature_gauge = Gauge(
                'water_temperature',
                'Water temperature by station',
                ['station_id', 'station_name']
            )
            
            self.do_gauge = Gauge(
                'water_dissolved_oxygen',
                'Dissolved oxygen level by station',
                ['station_id', 'station_name']
            )
            
            self.turbidity_gauge = Gauge(
                'water_turbidity',
                'Water turbidity by station',
                ['station_id', 'station_name']
            )
            
            # Metrics cho predictions
            self.prediction_gauge = Gauge(
                'wqi_prediction',
                'WQI prediction by model and station',
                ['station_id', 'model_type', 'station_name']
            )
            
            self.prediction_processing_time = Histogram(
                'prediction_processing_time_seconds',
                'Time spent processing predictions',
                ['station_id', 'model_type', 'station_name']
            )
            
            # Metrics cho station activity
            self.station_activity_counter = Counter(
                'station_data_processed_total',
                'Total number of data records processed by station',
                ['station_id', 'station_name']
            )
            
            # Metrics cho model performance
            self.model_accuracy_gauge = Gauge(
                'model_accuracy',
                'Model accuracy by type and station',
                ['station_id', 'model_type', 'station_name']
            )
            
            self.model_mae_gauge = Gauge(
                'model_mae',
                'Model Mean Absolute Error by type and station',
                ['station_id', 'model_type', 'station_name']
            )
            
            # Metrics cho alerts
            self.alert_counter = Counter(
                'water_quality_alerts_total',
                'Total number of alerts by type and station',
                ['station_id', 'alert_type', 'severity', 'station_name']
            )
            
            # Metrics cho training
            self.training_duration_histogram = Histogram(
                'model_training_duration_seconds',
                'Time spent training models',
                ['station_id', 'model_type', 'station_name']
            )
            
            self.training_records_gauge = Gauge(
                'model_training_records',
                'Number of records used for training',
                ['station_id', 'model_type', 'station_name']
            )
            
            # Metrics cho data processing
            self.data_processing_time = Histogram(
                'data_processing_time_seconds',
                'Time spent processing data',
                ['station_id', 'station_name']
            )
            
            # Metrics cho database operations
            self.db_operation_counter = Counter(
                'database_operations_total',
                'Total number of database operations',
                ['operation_type', 'table_name']
            )
            
            self.db_operation_duration = Histogram(
                'database_operation_duration_seconds',
                'Time spent on database operations',
                ['operation_type', 'table_name']
            )
            
            # Check if port is available before starting server
            if self._is_port_available(self.port):
                try:
                    start_http_server(self.port)
                    logger.info(f"Prometheus exporter started on port {self.port}")
                except Exception as e:
                    logger.error(f"Failed to start Prometheus exporter: {e}")
            else:
                logger.warning(f"Port {self.port} is already in use. Prometheus exporter not started.")
            
            self._initialized = True

    def _is_port_available(self, port):
        """Kiểm tra xem port có available không"""
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(('localhost', port))
                return True
        except OSError:
            return False

    def update_wqi_metric(self, station_id: int, wqi_value: float, station_name: str = None):
        """Cập nhật WQI metric"""
        try:
            station_name = station_name or f"Station_{station_id}"
            self.wqi_gauge.labels(
                station_id=str(station_id),
                station_name=station_name
            ).set(wqi_value)
        except Exception as e:
            logger.error(f"Error updating WQI metric: {e}")

    def update_ph_metric(self, station_id: int, ph_value: float, station_name: str = None):
        """Cập nhật pH metric"""
        try:
            station_name = station_name or f"Station_{station_id}"
            self.ph_gauge.labels(
                station_id=str(station_id),
                station_name=station_name
            ).set(ph_value)
        except Exception as e:
            logger.error(f"Error updating pH metric: {e}")

    def update_temperature_metric(self, station_id: int, temp_value: float, station_name: str = None):
        """Cập nhật temperature metric"""
        try:
            station_name = station_name or f"Station_{station_id}"
            self.temperature_gauge.labels(
                station_id=str(station_id),
                station_name=station_name
            ).set(temp_value)
        except Exception as e:
            logger.error(f"Error updating temperature metric: {e}")

    def update_do_metric(self, station_id: int, do_value: float, station_name: str = None):
        """Cập nhật DO metric"""
        try:
            station_name = station_name or f"Station_{station_id}"
            self.do_gauge.labels(
                station_id=str(station_id),
                station_name=station_name
            ).set(do_value)
        except Exception as e:
            logger.error(f"Error updating DO metric: {e}")

    def update_turbidity_metric(self, station_id: int, turbidity_value: float, station_name: str = None):
        """Cập nhật turbidity metric"""
        try:
            station_name = station_name or f"Station_{station_id}"
            self.turbidity_gauge.labels(
                station_id=str(station_id),
                station_name=station_name
            ).set(turbidity_value)
        except Exception as e:
            logger.error(f"Error updating turbidity metric: {e}")

    def update_prediction_metric(self, station_id: int, model_type: str, 
                               prediction_value: float, processing_time: float, 
                               station_name: str = None):
        """Cập nhật prediction metrics"""
        try:
            station_name = station_name or f"Station_{station_id}"
            
            # Cập nhật prediction value
            self.prediction_gauge.labels(
                station_id=str(station_id),
                model_type=model_type,
                station_name=station_name
            ).set(prediction_value)
            
            # Cập nhật processing time
            self.prediction_processing_time.labels(
                station_id=str(station_id),
                model_type=model_type,
                station_name=station_name
            ).observe(processing_time)
            
        except Exception as e:
            logger.error(f"Error updating prediction metric: {e}")

    def update_station_activity(self, station_id: int, station_name: str = None):
        """Cập nhật station activity metric"""
        try:
            station_name = station_name or f"Station_{station_id}"
            self.station_activity_counter.labels(
                station_id=str(station_id),
                station_name=station_name
            ).inc()
        except Exception as e:
            logger.error(f"Error updating station activity metric: {e}")

    def update_model_performance(self, station_id: int, model_type: str, 
                               accuracy: float, mae: float, station_name: str = None):
            """Cập nhật model performance metrics"""
            try:
                station_name = station_name or f"Station_{station_id}"
                
                # Cập nhật accuracy
                self.model_accuracy_gauge.labels(
                    station_id=str(station_id),
                    model_type=model_type,
                    station_name=station_name
                ).set(accuracy)
                
                # Cập nhật MAE
                self.model_mae_gauge.labels(
                    station_id=str(station_id),
                    model_type=model_type,
                    station_name=station_name
                ).set(mae)
                
            except Exception as e:
                logger.error(f"Error updating model performance metric: {e}")

    def record_alert(self, station_id: int, alert_type: str, severity: str, station_name: str = None):
        """Ghi lại alert metric"""
        try:
            station_name = station_name or f"Station_{station_id}"
            self.alert_counter.labels(
                station_id=str(station_id),
                alert_type=alert_type,
                severity=severity,
                station_name=station_name
            ).inc()
        except Exception as e:
            logger.error(f"Error recording alert metric: {e}")

    def record_training_metrics(self, station_id: int, model_type: str, 
                              duration: float, records_count: int, station_name: str = None):
        """Ghi lại training metrics"""
        try:
            station_name = station_name or f"Station_{station_id}"
            
            # Cập nhật training duration
            self.training_duration_histogram.labels(
                station_id=str(station_id),
                model_type=model_type,
                station_name=station_name
            ).observe(duration)
            
            # Cập nhật training records
            self.training_records_gauge.labels(
                station_id=str(station_id),
                model_type=model_type,
                station_name=station_name
            ).set(records_count)
            
        except Exception as e:
            logger.error(f"Error recording training metrics: {e}")

    def record_data_processing_time(self, station_id: int, processing_time: float, station_name: str = None):
        """Ghi lại data processing time"""
        try:
            station_name = station_name or f"Station_{station_id}"
            self.data_processing_time.labels(
                station_id=str(station_id),
                station_name=station_name
            ).observe(processing_time)
        except Exception as e:
            logger.error(f"Error recording data processing time: {e}")

    def record_db_operation(self, operation_type: str, table_name: str, duration: float = None):
        """Ghi lại database operation"""
        try:
            # Tăng counter
            self.db_operation_counter.labels(
                operation_type=operation_type,
                table_name=table_name
            ).inc()
            
            # Ghi lại duration nếu có
            if duration is not None:
                self.db_operation_duration.labels(
                    operation_type=operation_type,
                    table_name=table_name
                ).observe(duration)
                
        except Exception as e:
            logger.error(f"Error recording database operation: {e}")

    def get_metrics_summary(self) -> dict:
        """Lấy summary của metrics"""
        try:
            # Lấy tất cả metrics hiện tại
            metrics = {}
            
            # WQI metrics
            wqi_samples = list(self.wqi_gauge._metrics.values())
            if wqi_samples:
                metrics['wqi'] = {
                    'count': len(wqi_samples),
                    'values': [sample._value.get() for sample in wqi_samples]
                }
            
            # Prediction metrics
            pred_samples = list(self.prediction_gauge._metrics.values())
            if pred_samples:
                metrics['predictions'] = {
                    'count': len(pred_samples),
                    'values': [sample._value.get() for sample in pred_samples]
                }
            
            # Alert metrics
            alert_samples = list(self.alert_counter._metrics.values())
            if alert_samples:
                metrics['alerts'] = {
                    'count': len(alert_samples),
                    'values': [sample._value.get() for sample in alert_samples]
                }
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error getting metrics summary: {e}")
            return {}

# Get singleton instance
def get_prometheus_exporter():
    """Get the singleton instance of PrometheusExporter"""
    return PrometheusExporter() 