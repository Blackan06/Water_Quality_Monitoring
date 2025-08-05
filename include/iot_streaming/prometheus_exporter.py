import logging
from prometheus_client import start_http_server, Gauge, Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from datetime import datetime
import time
import threading
import socket
from flask import Flask, Response

logger = logging.getLogger(__name__)

class WaterQualityMetrics:
    def __init__(self):
        # Gauges for current values
        self.wqi_gauge = Gauge('water_quality_wqi', 'Water Quality Index', ['station_id', 'station_name'])
        self.ph_gauge = Gauge('water_quality_ph', 'pH Level', ['station_id', 'station_name'])
        self.temperature_gauge = Gauge('water_quality_temperature', 'Water Temperature', ['station_id', 'station_name'])
        self.do_gauge = Gauge('water_quality_do', 'Dissolved Oxygen', ['station_id', 'station_name'])
        
        # Additional water quality parameters
        self.turbidity_gauge = Gauge('water_quality_turbidity', 'Turbidity', ['station_id', 'station_name'])
        self.conductivity_gauge = Gauge('water_quality_conductivity', 'Conductivity', ['station_id', 'station_name'])
        self.tds_gauge = Gauge('water_quality_tds', 'Total Dissolved Solids', ['station_id', 'station_name'])
        self.chlorine_gauge = Gauge('water_quality_chlorine', 'Chlorine Level', ['station_id', 'station_name'])
        self.nitrate_gauge = Gauge('water_quality_nitrate', 'Nitrate Level', ['station_id', 'station_name'])
        self.nitrite_gauge = Gauge('water_quality_nitrite', 'Nitrite Level', ['station_id', 'station_name'])
        self.bacteria_gauge = Gauge('water_quality_bacteria', 'Bacteria Count', ['station_id', 'station_name'])
        
        # Counters for events
        self.alerts_generated = Counter('alerts_generated_total', 'Total alerts generated', ['alert_type', 'station_id', 'severity'])
        self.pipeline_runs = Counter('pipeline_runs_total', 'Total pipeline runs', ['dag_id', 'status'])
        self.data_points_processed = Counter('data_points_processed_total', 'Total data points processed', ['station_id'])
        
        # Histograms for performance metrics
        self.model_performance = Histogram('model_performance_r2_score', 'Model R2 Score', ['station_id', 'model_name'])
        self.data_drift_score = Histogram('data_drift_score', 'Data Drift Score', ['station_id'])
        self.processing_time = Histogram('processing_time_seconds', 'Processing time in seconds', ['operation'])
        
        # Database connection metrics
        self.db_connection_errors = Counter('db_connection_errors_total', 'Total database connection errors')
        self.kafka_connection_errors = Counter('kafka_connection_errors_total', 'Total Kafka connection errors')
        
        # Airflow specific metrics
        self.airflow_dag_success = Counter('airflow_dag_success_total', 'Total successful DAG runs', ['dag_id'])
        self.airflow_dag_failure = Counter('airflow_dag_failure_total', 'Total failed DAG runs', ['dag_id'])
        self.airflow_task_success = Counter('airflow_task_success_total', 'Total successful task runs', ['dag_id', 'task_id'])
        self.airflow_task_failure = Counter('airflow_task_failure_total', 'Total failed task runs', ['dag_id', 'task_id'])
        
        # Start metrics server
        self.start_metrics_server()
    
    def start_metrics_server(self):
        """Start Prometheus metrics server with better error handling"""
        try:
            port = 8020
            
            # Check if port is available
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(1)
            result = sock.connect_ex(('localhost', port))
            sock.close()
            
            if result != 0:  # Port is available
                # Start the server in a separate thread to avoid blocking
                def start_server():
                    try:
                        start_http_server(port, addr='0.0.0.0')
                        logger.info(f"✅ Prometheus metrics server started on port {port}")
                    except Exception as e:
                        logger.error(f"❌ Failed to start metrics server on port {port}: {e}")
                
                server_thread = threading.Thread(target=start_server, daemon=True)
                server_thread.start()
                
                # Wait a bit to see if server starts successfully
                time.sleep(2)
                
                # Test if server is responding
                try:
                    test_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    test_sock.settimeout(2)
                    result = test_sock.connect_ex(('localhost', port))
                    test_sock.close()
                    
                    if result == 0:
                        logger.info(f"✅ Metrics server confirmed running on port {port}")
                        return
                    else:
                        logger.error(f"❌ Port {port} test failed")
                except Exception as e:
                    logger.error(f"❌ Port {port} test failed: {e}")
            else:
                logger.error(f"❌ Port {port} is already in use")
                
        except Exception as e:
            logger.error(f"❌ Failed to start metrics server: {e}")
    
    def get_metrics_endpoint(self):
        """Return metrics in Prometheus format"""
        try:
            return generate_latest()
        except Exception as e:
            logger.error(f"Error generating metrics: {e}")
            return b"# Error generating metrics\n"
    
    def health_check(self):
        """Health check endpoint"""
        return {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "metrics_available": True
        }

    def update_wqi(self, station_id, station_name, wqi_value):
        """Update WQI metric"""
        try:
            self.wqi_gauge.labels(station_id=str(station_id), station_name=station_name).set(wqi_value)
            logger.debug(f"Updated WQI metric: station {station_id}, value {wqi_value}")
        except Exception as e:
            logger.error(f"Error updating WQI metric: {e}")

    def update_ph(self, station_id, station_name, ph_value):
        """Update pH metric"""
        try:
            self.ph_gauge.labels(station_id=str(station_id), station_name=station_name).set(ph_value)
        except Exception as e:
            logger.error(f"Error updating pH metric: {e}")

    def update_temperature(self, station_id, station_name, temp_value):
        """Update temperature metric"""
        try:
            self.temperature_gauge.labels(station_id=str(station_id), station_name=station_name).set(temp_value)
        except Exception as e:
            logger.error(f"Error updating temperature metric: {e}")

    def update_do(self, station_id, station_name, do_value):
        """Update dissolved oxygen metric"""
        try:
            self.do_gauge.labels(station_id=str(station_id), station_name=station_name).set(do_value)
        except Exception as e:
            logger.error(f"Error updating DO metric: {e}")

    def update_turbidity(self, station_id, station_name, turbidity_value):
        """Update turbidity metric"""
        try:
            self.turbidity_gauge.labels(station_id=str(station_id), station_name=station_name).set(turbidity_value)
        except Exception as e:
            logger.error(f"Error updating turbidity metric: {e}")

    def update_conductivity(self, station_id, station_name, conductivity_value):
        """Update conductivity metric"""
        try:
            self.conductivity_gauge.labels(station_id=str(station_id), station_name=station_name).set(conductivity_value)
        except Exception as e:
            logger.error(f"Error updating conductivity metric: {e}")

    def update_tds(self, station_id, station_name, tds_value):
        """Update TDS metric"""
        try:
            self.tds_gauge.labels(station_id=str(station_id), station_name=station_name).set(tds_value)
        except Exception as e:
            logger.error(f"Error updating TDS metric: {e}")

    def update_chlorine(self, station_id, station_name, chlorine_value):
        """Update chlorine metric"""
        try:
            self.chlorine_gauge.labels(station_id=str(station_id), station_name=station_name).set(chlorine_value)
        except Exception as e:
            logger.error(f"Error updating chlorine metric: {e}")

    def update_nitrate(self, station_id, station_name, nitrate_value):
        """Update nitrate metric"""
        try:
            self.nitrate_gauge.labels(station_id=str(station_id), station_name=station_name).set(nitrate_value)
        except Exception as e:
            logger.error(f"Error updating nitrate metric: {e}")

    def update_nitrite(self, station_id, station_name, nitrite_value):
        """Update nitrite metric"""
        try:
            self.nitrite_gauge.labels(station_id=str(station_id), station_name=station_name).set(nitrite_value)
        except Exception as e:
            logger.error(f"Error updating nitrite metric: {e}")

    def update_bacteria(self, station_id, station_name, bacteria_value):
        """Update bacteria count metric"""
        try:
            self.bacteria_gauge.labels(station_id=str(station_id), station_name=station_name).set(bacteria_value)
        except Exception as e:
            logger.error(f"Error updating bacteria metric: {e}")

    def record_alert(self, alert_type, station_id, severity):
        """Record alert generation"""
        try:
            self.alerts_generated.labels(alert_type=alert_type, station_id=str(station_id), severity=severity).inc()
            logger.info(f"Recorded alert: {alert_type} for station {station_id}")
        except Exception as e:
            logger.error(f"Error recording alert: {e}")
    
    def record_pipeline_run(self, dag_id, status):
        """Record pipeline run"""
        try:
            self.pipeline_runs.labels(dag_id=dag_id, status=status).inc()
            logger.info(f"Recorded pipeline run: {dag_id} - {status}")
        except Exception as e:
            logger.error(f"Error recording pipeline run: {e}")
    
    def record_data_point_processed(self, station_id):
        """Record data point processing"""
        try:
            self.data_points_processed.labels(station_id=str(station_id)).inc()
        except Exception as e:
            logger.error(f"Error recording data point: {e}")
    
    def record_model_performance(self, station_id, model_name, r2_score):
        """Record model performance"""
        try:
            self.model_performance.labels(station_id=str(station_id), model_name=model_name).observe(r2_score)
            logger.info(f"Recorded model performance: station {station_id}, model {model_name}, R2 {r2_score}")
        except Exception as e:
            logger.error(f"Error recording model performance: {e}")
    
    def record_data_drift(self, station_id, drift_score):
        """Record data drift score"""
        try:
            self.data_drift_score.labels(station_id=str(station_id)).observe(drift_score)
            logger.info(f"Recorded data drift: station {station_id}, score {drift_score}")
        except Exception as e:
            logger.error(f"Error recording data drift: {e}")
    
    def record_processing_time(self, operation, duration):
        """Record processing time"""
        try:
            self.processing_time.labels(operation=operation).observe(duration)
        except Exception as e:
            logger.error(f"Error recording processing time: {e}")
    
    def record_db_error(self):
        """Record database connection error"""
        try:
            self.db_connection_errors.inc()
        except Exception as e:
            logger.error(f"Error recording DB error: {e}")
    
    def record_kafka_error(self):
        """Record Kafka connection error"""
        try:
            self.kafka_connection_errors.inc()
        except Exception as e:
            logger.error(f"Error recording Kafka error: {e}")
    
    def record_airflow_dag_success(self, dag_id):
        """Record successful DAG run"""
        try:
            self.airflow_dag_success.labels(dag_id=dag_id).inc()
            logger.info(f"Recorded successful DAG run: {dag_id}")
        except Exception as e:
            logger.error(f"Error recording DAG success: {e}")
    
    def record_airflow_dag_failure(self, dag_id):
        """Record failed DAG run"""
        try:
            self.airflow_dag_failure.labels(dag_id=dag_id).inc()
            logger.info(f"Recorded failed DAG run: {dag_id}")
        except Exception as e:
            logger.error(f"Error recording DAG failure: {e}")
    
    def record_airflow_task_success(self, dag_id, task_id):
        """Record successful task run"""
        try:
            self.airflow_task_success.labels(dag_id=dag_id, task_id=task_id).inc()
            logger.info(f"Recorded successful task run: {dag_id}.{task_id}")
        except Exception as e:
            logger.error(f"Error recording task success: {e}")
    
    def record_airflow_task_failure(self, dag_id, task_id):
        """Record failed task run"""
        try:
            self.airflow_task_failure.labels(dag_id=dag_id, task_id=task_id).inc()
            logger.info(f"Recorded failed task run: {dag_id}.{task_id}")
        except Exception as e:
            logger.error(f"Error recording task failure: {e}")

    # Backward compatibility methods
    def update_wqi_metric(self, station_id, wqi_value, station_name=None):
        """Update WQI metric (backward compatibility)"""
        station_name = station_name or f"Station_{station_id}"
        self.update_wqi(station_id, station_name, wqi_value)
    
    def update_ph_metric(self, station_id, ph_value, station_name=None):
        """Update pH metric (backward compatibility)"""
        station_name = station_name or f"Station_{station_id}"
        self.update_ph(station_id, station_name, ph_value)
    
    def update_temperature_metric(self, station_id, temp_value, station_name=None):
        """Update temperature metric (backward compatibility)"""
        station_name = station_name or f"Station_{station_id}"
        self.update_temperature(station_id, station_name, temp_value)
    
    def update_do_metric(self, station_id, do_value, station_name=None):
        """Update dissolved oxygen metric (backward compatibility)"""
        station_name = station_name or f"Station_{station_id}"
        self.update_do(station_id, station_name, do_value)
    
    def update_turbidity_metric(self, station_id, turbidity_value, station_name=None):
        """Update turbidity metric (backward compatibility)"""
        station_name = station_name or f"Station_{station_id}"
        self.update_turbidity(station_id, station_name, turbidity_value)
    
    def update_conductivity_metric(self, station_id, conductivity_value, station_name=None):
        """Update conductivity metric (backward compatibility)"""
        station_name = station_name or f"Station_{station_id}"
        self.update_conductivity(station_id, station_name, conductivity_value)
    
    def update_tds_metric(self, station_id, tds_value, station_name=None):
        """Update TDS metric (backward compatibility)"""
        station_name = station_name or f"Station_{station_id}"
        self.update_tds(station_id, station_name, tds_value)
    
    def update_chlorine_metric(self, station_id, chlorine_value, station_name=None):
        """Update chlorine metric (backward compatibility)"""
        station_name = station_name or f"Station_{station_id}"
        self.update_chlorine(station_id, station_name, chlorine_value)
    
    def update_nitrate_metric(self, station_id, nitrate_value, station_name=None):
        """Update nitrate metric (backward compatibility)"""
        station_name = station_name or f"Station_{station_id}"
        self.update_nitrate(station_id, station_name, nitrate_value)
    
    def update_nitrite_metric(self, station_id, nitrite_value, station_name=None):
        """Update nitrite metric (backward compatibility)"""
        station_name = station_name or f"Station_{station_id}"
        self.update_nitrite(station_id, station_name, nitrite_value)
    
    def update_bacteria_metric(self, station_id, bacteria_value, station_name=None):
        """Update bacteria metric (backward compatibility)"""
        station_name = station_name or f"Station_{station_id}"
        self.update_bacteria(station_id, station_name, bacteria_value)
    
    def update_station_activity(self, station_id, station_name=None):
        """Update station activity (backward compatibility)"""
        station_name = station_name or f"Station_{station_id}"
        self.record_data_point_processed(station_id)
    
    def update_prediction_metric(self, station_id, model_type, prediction_value, processing_time, station_name=None):
        """Update prediction metric (backward compatibility)"""
        station_name = station_name or f"Station_{station_id}"
        self.record_processing_time(f"prediction_{model_type}", processing_time)
        # You can add more specific prediction metrics here if needed

# Global metrics instance
metrics = WaterQualityMetrics()

def get_metrics():
    """Get global metrics instance"""
    return metrics
            
def get_prometheus_exporter():
    """Get prometheus exporter instance (for backward compatibility)"""
    return metrics 