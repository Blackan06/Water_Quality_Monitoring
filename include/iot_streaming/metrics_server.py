#!/usr/bin/env python3
"""
Prometheus Metrics Server for Water Quality Monitoring
Chạy độc lập để cung cấp metrics cho Prometheus
"""

import logging
import time
import threading
import socket
from prometheus_client import start_http_server, generate_latest, CONTENT_TYPE_LATEST
from flask import Flask, Response
from datetime import datetime
import os
import sys

# Add the parent directory to the path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Removed prometheus_exporter import

# Cấu hình logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class MetricsServer:
    def __init__(self, port=8021):
        self.port = port
        # Removed prometheus_exporter metrics
        self.metrics = None
        self.app = Flask(__name__)
        self.setup_flask_routes()
        
    def setup_flask_routes(self):
        """Thiết lập các routes cho Flask app"""
        
        @self.app.route('/metrics')
        def metrics():
            """Endpoint cho Prometheus metrics"""
            try:
                return Response(generate_latest(), mimetype=CONTENT_TYPE_LATEST)
            except Exception as e:
                logger.error(f"Error generating metrics: {e}")
                return Response("# Error generating metrics\n", status=500)
        
        @self.app.route('/health')
        def health():
            """Health check endpoint"""
            return {
                "status": "healthy",
                "timestamp": datetime.now().isoformat(),
                "metrics_available": True,
                "port": self.port
            }
        
        @self.app.route('/')
        def root():
            """Root endpoint với thông tin server"""
            return {
                "service": "Water Quality Metrics Server",
                "version": "1.0.0",
                "endpoints": {
                    "/metrics": "Prometheus metrics endpoint",
                    "/health": "Health check endpoint"
                },
                "timestamp": datetime.now().isoformat()
            }
    
    def start_server(self):
        """Khởi động metrics server"""
        try:
            logger.info(f"🚀 Starting Water Quality Metrics Server on port {self.port}")
            
            # Kiểm tra port có available không
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(1)
            result = sock.connect_ex(('localhost', self.port))
            sock.close()
            
            if result == 0:
                logger.error(f"❌ Port {self.port} is already in use")
                return False
            
            # Khởi động Flask app
            def run_flask():
                try:
                    self.app.run(host='0.0.0.0', port=self.port, debug=False, threaded=True)
                except Exception as e:
                    logger.error(f"❌ Failed to start Flask server: {e}")
            
            # Chạy Flask trong thread riêng
            flask_thread = threading.Thread(target=run_flask, daemon=True)
            flask_thread.start()
            
            # Đợi một chút để server khởi động
            time.sleep(3)
            
            # Test kết nối
            try:
                test_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                test_sock.settimeout(2)
                result = test_sock.connect_ex(('localhost', self.port))
                test_sock.close()
                
                if result == 0:
                    logger.info(f"✅ Metrics server successfully started on port {self.port}")
                    logger.info(f"📊 Metrics endpoint: http://localhost:{self.port}/metrics")
                    logger.info(f"🏥 Health endpoint: http://localhost:{self.port}/health")
                    return True
                else:
                    logger.error(f"❌ Port {self.port} test failed")
                    return False
                    
            except Exception as e:
                logger.error(f"❌ Port {self.port} test failed: {e}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Failed to start metrics server: {e}")
            return False
    
    def generate_sample_metrics(self):
        """Tạo sample metrics để test"""
        try:
            logger.info("📊 Generating sample metrics...")
            
            # Sample data cho 3 stations
            stations = [
                {"id": 0, "name": "Station_0", "wqi": 85.5, "ph": 7.2, "temp": 25.3, "do": 6.8},
                {"id": 1, "name": "Station_1", "wqi": 78.2, "ph": 6.8, "temp": 28.1, "do": 5.9},
                {"id": 2, "name": "Station_2", "wqi": 92.1, "ph": 7.5, "temp": 22.7, "do": 7.2}
            ]
            
            for station in stations:
                # Update basic metrics
                self.metrics.update_wqi(station["id"], station["name"], station["wqi"])
                self.metrics.update_ph(station["id"], station["name"], station["ph"])
                self.metrics.update_temperature(station["id"], station["name"], station["temp"])
                self.metrics.update_do(station["id"], station["name"], station["do"])
                
                # Update additional metrics
                self.metrics.update_turbidity(station["id"], station["name"], 2.5)
                self.metrics.update_conductivity(station["id"], station["name"], 450.0)
                self.metrics.update_tds(station["id"], station["name"], 320.0)
                self.metrics.update_chlorine(station["id"], station["name"], 1.2)
                self.metrics.update_nitrate(station["id"], station["name"], 8.5)
                self.metrics.update_nitrite(station["id"], station["name"], 0.8)
                self.metrics.update_bacteria(station["id"], station["name"], 45.0)
                
                # Update system metrics
                self.metrics.record_data_point_processed(station["id"])
                self.metrics.record_model_performance(station["id"], "xgboost", 0.85)
                self.metrics.record_model_performance(station["id"], "lstm", 0.82)
                self.metrics.record_data_drift(station["id"], 0.15)
                self.metrics.record_processing_time("prediction", 0.5)
            
            # Update alert metrics
            self.metrics.record_alert("high_wqi", 0, "warning")
            self.metrics.record_alert("low_ph", 1, "critical")
            
            # Update pipeline metrics
            self.metrics.record_pipeline_run("streaming_data_processor", "success")
            self.metrics.record_pipeline_run("water_quality_processing", "success")
            
            # Update Airflow metrics
            self.metrics.record_airflow_dag_success("streaming_data_processor")
            self.metrics.record_airflow_task_success("streaming_data_processor", "process_streaming_data")
            
            logger.info("✅ Sample metrics generated successfully")
            
        except Exception as e:
            logger.error(f"❌ Error generating sample metrics: {e}")

def main():
    """Main function để chạy metrics server"""
    logger.info("🚀 Starting Water Quality Metrics Server")
    logger.info("=" * 60)
    
    # Lấy port từ environment variable hoặc dùng default
    port = int(os.getenv('METRICS_PORT', 8020))
    
    # Tạo và khởi động server
    server = MetricsServer(port=port)
    
    if server.start_server():
        logger.info("✅ Metrics server started successfully")
        
        # Tạo sample metrics
        server.generate_sample_metrics()
        
        # Giữ server chạy
        try:
            logger.info("🔄 Metrics server is running. Press Ctrl+C to stop.")
            while True:
                time.sleep(60)  # Sleep 1 minute
                logger.debug("Metrics server is still running...")
        except KeyboardInterrupt:
            logger.info("🛑 Stopping metrics server...")
            logger.info("✅ Metrics server stopped")
    else:
        logger.error("❌ Failed to start metrics server")
        sys.exit(1)

if __name__ == "__main__":
    main() 