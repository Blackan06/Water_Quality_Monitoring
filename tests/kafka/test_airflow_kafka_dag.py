#!/usr/bin/env python3
"""
Script test Airflow DAG kết nối Kafka
"""

import os
import sys
import logging
from datetime import datetime

# Thêm đường dẫn để import các module
sys.path.append(os.path.join(os.path.dirname(__file__), '../../include'))

# Cấu hình logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_kafka_producer_task():
    """Test kafka producer task từ Airflow DAG"""
    logger.info("=== TEST: Kafka Producer Task ===")
    
    try:
        from iot_streaming.kafka_producer_streaming import kafka_run
        
        # Mock context cho Airflow
        context = {
            'ti': MockTaskInstance(),
            'dag': None,
            'task': None
        }
        
        logger.info("🚀 Chạy kafka_producer_task...")
        kafka_run(**context)
        
        logger.info("✅ Kafka producer task chạy thành công!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Lỗi kafka producer task: {e}")
        return False

def test_kafka_consumer_task():
    """Test kafka consumer task từ Airflow DAG"""
    logger.info("=== TEST: Kafka Consumer Task ===")
    
    try:
        from iot_streaming.kafka_consumer import kafka_consumer_task
        
        # Mock context cho Airflow
        context = {
            'ti': MockTaskInstance(),
            'dag': None,
            'task': None
        }
        
        logger.info("🚀 Chạy kafka_consumer_task...")
        kafka_consumer_task(**context)
        
        logger.info("✅ Kafka consumer task chạy thành công!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Lỗi kafka consumer task: {e}")
        return False

def test_database_manager():
    """Test database manager"""
    logger.info("=== TEST: Database Manager ===")
    
    try:
        from iot_streaming.database_manager import db_manager
        
        logger.info("🚀 Kiểm tra kết nối database...")
        
        # Test kết nối
        connection = db_manager.get_connection()
        if connection:
            logger.info("✅ Kết nối database thành công!")
            connection.close()
        else:
            logger.warning("⚠️ Không thể kết nối database")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Lỗi database manager: {e}")
        return False

class MockTaskInstance:
    """Mock class để simulate Airflow TaskInstance"""
    
    def __init__(self):
        self.xcom_data = {}
    
    def xcom_push(self, key, value):
        """Mock xcom_push method"""
        self.xcom_data[key] = value
        logger.info(f"📤 XCom push: {key} = {value}")
    
    def xcom_pull(self, task_ids=None, key=None):
        """Mock xcom_pull method"""
        if key in self.xcom_data:
            return self.xcom_data[key]
        return None

def check_airflow_environment():
    """Kiểm tra môi trường Airflow"""
    logger.info("=== KIỂM TRA MÔI TRƯỜNG AIRFLOW ===")
    
    # Kiểm tra biến môi trường Airflow
    airflow_home = os.getenv('AIRFLOW_HOME')
    if airflow_home:
        logger.info(f"✅ AIRFLOW_HOME: {airflow_home}")
    else:
        logger.warning("⚠️ AIRFLOW_HOME không được set")
    
    # Kiểm tra thư mục dags
    dags_folder = os.path.join(os.path.dirname(__file__), '../../dags')
    if os.path.exists(dags_folder):
        logger.info(f"✅ DAGs folder: {dags_folder}")
    else:
        logger.warning(f"⚠️ DAGs folder không tồn tại: {dags_folder}")
    
    # Kiểm tra thư mục include
    include_folder = os.path.join(os.path.dirname(__file__), '../../include')
    if os.path.exists(include_folder):
        logger.info(f"✅ Include folder: {include_folder}")
    else:
        logger.warning(f"⚠️ Include folder không tồn tại: {include_folder}")

def main():
    """Chạy tất cả các test"""
    logger.info("🚀 Bắt đầu test Airflow Kafka DAG")
    logger.info("=" * 60)
    
    # Kiểm tra môi trường
    check_airflow_environment()
    logger.info("")
    
    results = []
    
    # Test các component
    results.append(("Database Manager", test_database_manager()))
    results.append(("Kafka Producer Task", test_kafka_producer_task()))
    results.append(("Kafka Consumer Task", test_kafka_consumer_task()))
    
    # Tổng kết
    logger.info("=" * 60)
    logger.info("📊 KẾT QUẢ TEST AIRFLOW DAG:")
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        logger.info(f"   {test_name}: {status}")
        if result:
            passed += 1
    
    logger.info(f"📈 Tổng kết: {passed}/{total} test thành công")
    
    if passed == total:
        logger.info("🎉 Tất cả Airflow DAG test đều thành công!")
    else:
        logger.warning("⚠️ Một số test thất bại. Vui lòng kiểm tra cấu hình.")
    
    logger.info("=" * 60)

if __name__ == "__main__":
    main() 