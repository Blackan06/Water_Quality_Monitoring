#!/usr/bin/env python3
"""
Script tổng hợp để chạy tất cả các test Kafka
"""

import os
import sys
import subprocess
import time
import logging
from datetime import datetime

# Cấu hình logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def run_script(script_name, description):
    """Chạy một script test"""
    logger.info(f"🚀 {description}")
    logger.info(f"📝 Script: {script_name}")
    logger.info("-" * 50)
    
    try:
        result = subprocess.run(
            [sys.executable, script_name],
            capture_output=True,
            text=True,
            timeout=60
        )
        
        if result.returncode == 0:
            logger.info("✅ Script chạy thành công!")
            logger.info("📄 Output:")
            print(result.stdout)
        else:
            logger.error("❌ Script chạy thất bại!")
            logger.error("📄 Error output:")
            print(result.stderr)
            logger.info("📄 Standard output:")
            print(result.stdout)
        
        return result.returncode == 0
        
    except subprocess.TimeoutExpired:
        logger.error(f"⏰ Script {script_name} timeout sau 60 giây")
        return False
    except FileNotFoundError:
        logger.error(f"❌ Không tìm thấy script {script_name}")
        return False
    except Exception as e:
        logger.error(f"❌ Lỗi chạy script {script_name}: {e}")
        return False

def check_requirements():
    """Kiểm tra các thư viện cần thiết"""
    logger.info("🔍 Kiểm tra thư viện cần thiết...")
    
    required_packages = [
        'kafka-python',
        'requests',
        'psycopg2-binary',
        'elasticsearch'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
            logger.info(f"✅ {package}")
        except ImportError:
            logger.warning(f"⚠️ Thiếu {package}")
            missing_packages.append(package)
    
    if missing_packages:
        logger.warning("📦 Cài đặt các thư viện thiếu:")
        for package in missing_packages:
            logger.info(f"   pip install {package}")
        return False
    
    logger.info("✅ Tất cả thư viện cần thiết đã được cài đặt!")
    return True

def check_environment():
    """Kiểm tra môi trường"""
    logger.info("🔍 Kiểm tra môi trường...")
    
    # Kiểm tra Python version
    python_version = sys.version_info
    logger.info(f"🐍 Python version: {python_version.major}.{python_version.minor}.{python_version.micro}")
    
    # Kiểm tra thư mục hiện tại
    current_dir = os.getcwd()
    logger.info(f"📁 Thư mục hiện tại: {current_dir}")
    
    # Kiểm tra các file cần thiết
    required_files = [
        '../kafka/test_kafka_connection.py',
        '../kafka/test_airflow_kafka_dag.py',
        '../monitoring/test_monitoring_services.py'
    ]
    
    missing_files = []
    for file in required_files:
        if os.path.exists(file):
            logger.info(f"✅ {file}")
        else:
            logger.warning(f"⚠️ Thiếu {file}")
            missing_files.append(file)
    
    if missing_files:
        logger.error("❌ Một số file test không tồn tại!")
        return False
    
    logger.info("✅ Môi trường sẵn sàng!")
    return True

def main():
    """Chạy tất cả các test"""
    logger.info("🚀 BẮT ĐẦU KIỂM TRA TOÀN BỘ HỆ THỐNG KAFKA")
    logger.info("=" * 80)
    logger.info(f"⏰ Thời gian bắt đầu: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 80)
    
    # Kiểm tra môi trường
    if not check_environment():
        logger.error("❌ Môi trường không sẵn sàng!")
        return
    
    logger.info("")
    
    # Kiểm tra requirements
    if not check_requirements():
        logger.warning("⚠️ Một số thư viện bị thiếu, có thể ảnh hưởng đến kết quả test")
    
    logger.info("")
    
    # Danh sách các test cần chạy
    tests = [
        ('../kafka/test_kafka_connection.py', 'Test kết nối Kafka cơ bản'),
        ('../monitoring/test_monitoring_services.py', 'Test monitoring services'),
        ('../kafka/test_airflow_kafka_dag.py', 'Test Airflow DAG components')
    ]
    
    results = []
    
    # Chạy từng test
    for script_name, description in tests:
        logger.info("")
        success = run_script(script_name, description)
        results.append((description, success))
        
        # Đợi 2 giây giữa các test
        time.sleep(2)
    
    # Tổng kết
    logger.info("")
    logger.info("=" * 80)
    logger.info("📊 KẾT QUẢ TỔNG HỢP:")
    logger.info("=" * 80)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        logger.info(f"   {test_name}: {status}")
        if result:
            passed += 1
    
    logger.info("")
    logger.info(f"📈 Tổng kết: {passed}/{total} test thành công")
    
    if passed == total:
        logger.info("🎉 TẤT CẢ TEST ĐỀU THÀNH CÔNG!")
        logger.info("✅ Hệ thống Kafka hoạt động bình thường")
    elif passed > 0:
        logger.warning("⚠️ MỘT SỐ TEST THẤT BẠI")
        logger.info("🔧 Vui lòng kiểm tra và khắc phục các vấn đề")
    else:
        logger.error("❌ TẤT CẢ TEST ĐỀU THẤT BẠI")
        logger.error("🚨 Hệ thống có vấn đề nghiêm trọng")
    
    logger.info("")
    logger.info("💡 HƯỚNG DẪN KHẮC PHỤC:")
    logger.info("   1. Kiểm tra Kafka broker có đang chạy không")
    logger.info("   2. Kiểm tra network connectivity đến 77.37.44.237:9092")
    logger.info("   3. Kiểm tra Docker containers có đang chạy không")
    logger.info("   4. Kiểm tra ports có bị conflict không")
    logger.info("   5. Kiểm tra cấu hình Airflow connections")
    
    logger.info("")
    logger.info("🔗 TÀI NGUYÊN HỮU ÍCH:")
    logger.info("   - Kafka UI: http://localhost:8085")
    logger.info("   - Airflow UI: http://localhost:8080")
    logger.info("   - Prometheus: http://localhost:9090")
    logger.info("   - Grafana: http://localhost:3000")
    
    logger.info("")
    logger.info(f"⏰ Thời gian kết thúc: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 80)

if __name__ == "__main__":
    main() 