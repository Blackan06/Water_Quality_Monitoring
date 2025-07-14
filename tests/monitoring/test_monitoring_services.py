#!/usr/bin/env python3
"""
Script kiểm tra Kafka UI và monitoring
"""

import requests
import json
import time
import logging
from datetime import datetime

# Cấu hình logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Cấu hình
KAFKA_UI_URL = "http://localhost:8085"
PROMETHEUS_URL = "http://localhost:9090"
GRAFANA_URL = "http://localhost:3000"
KAFKA_EXPORTER_URL = "http://localhost:9308"

def check_kafka_ui():
    """Kiểm tra Kafka UI"""
    logger.info("=== KIỂM TRA KAFKA UI ===")
    
    try:
        # Test kết nối cơ bản
        response = requests.get(f"{KAFKA_UI_URL}/api/clusters", timeout=10)
        
        if response.status_code == 200:
            logger.info("✅ Kafka UI đang hoạt động!")
            
            # Lấy thông tin cluster
            clusters = response.json()
            logger.info(f"📋 Số lượng clusters: {len(clusters)}")
            
            for cluster in clusters:
                logger.info(f"   - Cluster: {cluster.get('name', 'Unknown')}")
                logger.info(f"     Status: {cluster.get('status', 'Unknown')}")
                logger.info(f"     Bootstrap servers: {cluster.get('bootstrapServers', 'Unknown')}")
            
            # Lấy thông tin topics
            topics_response = requests.get(f"{KAFKA_UI_URL}/api/clusters/local-cluster/topics", timeout=10)
            if topics_response.status_code == 200:
                topics = topics_response.json()
                logger.info(f"📝 Số lượng topics: {len(topics)}")
                
                for topic in topics:
                    logger.info(f"   - Topic: {topic.get('name', 'Unknown')}")
                    logger.info(f"     Partitions: {topic.get('partitions', 'Unknown')}")
                    logger.info(f"     Replication: {topic.get('replicationFactor', 'Unknown')}")
            
            return True
        else:
            logger.error(f"❌ Kafka UI trả về status code: {response.status_code}")
            return False
            
    except requests.exceptions.ConnectionError:
        logger.error("❌ Không thể kết nối đến Kafka UI")
        logger.info("💡 Đảm bảo Kafka UI đang chạy trên port 8085")
        return False
    except Exception as e:
        logger.error(f"❌ Lỗi kiểm tra Kafka UI: {e}")
        return False

def check_prometheus():
    """Kiểm tra Prometheus"""
    logger.info("=== KIỂM TRA PROMETHEUS ===")
    
    try:
        # Test kết nối cơ bản
        response = requests.get(f"{PROMETHEUS_URL}/api/v1/status/config", timeout=10)
        
        if response.status_code == 200:
            logger.info("✅ Prometheus đang hoạt động!")
            
            # Lấy thông tin targets
            targets_response = requests.get(f"{PROMETHEUS_URL}/api/v1/targets", timeout=10)
            if targets_response.status_code == 200:
                targets_data = targets_response.json()
                targets = targets_data.get('data', {}).get('activeTargets', [])
                
                logger.info(f"🎯 Số lượng targets: {len(targets)}")
                
                for target in targets:
                    logger.info(f"   - Target: {target.get('labels', {}).get('job', 'Unknown')}")
                    logger.info(f"     URL: {target.get('scrapeUrl', 'Unknown')}")
                    logger.info(f"     Health: {target.get('health', 'Unknown')}")
            
            return True
        else:
            logger.error(f"❌ Prometheus trả về status code: {response.status_code}")
            return False
            
    except requests.exceptions.ConnectionError:
        logger.error("❌ Không thể kết nối đến Prometheus")
        logger.info("💡 Đảm bảo Prometheus đang chạy trên port 9090")
        return False
    except Exception as e:
        logger.error(f"❌ Lỗi kiểm tra Prometheus: {e}")
        return False

def check_grafana():
    """Kiểm tra Grafana"""
    logger.info("=== KIỂM TRA GRAFANA ===")
    
    try:
        # Test kết nối cơ bản
        response = requests.get(f"{GRAFANA_URL}/api/health", timeout=10)
        
        if response.status_code == 200:
            health_data = response.json()
            logger.info("✅ Grafana đang hoạt động!")
            logger.info(f"   Database: {health_data.get('database', 'Unknown')}")
            logger.info(f"   Version: {health_data.get('version', 'Unknown')}")
            
            # Kiểm tra datasources
            datasources_response = requests.get(f"{GRAFANA_URL}/api/datasources", timeout=10)
            if datasources_response.status_code == 200:
                datasources = datasources_response.json()
                logger.info(f"📊 Số lượng datasources: {len(datasources)}")
                
                for ds in datasources:
                    logger.info(f"   - Datasource: {ds.get('name', 'Unknown')}")
                    logger.info(f"     Type: {ds.get('type', 'Unknown')}")
                    logger.info(f"     URL: {ds.get('url', 'Unknown')}")
            
            return True
        else:
            logger.error(f"❌ Grafana trả về status code: {response.status_code}")
            return False
            
    except requests.exceptions.ConnectionError:
        logger.error("❌ Không thể kết nối đến Grafana")
        logger.info("💡 Đảm bảo Grafana đang chạy trên port 3000")
        return False
    except Exception as e:
        logger.error(f"❌ Lỗi kiểm tra Grafana: {e}")
        return False

def check_kafka_exporter():
    """Kiểm tra Kafka Exporter"""
    logger.info("=== KIỂM TRA KAFKA EXPORTER ===")
    
    try:
        # Test kết nối cơ bản
        response = requests.get(f"{KAFKA_EXPORTER_URL}/metrics", timeout=10)
        
        if response.status_code == 200:
            logger.info("✅ Kafka Exporter đang hoạt động!")
            
            # Parse metrics để tìm thông tin Kafka
            metrics = response.text
            lines = metrics.split('\n')
            
            kafka_metrics = []
            for line in lines:
                if line.startswith('kafka_'):
                    kafka_metrics.append(line.split(' ')[0])
            
            logger.info(f"📈 Số lượng Kafka metrics: {len(kafka_metrics)}")
            
            # Hiển thị một số metrics quan trọng
            important_metrics = [
                'kafka_brokers',
                'kafka_topic_partitions',
                'kafka_consumer_group_members',
                'kafka_consumer_lag_sum'
            ]
            
            for metric in important_metrics:
                for line in lines:
                    if line.startswith(metric):
                        logger.info(f"   - {line}")
                        break
            
            return True
        else:
            logger.error(f"❌ Kafka Exporter trả về status code: {response.status_code}")
            return False
            
    except requests.exceptions.ConnectionError:
        logger.error("❌ Không thể kết nối đến Kafka Exporter")
        logger.info("💡 Đảm bảo Kafka Exporter đang chạy trên port 9308")
        return False
    except Exception as e:
        logger.error(f"❌ Lỗi kiểm tra Kafka Exporter: {e}")
        return False

def check_docker_services():
    """Kiểm tra các Docker services"""
    logger.info("=== KIỂM TRA DOCKER SERVICES ===")
    
    try:
        import subprocess
        
        # Kiểm tra các containers đang chạy
        result = subprocess.run(
            ['docker', 'ps', '--format', 'table {{.Names}}\t{{.Status}}\t{{.Ports}}'],
            capture_output=True,
            text=True,
            timeout=10
        )
        
        if result.returncode == 0:
            logger.info("✅ Docker đang hoạt động!")
            
            lines = result.stdout.strip().split('\n')
            if len(lines) > 1:  # Có header
                logger.info("🐳 Các containers đang chạy:")
                for line in lines[1:]:  # Bỏ qua header
                    if line.strip():
                        parts = line.split('\t')
                        if len(parts) >= 3:
                            name, status, ports = parts[0], parts[1], parts[2]
                            logger.info(f"   - {name}: {status} ({ports})")
            else:
                logger.warning("⚠️ Không có containers nào đang chạy")
            
            return True
        else:
            logger.error(f"❌ Lỗi chạy docker ps: {result.stderr}")
            return False
            
    except FileNotFoundError:
        logger.error("❌ Docker không được cài đặt hoặc không có trong PATH")
        return False
    except Exception as e:
        logger.error(f"❌ Lỗi kiểm tra Docker services: {e}")
        return False

def main():
    """Chạy tất cả các kiểm tra"""
    logger.info("🚀 Bắt đầu kiểm tra Kafka Monitoring")
    logger.info("=" * 60)
    
    results = []
    
    # Kiểm tra các services
    results.append(("Docker Services", check_docker_services()))
    results.append(("Kafka UI", check_kafka_ui()))
    results.append(("Prometheus", check_prometheus()))
    results.append(("Grafana", check_grafana()))
    results.append(("Kafka Exporter", check_kafka_exporter()))
    
    # Tổng kết
    logger.info("=" * 60)
    logger.info("📊 KẾT QUẢ KIỂM TRA MONITORING:")
    
    passed = 0
    total = len(results)
    
    for service_name, result in results:
        status = "✅ ONLINE" if result else "❌ OFFLINE"
        logger.info(f"   {service_name}: {status}")
        if result:
            passed += 1
    
    logger.info(f"📈 Tổng kết: {passed}/{total} services đang hoạt động")
    
    if passed == total:
        logger.info("🎉 Tất cả monitoring services đều hoạt động bình thường!")
    else:
        logger.warning("⚠️ Một số services không hoạt động. Vui lòng kiểm tra:")
        logger.info("   1. Docker containers có đang chạy không?")
        logger.info("   2. Ports có bị conflict không?")
        logger.info("   3. Network configuration có đúng không?")
    
    logger.info("=" * 60)
    logger.info("🔗 URLs để truy cập:")
    logger.info(f"   Kafka UI: {KAFKA_UI_URL}")
    logger.info(f"   Prometheus: {PROMETHEUS_URL}")
    logger.info(f"   Grafana: {GRAFANA_URL}")
    logger.info(f"   Kafka Exporter: {KAFKA_EXPORTER_URL}")

if __name__ == "__main__":
    main() 