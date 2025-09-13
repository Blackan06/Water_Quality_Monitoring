#!/usr/bin/env python3
"""
Script test kết nối Kafka và khả năng nhận message
"""

import json
import time
import logging
from kafka import KafkaProducer, KafkaConsumer, KafkaAdminClient
from kafka.admin import NewTopic
from kafka.errors import TopicAlreadyExistsError, NoBrokersAvailable
from datetime import datetime

# Cấu hình logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Cấu hình Kafka
BROKERS = "194.238.16.14:9092"
TOPIC = "water-quality-data"
GROUP_ID = "test_consumer_group"

def test_kafka_connection():
    """Test kết nối cơ bản đến Kafka broker"""
    logger.info("=== TEST 1: Kiểm tra kết nối Kafka broker ===")
    
    try:
        # Test kết nối admin client
        admin_client = KafkaAdminClient(bootstrap_servers=BROKERS)
        logger.info("✅ Kết nối admin client thành công!")
        
        # Lấy danh sách topics
        topics = admin_client.list_topics()
        logger.info(f"📋 Danh sách topics hiện có: {topics}")
        
        admin_client.close()
        return True
        
    except NoBrokersAvailable as e:
        logger.error(f"❌ Không thể kết nối đến Kafka broker: {e}")
        return False
    except Exception as e:
        logger.error(f"❌ Lỗi kết nối: {e}")
        return False

def test_topic_creation():
    """Test tạo topic mới"""
    logger.info("=== TEST 2: Kiểm tra tạo topic ===")
    
    try:
        admin_client = KafkaAdminClient(bootstrap_servers=BROKERS)
        
        # Tạo topic test
        test_topic = "test-topic-connection"
        new_topic = NewTopic(test_topic, num_partitions=1, replication_factor=1)
        
        try:
            admin_client.create_topics([new_topic])
            logger.info(f"✅ Tạo topic '{test_topic}' thành công!")
            
            # Xóa topic test
            admin_client.delete_topics([test_topic])
            logger.info(f"✅ Xóa topic '{test_topic}' thành công!")
            
        except TopicAlreadyExistsError:
            logger.info(f"ℹ️ Topic '{test_topic}' đã tồn tại")
            admin_client.delete_topics([test_topic])
            
        admin_client.close()
        return True
        
    except Exception as e:
        logger.error(f"❌ Lỗi tạo topic: {e}")
        return False

def test_producer():
    """Test producer gửi message"""
    logger.info("=== TEST 3: Kiểm tra producer gửi message ===")
    
    try:
        producer = KafkaProducer(
            bootstrap_servers=BROKERS,
            value_serializer=lambda x: json.dumps(x).encode('utf-8'),
            acks='all'
        )
        
        # Tạo message test
        test_message = {
            "id": "test_001",
            "create_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "measurement_time": "2025-01-15T10:00:00",
            "ph": 7.2,
            "temperature": 25.5,
            "test": True
        }
        
        # Gửi message
        future = producer.send(TOPIC, test_message)
        record_metadata = future.get(timeout=10)
        
        logger.info(f"✅ Gửi message thành công!")
        logger.info(f"   Topic: {record_metadata.topic}")
        logger.info(f"   Partition: {record_metadata.partition}")
        logger.info(f"   Offset: {record_metadata.offset}")
        logger.info(f"   Message: {test_message}")
        
        producer.flush()
        producer.close()
        return True
        
    except Exception as e:
        logger.error(f"❌ Lỗi producer: {e}")
        return False

def test_consumer():
    """Test consumer nhận message"""
    logger.info("=== TEST 4: Kiểm tra consumer nhận message ===")
    
    try:
        consumer = KafkaConsumer(
            TOPIC,
            bootstrap_servers=BROKERS,
            auto_offset_reset='latest',  # Chỉ đọc message mới
            enable_auto_commit=True,
            group_id=GROUP_ID,
            consumer_timeout_ms=10000,  # Timeout 10s
            value_deserializer=lambda x: json.loads(x.decode('utf-8'))
        )
        
        logger.info(f"🔍 Đang chờ message từ topic '{TOPIC}'...")
        
        message_count = 0
        start_time = time.time()
        
        for message in consumer:
            message_count += 1
            logger.info(f"✅ Nhận message #{message_count}:")
            logger.info(f"   Topic: {message.topic}")
            logger.info(f"   Partition: {message.partition}")
            logger.info(f"   Offset: {message.offset}")
            logger.info(f"   Key: {message.key}")
            logger.info(f"   Value: {message.value}")
            logger.info(f"   Timestamp: {message.timestamp}")
            
            # Chỉ đọc 1 message để test
            break
            
        if message_count == 0:
            logger.warning("⚠️ Không nhận được message nào trong 10 giây")
            logger.info("💡 Gợi ý: Chạy test_producer() trước để tạo message")
        
        consumer.close()
        return message_count > 0
        
    except Exception as e:
        logger.error(f"❌ Lỗi consumer: {e}")
        return False

def test_consumer_with_earliest():
    """Test consumer đọc từ đầu topic"""
    logger.info("=== TEST 5: Kiểm tra consumer đọc từ đầu topic ===")
    
    try:
        consumer = KafkaConsumer(
            TOPIC,
            bootstrap_servers=BROKERS,
            auto_offset_reset='earliest',  # Đọc từ đầu topic
            enable_auto_commit=True,
            group_id=f"{GROUP_ID}_earliest",
            consumer_timeout_ms=5000,  # Timeout 5s
            value_deserializer=lambda x: json.loads(x.decode('utf-8'))
        )
        
        logger.info(f"🔍 Đang đọc từ đầu topic '{TOPIC}'...")
        
        message_count = 0
        for message in consumer:
            message_count += 1
            logger.info(f"✅ Message #{message_count}: {message.value}")
            
            # Chỉ đọc 3 message đầu tiên
            if message_count >= 3:
                break
        
        if message_count == 0:
            logger.warning("⚠️ Không có message nào trong topic")
        else:
            logger.info(f"📊 Tổng cộng đọc được {message_count} message")
        
        consumer.close()
        return message_count > 0
        
    except Exception as e:
        logger.error(f"❌ Lỗi consumer earliest: {e}")
        return False

def main():
    """Chạy tất cả các test"""
    logger.info("🚀 Bắt đầu test kết nối Kafka")
    logger.info(f"📍 Kafka broker: {BROKERS}")
    logger.info(f"📝 Topic: {TOPIC}")
    logger.info("=" * 50)
    
    results = []
    
    # Test 1: Kết nối cơ bản
    results.append(("Kết nối broker", test_kafka_connection()))
    
    # Test 2: Tạo topic
    results.append(("Tạo topic", test_topic_creation()))
    
    # Test 3: Producer
    results.append(("Producer", test_producer()))
    
    # Đợi 2 giây để đảm bảo message được gửi
    time.sleep(2)
    
    # Test 4: Consumer
    results.append(("Consumer", test_consumer()))
    
    # Test 5: Consumer earliest
    results.append(("Consumer earliest", test_consumer_with_earliest()))
    
    # Tổng kết
    logger.info("=" * 50)
    logger.info("📊 KẾT QUẢ TEST:")
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        logger.info(f"   {test_name}: {status}")
        if result:
            passed += 1
    
    logger.info(f"📈 Tổng kết: {passed}/{total} test thành công")
    
    if passed == total:
        logger.info("🎉 Tất cả test đều thành công! Kafka hoạt động bình thường.")
    else:
        logger.warning("⚠️ Một số test thất bại. Vui lòng kiểm tra cấu hình Kafka.")
    
    logger.info("=" * 50)

if __name__ == "__main__":
    main() 