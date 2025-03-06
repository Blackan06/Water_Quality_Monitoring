from kafka import KafkaProducer
from datetime import datetime
import time
import random
import logging
from json import dumps
from logs.LogService import write_log_to_postgres
# Configuration
TOPIC_NAME_CONS = "water-quality-data"
BOOTSTRAP_SERVERS_CONS = 'kafka:9092'

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_kafka_producer():
    """Tạo KafkaProducer với cấu hình tùy chỉnh"""
    return KafkaProducer(
        bootstrap_servers=BOOTSTRAP_SERVERS_CONS,
        value_serializer=lambda x: dumps(x).encode('utf-8'),
        acks='all',  # Đảm bảo dữ liệu được ghi vào ít nhất 1 broker
        retries=5,  # Retry 5 lần nếu có lỗi
        linger_ms=10  # Đợi thêm thời gian để gửi theo nhóm, tăng hiệu suất
    )

def kafka_producer_task():
    write_log_to_postgres('INFO', f"Water Quality Monitoring | Kafka Producer Application Started ...", 'kafka producer')
    # Tạo Kafka Producer
    kafka_producer_obj = create_kafka_producer()
    i = 0
    while True:
        try:
            # Giả lập dữ liệu cảm biến IoT
            pH = round(random.uniform(6.0, 8.5), 2)  # pH từ 6.0 đến 8.5
            turbidity = round(random.uniform(0.1, 5.0), 2)  # Độ đục từ 0.1 đến 5.0 NTU
            temperature = round(random.uniform(10.0, 30.0), 2)  # Nhiệt độ từ 10.0 đến 30.0°C

            # Tạo thông điệp sự kiện
            event_message = {}
            event_datetime = datetime.now()

            event_message["id"] = str(i + 1)
            event_message["create_at"] = event_datetime.strftime("%Y-%m-%d %H:%M:%S")
            event_message["ph"] = pH
            event_message["turbidity"] = turbidity
            event_message["temperature"] = temperature

            logger.info(f"Sending message to Kafka topic: {TOPIC_NAME_CONS}")
            write_log_to_postgres('INFO',f"Sending message to Kafka topic: {TOPIC_NAME_CONS}", 'kafka producer')

            logger.info(f"Message to be sent: {event_message}")
            write_log_to_postgres('INFO',f"Message to be sent: {event_message}", 'kafka producer')

            # Gửi thông điệp vào Kafka
            kafka_producer_obj.send(TOPIC_NAME_CONS, event_message)
            logger.info(f"Message successfully sent to Kafka topic: {TOPIC_NAME_CONS}")
            write_log_to_postgres('INFO',f"Message successfully sent to Kafka topic: {TOPIC_NAME_CONS}", 'kafka producer')

        except Exception as ex:
            logger.error(f"Event Message Construction Failed: {ex}")
            write_log_to_postgres('ERROR',f"Event Message Construction Failed: {ex}", 'kafka producer')

        # Chờ 1 giây trước khi gửi thông điệp tiếp theo
        time.sleep(1)
        i += 1

def kafka_producer_close(kafka_producer_obj):

    # Đảm bảo tất cả thông điệp đã được gửi và đóng Kafka producer
    try:
        logger.info("Flushing and closing Kafka producer...")
        write_log_to_postgres('INFO',f"Flushing and closing Kafka producer...", 'kafka producer')

        kafka_producer_obj.flush()
        kafka_producer_obj.close()
        logger.info("Kafka producer closed successfully.")
        write_log_to_postgres('INFO',f"Kafka producer closed successfully.", 'kafka producer')

    except Exception as ex:
        logger.error(f"Failed to flush and close Kafka producer: {ex}")
        write_log_to_postgres('ERROR',f"Failed to flush and close Kafka producer: {ex}", 'kafka producer')


    logger.info("Water Quality Monitoring | Kafka Producer Application Completed.")
    write_log_to_postgres('INFO',f"Water Quality Monitoring | Kafka Producer Application Completed.", 'kafka producer')

if __name__ == '__main__':
    kafka_producer_task()