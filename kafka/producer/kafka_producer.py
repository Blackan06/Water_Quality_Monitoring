from kafka import KafkaProducer
from datetime import datetime
import time
import random
import logging
from json import dumps
from logs.LogService import write_log_to_postgres

# Configuration
TOPIC_PH = "water-quality-ph"
TOPIC_TURBIDITY = "water-quality-turbidity"
TOPIC_TEMPERATURE = "water-quality-temperature"
BOOTSTRAP_SERVERS = 'kafka:9092'

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_kafka_producer():
    """Tạo KafkaProducer với cấu hình tùy chỉnh"""
    return KafkaProducer(
        bootstrap_servers=BOOTSTRAP_SERVERS,
        value_serializer=lambda x: dumps(x).encode('utf-8'),
        acks='all',  
        retries=5,  
        linger_ms=10  
    )

def kafka_producer_task():
    write_log_to_postgres('INFO', "Water Quality Monitoring | Kafka Producer Application Started ...", 'kafka producer')
    
    kafka_producer_obj = create_kafka_producer()
    i = 0

    while True:
        try:
            # Giả lập dữ liệu cảm biến
            pH = round(random.uniform(6.0, 8.5), 2)
            turbidity = round(random.uniform(0.1, 5.0), 2)
            temperature = round(random.uniform(10.0, 30.0), 2)
            event_datetime = datetime.now()

            # Tạo message riêng cho từng topic
            ph_message = {
                "id": str(i + 1),
                "create_at": event_datetime.strftime("%Y-%m-%d %H:%M:%S"),
                "ph": pH
            }

            turbidity_message = {
                "id": str(i + 1),
                "create_at": event_datetime.strftime("%Y-%m-%d %H:%M:%S"),
                "turbidity": turbidity
            }

            temperature_message = {
                "id": str(i + 1),
                "create_at": event_datetime.strftime("%Y-%m-%d %H:%M:%S"),
                "temperature": temperature
            }

            # Gửi dữ liệu đến từng topic
            kafka_producer_obj.send(TOPIC_PH, ph_message)
            kafka_producer_obj.send(TOPIC_TURBIDITY, turbidity_message)
            kafka_producer_obj.send(TOPIC_TEMPERATURE, temperature_message)

            logger.info(f"Sent data to Kafka: {ph_message} -> {TOPIC_PH}")
            logger.info(f"Sent data to Kafka: {turbidity_message} -> {TOPIC_TURBIDITY}")
            logger.info(f"Sent data to Kafka: {temperature_message} -> {TOPIC_TEMPERATURE}")

            write_log_to_postgres('INFO', f"Sent data to Kafka: {ph_message} -> {TOPIC_PH}", 'kafka producer')
            write_log_to_postgres('INFO', f"Sent data to Kafka: {turbidity_message} -> {TOPIC_TURBIDITY}", 'kafka producer')
            write_log_to_postgres('INFO', f"Sent data to Kafka: {temperature_message} -> {TOPIC_TEMPERATURE}", 'kafka producer')

        except Exception as ex:
            logger.error(f"Event Message Construction Failed: {ex}")
            write_log_to_postgres('ERROR', f"Event Message Construction Failed: {ex}", 'kafka producer')

        time.sleep(1)
        i += 1

def kafka_producer_close(kafka_producer_obj):
    try:
        logger.info("Flushing and closing Kafka producer...")
        write_log_to_postgres('INFO', "Flushing and closing Kafka producer...", 'kafka producer')

        kafka_producer_obj.flush()
        kafka_producer_obj.close()
        logger.info("Kafka producer closed successfully.")
        write_log_to_postgres('INFO', "Kafka producer closed successfully.", 'kafka producer')

    except Exception as ex:
        logger.error(f"Failed to flush and close Kafka producer: {ex}")
        write_log_to_postgres('ERROR', f"Failed to flush and close Kafka producer: {ex}", 'kafka producer')

    logger.info("Water Quality Monitoring | Kafka Producer Application Completed.")
    write_log_to_postgres('INFO', "Water Quality Monitoring | Kafka Producer Application Completed.", 'kafka producer')

if __name__ == '__main__':
    kafka_producer_task()
