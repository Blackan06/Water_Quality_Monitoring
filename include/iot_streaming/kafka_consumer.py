# kafka_consumer_one_record.py

from kafka import KafkaConsumer
from json import loads
import logging

# ——— Cấu hình ———
TOPIC    = "water-quality-data"
BROKERS  = "kafka:9092"
GROUP_ID = "wqi_consumer_one"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def kafka_consumer_task(**kwargs):
    """
    - Mở kết nối tới Kafka
    - Đọc một bản ghi đầu tiên (sẽ là của ngày 2025-04-15 nếu producer đã gửi đúng)
    - Đẩy giá trị đó lên XCom
    """
    logger.info("Kafka Consumer (single record) started…")

    consumer = KafkaConsumer(
        TOPIC,
        bootstrap_servers=BROKERS,
        auto_offset_reset='earliest',  # nếu muốn đọc từ đầu
        enable_auto_commit=False,
        group_id=GROUP_ID,
        consumer_timeout_ms=5000,       # timeout 5s nếu không có message
        value_deserializer=lambda x: loads(x.decode('utf-8'))
    )

    record = None
    try:
        for msg in consumer:
            record = msg.value
            logger.info(f"Consumed one record: {record}")
            break
    except Exception as e:
        logger.error(f"Error while consuming: {e}")
    finally:
        consumer.close()

    if record is None:
        logger.warning("No record consumed within timeout.")
    else:
        # Đẩy lên XCom cho downstream task
        if 'ti' in kwargs:
            kwargs['ti'].xcom_push(key='consumed_data', value=record)
        logger.info("Pushed record to XCom under key='consumed_data'")

    logger.info("Kafka Consumer (single record) completed.")

if __name__ == "__main__":
    kafka_consumer_task()
