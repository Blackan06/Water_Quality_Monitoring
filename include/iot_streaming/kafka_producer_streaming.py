# kafka_producer_one_record.py

from kafka import KafkaProducer, KafkaAdminClient
from kafka.admin import NewTopic, ConfigResource, ConfigResourceType
from datetime import datetime
from json import dumps
from elasticsearch import Elasticsearch
import logging

# ——— Cấu hình ———
TOPIC = "water-quality-data"
BROKERS = "kafka:9092"
ES_HOST   = "https://elasticsearch.anhkiet.xyz"
ES_USER   = "elastic"
ES_PASS   = "6F2A0Ib+Tqm9Lti9Fpfl"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_producer():
    return KafkaProducer(
        bootstrap_servers=BROKERS,
        value_serializer=lambda x: dumps(x).encode(),
        acks='all'
    )

def set_retention(topic, retention_ms=86400000):
    admin_client = KafkaAdminClient(bootstrap_servers=BROKERS)
    configs = {"retention.ms": str(retention_ms)}
    resource = ConfigResource(ConfigResourceType.TOPIC, topic, configs=configs)
    admin_client.alter_configs([resource])
    logger.info(f"Set retention {retention_ms} ms for topic '{topic}'")

def kafka_run(**kwargs):
    # Set retention (chỉ cần chạy lần đầu hoặc khi cần thiết)
    set_retention(TOPIC, retention_ms=86400000)  # giữ dữ liệu 24 giờ

    # 1 bản ghi mẫu
    rec = {
        "measurement_time": "2025-04-15T00:00:00",
        "ph":                7.2,
        "temperature":      26.5,
        "do":                8.0
    }

    # — Gửi Kafka —
    prod = get_producer()
    logger.info(f"Sending to Kafka: {rec}")
    prod.send(TOPIC, rec)
    prod.flush()

    # — Push lên XCom để downstream task / Spark job pull về —
    if 'ti' in kwargs:
        kwargs['ti'].xcom_push(key='kafka_data', value=[rec])

    logger.info("Done.")

if __name__ == "__main__":
    kafka_run()