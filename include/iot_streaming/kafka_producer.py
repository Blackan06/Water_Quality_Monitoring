from kafka import KafkaProducer
from datetime import datetime, timedelta
import time
import random
import logging
from json import dumps
import requests
from elasticsearch import Elasticsearch, ConnectionError

# Configuration
TOPIC_NAME_CONS = "water-quality-data"
BOOTSTRAP_SERVERS_CONS = 'kafka:9092'
ES_HOST = "https://elasticsearch.anhkiet.xyz"

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_kafka_producer():
    """Tạo KafkaProducer với cấu hình tùy chỉnh"""
    return KafkaProducer(
        bootstrap_servers=BOOTSTRAP_SERVERS_CONS,
        value_serializer=lambda x: dumps(x).encode('utf-8'),
        acks='all',
        retries=5,
        linger_ms=10
    )

def fetch_data_from_api(start_date):
    """Lấy dữ liệu pH và nhiệt độ từ API"""
    # Format the date to use in the API URL
    date_filter = start_date.strftime("%Y-%m-%d")
    API_URL = f"https://wise-bird-still.ngrok-free.app/api/ph/logs?date_filter={date_filter}&pond_id=1&utm_source=zalo&utm_medium=zalo&utm_campaign=zalo"
    
    try:
        response = requests.get(API_URL)
        response.raise_for_status()  # Kiểm tra nếu API trả về lỗi
        data = response.json()  # Lấy dữ liệu JSON
        return data
    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to fetch data from API: {e}")
        return []

def create_elasticsearch_client():
    """Tạo Elasticsearch client và kiểm tra kết nối"""
    es_client = Elasticsearch([ES_HOST])
    
    try:
        # Kiểm tra xem Elasticsearch có hoạt động không
        if es_client.ping():
            logger.info("Successfully connected to Elasticsearch!")
        else:
            logger.error("Failed to connect to Elasticsearch!")
            es_client = None
    except ConnectionError as e:
        logger.error(f"Elasticsearch connection error: {e}")
        es_client = None
    return es_client

def save_to_elasticsearch(data, es_client):
    """Lưu dữ liệu vào Elasticsearch"""
    if es_client is None:
        logger.error("Elasticsearch client is not available. Skipping save.")
        return
    
    for entry in data:
        ph = entry.get("ph_value")
        temperature = entry.get("temperature")
        measurement_time = entry.get("measurement_time")
        document = {
            "ph": ph,
            "temperature": temperature,
            "measurement_time": measurement_time,
        }
        try:
            # Lưu dữ liệu vào Elasticsearch
            es_client.index(index="water_quality_logs", id=measurement_time, body=document)
            logger.info(f"Document saved to Elasticsearch: {document}")
        except Exception as e:
            logger.error(f"Failed to save document to Elasticsearch: {e}")

def kafka_producer_task(**kwargs):
    # Tạo Kafka Producer và Elasticsearch client
    kafka_producer_obj = create_kafka_producer()
    es_client = create_elasticsearch_client()
    
    # Get today's date
    today = datetime.today()

    data_received = False  # Flag to check if data is received
    retries = 0  # Counter for retry attempts
    max_retries = 5  # Maximum number of retries before stopping

    while True:
        try:
            # Fetch data for the current date
            data = fetch_data_from_api(today)
            
            if data:
                data_received = True
                # Gửi dữ liệu vào Kafka và Elasticsearch
                kwargs['ti'].xcom_push(key='kafka_data', value=data)
                
                for entry in data:
                    pH = entry.get("ph_value", None)
                    temperature = entry.get("temperature", None)
                    measurement_time = entry.get("measurement_time", None)

                    if pH is not None and temperature is not None:
                        # Tạo thông điệp sự kiện
                        event_message = {
                            "id": str(random.randint(1, 100000)),  # Random id to avoid conflicts
                            "create_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            "measurement_time": measurement_time,
                            "ph": pH,
                            "temperature": temperature
                        }

                        # Gửi thông điệp vào Kafka
                        logger.info(f"Sending message to Kafka topic: {TOPIC_NAME_CONS}")
                        logger.info(f"Message to be sent: {event_message}")
                        kafka_producer_obj.send(TOPIC_NAME_CONS, event_message)
                        logger.info(f"Message successfully sent to Kafka topic: {TOPIC_NAME_CONS}")

                        # Lưu vào Elasticsearch
                        save_to_elasticsearch(data, es_client)

                # Reset retry counter after successful data processing
                retries = 0

            else:
                # If no data is received, increment retry counter
                retries += 1
                logger.warning(f"No data received. Retry attempt {retries}/{max_retries}.")

                # If retries exceed max_retries, stop the process
                if retries >= max_retries:
                    logger.info("Max retries reached. No new data available. Exiting.")
                    kwargs['ti'].xcom_push(key='status', value='success')
                    break

        except Exception as ex:
            logger.error(f"Event Message Construction Failed: {ex}")
            kwargs['ti'].xcom_push(key='status', value='failed')
            raise ex  # This will cause the task to fail

        # Chờ 1 giây trước khi gửi thông điệp tiếp theo
        time.sleep(1)


def kafka_producer_close(kafka_producer_obj):
    # Đảm bảo tất cả thông điệp đã được gửi và đóng Kafka producer
    try:
        logger.info("Flushing and closing Kafka producer...")
        kafka_producer_obj.flush()
        kafka_producer_obj.close()
        logger.info("Kafka producer closed successfully.")
    except Exception as ex:
        logger.error(f"Failed to flush and close Kafka producer: {ex}")

    logger.info("Water Quality Monitoring | Kafka Producer Application Completed.")

if __name__ == '__main__':
    kafka_producer_task()
