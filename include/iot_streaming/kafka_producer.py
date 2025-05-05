from kafka import KafkaProducer
from datetime import datetime, timedelta
import time
import logging
from json import dumps
import requests
from elasticsearch import Elasticsearch, ConnectionError
import random

# Configuration
TOPIC = "water-quality-data"
BROKERS = "77.37.44.237:9092"  # VPS Kafka address
GROUP_ID = "wqi_producer"
ES_HOST = "https://elasticsearch.anhkiet.xyz"
ES_NAME = 'elastic'
ES_PASSWORD = '6F2A0Ib+Tqm9Lti9Fpfl'

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
kafka_producer_instance = None

def get_kafka_producer():
    global kafka_producer_instance
    if kafka_producer_instance is None:
        kafka_producer_instance = create_kafka_producer()
    return kafka_producer_instance

def check_kafka_producer(**kwargs):
    global kafka_producer_instance
    if kafka_producer_instance is None:
        kafka_producer_instance = create_kafka_producer()
        # Chỉ push thông báo trạng thái đơn giản vào XCom, không phải đối tượng KafkaProducer
        kwargs['ti'].xcom_push(key='status', value='Kafka producer created successfully')
        logger.info("DAG completed successfully and is now stopped.")
    # Trả về một thông báo thay vì đối tượng KafkaProducer
    return "Kafka producer created"

def create_kafka_producer():
    """Tạo KafkaProducer với cấu hình tùy chỉnh"""
    try:
        kafka_producer = KafkaProducer(
            bootstrap_servers=BROKERS,
            value_serializer=lambda x: dumps(x).encode('utf-8'),
            acks='all',
            retries=5,
            linger_ms=10
        )
        logger.info("Kafka Producer created successfully.")
        return kafka_producer
    except Exception as ex:
        logger.error(f"Error creating Kafka Producer: {ex}")
        raise

def fetch_data_from_api(start_date):
    """Lấy dữ liệu pH và nhiệt độ từ API"""
    date_filter = start_date.strftime("%Y-%m-%d")
    API_URL = f"https://wise-bird-still.ngrok-free.app/api/ph/logs?date_filter={date_filter}&pond_id=1&utm_source=zalo&utm_medium=zalo&utm_campaign=zalo"
    
    try:
        response = requests.get(API_URL)
        response.raise_for_status()
        data = response.json()
        return data
    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to fetch data from API: {e}")
        return []

def create_elasticsearch_client():
    """Tạo Elasticsearch client và kiểm tra kết nối"""
    es_client = Elasticsearch(
        [ES_HOST],
        basic_auth=(ES_NAME, ES_PASSWORD),
    )    
    try:
        if es_client.ping():
            logger.info("Successfully connected to Elasticsearch!")
        else:
            logger.error("Failed to connect to Elasticsearch!")
            es_client = None
    except ConnectionError as e:
        logger.error(f"Elasticsearch connection error: {e}")
        es_client = None
    return es_client

def calculate_wqi(ph, temperature):
    """Tính Water Quality Index (WQI) từ pH và nhiệt độ"""
    if ph is None or temperature is None:
        return None
    try:
        # Công thức ví dụ cho WQI
        wqi = (7.0 - abs(ph - 7.0)) * (30 - abs(temperature - 25))
        return round(wqi, 2)
    except Exception as e:
        logger.error(f"Error calculating WQI: {e}")
        return None

def document_exists_in_es(es_client, measurement_time):
    """Kiểm tra xem đã có document nào có measurement_time (đầy đủ) chưa."""
    try:
        # Truy vấn Elasticsearch kiểm tra sự tồn tại của measurement_time
        query = {
            "query": {
                "term": {
                    "measurement_time": measurement_time
                }
            }
        }
        response = es_client.search(index="water_quality_logs", body=query)
        if response['hits']['total']['value'] > 0:
            logger.info(f"Document với measurement_time {measurement_time} đã tồn tại trong Elasticsearch.")
            return True
        else:
            logger.info(f"Document với measurement_time {measurement_time} chưa tồn tại trong Elasticsearch.")
            return False
    except Exception as e:
        logger.error(f"Error checking document existence: {e}")
        return False


def save_to_elasticsearch(data, es_client):
    """Lưu dữ liệu vào Elasticsearch nếu document (theo measurement_time) chưa tồn tại."""
    if es_client is None:
        logger.error("Elasticsearch client is not available. Skipping save.")
        return
    
    for entry in data:
        ph = entry.get("ph_value")
        temperature = entry.get("temperature")
        measurement_time = entry.get("measurement_time")
        
        # Chuyển đổi measurement_time thành datetime để tách date và time_hour
        if measurement_time:
            try:
                measurement_datetime = datetime.fromisoformat(measurement_time)
                measurement_date = measurement_datetime.strftime("%Y-%m-%d")
                measurement_time_hour = measurement_datetime.strftime("%H:%M:%S")
                # Sử dụng measurement_time đầy đủ (định dạng ISO) để kiểm tra trùng lặp
            except ValueError as e:
                logger.error(f"Failed to parse measurement_time {measurement_time}: {e}")
                continue
        else:
            measurement_date = None
            measurement_time_hour = None

        # Kiểm tra xem document với measurement_time này đã tồn tại chưa
        if measurement_time and document_exists_in_es(es_client, measurement_time):
            logger.info(f"Document cho measurement_time {measurement_time} đã tồn tại. Bỏ qua lưu.")
            continue

        # Tính toán WQI
        wqi = calculate_wqi(ph, temperature)

        # Tạo document để lưu vào Elasticsearch
        document = {
            "ph": ph,
            "temperature": temperature,
            "measurement_time": measurement_time,
            "measurement_date": measurement_date,
            "measurement_time_hour": measurement_time_hour,
            "wqi": wqi
        }

        try:
            # Sử dụng measurement_time làm id để đảm bảo tính duy nhất
            doc_id = measurement_time
            es_client.index(index="water_quality_logs", id=doc_id, body=document)
            logger.info(f"Document saved to Elasticsearch: {document}")
        except Exception as e:
            logger.error(f"Failed to save document to Elasticsearch: {e}")

def check_last_measurement_time_in_es(es_client):
    """Lấy measurement_time mới nhất từ Elasticsearch"""
    try:
        # Lấy một số lượng document (ở đây là 1000) và tự tìm giá trị measurement_time lớn nhất
        query = {
            "size": 1000,
            "query": {
                "match_all": {}
            },
            "_source": ["measurement_time"]
        }
        response = es_client.search(index="water_quality_logs", body=query)
        
        latest_time = None
        if response['hits']['total']['value'] > 0:
            for hit in response['hits']['hits']:
                time_str = hit['_source'].get('measurement_time')
                if time_str:
                    if latest_time is None or time_str > latest_time:
                        latest_time = time_str
            if latest_time:
                logger.info(f"Last measurement time retrieved from Elasticsearch: {latest_time}")
                return latest_time
        
        logger.info("No measurement time found in Elasticsearch.")
        return None
    except Exception as e:
        logger.error(f"Error checking last measurement time in Elasticsearch: {e}")
        return None

def kafka_producer_task(**kwargs):
    kafka_producer_obj = get_kafka_producer()
    es_client = create_elasticsearch_client()

    today = datetime.today()

    data_received = False
    retries = 0
    max_retries = 5

    # Lấy measurement_time cuối cùng trong Elasticsearch
    last_measurement_time_es = check_last_measurement_time_in_es(es_client)
    if last_measurement_time_es:
        logger.info(f"Last measurement time in Elasticsearch: {last_measurement_time_es}")
    else:
        logger.info("No measurement time found in Elasticsearch.")

    while True:
        try:
            data = fetch_data_from_api(today)
            if data:
                new_data = []
                for entry in data:
                    measurement_time = entry.get("measurement_time")
                    # Chỉ xử lý entry có measurement_time lớn hơn thời gian mới nhất đã lưu
                    if last_measurement_time_es is None or measurement_time > last_measurement_time_es:
                        new_data.append(entry)
                        # Cập nhật last_measurement_time_es với giá trị mới nhất (nếu cần)
                        if last_measurement_time_es is None or measurement_time > last_measurement_time_es:
                            last_measurement_time_es = measurement_time

                if new_data:
                    data_received = True
                    kwargs['ti'].xcom_push(key='kafka_data', value=new_data)

                    # Gửi dữ liệu mới tới Kafka và lưu vào Elasticsearch
                    for entry in new_data:
                        pH = entry.get("ph_value", None)
                        temperature = entry.get("temperature", None)
                        measurement_time = entry.get("measurement_time", None)

                        if pH is not None and temperature is not None:
                            event_message = {
                                "id": str(random.randint(1, 100000)),
                                "create_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                "measurement_time": measurement_time,
                                "ph": pH,
                                "temperature": temperature
                            }

                            logger.info(f"Sending message to Kafka topic: {TOPIC}")
                            kafka_producer_obj.send(TOPIC, event_message)
                            logger.info(f"Message successfully sent to Kafka topic: {TOPIC}")

                    # Lưu dữ liệu mới vào Elasticsearch
                    save_to_elasticsearch(new_data, es_client)

                else:
                    logger.info("No new data to process. Skipping...")
                    kwargs['ti'].xcom_push(key='status', value='success')
                    break
            else:
                retries += 1
                logger.warning(f"No data received. Retry attempt {retries}/{max_retries}.")
                if retries >= max_retries:
                    logger.info("Max retries reached. No new data available. Exiting.")
                    kwargs['ti'].xcom_push(key='status', value='success')
                    break

        except Exception as ex:
            logger.error(f"Event Message Construction Failed: {ex}")
            kwargs['ti'].xcom_push(key='status', value='failed')
            raise ex

        time.sleep(1)
    
    kafka_producer_close(kafka_producer_obj)
    kwargs['ti'].xcom_push(key='status', value='completed')
    logger.info("DAG completed successfully and is now stopped.")

def kafka_producer_close(kafka_producer_obj):
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
