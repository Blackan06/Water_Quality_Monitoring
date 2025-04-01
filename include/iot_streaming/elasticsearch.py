from elasticsearch import Elasticsearch, NotFoundError
import requests
import logging
from datetime import datetime, timedelta

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Elasticsearch configuration
ES_HOST = "https://elasticsearch.anhkiet.xyz"
ES_INDEX = "water_quality_logs"
ES_NAME = 'elastic'
ES_PASSWORD = '6F2A0Ib+Tqm9Lti9Fpfl'
es_client = Elasticsearch(
    [ES_HOST],
    basic_auth=(ES_NAME, ES_PASSWORD),
)

def check_and_delete_index():
    """Check if the index exists in Elasticsearch and delete if it does."""
    try:
        # Check if index exists
        if es_client.indices.exists(index=ES_INDEX):
            # Delete the index if it exists
            es_client.indices.delete(index=ES_INDEX)
            print(f"Index {ES_INDEX} deleted successfully.")
        else:
            print(f"Index {ES_INDEX} does not exist.")
            create_index()
    except NotFoundError:
        print(f"Index {ES_INDEX} does not exist.")
    except Exception as e:
        print(f"Error occurred while checking/deleting index: {e}")
        raise
def check_index():
    """Check if the index exists in Elasticsearch and delete if it does."""
    try:
        # Check if index exists
        if es_client.indices.exists(index=ES_INDEX):
            # Delete the index if it exists
            print(f"Index {ES_INDEX} connected successfully.")
        else:
            print(f"Index {ES_INDEX} does not exist.")
            create_index()
    except NotFoundError:
        print(f"Index {ES_INDEX} does not exist.")
    except Exception as e:
        print(f"Error occurred while checking/deleting index: {e}")
        raise
def create_index():
    """Tạo index với mappings trong Elasticsearch"""
    index_settings = {
        "settings": {
            "number_of_shards": 1,
            "number_of_replicas": 1
        },
        "mappings": {
            "properties": {
                "ph": {
                    "type": "float"
                },
                "temperature": {
                    "type": "float"
                },
                "measurement_time": {
                    "type": "keyword"  # Sử dụng keyword cho thời gian để có thể sắp xếp và lọc
                },
                "measurement_date": {
                    "type": "date",  # Trường ngày riêng biệt
                    "format": "yyyy-MM-dd"  # Định dạng chỉ chứa ngày
                },
                "measurement_time_hour": {
                    "type": "date",  # Sử dụng kiểu date cho giờ (với định dạng giờ phút giây)
                    "format": "HH:mm:ss"  # Định dạng giờ, phút, giây
                },
                "wqi": {
                    "type": "float"  # Trường Water Quality Index (WQI)
                }
            }
        }
    }
    # Kiểm tra nếu index chưa tồn tại, tạo mới
    if not es_client.indices.exists(index=ES_INDEX):
        es_client.indices.create(index=ES_INDEX, body=index_settings)
        logger.info(f"Index '{ES_INDEX}' created successfully.")
    else:
        logger.info(f"Index '{ES_INDEX}' already exists.")

def fetch_and_save_data_from_api():
    """Lấy dữ liệu từ API trong khoảng thời gian từ start_date đến end_date và lưu vào Elasticsearch"""
    start_date = datetime(2025, 3, 16)
    end_date = datetime(2025, 3, 30)
    current_date = start_date
    while current_date <= end_date:
        date_filter = current_date.strftime("%Y-%m-%d")
        api_url = f"https://wise-bird-still.ngrok-free.app/api/ph/logs?date_filter={date_filter}&pond_id=1&utm_source=zalo&utm_medium=zalo&utm_campaign=zalo"
        
        try:
            response = requests.get(api_url)
            response.raise_for_status()
            data = response.json()
            save_to_elasticsearch(data)
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to fetch data for {date_filter}: {e}")
        
        # Tiến tới ngày tiếp theo
        current_date += timedelta(days=1)

def calculate_wqi(ph, temperature):
    """Tính Water Quality Index (WQI) từ pH và nhiệt độ"""
    if ph is None or temperature is None:
        return None
    try:
        # Công thức tính WQI đơn giản (có thể thay đổi theo yêu cầu)
        wqi = (7.0 - abs(ph - 7.0)) * (30 - abs(temperature - 25))
        return round(wqi, 2)
    except Exception as e:
        logger.error(f"Error calculating WQI: {e}")
        return None

def save_to_elasticsearch(data):
    """Lưu dữ liệu vào Elasticsearch"""
    for entry in data:
        ph = entry.get("ph_value")
        temperature = entry.get("temperature")
        measurement_time = entry.get("measurement_time")

        # Kiểm tra nếu trường measurement_time có dữ liệu
        if measurement_time:
            # Chuyển đổi measurement_time thành đối tượng datetime
            try:
                measurement_datetime = datetime.fromisoformat(measurement_time)
                # Tách phần ngày và giờ
                measurement_date = measurement_datetime.strftime("%Y-%m-%d")
                measurement_time_hour = measurement_datetime.strftime("%H:%M:%S")
            except ValueError as e:
                logger.error(f"Failed to parse measurement_time {measurement_time}: {e}")
                continue
        else:
            measurement_date = None
            measurement_time_hour = None

        # Tính toán WQI
        wqi = calculate_wqi(ph, temperature)

        # Tạo document để lưu vào Elasticsearch
        document = {
            "ph": ph,
            "temperature": temperature,
            "measurement_time": measurement_time,
            "measurement_date": measurement_date,
            "measurement_time_hour": measurement_time_hour,
            "wqi": wqi  # Thêm WQI vào document
        }

        try:
            es_client.index(index=ES_INDEX, body=document)
            logger.info(f"Document saved to Elasticsearch: {document}")
        except Exception as e:
            logger.error(f"Failed to save document to Elasticsearch: {e}")
