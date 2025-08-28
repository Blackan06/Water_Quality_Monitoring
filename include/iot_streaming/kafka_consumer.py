# kafka_consumer_one_record.py

from kafka import KafkaConsumer, TopicPartition
from json import loads
import logging
from datetime import datetime
import sys
import os

# Add paths for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
from include.iot_streaming.database_manager import db_manager

# ——— Cấu hình ———
TOPIC    = "water-quality-data"
# Ưu tiên lấy từ env, mặc định dùng địa chỉ VPS public để tiện client ngoài
BROKERS  = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "77.37.44.237:19092")
GROUP_ID = "wqi_consumer_one"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_kafka_offset_info():
    """Lấy thông tin offset hiện tại của consumer group"""
    try:
        consumer = KafkaConsumer(
            TOPIC,
            bootstrap_servers=BROKERS,
            group_id=GROUP_ID,
            enable_auto_commit=False,
            value_deserializer=lambda x: loads(x.decode('utf-8'))
        )
        
        # Lấy thông tin partitions
        partitions = consumer.partitions_for_topic(TOPIC)
        if not partitions:
            logger.warning(f"No partitions found for topic {TOPIC}")
            consumer.close()
            return None
        
        offset_info = {}
        for partition in partitions:
            try:
                # Đảm bảo partition là integer
                partition_id = int(partition)
                tp = TopicPartition(TOPIC, partition_id)
                
                # Lấy committed offset
                committed = consumer.committed([tp])
                committed_offset = committed[tp] if committed[tp] is not None else -1
                
                # Lấy beginning offset
                beginning = consumer.beginning_offsets([tp])
                beginning_offset = beginning[tp]
                
                # Lấy end offset
                end = consumer.end_offsets([tp])
                end_offset = end[tp]
                
                offset_info[partition_id] = {
                    'committed_offset': committed_offset,
                    'beginning_offset': beginning_offset,
                    'end_offset': end_offset,
                    'lag': end_offset - committed_offset if committed_offset >= 0 else 0
                }
            except (ValueError, TypeError) as e:
                logger.warning(f"Invalid partition {partition}: {e}")
                continue
        
        consumer.close()
        return offset_info
        
    except Exception as e:
        logger.error(f"Error getting offset info: {e}")
        return None

def reset_kafka_offset(reset_to='earliest'):
    """Reset offset của consumer group"""
    try:
        consumer = KafkaConsumer(
            TOPIC,
            bootstrap_servers=BROKERS,
            group_id=GROUP_ID,
            enable_auto_commit=False,
            value_deserializer=lambda x: loads(x.decode('utf-8'))
        )
        
        partitions = consumer.partitions_for_topic(TOPIC)
        if not partitions:
            logger.warning(f"No partitions found for topic {TOPIC}")
            consumer.close()
            return False
        
        topic_partitions = [TopicPartition(TOPIC, p) for p in partitions]
        
        if reset_to == 'earliest':
            # Reset về offset đầu tiên
            beginning_offsets = consumer.beginning_offsets(topic_partitions)
            for tp in topic_partitions:
                consumer.seek(tp, beginning_offsets[tp])
            logger.info(f"Reset offset to earliest for {len(partitions)} partitions")
        elif reset_to == 'latest':
            # Reset về offset cuối cùng
            end_offsets = consumer.end_offsets(topic_partitions)
            for tp in topic_partitions:
                consumer.seek(tp, end_offsets[tp])
            logger.info(f"Reset offset to latest for {len(partitions)} partitions")
        
        # Commit offset mới
        consumer.commit()
        consumer.close()
        return True
        
    except Exception as e:
        logger.error(f"Error resetting offset: {e}")
        return False

def kafka_consumer_task(**kwargs):
    """
    - Mở kết nối tới Kafka
    - Đọc nhiều bản ghi (batch processing)
    - Lưu vào database raw_sensor_data
    - Commit offset sau khi xử lý thành công
    - Đẩy giá trị đó lên XCom
    """
    logger.info("Kafka Consumer (batch processing) started…")

    consumer = KafkaConsumer(
        TOPIC,
        bootstrap_servers=BROKERS,
        auto_offset_reset='earliest',  # nếu muốn đọc từ đầu
        enable_auto_commit=False,      # Tắt auto-commit để manual control
        group_id=GROUP_ID,
        consumer_timeout_ms=10000,     # timeout 10s nếu không có message
        value_deserializer=lambda x: loads(x.decode('utf-8'))
    )

    records = []
    messages = []
    processed_count = 0
    error_count = 0
    
    try:
        # Đọc tất cả messages có sẵn trong batch
        for msg in consumer:
            record = msg.value
            messages.append(msg)
            records.append(record)
            logger.info(f"Consumed record {len(records)}: {record}")
            
            # Giới hạn batch size để tránh memory issues
            if len(records) >= 50:  # Process tối đa 50 messages mỗi batch
                logger.info(f"Batch size limit reached ({len(records)} records)")
                break
                
    except Exception as e:
        logger.error(f"Error while consuming: {e}")
        consumer.close()
        return f"Error consuming: {str(e)}"

    if not records:
        logger.warning("No records consumed within timeout.")
        consumer.close()
        return "No records consumed"
    
    logger.info(f"📦 Processing batch of {len(records)} messages")
    
    # Xử lý từng record trong batch
    for i, (record, message) in enumerate(zip(records, messages)):
        try:
            # Parse dữ liệu từ Kafka message
            station_id = int(record.get('station_id'))
            measurement_time_str = record.get('measurement_time')
            ph = record.get('ph')
            temperature = record.get('temperature')
            do = record.get('do')
            
            # Convert measurement_time string to datetime
            if measurement_time_str:
                # Handle different time formats
                if measurement_time_str.endswith('Z'):
                    measurement_time_str = measurement_time_str[:-1]  # Remove 'Z'
                try:
                    measurement_time = datetime.fromisoformat(measurement_time_str)
                except ValueError:
                    logger.warning(f"Invalid measurement_time format: {measurement_time_str}, using current time")
                    measurement_time = datetime.now()
            else:
                measurement_time = datetime.now()
            
            # Validate required fields
            missing_fields = []
            if ph is None:
                missing_fields.append('ph')
            if temperature is None:
                missing_fields.append('temperature')
            if do is None:
                missing_fields.append('do')
            
            if missing_fields:
                logger.warning(f"Missing required fields: {missing_fields} in record {i+1}: {record}")
                error_count += 1
                continue  # Skip invalid record, continue with next
            
            # Tạo station_id mặc định nếu không có
            if station_id is None:
                station_id = "unknown_station"
                logger.info(f"⚠️ No station_id provided for record {i+1}, using default: {station_id}")
            else:
                # Đảm bảo station_id là integer
                try:
                    station_id = int(station_id)
                except (ValueError, TypeError):
                    logger.warning(f"Invalid station_id format for record {i+1}: {station_id}, using default: 0")
                    station_id = 0
            
            # Prepare data for database
            raw_data = {
                'station_id': station_id,
                'measurement_time': measurement_time,
                'ph': float(ph),
                'temperature': float(temperature),
                'do': float(do)
            }
            
            # Insert into raw_sensor_data
            success = db_manager.insert_raw_data(raw_data)
            
            if success:
                logger.info(f"✅ Successfully saved record {i+1} to database: Station {station_id}, Time {measurement_time}")
                processed_count += 1
            else:
                logger.error(f"❌ Failed to save record {i+1} to database")
                error_count += 1
                
        except Exception as e:
            logger.error(f"❌ Error processing record {i+1}: {e}")
            error_count += 1
            continue
    
    # Commit offset cho tất cả messages đã xử lý
    if messages:
        try:
            consumer.commit()
            logger.info(f"✅ Successfully committed offset for batch: {len(messages)} messages")
        except Exception as commit_error:
            logger.error(f"❌ Failed to commit offset: {commit_error}")
    
    consumer.close()
    
    # Đẩy thông tin batch lên XCom cho downstream task
    if 'ti' in kwargs:
        kwargs['ti'].xcom_push(key='batch_size', value=len(records))
        kwargs['ti'].xcom_push(key='processed_count', value=processed_count)
        kwargs['ti'].xcom_push(key='error_count', value=error_count)
        kwargs['ti'].xcom_push(key='saved_to_db', value=processed_count > 0)
        
        # Lưu thông tin offset của message cuối cùng
        if messages:
            last_message = messages[-1]
            kwargs['ti'].xcom_push(key='kafka_offset', value={
                'partition': last_message.partition,
                'offset': last_message.offset,
                'topic': last_message.topic
            })
    
    logger.info(f"📊 Batch processing completed: {processed_count} processed, {error_count} errors out of {len(records)} total")
    return f"Batch processed: {processed_count} successful, {error_count} errors out of {len(records)} total"

if __name__ == "__main__":
    kafka_consumer_task()
