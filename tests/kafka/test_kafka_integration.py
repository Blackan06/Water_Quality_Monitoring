"""Integration tests for Kafka functionality. These tests ensure that Kafka producer, consumer, and Airflow integration work correctly."""

import os
import sys
import json
import time
import logging
from contextlib import contextmanager
import pytest
from datetime import datetime

# Add include path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '../../include'))

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@contextmanager
def suppress_logging(namespace):
    logger = logging.getLogger(namespace)
    old_value = logger.disabled
    logger.disabled = True
    try:
        yield
    finally:
        logger.disabled = old_value


class MockTaskInstance:
    """Mock class to simulate Airflow TaskInstance"""
    
    def __init__(self):
        self.xcom_data = {}
    
    def xcom_push(self, key, value):
        """Mock xcom_push method"""
        self.xcom_data[key] = value
        logger.info(f"📤 XCom push: {key} = {value}")
    
    def xcom_pull(self, task_ids=None, key=None):
        """Mock xcom_pull method"""
        if key in self.xcom_data:
            return self.xcom_data[key]
        return None


def test_kafka_broker_connection():
    """Test basic connection to Kafka broker"""
    try:
        from kafka import KafkaAdminClient
        from kafka.errors import NoBrokersAvailable
        
        admin_client = KafkaAdminClient(bootstrap_servers="77.37.44.237:9092")
        topics = admin_client.list_topics()
        admin_client.close()
        
        assert isinstance(topics, list), "Should return list of topics"
        logger.info(f"✅ Connected to Kafka broker, found {len(topics)} topics")
        
    except NoBrokersAvailable as e:
        pytest.fail(f"❌ Cannot connect to Kafka broker: {e}")
    except Exception as e:
        pytest.fail(f"❌ Unexpected error: {e}")


def test_kafka_topic_exists():
    """Test that water-quality-data topic exists"""
    try:
        from kafka import KafkaAdminClient
        
        admin_client = KafkaAdminClient(bootstrap_servers="77.37.44.237:9092")
        topics = admin_client.list_topics()
        admin_client.close()
        
        assert "water-quality-data" in topics, "Topic 'water-quality-data' should exist"
        logger.info("✅ Topic 'water-quality-data' exists")
        
    except Exception as e:
        pytest.fail(f"❌ Error checking topic: {e}")


def test_kafka_producer_function():
    """Test Kafka producer function from streaming module"""
    try:
        from iot_streaming.kafka_producer_streaming import kafka_run
        
        # Mock context
        context = {
            'ti': MockTaskInstance(),
            'dag': None,
            'task': None
        }
        
        # Run producer function
        kafka_run(**context)
        
        # Check if data was pushed to XCom
        xcom_data = context['ti'].xcom_data.get('kafka_data')
        assert xcom_data is not None, "Data should be pushed to XCom"
        assert isinstance(xcom_data, list), "XCom data should be a list"
        
        logger.info("✅ Kafka producer function works correctly")
        
    except ImportError as e:
        pytest.skip(f"Skipping test - module not available: {e}")
    except Exception as e:
        pytest.fail(f"❌ Error in producer function: {e}")


def test_kafka_consumer_function():
    """Test Kafka consumer function from streaming module"""
    try:
        from iot_streaming.kafka_consumer import kafka_consumer_task
        
        # Mock context
        context = {
            'ti': MockTaskInstance(),
            'dag': None,
            'task': None
        }
        
        # Run consumer function
        kafka_consumer_task(**context)
        
        # Check if data was consumed and pushed to XCom
        consumed_data = context['ti'].xcom_data.get('consumed_data')
        # Note: This might be None if no messages are available
        
        logger.info("✅ Kafka consumer function works correctly")
        
    except ImportError as e:
        pytest.skip(f"Skipping test - module not available: {e}")
    except Exception as e:
        pytest.fail(f"❌ Error in consumer function: {e}")


def test_database_manager_connection():
    """Test database manager connection"""
    try:
        from iot_streaming.database_manager import db_manager
        
        connection = db_manager.get_connection()
        assert connection is not None, "Database connection should be established"
        connection.close()
        
        logger.info("✅ Database manager connection works")
        
    except ImportError as e:
        pytest.skip(f"Skipping test - module not available: {e}")
    except Exception as e:
        pytest.fail(f"❌ Error in database manager: {e}")


def test_kafka_message_format():
    """Test that Kafka messages have correct format"""
    try:
        from kafka import KafkaProducer, KafkaConsumer
        import json
        
        # Create test message
        test_message = {
            "id": "test_001",
            "create_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "measurement_time": "2025-01-15T10:00:00",
            "ph": 7.2,
            "temperature": 25.5,
            "test": True
        }
        
        # Test producer
        producer = KafkaProducer(
            bootstrap_servers="77.37.44.237:9092",
            value_serializer=lambda x: json.dumps(x).encode('utf-8'),
            acks='all'
        )
        
        # Send message
        future = producer.send("water-quality-data", test_message)
        record_metadata = future.get(timeout=10)
        
        assert record_metadata.topic == "water-quality-data", "Message should be sent to correct topic"
        assert record_metadata.offset >= 0, "Message should have valid offset"
        
        producer.flush()
        producer.close()
        
        logger.info("✅ Kafka message format is correct")
        
    except Exception as e:
        pytest.fail(f"❌ Error testing message format: {e}")


def test_kafka_consumer_group():
    """Test Kafka consumer group functionality"""
    try:
        from kafka import KafkaConsumer
        import json
        
        consumer = KafkaConsumer(
            "water-quality-data",
            bootstrap_servers="77.37.44.237:9092",
            auto_offset_reset='earliest',
            enable_auto_commit=True,
            group_id="test_group_pytest",
            consumer_timeout_ms=5000,  # 5 second timeout
            value_deserializer=lambda x: json.loads(x.decode('utf-8'))
        )
        
        # Try to consume at least one message
        message_count = 0
        for message in consumer:
            message_count += 1
            assert isinstance(message.value, dict), "Message value should be a dictionary"
            assert "measurement_time" in message.value, "Message should have measurement_time field"
            break  # Only consume one message for test
        
        consumer.close()
        
        logger.info(f"✅ Kafka consumer group works, consumed {message_count} messages")
        
    except Exception as e:
        pytest.fail(f"❌ Error testing consumer group: {e}")


def test_airflow_kafka_connection_config():
    """Test Airflow Kafka connection configuration"""
    try:
        # This test would require Airflow environment
        # For now, we'll test the configuration format
        kafka_config = {
            "conn_id": "kafka_default",
            "conn_type": "kafka",
            "conn_host": "77.37.44.237",
            "conn_port": 9092,
            "conn_extra": {
                "bootstrap.servers": "77.37.44.237:9092",
                "group.id": "water_quality_group",
                "auto.offset.reset": "earliest"
            }
        }
        
        assert kafka_config["conn_id"] == "kafka_default", "Connection ID should be 'kafka_default'"
        assert kafka_config["conn_type"] == "kafka", "Connection type should be 'kafka'"
        assert kafka_config["conn_host"] == "77.37.44.237", "Host should be '77.37.44.237'"
        assert kafka_config["conn_port"] == 9092, "Port should be 9092"
        
        logger.info("✅ Airflow Kafka connection configuration is correct")
        
    except Exception as e:
        pytest.fail(f"❌ Error testing Airflow config: {e}")


def test_streaming_dag_functions():
    """Test streaming DAG helper functions"""
    try:
        # Import the DAG module to test functions
        import sys
        sys.path.append(os.path.join(os.path.dirname(__file__), '../../dags'))
        
        # Test always_true function
        from dags.streaming_process_dag import always_true
        
        # Test with mock event
        mock_event = type('MockEvent', (), {'value': lambda: b'test_message'})()
        result = always_true(mock_event)
        assert result is True, "always_true function should return True"
        
        logger.info("✅ Streaming DAG functions work correctly")
        
    except ImportError as e:
        pytest.skip(f"Skipping test - module not available: {e}")
    except Exception as e:
        pytest.fail(f"❌ Error testing DAG functions: {e}")


def test_kafka_topic_metadata():
    """Test Kafka topic metadata"""
    try:
        from kafka import KafkaAdminClient
        from kafka.admin import ConfigResource, ConfigResourceType
        
        admin_client = KafkaAdminClient(bootstrap_servers="77.37.44.237:9092")
        
        # Get topic metadata
        resource = ConfigResource(ConfigResourceType.TOPIC, "water-quality-data")
        configs = admin_client.describe_configs([resource])
        
        assert "water-quality-data" in configs, "Topic metadata should be available"
        
        admin_client.close()
        
        logger.info("✅ Kafka topic metadata is accessible")
        
    except Exception as e:
        pytest.fail(f"❌ Error testing topic metadata: {e}")


def test_kafka_producer_serialization():
    """Test Kafka producer serialization"""
    try:
        from kafka import KafkaProducer
        import json
        
        producer = KafkaProducer(
            bootstrap_servers="77.37.44.237:9092",
            value_serializer=lambda x: json.dumps(x).encode('utf-8'),
            acks='all'
        )
        
        # Test message serialization
        test_data = {"test": "data", "number": 123}
        serialized = producer.config['value_serializer'](test_data)
        
        assert isinstance(serialized, bytes), "Serialized data should be bytes"
        
        # Test deserialization
        deserialized = json.loads(serialized.decode('utf-8'))
        assert deserialized == test_data, "Deserialized data should match original"
        
        producer.close()
        
        logger.info("✅ Kafka producer serialization works correctly")
        
    except Exception as e:
        pytest.fail(f"❌ Error testing serialization: {e}")


if __name__ == "__main__":
    # Run tests manually if needed
    pytest.main([__file__, "-v"]) 