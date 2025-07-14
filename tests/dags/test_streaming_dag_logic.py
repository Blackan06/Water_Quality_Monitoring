"""Test logic for streaming DAG with AwaitMessageTriggerFunctionSensor. This test ensures that the DAG logic works correctly."""

import os
import sys
import json
import logging
from contextlib import contextmanager
import pytest
from datetime import datetime

# Add paths for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '../../dags'))
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


class MockEvent:
    """Mock Kafka event"""
    
    def __init__(self, message):
        self._message = message
    
    def value(self):
        return json.dumps(self._message).encode('utf-8')


def test_notify_trigger_function():
    """Test notify_trigger function"""
    try:
        from dags.streaming_process_dag import notify_trigger
        
        # Create mock context
        context = {
            'ti': MockTaskInstance(),
            'dag': None,
            'task': None
        }
        
        # Create mock event
        test_message = {
            "id": "test_001",
            "measurement_time": "2025-01-15T10:00:00",
            "ph": 7.2,
            "temperature": 25.5
        }
        mock_event = MockEvent(test_message)
        
        # Test notify_trigger function
        notify_trigger(mock_event, **context)
        
        # Check if message was pushed to XCom
        xcom_message = context['ti'].xcom_data.get('kafka_message')
        assert xcom_message is not None, "Message should be pushed to XCom"
        
        # Parse the message
        parsed_message = json.loads(xcom_message)
        assert parsed_message['id'] == 'test_001', "Message ID should match"
        assert parsed_message['ph'] == 7.2, "pH value should match"
        
        logger.info("✅ notify_trigger function works correctly")
        
    except ImportError as e:
        pytest.skip(f"Skipping test - module not available: {e}")
    except Exception as e:
        pytest.fail(f"❌ Error testing notify_trigger: {e}")


def test_enhanced_kafka_consumer_task():
    """Test enhanced_kafka_consumer_task function"""
    try:
        from dags.streaming_process_dag import enhanced_kafka_consumer_task
        
        # Create mock context with XCom data
        context = {
            'ti': MockTaskInstance(),
            'dag': None,
            'task': None
        }
        
        # Simulate XCom data from sensor
        context['ti'].xcom_data['kafka_message'] = json.dumps({
            "id": "test_001",
            "measurement_time": "2025-01-15T10:00:00",
            "ph": 7.2,
            "temperature": 25.5
        })
        
        # Test enhanced consumer task
        result = enhanced_kafka_consumer_task(**context)
        
        assert result is not None, "Enhanced consumer task should return result"
        logger.info("✅ enhanced_kafka_consumer_task works correctly")
        
    except ImportError as e:
        pytest.skip(f"Skipping test - module not available: {e}")
    except Exception as e:
        pytest.fail(f"❌ Error testing enhanced consumer task: {e}")


def test_always_true_function():
    """Test always_true function"""
    try:
        from dags.streaming_process_dag import always_true
        
        # Create mock event
        mock_event = MockEvent({"test": "data"})
        
        # Test always_true function
        result = always_true(mock_event)
        
        assert result is True, "always_true function should return True"
        logger.info("✅ always_true function works correctly")
        
    except ImportError as e:
        pytest.skip(f"Skipping test - module not available: {e}")
    except Exception as e:
        pytest.fail(f"❌ Error testing always_true: {e}")


def test_trigger_ml_pipeline_after_processing():
    """Test trigger_ml_pipeline_after_processing function"""
    try:
        from dags.streaming_process_dag import trigger_ml_pipeline_after_processing
        
        # Create mock context
        context = {
            'ti': MockTaskInstance(),
            'dag': None,
            'task': None
        }
        
        # Test trigger ML pipeline function
        result = trigger_ml_pipeline_after_processing(**context)
        
        assert result is not None, "Function should return result"
        logger.info("✅ trigger_ml_pipeline_after_processing works correctly")
        
    except ImportError as e:
        pytest.skip(f"Skipping test - module not available: {e}")
    except Exception as e:
        pytest.fail(f"❌ Error testing trigger ML pipeline: {e}")


def test_xcom_communication():
    """Test XCom communication between tasks"""
    try:
        from dags.streaming_process_dag import notify_trigger, enhanced_kafka_consumer_task
        
        # Create mock context
        context = {
            'ti': MockTaskInstance(),
            'dag': None,
            'task': None
        }
        
        # Simulate sensor receiving message
        test_message = {
            "id": "test_002",
            "measurement_time": "2025-01-15T11:00:00",
            "ph": 6.8,
            "temperature": 24.0
        }
        mock_event = MockEvent(test_message)
        
        # Step 1: Sensor receives message and pushes to XCom
        notify_trigger(mock_event, **context)
        
        # Check XCom data
        xcom_message = context['ti'].xcom_data.get('kafka_message')
        assert xcom_message is not None, "Message should be in XCom"
        
        # Step 2: Consumer task reads from XCom
        result = enhanced_kafka_consumer_task(**context)
        
        assert result is not None, "Consumer task should process message"
        logger.info("✅ XCom communication works correctly")
        
    except ImportError as e:
        pytest.skip(f"Skipping test - module not available: {e}")
    except Exception as e:
        pytest.fail(f"❌ Error testing XCom communication: {e}")


def test_dag_task_dependencies():
    """Test DAG task dependencies"""
    try:
        # Import DAG functions
        from dags.streaming_process_dag import streaming_process_dag
        
        # Get the DAG
        dag = streaming_process_dag()
        
        # Check that all expected tasks exist
        expected_tasks = ['wait_for_kafka', 'kafka_consumer_task', 'trigger_ml_pipeline_after_processing']
        
        for task_id in expected_tasks:
            assert task_id in dag.task_dict, f"Task {task_id} should exist in DAG"
        
        # Check task dependencies
        wait_for_kafka = dag.task_dict['wait_for_kafka']
        consumer_task = dag.task_dict['kafka_consumer_task']
        trigger_task = dag.task_dict['trigger_ml_pipeline_after_processing']
        
        # Check that consumer_task depends on wait_for_kafka
        assert consumer_task in wait_for_kafka.downstream_list, "Consumer task should depend on sensor"
        
        # Check that trigger_task depends on consumer_task
        assert trigger_task in consumer_task.downstream_list, "Trigger task should depend on consumer task"
        
        # Check trigger rules
        assert consumer_task.trigger_rule == 'all_done', "Consumer task should have trigger_rule='all_done'"
        assert trigger_task.trigger_rule == 'all_done', "Trigger task should have trigger_rule='all_done'"
        
        logger.info("✅ DAG task dependencies are correct")
        
    except ImportError as e:
        pytest.skip(f"Skipping test - module not available: {e}")
    except Exception as e:
        pytest.fail(f"❌ Error testing DAG dependencies: {e}")


def test_sensor_configuration():
    """Test sensor configuration"""
    try:
        from dags.streaming_process_dag import streaming_process_dag
        
        # Get the DAG
        dag = streaming_process_dag()
        
        # Get the sensor task
        sensor_task = dag.task_dict['wait_for_kafka']
        
        # Check sensor configuration
        assert sensor_task.kafka_config_id == 'kafka_default', "Kafka config ID should be 'kafka_default'"
        assert 'water-quality-data' in sensor_task.topics, "Topic should be 'water-quality-data'"
        assert sensor_task.poll_timeout == 1, "Poll timeout should be 1"
        assert sensor_task.poll_interval == 10, "Poll interval should be 10"
        assert sensor_task.poke_interval == 10, "Poke interval should be 10"
        assert sensor_task.mode == 'poke', "Mode should be 'poke'"
        
        logger.info("✅ Sensor configuration is correct")
        
    except ImportError as e:
        pytest.skip(f"Skipping test - module not available: {e}")
    except Exception as e:
        pytest.fail(f"❌ Error testing sensor configuration: {e}")


def test_message_flow():
    """Test complete message flow through the DAG"""
    try:
        from dags.streaming_process_dag import notify_trigger, enhanced_kafka_consumer_task, trigger_ml_pipeline_after_processing
        
        # Create mock context
        context = {
            'ti': MockTaskInstance(),
            'dag': None,
            'task': None
        }
        
        # Step 1: Simulate Kafka message arrival
        test_message = {
            "id": "test_003",
            "measurement_time": "2025-01-15T12:00:00",
            "ph": 7.5,
            "temperature": 26.2
        }
        mock_event = MockEvent(test_message)
        
        # Step 2: Sensor processes message
        notify_trigger(mock_event, **context)
        
        # Step 3: Consumer task processes message
        consumer_result = enhanced_kafka_consumer_task(**context)
        
        # Step 4: Trigger ML pipeline
        trigger_result = trigger_ml_pipeline_after_processing(**context)
        
        # Verify flow
        assert context['ti'].xcom_data.get('kafka_message') is not None, "Message should be in XCom"
        assert consumer_result is not None, "Consumer should process message"
        assert trigger_result is not None, "ML pipeline should be triggered"
        
        logger.info("✅ Complete message flow works correctly")
        
    except ImportError as e:
        pytest.skip(f"Skipping test - module not available: {e}")
    except Exception as e:
        pytest.fail(f"❌ Error testing message flow: {e}")


if __name__ == "__main__":
    # Run tests manually if needed
    pytest.main([__file__, "-v"]) 