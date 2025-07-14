"""Test for streaming DAG with AwaitMessageTriggerFunctionSensor. This test ensures that the streaming DAG can be imported, has proper configuration, and the Kafka sensor is properly configured."""

import os
import logging
from contextlib import contextmanager
import pytest
from airflow.models import DagBag
from airflow.providers.apache.kafka.sensors.kafka import AwaitMessageTriggerFunctionSensor
from airflow.operators.python import PythonOperator
from airflow.operators.trigger_dagrun import TriggerDagRunOperator


@contextmanager
def suppress_logging(namespace):
    logger = logging.getLogger(namespace)
    old_value = logger.disabled
    logger.disabled = True
    try:
        yield
    finally:
        logger.disabled = old_value


def get_streaming_dag():
    """
    Get the streaming_process_dag specifically
    """
    with suppress_logging("airflow"):
        dag_bag = DagBag(include_examples=False)
        
    # Find the streaming DAG
    streaming_dag = None
    for dag_id, dag in dag_bag.dags.items():
        if 'streaming' in dag_id.lower():
            streaming_dag = dag
            break
    
    return streaming_dag


def test_streaming_dag_import():
    """Test that streaming DAG can be imported without errors"""
    with suppress_logging("airflow"):
        dag_bag = DagBag(include_examples=False)
    
    # Check for import errors
    import_errors = dag_bag.import_errors
    streaming_errors = {k: v for k, v in import_errors.items() if 'streaming' in k.lower()}
    
    assert not streaming_errors, f"Streaming DAG has import errors: {streaming_errors}"


def test_streaming_dag_exists():
    """Test that streaming DAG exists in the DagBag"""
    streaming_dag = get_streaming_dag()
    assert streaming_dag is not None, "Streaming DAG not found in DagBag"


def test_streaming_dag_has_kafka_sensor():
    """Test that streaming DAG has AwaitMessageTriggerFunctionSensor"""
    streaming_dag = get_streaming_dag()
    assert streaming_dag is not None, "Streaming DAG not found"
    
    # Find Kafka sensor task
    kafka_sensor = None
    for task_id, task in streaming_dag.tasks.items():
        if isinstance(task, AwaitMessageTriggerFunctionSensor):
            kafka_sensor = task
            break
    
    assert kafka_sensor is not None, "AwaitMessageTriggerFunctionSensor not found in streaming DAG"


def test_kafka_sensor_configuration():
    """Test Kafka sensor configuration"""
    streaming_dag = get_streaming_dag()
    assert streaming_dag is not None, "Streaming DAG not found"
    
    # Find Kafka sensor task
    kafka_sensor = None
    for task_id, task in streaming_dag.tasks.items():
        if isinstance(task, AwaitMessageTriggerFunctionSensor):
            kafka_sensor = task
            break
    
    assert kafka_sensor is not None, "AwaitMessageTriggerFunctionSensor not found"
    
    # Test configuration
    assert kafka_sensor.kafka_config_id == 'kafka_default', "Kafka config ID should be 'kafka_default'"
    assert 'water-quality-data' in kafka_sensor.topics, "Topic 'water-quality-data' should be in topics list"
    assert kafka_sensor.poll_timeout == 1, "Poll timeout should be 1"
    assert kafka_sensor.poll_interval == 10, "Poll interval should be 10"


def test_streaming_dag_has_consumer_task():
    """Test that streaming DAG has Kafka consumer task"""
    streaming_dag = get_streaming_dag()
    assert streaming_dag is not None, "Streaming DAG not found"
    
    # Find consumer task
    consumer_task = None
    for task_id, task in streaming_dag.tasks.items():
        if isinstance(task, PythonOperator) and 'consumer' in task_id.lower():
            consumer_task = task
            break
    
    assert consumer_task is not None, "Kafka consumer task not found in streaming DAG"


def test_streaming_dag_has_trigger_task():
    """Test that streaming DAG has ML pipeline trigger task"""
    streaming_dag = get_streaming_dag()
    assert streaming_dag is not None, "Streaming DAG not found"
    
    # Find trigger task
    trigger_task = None
    for task_id, task in streaming_dag.tasks.items():
        if isinstance(task, PythonOperator) and 'trigger' in task_id.lower():
            trigger_task = task
            break
    
    assert trigger_task is not None, "ML pipeline trigger task not found in streaming DAG"


def test_streaming_dag_task_dependencies():
    """Test that streaming DAG has correct task dependencies"""
    streaming_dag = get_streaming_dag()
    assert streaming_dag is not None, "Streaming DAG not found"
    
    # Get task dependencies
    task_dependencies = streaming_dag.task_dict
    
    # Check that we have the expected tasks
    expected_tasks = ['wait_for_kafka', 'kafka_consumer_task', 'trigger_ml_pipeline_after_processing']
    
    for task_id in expected_tasks:
        assert task_id in task_dependencies, f"Task {task_id} not found in streaming DAG"


def test_streaming_dag_tags():
    """Test that streaming DAG has proper tags"""
    streaming_dag = get_streaming_dag()
    assert streaming_dag is not None, "Streaming DAG not found"
    
    assert streaming_dag.tags, "Streaming DAG should have tags"
    assert 'iot_pipeline_dag' in streaming_dag.tags, "Streaming DAG should have 'iot_pipeline_dag' tag"


def test_streaming_dag_schedule():
    """Test that streaming DAG has correct schedule"""
    streaming_dag = get_streaming_dag()
    assert streaming_dag is not None, "Streaming DAG not found"
    
    # Streaming DAG should be triggered manually (schedule_interval=None)
    assert streaming_dag.schedule_interval is None, "Streaming DAG should have schedule_interval=None for manual triggering"


def test_kafka_connection_config():
    """Test that Kafka connection is properly configured"""
    from airflow.models import Connection
    from airflow.settings import Session
    
    session = Session()
    try:
        kafka_conn = session.query(Connection).filter(Connection.conn_id == 'kafka_default').first()
        assert kafka_conn is not None, "Kafka connection 'kafka_default' not found"
        assert kafka_conn.conn_type == 'kafka', "Kafka connection type should be 'kafka'"
        assert kafka_conn.host == '77.37.44.237', "Kafka host should be '77.37.44.237'"
        assert kafka_conn.port == 9092, "Kafka port should be 9092"
    finally:
        session.close()


def test_streaming_dag_description():
    """Test that streaming DAG has proper description"""
    streaming_dag = get_streaming_dag()
    assert streaming_dag is not None, "Streaming DAG not found"
    
    description = streaming_dag.description
    assert description is not None, "Streaming DAG should have a description"
    assert 'IoT' in description or 'Kafka' in description, "Description should mention IoT or Kafka"


def test_streaming_dag_catchup():
    """Test that streaming DAG has catchup disabled"""
    streaming_dag = get_streaming_dag()
    assert streaming_dag is not None, "Streaming DAG not found"
    
    assert streaming_dag.catchup is False, "Streaming DAG should have catchup=False"


def test_streaming_dag_start_date():
    """Test that streaming DAG has proper start date"""
    streaming_dag = get_streaming_dag()
    assert streaming_dag is not None, "Streaming DAG not found"
    
    assert streaming_dag.start_date is not None, "Streaming DAG should have a start date"


def test_kafka_sensor_apply_function():
    """Test that Kafka sensor has proper apply function"""
    streaming_dag = get_streaming_dag()
    assert streaming_dag is not None, "Streaming DAG not found"
    
    # Find Kafka sensor task
    kafka_sensor = None
    for task_id, task in streaming_dag.tasks.items():
        if isinstance(task, AwaitMessageTriggerFunctionSensor):
            kafka_sensor = task
            break
    
    assert kafka_sensor is not None, "AwaitMessageTriggerFunctionSensor not found"
    
    # Test apply function
    assert kafka_sensor.apply_function == 'streaming_process_dag.always_true', "Apply function should be 'streaming_process_dag.always_true'"


def test_kafka_sensor_event_triggered_function():
    """Test that Kafka sensor has proper event triggered function"""
    streaming_dag = get_streaming_dag()
    assert streaming_dag is not None, "Streaming DAG not found"
    
    # Find Kafka sensor task
    kafka_sensor = None
    for task_id, task in streaming_dag.tasks.items():
        if isinstance(task, AwaitMessageTriggerFunctionSensor):
            kafka_sensor = task
            break
    
    assert kafka_sensor is not None, "AwaitMessageTriggerFunctionSensor not found"
    
    # Test event triggered function
    assert kafka_sensor.event_triggered_function == 'streaming_process_dag.notify_trigger', "Event triggered function should be 'streaming_process_dag.notify_trigger'"


if __name__ == "__main__":
    # Run tests manually if needed
    pytest.main([__file__, "-v"]) 