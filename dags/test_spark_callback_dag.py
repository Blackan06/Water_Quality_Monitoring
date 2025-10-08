# Test DAG to verify Spark callback function works
from datetime import datetime, timezone
import logging
from airflow.decorators import dag
from airflow.providers.apache.kafka.sensors.kafka import AwaitMessageTriggerFunctionSensor

logger = logging.getLogger(__name__)

def simple_callback(event=None, **kwargs):
    """Simple callback function to test"""
    logger.info("🚀 SIMPLE CALLBACK CALLED!")
    logger.info(f"📥 Event: {event}")
    logger.info(f"📥 Kwargs: {kwargs}")
    
    # Test Spark consumer
    try:
        from include.iot_streaming.spark_consumer import process_kafka_message_with_spark
        logger.info("✅ Successfully imported Spark consumer")
        
        result = process_kafka_message_with_spark(event)
        logger.info(f"📊 Spark result: {result}")
        
        return True
    except Exception as e:
        logger.error(f"❌ Spark error: {e}")
        return True

@dag(
    description="Test Spark callback function",
    start_date=datetime(2024, 1, 1),
    schedule=None,
    catchup=False,
    tags=["test"],
    max_active_runs=1,
)
def test_spark_callback_dag():
    """Test DAG to verify Spark callback works"""
    
    wait_for_kafka = AwaitMessageTriggerFunctionSensor(
        task_id="test_kafka_callback",
        kafka_config_id="kafka_default",
        topics=["water-quality-data"],
        apply_function="include.iot_streaming.kafka_handlers.extract_value",
        event_triggered_function=simple_callback,
        poll_timeout=1,
        poll_interval=10,
    )

# Instantiate DAG
test_spark_callback_dag()
