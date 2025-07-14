from datetime import datetime
from airflow.decorators import dag
from airflow.providers.apache.kafka.sensors.kafka import AwaitMessageTriggerFunctionSensor
from airflow.operators.python import PythonOperator
from airflow.operators.trigger_dagrun import TriggerDagRunOperator
from airflow.models.baseoperator import chain
from include.iot_streaming.kafka_consumer import kafka_consumer_task, get_kafka_offset_info
from airflow.exceptions import AirflowException
from airflow.models import Variable
from airflow.utils.log.logging_mixin import LoggingMixin
import logging

# Setup logger
logger = logging.getLogger(__name__)

def notify_trigger(event, **kwargs):
    """Handler function khi nhận được message từ Kafka - trigger tất cả downstream tasks"""
    logger.info("🔔 Notify trigger function called")
    
    # extract message value
    try:
        msg = event.value().decode("utf-8")
    except Exception:
        msg = str(event)
    
    logger.info("🎉 New Kafka message received: " + msg)
    
    # Push message to XCom để downstream tasks có thể sử dụng
    if 'ti' in kwargs:
        kwargs['ti'].xcom_push(key='kafka_message', value=msg)
        kwargs['ti'].xcom_push(key='trigger_time', value=datetime.now().isoformat())
        logger.info(f"📤 Pushed message to XCom: {msg}")
    
    # Trigger tất cả downstream tasks ngay lập tức
    try:
        # Trigger consumer task
        consumer_result = enhanced_kafka_consumer_task(**kwargs)
        logger.info("✅ Consumer task executed successfully")
        
        # Trigger external DAG task (chỉ trigger DAG, không gọi trực tiếp ML pipeline)
        external_result = trigger_external_dag(**kwargs)
        logger.info("✅ External DAG task executed successfully")
        
        logger.info("🎉 All downstream tasks completed successfully")
        
    except Exception as e:
        logger.error(f"❌ Error executing downstream tasks: {str(e)}")
        raise AirflowException(f"Failed to execute downstream tasks: {str(e)}")

def enhanced_kafka_consumer_task(**context):
    """Enhanced consumer task that can access Kafka message from XCom"""
    from include.iot_streaming.kafka_consumer import kafka_consumer_task
    
    logger.info("🔄 Starting enhanced Kafka consumer task...")
    
    try:
        # Get message from XCom (if available from sensor)
        kafka_message = context['ti'].xcom_pull(key='kafka_message', task_ids='wait_for_kafka')
        
        if kafka_message:
            logger.info(f"📥 Processing Kafka message from sensor: {kafka_message}")
        
        # Run the original consumer task
        result = kafka_consumer_task(**context)
        
        logger.info("✅ Enhanced Kafka consumer task completed")
        return result
        
    except Exception as e:
        logger.error(f"❌ Error in enhanced consumer task: {str(e)}")
        raise e

def trigger_ml_pipeline_after_processing(**context):
    """Trigger ML pipeline sau khi xử lý dữ liệu Kafka"""
    try:
        from include.iot_streaming.database_manager import db_manager
        
        # Trigger ML pipeline
        success = db_manager.trigger_ml_pipeline()
        
        if success:
            logger.info("✅ Successfully triggered ML pipeline")
            return "ML pipeline triggered successfully"
        else:
            logger.error("❌ Failed to trigger ML pipeline")
            return "Failed to trigger ML pipeline"
            
    except Exception as e:
        logger.error(f"❌ Error triggering ML pipeline: {str(e)}")
        return f"Error: {str(e)}"

def trigger_external_dag(**context):
    """Trigger external ML pipeline DAG"""
    try:
        # Get message from XCom
        kafka_message = context['ti'].xcom_pull(key='kafka_message', task_ids='wait_for_kafka')
        
        # Create trigger operator for ML pipeline (external DAG)
        trigger = TriggerDagRunOperator(
            task_id="trigger_ml_pipeline",
            trigger_dag_id="streaming_data_processor",
            conf={"kafka_msg": kafka_message},
            wait_for_completion=False,
        )
        
        # Execute trigger
        trigger.execute(context=context)
        logger.info("✅ Successfully triggered external ML pipeline DAG")
        return "External DAG triggered successfully"
        
    except Exception as e:
        logger.error(f"❌ Error triggering external DAG: {str(e)}")
        return f"Error: {str(e)}"

# simple apply function
def always_true(event, **kwargs):
    return True

# Default args for DAG
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
}

@dag(
    default_args=default_args,
    description='Continuous IoT Pipeline - processes messages as they arrive without restarting',
    schedule_interval=None,
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=['iot_pipeline_continuous'],
    max_active_runs=1,  # Chỉ cho phép 1 DAG run tại một thời điểm
)
def streaming_process_dag():
    # Sensor: wait for new message on Kafka topic
    # DAG sẽ chạy liên tục, không bao giờ đóng
    # Khi có message, notify_trigger sẽ tự động chạy tất cả downstream tasks
    # Và sensor tiếp tục đợi message tiếp theo
    wait_for_kafka = AwaitMessageTriggerFunctionSensor(
        task_id='wait_for_kafka',
        kafka_config_id='kafka_default',
        topics=['water-quality-data'],
        apply_function='streaming_process_dag.always_true',
        event_triggered_function=notify_trigger,
        poll_timeout=1,
        poll_interval=10,
    )

    # Chỉ return sensor task, các task khác sẽ được trigger bởi notify_trigger
    # Sensor sẽ chạy liên tục và xử lý messages khi chúng đến
    return wait_for_kafka

# Instantiate DAG
streaming_process_dag()
