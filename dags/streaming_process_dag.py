"""
## Streaming Process DAG

This DAG handles Kafka message processing and triggers downstream ML pipelines.
It monitors Kafka topics for new messages and orchestrates the processing workflow.

The pipeline consists of several tasks:
1. Wait for Kafka messages using sensors
2. Process Kafka messages in batch
3. Trigger external ML pipeline DAGs
4. Handle batch processing results

For more information about the water quality monitoring system, see the project documentation.

![Kafka Streaming](https://kafka.apache.org/images/streams-architecture-overview.png)
"""

from airflow.decorators import dag, task
from airflow.providers.apache.kafka.sensors.kafka import AwaitMessageTriggerFunctionSensor
from airflow.operators.python import PythonOperator
from airflow.operators.trigger_dagrun import TriggerDagRunOperator
from airflow.models.baseoperator import chain
from include.iot_streaming.kafka_consumer import kafka_consumer_task, get_kafka_offset_info
from airflow.exceptions import AirflowException
from airflow.models import Variable
from airflow.utils.log.logging_mixin import LoggingMixin
from pendulum import datetime
import logging

# Setup logger
logger = logging.getLogger(__name__)

# Define the basic parameters of the DAG
@dag(
    start_date=datetime(2025, 1, 1),
    schedule="@hourly",
    catchup=False,
    doc_md=__doc__,
    default_args={"owner": "streaming_team", "retries": 2},
    tags=["kafka", "streaming", "ml-pipeline"],
)
def streaming_process():
    """
    Kafka message processing and ML pipeline orchestration.
    """
    
    @task
    def notify_trigger(event, **kwargs) -> str:
        """
        Handler function when receiving messages from Kafka.
        This task triggers all downstream tasks immediately.
        """
        logger.info("🔔 Notify trigger function called")
        
        # Extract message value
        try:
            msg = event.value().decode("utf-8")
        except Exception:
            msg = str(event)
        
        logger.info("🎉 New Kafka message received: " + msg)
        
        # Push message to XCom for downstream tasks
        if 'ti' in kwargs:
            kwargs['ti'].xcom_push(key='kafka_message', value=msg)
            kwargs['ti'].xcom_push(key='trigger_time', value=datetime.now().isoformat())
            logger.info(f"📤 Pushed message to XCom: {msg}")
        
        # Trigger all downstream tasks immediately
        try:
            # Trigger consumer task (now handles batch processing)
            consumer_result = enhanced_kafka_consumer_task(**kwargs)
            logger.info("✅ Consumer task executed successfully")
            
            # Get batch processing results
            batch_size = kwargs['ti'].xcom_pull(key='batch_size', task_ids='enhanced_kafka_consumer_task')
            processed_count = kwargs['ti'].xcom_pull(key='processed_count', task_ids='enhanced_kafka_consumer_task')
            error_count = kwargs['ti'].xcom_pull(key='error_count', task_ids='enhanced_kafka_consumer_task')
            
            if batch_size:
                logger.info(f"📊 Batch processing results: {processed_count}/{batch_size} processed, {error_count} errors")
            
            # Trigger external DAG task (only trigger DAG, don't call ML pipeline directly)
            external_result = trigger_external_dag(**kwargs)
            logger.info("✅ External DAG task executed successfully")
            
            logger.info("🎉 All downstream tasks completed successfully")
            return "All downstream tasks completed successfully"
            
        except Exception as e:
            logger.error(f"❌ Error executing downstream tasks: {str(e)}")
            raise AirflowException(f"Failed to execute downstream tasks: {str(e)}")
    
    @task
    def enhanced_kafka_consumer_task(**context) -> str:
        """
        Enhanced consumer task that can access Kafka message from XCom and handles batch processing.
        This task processes Kafka messages in batches for efficiency.
        """
        from include.iot_streaming.kafka_consumer import kafka_consumer_task
        
        logger.info("🔄 Starting enhanced Kafka consumer task (batch processing)...")
        
        try:
            # Get message from XCom (if available from sensor)
            kafka_message = context['ti'].xcom_pull(key='kafka_message', task_ids='wait_for_kafka')
            
            if kafka_message:
                logger.info(f"📥 Processing Kafka message from sensor: {kafka_message}")
            
            # Run the original consumer task (now with batch processing)
            result = kafka_consumer_task(**context)
            
            # Get batch processing results
            batch_size = context['ti'].xcom_pull(key='batch_size')
            processed_count = context['ti'].xcom_pull(key='processed_count')
            error_count = context['ti'].xcom_pull(key='error_count')
            
            if batch_size and processed_count is not None:
                logger.info(f"📊 Batch processing completed: {processed_count}/{batch_size} processed, {error_count} errors")
            
            logger.info("✅ Enhanced Kafka consumer task completed")
            return result
            
        except Exception as e:
            logger.error(f"❌ Error in enhanced consumer task: {str(e)}")
            raise e
    
    @task
    def trigger_ml_pipeline_after_processing(**context) -> str:
        """
        Trigger ML pipeline after processing Kafka data.
        This task orchestrates the ML pipeline execution.
        """
        try:
            from include.iot_streaming.database_manager import db_manager
            
            # Trigger ML pipeline
            success = db_manager.trigger_ml_pipeline()
            
            if success:
                logger.info("✅ Successfully triggered ML pipeline")
                return "ML pipeline triggered successfully"
            else:
                logger.warning("⚠️ ML pipeline trigger failed")
                return "ML pipeline trigger failed"
                
        except Exception as e:
            logger.error(f"❌ Error triggering ML pipeline: {str(e)}")
            return f"Error triggering ML pipeline: {str(e)}"
    
    @task
    def trigger_external_dag(**context) -> str:
        """
        Trigger external DAG for ML pipeline processing.
        This task starts the external ML pipeline DAG.
        """
        try:
            # Trigger the streaming data processor DAG
            trigger_operator = TriggerDagRunOperator(
                task_id='trigger_streaming_processor',
                trigger_dag_id='streaming_data_processor',
                wait_for_completion=False,
                poke_interval=30,
                timeout=300
            )
            
            result = trigger_operator.execute(context=context)
            logger.info("✅ External DAG triggered successfully")
            return "External DAG triggered successfully"
            
        except Exception as e:
            logger.error(f"❌ Error triggering external DAG: {str(e)}")
            return f"Error triggering external DAG: {str(e)}"
    
    @task
    def wait_for_kafka(**context) -> str:
        """
        Wait for Kafka messages using sensor.
        This task monitors Kafka topics for new messages.
        """
        try:
            # Configure Kafka sensor
            kafka_sensor = AwaitMessageTriggerFunctionSensor(
                task_id='kafka_message_sensor',
                kafka_config_id='kafka_default',
                topics=['water-quality-data'],
                apply_function=notify_trigger,
                poll_timeout=60,
                poke_interval=30
            )
            
            result = kafka_sensor.execute(context=context)
            logger.info("✅ Kafka sensor completed successfully")
            return "Kafka sensor completed successfully"
            
        except Exception as e:
            logger.error(f"❌ Error in Kafka sensor: {str(e)}")
            return f"Error in Kafka sensor: {str(e)}"
    
    # Define task dependencies using TaskFlow API
    kafka_wait = wait_for_kafka()
    consumer_task = enhanced_kafka_consumer_task()
    ml_pipeline = trigger_ml_pipeline_after_processing()
    external_dag = trigger_external_dag()
    
    # Set up the pipeline flow
    kafka_wait >> consumer_task >> ml_pipeline >> external_dag


# Instantiate the DAG
streaming_process()
