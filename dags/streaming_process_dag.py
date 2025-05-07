from datetime import datetime
from airflow.decorators import dag
from airflow.providers.apache.kafka.sensors.kafka import AwaitMessageTriggerFunctionSensor
from airflow.operators.python import PythonOperator
from airflow.operators.trigger_dagrun import TriggerDagRunOperator
from airflow.models.baseoperator import chain
from include.iot_streaming.kafka_consumer import kafka_consumer_task
from airflow.exceptions import AirflowException

# Handler functions must accept **kwargs to absorb context

def notify_trigger(event, **kwargs):
    # extract message value
    try:
        msg = event.value().decode("utf-8")
    except Exception:
        msg = str(event)
    print("New Kafka message matched:", msg)
    
    # Create trigger operator
    trigger = TriggerDagRunOperator(
        task_id="trigger_process_water_quality",
        trigger_dag_id="iot_pipeline_dag",
        conf={"kafka_msg": msg},
        wait_for_completion=False,
    )
    
    # Execute trigger with proper context
    try:
        trigger.execute(context=kwargs)
    except Exception as e:
        print(f"Error triggering DAG: {str(e)}")
        raise AirflowException(f"Failed to trigger DAG: {str(e)}")

# simple apply function

def always_true(event, **kwargs):
    return True

# Default args for DAG

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'email_on_failure': False,   # disable until SMTP is configured
    'email_on_retry': False,
}

@dag(
    default_args=default_args,
    description='IoT Pipeline using KafkaSensor to trigger downstream tasks',
    schedule_interval=None,
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=['iot_pipeline_dag'],
)
def streaming_process_dag():
    # Sensor: wait for new message on Kafka topic
    wait_for_kafka = AwaitMessageTriggerFunctionSensor(
        task_id='wait_for_kafka',
        kafka_config_id='kafka_default',
        topics=['water-quality-data'],
        apply_function='streaming_process_dag.always_true',
        event_triggered_function=notify_trigger,
        poll_timeout=1,
        poll_interval=10,
    )

    # Task: consume from Kafka (optional downstream logic)
    consumer_task = PythonOperator(
        task_id='kafka_consumer_task',
        python_callable=kafka_consumer_task,
        provide_context=True,
    )

    # Define task order: sensor -> consume
    chain(wait_for_kafka, consumer_task)

# Instantiate DAG
streaming_process_dag()
