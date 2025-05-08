from datetime import datetime, timedelta
from airflow.decorators import dag, task
from airflow.providers.docker.operators.docker import DockerOperator
from airflow.operators.python import PythonOperator
from airflow.models.baseoperator import chain
from include.iot_streaming.kafka_consumer import kafka_consumer_task
from include.iot_streaming.kafka_producer import kafka_producer_task,check_kafka_producer
from include.iot_streaming.kafka_producer_streaming import kafka_run
from airflow.models import Variable

openai_key = Variable.get("openai_api_key")

# Định nghĩa các default_args cho DAG
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'email_on_failure': True,  # Enable email on failure
    'email_on_retry': False,
    'email': 'huynhanhkiet2222@gmail.com', 
}

@dag(
    default_args=default_args,
    description='Optimized IoT Data Pipeline using Kafka and Spark',
    schedule_interval=None,  # Can be set to an interval if needed
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=['iot_pipeline_dag'],
)
def iot_pipeline_dag():


    # Task gửi dữ liệu vào Kafka
    producer_task = PythonOperator(
        task_id='kafka_producer_task',
        python_callable=kafka_run,
        provide_context=True,
    )

    # Task nhận dữ liệu từ Kafka
    consumer_task = PythonOperator(
        task_id='kafka_consumer_task',
        python_callable=kafka_consumer_task,
        provide_context=True,
    )

    # Task chạy Spark job
    run_spark_job = DockerOperator(
        task_id='run_spark_job',
        image='airflow/iot_stream',
        api_version='auto',
        auto_remove=True,
        tty=True,
        xcom_all=False,
        mount_tmp_dir=False,
        container_name='run_spark_job_container',
        docker_url='tcp://docker-proxy:2375',
        environment={
            "OPENAI_API_KEY": openai_key,
            'SPARK_APPLICATION_ARGS': '{{ ti.xcom_pull(task_ids="kafka_consumer_task") }}',
            'SHARED_VOLUME_PATH': '/shared_volume',
            'MLFLOW_TRACKING_URI': 'http://mlflow:5003',
            'MLFLOW_DEFAULT_ARTIFACT_ROOT': 'file:///mlflow_data/artifacts',
            'MODEL_NAME': 'water_quality_xgb',
            'KAFKA_BOOTSTRAP_SERVERS': '77.37.44.237:9092',  # VPS Kafka address
        },
        do_xcom_push=True,
    )
  
    chain(producer_task,[consumer_task,run_spark_job])
     

iot_pipeline_dag()
