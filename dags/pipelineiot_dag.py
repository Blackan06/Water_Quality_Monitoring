from datetime import datetime, timedelta
from airflow.decorators import dag, task
from airflow.providers.docker.operators.docker import DockerOperator
from airflow.operators.python import PythonOperator
from airflow.models.baseoperator import chain
from include.iot_streaming.kafka_consumer import kafka_consumer_task
from include.iot_streaming.kafka_producer import kafka_producer_task
from include.iot_streaming.elasticsearch import check_and_delete_index

# Định nghĩa các default arguments cho DAG
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'email': 'huynhanhkiet2222@gmail.com', 
}

# Định nghĩa DAG với decorator
@dag(
    default_args=default_args,
    description='DAG for IoT Data Pipeline using Kafka and Spark',
    schedule_interval=None,  # Thời gian chạy tự động nếu có dữ liệu mới
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=['iot_pipeline_dag'],
)
def iot_pipeline_dag():
    #ElasticSearch_Task
    delete_index_task = PythonOperator(
        task_id='elasticsearch_task',
        python_callable=check_and_delete_index,
        provide_context=True,
    )

    # Producer task
    producer_task = PythonOperator(
        task_id='kafka_producer_task',
        python_callable=kafka_producer_task,
        provide_context=True,
    )

    # Consumer task
    consumer_task = PythonOperator(
        task_id='kafka_consumer_task',
        python_callable=kafka_consumer_task,
        provide_context=True,
    )
    # Spark job task
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
        network_mode='container:elasticsearch',
        environment={
            'SPARK_APPLICATION_ARGS': '{{ ti.xcom_pull(task_ids="kafka_consumer_task") }}',
            'SHARED_VOLUME_PATH': '/shared_volume',
        },
        do_xcom_push=True,
    )
    def trigger_consumer_and_spark_jobs(**kwargs):
        data = kwargs['ti'].xcom_pull(task_ids='kafka_producer_task', key='kafka_data')

        if data:
            print(f"Data received: {data}. Triggering consumer and spark job tasks...")
            # Trigger the consumer task and run_spark_job dynamically
            kwargs['ti'].xcom_push(key='status', value='data_received')
            # Run both consumer_task and run_spark_job immediately when data is received
            consumer_task.run(context=kwargs)  # Trigger the consumer task
            run_spark_job.run(context=kwargs)  # Trigger the spark job task

    trigger_consumer_and_spark_jobs_operator = PythonOperator(
        task_id='trigger_consumer_and_spark_jobs_operator',
        python_callable=trigger_consumer_and_spark_jobs,
        provide_context=True,
    )
    chain(delete_index_task,producer_task,trigger_consumer_and_spark_jobs_operator)

iot_pipeline_dag()
