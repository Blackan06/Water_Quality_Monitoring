from datetime import datetime, timedelta
from airflow.decorators import dag, task
from airflow.providers.docker.operators.docker import DockerOperator
from airflow.operators.python import PythonOperator
from airflow.models.baseoperator import chain
from include.iot_streaming.kafka_consumer import kafka_consumer_task
from include.iot_streaming.kafka_producer import kafka_producer_task,get_kafka_producer
from include.iot_streaming.elasticsearch import check_and_delete_index, fetch_and_save_data_from_api, check_index

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

    # Task kiểm tra Elasticsearch index
    connect_Elastic_Task = PythonOperator(
        task_id='elasticsearch_task',
        python_callable=check_index,
        provide_context=True,
    )
    
    # Task tải và lưu dữ liệu từ API
    fetch_index_task = PythonOperator(
        task_id='data_task',
        python_callable=fetch_and_save_data_from_api,
        provide_context=True,
    )
    #Check Kafka Producer 
    check_kafka_producer = PythonOperator(
        task_id='check_kafka_producer_task',
        python_callable=get_kafka_producer,
        provide_context=True,
    )
    # Task gửi dữ liệu vào Kafka
    producer_task = PythonOperator(
        task_id='kafka_producer_task',
        python_callable=kafka_producer_task,
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
            'SPARK_APPLICATION_ARGS': '{{ ti.xcom_pull(task_ids="kafka_consumer_task") }}',
            'SHARED_VOLUME_PATH': '/shared_volume',
        },
        do_xcom_push=True,
    )

    # Task gửi email thông báo khi pipeline hoàn thành
    # send_success_email = EmailOperator(
    #     task_id='send_success_email',
    #     to='huynhanhkiet2222@gmail.com',
    #     subject='IoT Data Pipeline Success',
    #     html_content='The IoT Data Pipeline has completed successfully.',
    # )

    # # Task gửi thông báo nếu có lỗi
    # send_failure_email = EmailOperator(
    #     task_id='send_failure_email',
    #     to='huynhanhkiet2222@gmail.com',
    #     subject='IoT Data Pipeline Failure',
    #     html_content='There was an error in the IoT Data Pipeline.',
    # )
    chain(connect_Elastic_Task,fetch_index_task,check_kafka_producer,producer_task,consumer_task,run_spark_job)
     

iot_pipeline_dag()
