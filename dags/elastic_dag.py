from datetime import datetime, timedelta
from airflow.decorators import dag, task
from airflow.operators.python import PythonOperator
from include.iot_streaming.kafka_consumer import kafka_consumer_task
from include.iot_streaming.kafka_producer import kafka_producer_task,check_kafka_producer
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
    tags=['elastic_dag'],
)
def elastic_dag():

    # Task kiểm tra Elasticsearch index
    connect_Elastic_Task = PythonOperator(
        task_id='delete_index_task',
        python_callable=check_and_delete_index,
        provide_context=True,
    )
    
    connect_Elastic_Task
     

elastic_dag()
