from airflow import DAG
from airflow.providers.docker.operators.docker import DockerOperator
from docker.types import Mount
from datetime import datetime, timedelta
from airflow.models import Variable
from airflow.decorators import dag
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2024, 1, 1),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': False,
    'retry_delay': timedelta(minutes=5)
}

@dag(
    'water_quality_processing',
    default_args=default_args,
    description='Water Quality Monitoring Pipeline',
    schedule=timedelta(days=1),
    catchup=False
)





def water_quality_processing():
    # mount local models folder vào /app/models
    model_mount = Mount(
        source='/Users/kiethuynhanh/Documents/THACSIDOCUMENT/Water_Quality_Monitoring/models',
        target='/app/models',
        type='bind'
    )
    
    # chung network với host để truy cập MLflow ở localhost:5003
    common_op_kwargs = {
        'image': 'water-quality-processor:latest',
        'api_version': 'auto',
        'auto_remove': 'success',
        'docker_url': 'tcp://docker-proxy:2375',
        'mount_tmp_dir': False,
        'mounts': [model_mount],
        'network_mode': 'host',
        'environment': {  # <<< đây đổi thành dict
            'MLFLOW_TRACKING_URI': 'http://mlflow:5003'
        }
    }
    load_data = DockerOperator(
        task_id='load_data',
        command='python /app/spark_jobs/load_data.py',
        **common_op_kwargs
    )

    preprocess_data = DockerOperator(
        task_id='preprocess_data',
        command='python /app/spark_jobs/preprocess_data.py',
        **common_op_kwargs
    )

    train_model = DockerOperator(
        task_id='train_model',
        command='python /app/spark_jobs/train_model.py',
        **common_op_kwargs
    )

    save_results = DockerOperator(
        task_id='save_results',
        command='python /app/spark_jobs/save_results.py',
        **common_op_kwargs
    )


    load_data >> preprocess_data >> train_model >> save_results
water_quality_processing()