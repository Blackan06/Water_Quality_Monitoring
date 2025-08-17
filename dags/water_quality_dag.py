"""
## Water Quality Processing DAG

This DAG orchestrates the water quality monitoring pipeline using Docker containers.
It loads data, preprocesses it, trains models, and saves results using Spark jobs.

The pipeline consists of four main tasks:
1. Load data from various sources
2. Preprocess and clean the data
3. Train machine learning models
4. Save results and metrics

For more information about the water quality monitoring system, see the project documentation.

![Water Quality Monitoring](https://www.epa.gov/sites/default/files/2018-07/water-quality-monitoring.jpg)
"""

from airflow.decorators import dag, task
from airflow.providers.docker.operators.docker import DockerOperator
from docker.types import Mount
from pendulum import datetime
import logging

# Setup logging
logger = logging.getLogger(__name__)

# Define the basic parameters of the DAG
@dag(
    start_date=datetime(2024, 1, 1),
    schedule="@daily",
    catchup=False,
    doc_md=__doc__,
    default_args={"owner": "water_quality_team", "retries": 3},
    tags=["water-quality", "processing", "ml"],
)
def water_quality_processing():
    """
    Water Quality Monitoring Pipeline using Docker containers and Spark jobs.
    """
    
    # Define Docker mount for models
    model_mount = Mount(
        source='/Users/kiethuynhanh/Documents/THACSIDOCUMENT/Water_Quality_Monitoring/models',
        target='/app/models',
        type='bind'
    )
    
    # Common Docker operator arguments
    common_op_kwargs = {
        'image': 'water-quality-processor:latest',
        'api_version': 'auto',
        'auto_remove': True,
        'docker_url': 'unix://var/run/docker.sock',
        'mount_tmp_dir': False,
        'mounts': [model_mount],
        'network_mode': 'host',
        'environment': {
            'MLFLOW_TRACKING_URI': 'http://localhost:5003'
        }
    }
    
    @task
    def load_data_task(**context) -> str:
        """
        Load water quality data from various sources using Spark.
        This task retrieves raw data and prepares it for preprocessing.
        """
        logger.info("Starting data loading process...")
        
        # Execute Docker container for data loading
        load_operator = DockerOperator(
            task_id='load_data_container',
            command='python /app/spark_jobs/load_data.py',
            **common_op_kwargs
        )
        
        result = load_operator.execute(context=context)
        logger.info("✅ Data loading completed successfully")
        return "Data loaded successfully"
    
    @task
    def preprocess_data_task(**context) -> str:
        """
        Preprocess and clean the loaded water quality data.
        This task handles data cleaning, feature engineering, and validation.
        """
        logger.info("Starting data preprocessing...")
        
        # Execute Docker container for data preprocessing
        preprocess_operator = DockerOperator(
            task_id='preprocess_data_container',
            command='python /app/spark_jobs/preprocess_data.py',
            **common_op_kwargs
        )
        
        result = preprocess_operator.execute(context=context)
        logger.info("✅ Data preprocessing completed successfully")
        return "Data preprocessed successfully"
    
    @task
    def train_model_task(**context) -> str:
        """
        Train machine learning models on the preprocessed data.
        This task trains multiple models and selects the best performing one.
        """
        logger.info("Starting model training...")
        
        # Execute Docker container for model training
        train_operator = DockerOperator(
            task_id='train_model_container',
            command='python /app/spark_jobs/train_model.py',
            **common_op_kwargs
        )
        
        result = train_operator.execute(context=context)
        logger.info("✅ Model training completed successfully")
        return "Models trained successfully"
    
    @task
    def save_results_task(**context) -> str:
        """
        Save training results, metrics, and model artifacts.
        This task persists the final results and model metadata.
        """
        logger.info("Starting results saving...")
        
        # Execute Docker container for saving results
        save_operator = DockerOperator(
            task_id='save_results_container',
            command='python /app/spark_jobs/save_results.py',
            **common_op_kwargs
        )
        
        result = save_operator.execute(context=context)
        logger.info("✅ Results saved successfully")
        return "Results saved successfully"
    
    # Define task dependencies using TaskFlow API
    load_result = load_data_task()
    preprocess_result = preprocess_data_task()
    train_result = train_model_task()
    save_result = save_results_task()
    
    # Set up the pipeline flow
    load_result >> preprocess_result >> train_result >> save_result


# Instantiate the DAG
water_quality_processing()
