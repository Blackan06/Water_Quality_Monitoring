from datetime import datetime, timedelta
import os
import logging
from pathlib import Path
from airflow.decorators import dag, task
from airflow.operators.python import PythonOperator
from airflow.providers.docker.operators.docker import DockerOperator
from airflow.models.baseoperator import chain
from docker.types import Mount
import pandas as pd
import psycopg2
from sqlalchemy import create_engine

# Project paths - these are paths inside the Airflow container
AIRFLOW_HOME = os.environ.get('AIRFLOW_HOME', '/usr/local/airflow')
DATA_DIR = os.path.join(AIRFLOW_HOME, 'data')
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, 'processed')
SRC_DIR = os.path.join(AIRFLOW_HOME, 'dags/spark/spark_batch')

# Database configuration
DB_CONFIG = {
    'host': '149.28.145.56',
    'port': '5432',
    'database': 'wqi_db',
    'user': 'postgres',
    'password': 'postgres1234'
}

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def create_table_if_not_exists():
    """Create water_quality table if it doesn't exist"""
    try:
        conn_string = f"postgresql://{DB_CONFIG['user']}:{DB_CONFIG['password']}@{DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['database']}"
        engine = create_engine(conn_string)
        
        # Create table if not exists
        with engine.connect() as connection:
            connection.execute("""
                CREATE TABLE IF NOT EXISTS water_quality (
                    id SERIAL PRIMARY KEY,
                    wq_date DATE,
                    temperature DOUBLE PRECISION,
                    DO DOUBLE PRECISION,
                    ph DOUBLE PRECISION
                )
            """)
            connection.commit()
        
        logger.info("Table water_quality created or already exists")
        return True
        
    except Exception as e:
        logger.error(f"Error creating table: {str(e)}", exc_info=True)
        raise

@dag(
    dag_id='water_quality_batch_processing',
    default_args={
        'owner': 'airflow',
        'depends_on_past': False,
        'email_on_failure': False,
        'email_on_retry': False,
        'retries': False,
        'retry_delay': timedelta(minutes=5)
    },
    description='Water Quality Data Processing Pipeline',
    schedule_interval=timedelta(days=1),
    start_date=datetime(2024, 1, 1),
    catchup=False
)
def water_quality_dag():
    @task
    def setup_directories():
        """Set up required directories"""
        try:
            # Create directories if they don't exist
            for dir_path in [DATA_DIR, PROCESSED_DATA_DIR, SRC_DIR]:
                if not os.path.exists(dir_path):
                    os.makedirs(dir_path, exist_ok=True)
                logger.info(f"Directory ready: {dir_path}")
            
            logger.info("All directories set up successfully")
            return True
        
        except Exception as e:
            logger.error(f"Error setting up directories: {str(e)}", exc_info=True)
            raise

    @task
    def init_database():
        """Initialize database tables"""
        return create_table_if_not_exists()

    @task
    def validate_and_load_data():
        """Validate CSV data and load to PostgreSQL"""
        try:
            # Read and validate CSV
            csv_path = os.path.join(DATA_DIR, "WQI_data.csv")
            if not os.path.exists(csv_path):
                raise FileNotFoundError(f"Data file not found: {csv_path}")
            
            df = pd.read_csv(csv_path)
            logger.info(f"Successfully loaded data: {len(df)} rows")
            logger.info(f"Columns in CSV: {df.columns.tolist()}")
            
            # Map columns to match PostgreSQL table schema
            column_mapping = {
                'Date': 'wq_date',
                'Temperature': 'temperature',
                'DO': 'DO',
                'PH': 'ph'
            }
            
            # Check which columns are present
            available_columns = []
            for old_name, new_name in column_mapping.items():
                if old_name in df.columns:
                    available_columns.append((old_name, new_name))
            
            if not available_columns:
                raise ValueError(f"No matching columns found. Available columns: {df.columns.tolist()}")
            
            # Rename only the columns that exist
            rename_dict = {old: new for old, new in available_columns}
            df = df.rename(columns=rename_dict)
            
            # Ensure required columns exist after renaming
            required_columns = ['wq_date', 'temperature', 'DO', 'ph']
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                raise ValueError(f"Missing required columns after renaming: {missing_columns}. Available columns: {df.columns.tolist()}")
            
            # Convert date format
            df['wq_date'] = pd.to_datetime(df['wq_date']).dt.date
            
            # Select only required columns in correct order
            df = df[required_columns]
            
            # Connect to PostgreSQL and load data
            conn_string = f"postgresql://{DB_CONFIG['user']}:{DB_CONFIG['password']}@{DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['database']}"
            engine = create_engine(conn_string)
            
            # Load data
            df.to_sql('water_quality', engine, if_exists='append', index=False)
            logger.info(f"Successfully loaded {len(df)} rows to PostgreSQL")
            
            return True
            
        except Exception as e:
            logger.error(f"Error in data validation and loading: {str(e)}", exc_info=True)
            raise

    # Create tasks
    setup_task = setup_directories()
    validate_task = validate_and_load_data()
    
    # Create Docker task
    process_task = DockerOperator(
        task_id='data_preprocessing',
        image='airflow/spark_batch',
        command='python3 /app/spark_job.py',
        network_mode='bridge',
        container_name='spark_batch_job',
        mounts=[],
        docker_url='tcp://docker-proxy:2375',
        mount_tmp_dir=False,
        environment={
            'DB_HOST': DB_CONFIG['host'],
            'DB_PORT': DB_CONFIG['port'],
            'DB_NAME': DB_CONFIG['database'],
            'DB_USER': DB_CONFIG['user'],
            'DB_PASSWORD': DB_CONFIG['password']
        }
    )

    # Set task dependencies
    setup_task  >> validate_task >> process_task

# Create DAG instance
dag = water_quality_dag()