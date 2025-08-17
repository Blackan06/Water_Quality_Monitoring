"""
## Historical Data Loading and Training DAG

This DAG loads historical water quality data and trains machine learning models.
It processes training results, compares model performance, and manages model artifacts.

The pipeline consists of several tasks:
1. Load and process historical data
2. Train multiple machine learning models
3. Compare model performance and select the best one
4. Clean up inferior models and save results

For more information about the water quality monitoring system, see the project documentation.

![Machine Learning Pipeline](https://www.databricks.com/wp-content/uploads/2020/04/mlflow-model-registry.png)
"""

from airflow.decorators import dag, task
from airflow.models import Variable
from airflow.operators.python import PythonOperator
from airflow.providers.docker.operators.docker import DockerOperator
from docker.types import Mount
from pendulum import datetime
import logging
import os
import mlflow
import mlflow.sklearn
import mlflow.xgboost
import tempfile
import pickle
import json
import numpy as np

# Cấu hình logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('logs/training.log')
    ]
)
logger = logging.getLogger(__name__)

# Define the basic parameters of the DAG
@dag(
    start_date=datetime(2024, 1, 1),
    schedule="@weekly",
    catchup=False,
    doc_md=__doc__,
    default_args={"owner": "ml_team", "retries": 3},
    tags=["historical-data", "training", "ml"],
)
def load_historical_data():
    """
    Historical data loading and model training pipeline.
    """
    
    @task
    def process_training_results(**context) -> str:
        """
        Process results from training and show best model information.
        This task analyzes model performance and selects the best performing model.
        """
        try:
            logger.info("Processing training results...")
            
            # Check for new comprehensive metrics first, then fallback to enhanced metadata
            metrics_file = './models/metrics.json'
            metadata_file = './models/enhanced_metadata.json'
            
            if os.path.exists(metrics_file):
                # Use new comprehensive metrics
                with open(metrics_file, 'r') as f:
                    metrics = json.load(f)
                logger.info("📊 Using comprehensive ensemble metrics")
            elif os.path.exists(metadata_file):
                # Fallback to enhanced metadata
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
                metrics = metadata['metrics']
                logger.info("📊 Using enhanced metadata (fallback)")
            else:
                logger.error("❌ No metrics files found")
                return "Training completed but no metrics found"
            
            # Extract metrics
            xgb_r2 = metrics.get('xgb', {}).get('r2', 0.0)
            rf_r2 = metrics.get('rf', {}).get('r2', 0.0)
            ensemble_r2 = metrics.get('ensemble', {}).get('r2', 0.0)
            
            # Compare all models and select the best one
            model_scores = {
                'xgb': xgb_r2,
                'rf': rf_r2,
                'ensemble': ensemble_r2
            }
            
            best_model = max(model_scores, key=model_scores.get)
            best_score = model_scores[best_model]
            
            logger.info(f"📊 Model Performance Comparison:")
            logger.info(f"  XGBoost: R² = {xgb_r2:.4f}")
            logger.info(f"  Random Forest: R² = {rf_r2:.4f}")
            logger.info(f"  Ensemble: R² = {ensemble_r2:.4f}")
            logger.info(f"🏆 Best model selected: {best_model.upper()} (R²: {best_score:.4f})")
            
            # Cleanup inferior models to save space
            _cleanup_inferior_models(best_model)
            
            # Push results to XCom
            context['task_instance'].xcom_push(key='training_metrics', value=metrics)
            context['task_instance'].xcom_push(key='best_model_info', value={
                'feature_count': 105,  # Default feature count for comprehensive model
                'best_model': best_model,
                'best_score': best_score
            })
            
            return f"Training completed: {best_model.upper()} (R²: {best_score:.4f}) - Kept only best model"
            
        except Exception as e:
            logger.error(f"❌ Error processing training results: {e}")
            return f"Error processing results: {e}"
    
    def _cleanup_inferior_models(best_model: str):
        """Clean up inferior models to save storage space"""
        try:
            models_to_remove = []
            if best_model != 'xgb':
                models_to_remove.append('./models/xgb.pkl')
            if best_model != 'rf':
                models_to_remove.append('./models/rf.pkl')
            
            for model_path in models_to_remove:
                if os.path.exists(model_path):
                    os.remove(model_path)
                    logger.info(f"🗑️ Removed inferior model: {model_path}")
                    
        except Exception as e:
            logger.warning(f"⚠️ Error during model cleanup: {e}")
    
    @task
    def load_historical_data_task(**context) -> str:
        """
        Load historical water quality data from various sources.
        This task retrieves and prepares historical data for training.
        """
        logger.info("Starting historical data loading...")
        
        # Define Docker mount for models
        model_mount = Mount(
            source='./models',
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
        
        # Execute Docker container for data loading
        load_operator = DockerOperator(
            task_id='load_historical_data_container',
            command='python /app/spark_jobs/load_data.py',
            **common_op_kwargs
        )
        
        result = load_operator.execute(context=context)
        logger.info("✅ Historical data loading completed successfully")
        return "Historical data loaded successfully"
    
    @task
    def train_ensemble_model_task(**context) -> str:
        """
        Train ensemble machine learning models on historical data.
        This task trains multiple models and creates an ensemble.
        """
        logger.info("Starting ensemble model training...")
        
        # Define Docker mount for models
        model_mount = Mount(
            source='./models',
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
        
        # Execute Docker container for ensemble training
        train_operator = DockerOperator(
            task_id='train_ensemble_model_container',
            command='python /app/spark_jobs/train_ensemble_model.py',
            **common_op_kwargs
        )
        
        result = train_operator.execute(context=context)
        logger.info("✅ Ensemble model training completed successfully")
        return "Ensemble models trained successfully"
    
    # Define task dependencies using TaskFlow API
    load_result = load_historical_data_task()
    train_result = train_ensemble_model_task()
    process_result = process_training_results()
    
    # Set up the pipeline flow
    load_result >> train_result >> process_result


# Instantiate the DAG
load_historical_data() 