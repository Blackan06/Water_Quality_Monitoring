"""
## Model Promotion to Production DAG

This DAG promotes the best performing machine learning model to production.
It checks MLflow connections, retrieves model versions, and manages model lifecycle.

The pipeline consists of several tasks:
1. Check MLflow server connectivity
2. Get latest model versions from MLflow
3. Promote best model to production stage
4. Update model metadata and tags

For more information about the water quality monitoring system, see the project documentation.

![MLflow Model Registry](https://mlflow.org/docs/latest/_images/model-registry-concept.png)
"""

from airflow.decorators import dag, task
from airflow.operators.python import PythonOperator
from pendulum import datetime
import logging
import mlflow
from mlflow.tracking import MlflowClient
import time
import requests

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define the basic parameters of the DAG
@dag(
    start_date=datetime(2024, 1, 1),
    schedule="@daily",
    catchup=False,
    doc_md=__doc__,
    default_args={"owner": "ml_team", "retries": 3},
    tags=["model-promotion", "mlflow", "production"],
)
def promote_model_to_production():
    """
    Model promotion to production pipeline using MLflow.
    """
    
    @task
    def check_mlflow_connection(**context) -> str:
        """
        Check if MLflow server is accessible.
        This task verifies connectivity to the MLflow tracking server.
        """
        try:
            response = requests.get("http://77.37.44.237:5003/health", timeout=10)
            if response.status_code == 200:
                logger.info("✅ MLflow server is accessible")
                return "MLflow server is accessible"
            else:
                logger.error(f"❌ MLflow server returned status {response.status_code}")
                return f"MLflow server returned status {response.status_code}"
        except Exception as e:
            logger.error(f"❌ Cannot connect to MLflow server: {e}")
            return f"Cannot connect to MLflow server: {e}"
    
    @task
    def get_latest_model_version(**context) -> dict:
        """
        Get the latest version of a model using MLflow API.
        This task retrieves the most recent model version for promotion.
        """
        try:
            # Set up MLflow client
            mlflow.set_tracking_uri("http://77.37.44.237:5003")
            client = MlflowClient()
            
            # Model name to promote
            model_name = "water_quality_best_model"
            
            logger.info(f"🔍 Looking for model: {model_name}")
            
            # Get all versions of the model
            versions = client.search_model_versions(f"name='{model_name}'")
            if not versions:
                logger.error(f"❌ No versions found for model {model_name}")
                return {"error": f"No versions found for model {model_name}"}
            
            # Sort by version number and get the latest
            latest_version = max(versions, key=lambda v: v.version)
            version = latest_version.version
            
            logger.info(f"📋 Found model {model_name} version {version}")
            
            return {
                "model_name": model_name,
                "version": version,
                "latest_version": latest_version
            }
            
        except Exception as e:
            logger.error(f"❌ Error getting latest version for {model_name}: {e}")
            return {"error": f"Error getting latest version: {e}"}
    
    @task
    def promote_best_model_to_production(**context) -> str:
        """
        Promote the best performing model to Production stage.
        This task manages the model lifecycle and promotion process.
        """
        try:
            # Get model information from previous task
            model_info = context['task_instance'].xcom_pull(key='return_value', task_ids='get_latest_model_version')
            
            if not model_info or 'error' in model_info:
                error_msg = model_info.get('error', 'Unknown error') if model_info else 'No model info received'
                logger.error(f"❌ Cannot promote model: {error_msg}")
                return f"Cannot promote model: {error_msg}"
            
            model_name = model_info['model_name']
            version = model_info['version']
            latest_version = model_info['latest_version']
            
            # Set up MLflow client
            mlflow.set_tracking_uri("http://77.37.44.237:5003")
            client = MlflowClient()
            
            # Check if model is already in production
            try:
                # Get model version details
                model_version = client.get_model_version(model_name, version)
                
                # Check if this version is already marked as production
                production_tag = None
                for tag in model_version.tags:
                    if tag.key == "stage" and tag.value == "Production":
                        production_tag = tag
                        break
                
                if production_tag:
                    logger.info(f"ℹ️ Model {model_name} v{version} is already in Production")
                    return f"✅ Model {model_name} v{version} already in Production"
                
                # Mark this version as production
                client.set_registered_model_alias(
                    name=model_name,
                    alias="Production",
                    version=version
                )
                
                # Add production tag
                client.set_model_version_tag(
                    name=model_name,
                    version=version,
                    key="stage",
                    value="Production"
                )
                
                # Add promotion timestamp
                client.set_model_version_tag(
                    name=model_name,
                    version=version,
                    key="promoted_at",
                    value=datetime.now().isoformat()
                )
                
                logger.info(f"✅ Successfully promoted {model_name} v{version} to Production")
                return f"✅ Successfully promoted {model_name} v{version} to Production"
                
            except Exception as e:
                logger.error(f"❌ Error promoting model to production: {e}")
                return f"Error promoting model to production: {e}"
                
        except Exception as e:
            logger.error(f"❌ Error in model promotion task: {e}")
            return f"Error in model promotion task: {e}"
    
    @task
    def update_model_metadata(**context) -> str:
        """
        Update model metadata and create promotion summary.
        This task finalizes the promotion process with metadata updates.
        """
        try:
            # Get promotion result from previous task
            promotion_result = context['task_instance'].xcom_pull(key='return_value', task_ids='promote_best_model_to_production')
            
            if not promotion_result or "Error" in promotion_result:
                logger.warning(f"⚠️ Model promotion may have failed: {promotion_result}")
                return f"Model promotion may have failed: {promotion_result}"
            
            # Set up MLflow client for metadata update
            mlflow.set_tracking_uri("http://77.37.44.237:5003")
            client = MlflowClient()
            
            # Get model information
            model_info = context['task_instance'].xcom_pull(key='return_value', task_ids='get_latest_model_version')
            
            if model_info and 'model_name' in model_info:
                model_name = model_info['model_name']
                version = model_info['version']
                
                # Add additional metadata
                client.set_model_version_tag(
                    name=model_name,
                    version=version,
                    key="promotion_method",
                    value="automated_dag"
                )
                
                client.set_model_version_tag(
                    name=model_name,
                    version=version,
                    key="dag_run_id",
                    value=context['dag_run'].run_id
                )
                
                logger.info(f"✅ Updated metadata for {model_name} v{version}")
                return f"✅ Updated metadata for {model_name} v{version}"
            else:
                logger.warning("⚠️ No model info available for metadata update")
                return "No model info available for metadata update"
                
        except Exception as e:
            logger.error(f"❌ Error updating model metadata: {e}")
            return f"Error updating model metadata: {e}"
    
    # Define task dependencies using TaskFlow API
    connection_check = check_mlflow_connection()
    model_version = get_latest_model_version()
    model_promotion = promote_best_model_to_production()
    metadata_update = update_model_metadata()
    
    # Set up the pipeline flow
    connection_check >> model_version >> model_promotion >> metadata_update


# Instantiate the DAG
promote_model_to_production()