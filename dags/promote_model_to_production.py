from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import logging
import mlflow
from mlflow.tracking import MlflowClient
import time
import requests
from airflow.decorators import dag, task
from pendulum import datetime
# Cấu hình logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Default arguments
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2024, 1, 1),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 3,
    'retry_delay': timedelta(minutes=2),
}

def check_mlflow_connection():
    """Check if MLflow server is accessible"""
    try:
        response = requests.get("http://mlflow:5003/health", timeout=10)
        if response.status_code == 200:
            logger.info("✅ MLflow server is accessible")
            return True
        else:
            logger.error(f"❌ MLflow server returned status {response.status_code}")
            return False
    except Exception as e:
        logger.error(f"❌ Cannot connect to MLflow server: {e}")
        return False

def get_latest_model_version(client, model_name):
    """Get the latest version of a model using new API"""
    try:
        # Get all versions of the model
        versions = client.search_model_versions(f"name='{model_name}'")
        if not versions:
            return None
        
        # Sort by version number and get the latest
        latest_version = max(versions, key=lambda v: v.version)
        return latest_version
    except Exception as e:
        logger.error(f"❌ Error getting latest version for {model_name}: {e}")
        return None

def promote_best_model_to_production(**context):
    """Promote the best performing model to Production stage"""
    try:
        # Check MLflow connection first
        if not check_mlflow_connection():
            return "❌ MLflow server is not accessible"
        
        # Thiết lập MLflow client (VPS)
        mlflow.set_tracking_uri("http://mlflow:5003")
        client = MlflowClient()
        
        # Tên model để promote
        model_name = "water_quality_best_model"
        
        logger.info(f"🔍 Looking for model: {model_name}")
        
        # Get latest version using new API
        latest_version = get_latest_model_version(client, model_name)
        if not latest_version:
            logger.error(f"❌ No versions found for model {model_name}")
            return f"No versions found for model {model_name}"
        
        version = latest_version.version
        logger.info(f"📋 Found model {model_name} version {version}")
        
        # Check if model is already in production (using tags instead of stages)
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
            
            logger.info(f"🚀 Successfully promoted {model_name} v{version} to Production!")
            
            # Log model details
            logger.info(f"📊 Production Model Details:")
            logger.info(f"   Name: {model_name}")
            logger.info(f"   Version: {version}")
            logger.info(f"   Stage: Production")
            logger.info(f"   Created: {model_version.creation_timestamp}")
            
            return f"✅ Model {model_name} v{version} promoted to Production"
            
        except Exception as e:
            logger.error(f"❌ Error promoting model {model_name}: {e}")
            return f"Error promoting model: {str(e)}"
            
    except Exception as e:
        logger.error(f"❌ Error in promote_best_model_to_production: {e}")
        return f"Error promoting model: {str(e)}"

def promote_scaler_to_production(**context):
    """Promote scaler to Production stage"""
    try:
        # Check MLflow connection first
        if not check_mlflow_connection():
            return "❌ MLflow server is not accessible"
        
        mlflow.set_tracking_uri("http://mlflow:5003")
        client = MlflowClient()
        
        model_name = "water_quality_scaler"
        
        logger.info(f"🔍 Looking for scaler: {model_name}")
        
        # Get latest version using new API
        latest_version = get_latest_model_version(client, model_name)
        if not latest_version:
            logger.warning(f"⚠️ No versions found for scaler {model_name}")
            return f"No versions found for scaler {model_name}"
        
        version = latest_version.version
        logger.info(f"📋 Found scaler {model_name} version {version}")
        
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
                logger.info(f"ℹ️ Scaler {model_name} v{version} is already in Production")
                return f"✅ Scaler {model_name} v{version} already in Production"
            
            # Mark this version as production
            client.set_registered_model_alias(
                name=model_name,
                alias="Production",
                version=version
            )
            
            logger.info(f"🚀 Successfully promoted scaler v{version} to Production!")
            return f"✅ Scaler v{version} promoted to Production"
            
        except Exception as e:
            logger.error(f"❌ Error promoting scaler {model_name}: {e}")
            return f"Error promoting scaler: {str(e)}"
            
    except Exception as e:
        logger.error(f"❌ Error in promote_scaler_to_production: {e}")
        return f"Error promoting scaler: {str(e)}"

def verify_production_models(**context):
    """Verify that models are properly deployed in Production"""
    try:
        # Check MLflow connection first
        if not check_mlflow_connection():
            return "❌ MLflow server is not accessible"
        
        mlflow.set_tracking_uri("http://mlflow:5003")
        client = MlflowClient()
        
        models_to_check = ["water_quality_best_model", "water_quality_scaler"]
        production_models = []
        
        for model_name in models_to_check:
            try:
                # Check if model has Production alias
                try:
                    production_version = client.get_model_version_by_alias(model_name, "Production")
                    production_models.append(f"{model_name} v{production_version.version}")
                    logger.info(f"✅ {model_name} v{production_version.version} is in Production")
                except Exception as e:
                    logger.warning(f"⚠️ {model_name} not found in Production: {e}")
                    
            except Exception as e:
                logger.error(f"❌ Error checking {model_name}: {e}")
        
        if len(production_models) >= 2:  # Best model + Scaler
            logger.info("🎉 All required models are in Production stage!")
            logger.info(f"📋 Production Models: {', '.join(production_models)}")
            return f"✅ Production ready: {', '.join(production_models)}"
        else:
            logger.warning("⚠️ Some models missing from Production")
            return f"⚠️ Only {len(production_models)} models in Production: {', '.join(production_models)}"
            
    except Exception as e:
        logger.error(f"❌ Error in verify_production_models: {e}")
        return f"Error verifying models: {str(e)}"

# Tạo DAG
@dag(
    'promote_models_to_production',
    default_args=default_args,
    description='Promote trained models to Production stage in MLflow using new API',
    schedule=None,  # Manual trigger
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=['mlflow', 'production', 'model-deployment']
)
def promote_models_to_production(): 
    # Task 1: Promote best model to Production
    promote_best_model_task = PythonOperator(
        task_id='promote_best_model',
        python_callable=promote_best_model_to_production,
    )

    # Task 2: Promote scaler to Production
    promote_scaler_task = PythonOperator(
        task_id='promote_scaler',
        python_callable=promote_scaler_to_production,
    )

    # Task 3: Verify production deployment
    verify_models_task = PythonOperator(
        task_id='verify_production_models',
        python_callable=verify_production_models,
    )

    # Định nghĩa dependencies
    [promote_best_model_task, promote_scaler_task] >> verify_models_task

promote_models_to_production()