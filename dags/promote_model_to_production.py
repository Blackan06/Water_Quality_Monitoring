from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import logging
import mlflow
from mlflow.tracking import MlflowClient

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
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

def promote_best_model_to_production(**context):
    """Promote the best performing model to Production stage"""
    try:
        # Thiết lập MLflow client (VPS)
        mlflow.set_tracking_uri("http://77.37.44.237:5003")
        client = MlflowClient()
        
        # Tên model để promote
        model_name = "water_quality_best_model"
        
        logger.info(f"🔍 Looking for model: {model_name}")
        
        # Lấy latest version của best model
        try:
            latest_versions = client.get_latest_versions(model_name, stages=["None"])
            if not latest_versions:
                logger.error(f"❌ No versions found for model {model_name}")
                return f"No versions found for model {model_name}"
            
            latest_version = latest_versions[0]
            version = latest_version.version
            
            logger.info(f"📋 Found model {model_name} version {version}")
            
            # Kiểm tra xem có version nào đang ở Production không
            try:
                production_versions = client.get_latest_versions(model_name, stages=["Production"])
                if production_versions:
                    old_version = production_versions[0].version
                    # Archive old production version
                    client.transition_model_version_stage(
                        name=model_name,
                        version=old_version,
                        stage="Archived",
                        archive_existing_versions=False
                    )
                    logger.info(f"📦 Archived old production version {old_version}")
            except Exception as e:
                logger.info(f"ℹ️ No existing production version to archive: {e}")
            
            # Promote latest version to Production
            client.transition_model_version_stage(
                name=model_name,
                version=version,
                stage="Production",
                archive_existing_versions=False
            )
            
            logger.info(f"🚀 Successfully promoted {model_name} v{version} to Production!")
            
            # Log model details
            model_version = client.get_model_version(model_name, version)
            logger.info(f"📊 Production Model Details:")
            logger.info(f"   Name: {model_name}")
            logger.info(f"   Version: {version}")
            logger.info(f"   Stage: Production")
            logger.info(f"   Created: {model_version.creation_timestamp}")
            
            return f"✅ Model {model_name} v{version} promoted to Production"
            
        except Exception as e:
            logger.error(f"❌ Error accessing model {model_name}: {e}")
            return f"Error accessing model: {str(e)}"
            
    except Exception as e:
        logger.error(f"❌ Error in promote_best_model_to_production: {e}")
        return f"Error promoting model: {str(e)}"

def promote_scaler_to_production(**context):
    """Promote scaler to Production stage"""
    try:
        mlflow.set_tracking_uri("http://77.37.44.237:5003")
        client = MlflowClient()
        
        model_name = "water_quality_scaler"
        
        logger.info(f"🔍 Looking for scaler: {model_name}")
        
        try:
            latest_versions = client.get_latest_versions(model_name, stages=["None"])
            if not latest_versions:
                logger.warning(f"⚠️ No versions found for scaler {model_name}")
                return f"No versions found for scaler {model_name}"
            
            latest_version = latest_versions[0]
            version = latest_version.version
            
            # Archive old production scaler if exists
            try:
                production_versions = client.get_latest_versions(model_name, stages=["Production"])
                if production_versions:
                    old_version = production_versions[0].version
                    client.transition_model_version_stage(
                        name=model_name,
                        version=old_version,
                        stage="Archived"
                    )
                    logger.info(f"📦 Archived old production scaler v{old_version}")
            except Exception as e:
                logger.info(f"ℹ️ No existing production scaler to archive: {e}")
            
            # Promote latest scaler to Production
            client.transition_model_version_stage(
                name=model_name,
                version=version,
                stage="Production"
            )
            
            logger.info(f"🚀 Successfully promoted scaler v{version} to Production!")
            return f"✅ Scaler v{version} promoted to Production"
            
        except Exception as e:
            logger.error(f"❌ Error accessing scaler {model_name}: {e}")
            return f"Error accessing scaler: {str(e)}"
            
    except Exception as e:
        logger.error(f"❌ Error in promote_scaler_to_production: {e}")
        return f"Error promoting scaler: {str(e)}"

def verify_production_models(**context):
    """Verify that models are properly deployed in Production"""
    try:
        mlflow.set_tracking_uri("http://77.37.44.237:5003")
        client = MlflowClient()
        
        models_to_check = ["water_quality_best_model", "water_quality_scaler"]
        production_models = []
        
        for model_name in models_to_check:
            try:
                production_versions = client.get_latest_versions(model_name, stages=["Production"])
                if production_versions:
                    version = production_versions[0].version
                    production_models.append(f"{model_name} v{version}")
                    logger.info(f"✅ {model_name} v{version} is in Production")
                else:
                    logger.warning(f"⚠️ {model_name} not found in Production")
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
dag = DAG(
    'promote_models_to_production',
    default_args=default_args,
    description='Promote trained models to Production stage in MLflow',
    schedule_interval=None,  # Manual trigger
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=['mlflow', 'production', 'model-deployment']
)

# Task 1: Promote best model to Production
promote_best_model_task = PythonOperator(
    task_id='promote_best_model',
    python_callable=promote_best_model_to_production,
    dag=dag
)

# Task 2: Promote scaler to Production
promote_scaler_task = PythonOperator(
    task_id='promote_scaler',
    python_callable=promote_scaler_to_production,
    dag=dag
)

# Task 3: Verify production deployment
verify_models_task = PythonOperator(
    task_id='verify_production_models',
    python_callable=verify_production_models,
    dag=dag
)

# Định nghĩa dependencies
[promote_best_model_task, promote_scaler_task] >> verify_models_task