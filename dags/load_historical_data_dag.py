from airflow.operators.python import PythonOperator
from airflow.providers.docker.operators.docker import DockerOperator
from docker.types import Mount
from datetime import datetime, timedelta
import logging
import os
os.environ['GIT_PYTHON_REFRESH'] = 'quiet'
from airflow.decorators import dag, task
from pendulum import datetime
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

# Default arguments
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2024, 1, 1),
    'email_on_failure': False,
    'email_on_retry': False,
}

def process_training_results(**context):
    """Process results from training and show best model information"""
    try:
        logger.info("Processing training results...")
        
        import os
        import json
        
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
        logger.error(f"Error processing training results: {e}")
        return f"Training processing failed: {str(e)}"

def save_models_to_mlflow(**context):
    """Save trained models to MLflow registry (with dynamic input_example and correct log_model)."""
    logger.info("Saving models to MLflow registry...")

    # 1) Thiết lập MLflow tracking URI (VPS backend)
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "http://77.37.44.237:5003")
    mlflow.set_tracking_uri(tracking_uri)
    
    # Handle experiment creation/selection with error handling
    experiment_name = "water_quality_models"
    try:
        # Try to set existing experiment
        mlflow.set_experiment(experiment_name)
        logger.info(f"✅ Using existing experiment: {experiment_name}")
    except Exception as e:
        logger.warning(f"⚠️ Experiment '{experiment_name}' not found or deleted: {e}")
        logger.info("🔄 Creating new experiment...")
        
        # Create new experiment with timestamp to avoid conflicts
        from datetime import datetime
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        new_experiment_name = f"water_quality_models_{timestamp}"
        
        try:
            # Create experiment via MLflow client
            client = mlflow.tracking.MlflowClient()
            experiment = client.create_experiment(new_experiment_name)
            mlflow.set_experiment(new_experiment_name)
            logger.info(f"✅ Created new experiment: {new_experiment_name}")
        except Exception as create_error:
            logger.error(f"❌ Failed to create experiment: {create_error}")
            # Fallback to default experiment
            mlflow.set_experiment("Default")
            logger.info("ℹ️ Using default experiment as fallback")

    # 2) Đường dẫn file model
    xgb_path     = './models/xgb.pkl'
    rf_path      = './models/rf.pkl'
    scaler_path  = './models/scaler.pkl'
    metrics_path = './models/metrics.json'

    # 3) Kiểm tra tồn tại của metrics (bắt buộc)
    if not os.path.exists(metrics_path):
        logger.error(f"Metrics file not found: {metrics_path}")
        return f"Metrics file not found: {metrics_path}"

    # 4) Load metrics
    with open(metrics_path, 'r') as f:
        metrics = json.load(f)
    
    # 5) Load scaler nếu có
    scaler = None
    if os.path.exists(scaler_path):
        try:
            with open(scaler_path, 'rb') as f:
                scaler = pickle.load(f)
            logger.info("✅ Scaler loaded successfully")
        except (EOFError, pickle.UnpicklingError) as e:
            logger.error(f"❌ Failed to load scaler: {e}")
            logger.info("ℹ️ Continuing without scaler")
    else:
        logger.warning("⚠️ Scaler file not found")
    
    # 6) Load models with error handling
    models = {}
    
    # Load XGBoost model
    if os.path.exists(xgb_path):
        try:
            with open(xgb_path, 'rb') as f:
                models['xgb'] = pickle.load(f)
            logger.info("✅ XGBoost model loaded successfully")
        except (EOFError, pickle.UnpicklingError) as e:
            logger.error(f"❌ Failed to load XGBoost model: {e}")
    else:
        logger.warning("⚠️ XGBoost model file not found")
    
    # Load Random Forest model (stored as Spark pipeline)
    rf_pipeline_path = './models/rf_pipeline'
    if os.path.exists(rf_pipeline_path):
        try:
            # RF được lưu như Spark pipeline, chúng ta sẽ note nó có sẵn
            # nhưng không load vào memory vì cần Spark context
            logger.info("✅ Random Forest pipeline found (Spark format)")
            # Tạo placeholder để đánh dấu RF có sẵn trong metrics
            models['rf'] = "spark_pipeline_available"
        except Exception as e:
            logger.error(f"❌ Failed to access Random Forest pipeline: {e}")
    else:
        logger.warning("⚠️ Random Forest pipeline not found")
    
    # Kiểm tra có ít nhất 1 model
    if not models:
        logger.error("❌ No models could be loaded!")
        return "No valid models found to register"

    # 7) Chọn best model từ các models có sẵn
    available_metrics = {k: v for k, v in metrics.items() if k in models}
    if not available_metrics:
        logger.error("❌ No metrics found for available models!")
        return "No metrics found for loaded models"
    
    best_name, best_info = max(available_metrics.items(), key=lambda kv: kv[1].get('r2', -float('inf')))
    best_r2 = best_info.get('r2', 0.0)
    logger.info(f"🏆 Best model: {best_name} (R²: {best_r2:.4f}) from available models: {list(models.keys())}")

    # 8) Bắt đầu MLflow run
    with mlflow.start_run(run_name="comprehensive_ensemble_training"):
        # Log các tham số chung
        mlflow.log_param("best_model", best_name)
        mlflow.log_param("training_date", datetime.now().isoformat())
        mlflow.log_param("feature_count", len(models))

        # Log metrics của tất cả models
        for model_name, model_metrics in metrics.items():
            for metric_name, metric_val in model_metrics.items():
                mlflow.log_metric(f"{model_name}_{metric_name}", metric_val)

        # Log scaler (nếu có)
        if scaler is not None:
            mlflow.sklearn.log_model(
                scaler,
                artifact_path="scaler_model",
                registered_model_name="water_quality_scaler"
            )
            logger.info("✅ Scaler registered to MLflow")

        # Log tất cả models có sẵn (chỉ log models thực sự)
        for model_name, model_obj in models.items():
            if model_name == 'xgb' and model_obj != "spark_pipeline_available":
                # Log XGBoost model
                n_features = getattr(model_obj, "n_features_in_", 10)  # default 10 features
                sample_input = np.zeros((1, n_features))
                
                mlflow.xgboost.log_model(
                    model_obj,
                    artifact_path=f"{model_name}_model",
                    registered_model_name=f"water_quality_{model_name}_model",
                    input_example=sample_input
                )
                logger.info(f"✅ {model_name.upper()} model registered to MLflow")
                
            elif model_name == 'rf' and model_obj == "spark_pipeline_available":
                # RF là Spark pipeline, không thể log vào MLflow trực tiếp
                logger.info(f"ℹ️ {model_name.upper()} pipeline available but not logged (Spark format)")
                # Log artifact path thay thế
                mlflow.log_artifact('./models/rf_pipeline', artifact_path="rf_pipeline_model")
                logger.info(f"✅ {model_name.upper()} pipeline logged as artifact")
        
        # Log best model với tên riêng
        if best_name in models:
            best_model_obj = models[best_name]
            if best_name == 'xgb' and best_model_obj != "spark_pipeline_available":
                n_features = getattr(best_model_obj, "n_features_in_", 10)
                sample_input = np.zeros((1, n_features))
                mlflow.xgboost.log_model(
                    best_model_obj,
                    artifact_path="best_model",
                    registered_model_name="water_quality_best_model",
                    input_example=sample_input
                )
                logger.info(f"✅ Best model ({best_name.upper()}) registered as water_quality_best_model")
            elif best_name == 'rf' and best_model_obj == "spark_pipeline_available":
                # RF pipeline không thể register như model, chỉ log artifact
                mlflow.log_artifact('./models/rf_pipeline', artifact_path="best_model_rf_pipeline")
                logger.info(f"✅ Best model ({best_name.upper()}) pipeline logged as artifact")
            else:
                logger.warning(f"⚠️ Cannot register best model {best_name} - unsupported format")

        # Log toàn bộ metrics.json như artifact
        with tempfile.NamedTemporaryFile("w", delete=False, suffix=".json") as tmp:
            json.dump(metrics, tmp, indent=2)
            tmp_path = tmp.name
        mlflow.log_artifact(tmp_path, artifact_path="metrics")
        os.remove(tmp_path)

        run_id = mlflow.active_run().info.run_id
        logger.info(f"✅ MLflow run completed: {run_id}")

    return f"Best model saved to MLflow: {best_name} (R²: {best_r2:.4f})"

def _cleanup_inferior_models(best_model):
    """Xóa các model không tốt nhất để tiết kiệm dung lượng"""
    try:
        import os
        
        # Danh sách tất cả model files
        model_files = [
            'models/enhanced_xgb_model.pkl',
            'models/enhanced_rf_model.pkl',
            'models/enhanced_feature_pipeline.pkl',
            'models/enhanced_scaler.pkl'
        ]
        
        # Xác định file cần giữ lại
        best_model_file = f'models/enhanced_{best_model}_model.pkl'
        
        deleted_count = 0
        for model_file in model_files:
            if os.path.exists(model_file) and model_file != best_model_file:
                try:
                    os.remove(model_file)
                    logger.info(f"🗑️ Deleted inferior model: {model_file}")
                    deleted_count += 1
                except Exception as e:
                    logger.warning(f"Could not delete {model_file}: {e}")
        
        logger.info(f"🧹 Cleanup completed: deleted {deleted_count} inferior models, kept {best_model}")
        
    except Exception as e:
        logger.error(f"Error during model cleanup: {e}")

def show_best_model_info(**context):
    """Hiển thị thông tin model tốt nhất"""
    try:
        logger.info("Displaying best model information...")
        
        # Get training results from XCom
        training_metrics = context['task_instance'].xcom_pull(
            task_ids='process_training_results', key='training_metrics'
        )
        best_model_info = context['task_instance'].xcom_pull(
            task_ids='process_training_results', key='best_model_info'
        )
        
        if best_model_info:
            best_model = best_model_info.get('best_model', 'Unknown')
            best_score = best_model_info.get('best_score', 0.0)
            feature_count = best_model_info.get('feature_count', 0)
            
            logger.info("🏆 BEST MODEL INFORMATION:")
            logger.info(f"  Model Type: {best_model.upper()}")
            logger.info(f"  R² Score: {best_score:.4f}")
            logger.info(f"  Feature Count: {feature_count}")
            logger.info(f"  Model File: enhanced_{best_model}_model.pkl")
            
            if training_metrics:
                logger.info("📊 All Model Performance:")
                for model_name, metrics in training_metrics.items():
                    r2 = metrics.get('r2', 0.0)
                    mae = metrics.get('mae', 0.0)
                    rmse = metrics.get('rmse', 0.0)
                    logger.info(f"  {model_name.upper()}: R²={r2:.4f}, MAE={mae:.4f}, RMSE={rmse:.4f}")
            
            return f"Best model: {best_model.upper()} (R²: {best_score:.4f}) with {feature_count} features"
        else:
            logger.warning("No best model information found")
            return "No best model information available"
        
    except Exception as e:
        logger.error(f"Error showing best model info: {e}")
        return f"Error displaying best model info: {str(e)}"

# Tạo DAG
@dag(
    'load_historical_data_and_train_ensemble',
    default_args=default_args,
    description='Load historical WQI data and train enhanced ensemble models',
    schedule=None,  # Chạy thủ công
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=['water-quality', 'ensemble-ml', 'wqi-forecasting', 'xgboost', 'random-forest']
)
def load_historical_data_and_train_ensemble() : 
    # Định nghĩa các tasks
    # Task 1: Train model
    train_model_task = DockerOperator(
        task_id='train_model',
        image='airflow/iot_stream:ensemble',
        container_name='spark_ensemble_training',
        api_version='auto',
        auto_remove=True,
        docker_url='unix://var/run/docker.sock',
        network_mode='bridge',
        mount_tmp_dir=False,  # Disable automatic tmp directory mounting
        mounts=[
            Mount(source=os.getenv('PROJECT_ROOT', '/root/Water_Quality_Monitoring') + '/models', 
                target='/app/models', type='bind'),
            Mount(source=os.getenv('PROJECT_ROOT', '/root/Water_Quality_Monitoring') + '/spark', 
                target='/app/spark', type='bind')
        ],
        environment={
            'DB_HOST': '194.238.16.14',
            'DB_PORT': '5432',
            'DB_NAME': 'wqi_db',
            'DB_USER': 'postgres',
            'DB_PASSWORD': 'postgres1234',
            'DB_SCHEMA': 'public',
            'OUTPUT_MODEL_DIR': '/app/models',
            'MLFLOW_TRACKING_URI': 'http://77.37.44.237:5003',
            # Force Spark to use only 1 core and 1G memory for this job
            'PYSPARK_SUBMIT_ARGS': '--master local[1] --conf spark.executor.cores=1 --conf spark.cores.max=1 --conf spark.driver.memory=1g --conf spark.executor.memory=1g pyspark-shell'
        },
        command='python /app/spark/spark_jobs/train_ensemble_model.py',
    )

    # Task 2: Process training results
    process_results_task = PythonOperator(
        task_id='process_training_results',
        python_callable=process_training_results,
    )

    # Task 3: Save models to MLflow
    save_mlflow_task = PythonOperator(
        task_id='save_models_to_mlflow',
        python_callable=save_models_to_mlflow,
    )

    # Task 4: Show best model information
    show_best_model_task = PythonOperator(
        task_id='show_best_model_info',
        python_callable=show_best_model_info,
    )

    # Định nghĩa dependencies - Sequential execution
    train_model_task >> process_results_task >> save_mlflow_task >> show_best_model_task 

load_historical_data_and_train_ensemble()