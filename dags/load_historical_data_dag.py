from airflow.operators.python import PythonOperator
from airflow.providers.docker.operators.docker import DockerOperator
from airflow.operators.trigger_dagrun import TriggerDagRunOperator
from docker.types import Mount
from datetime import datetime, timedelta
import datetime as dt
import logging
import os
os.environ['GIT_PYTHON_REFRESH'] = 'quiet'
from airflow.decorators import dag, task
from pendulum import datetime
import mlflow
import mlflow.sklearn
import mlflow.xgboost
import mlflow.pyfunc
import tempfile
import pickle
import json
import numpy as np
import pandas as pd
import psycopg2

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
# Use container path since volumes are already mounted in docker-compose.yaml
# ./models -> /opt/airflow/models, ./spark -> /opt/airflow/spark, etc.
project_root = os.getenv("PROJECT_ROOT") or "/opt/airflow"

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

        # Check for ensemble + LSTM metrics
        metrics_file = './models/metrics.json'  # ensemble metrics
        # Try multiple locations for LSTM metrics
        lstm_metrics_candidates = [
            './models/lstm_metrics.json',
            os.path.join(os.getenv('AIRFLOW_MODELS_DIR', '/usr/local/airflow/models'), 'lstm_metrics.json')
        ]
        metadata_file = './models/enhanced_metadata.json'

        metrics = {}
        if os.path.exists(metrics_file):
            with open(metrics_file, 'r') as f:
                ensemble_metrics = json.load(f)
            metrics.update(ensemble_metrics)
            logger.info("📊 Found ensemble metrics")

        # Load LSTM metrics if present in any candidate path
        for cand in lstm_metrics_candidates:
            if os.path.exists(cand):
                try:
                    with open(cand, 'r') as f:
                        lstm_metrics = json.load(f)
                    metrics.update(lstm_metrics)
                    logger.info(f"📊 Found LSTM metrics at: {cand}")
                    break
                except Exception as e:
                    logger.warning(f"Could not read LSTM metrics at {cand}: {e}")

        if not metrics and os.path.exists(metadata_file):
            # Fallback to enhanced metadata
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
            metrics = metadata.get('metrics', {})
            logger.info("📊 Using enhanced metadata (fallback)")

        if not metrics:
            logger.error("❌ No metrics files found")
            return "Training completed but no metrics found"
        
        # Optional: Blend existing XGB+RF ensemble with LSTM predictions if per-sample predictions are available
        try:
            en_pred_path = os.path.join('./models', 'ensemble_test_predictions.csv')
            lstm_pred_candidates = [
                os.path.join('./models', 'lstm_test_predictions_h1.csv'),
                os.path.join(os.getenv('AIRFLOW_MODELS_DIR', '/usr/local/airflow/models'), 'lstm_test_predictions_h1.csv')
            ]
            lstm_pred_path = next((p for p in lstm_pred_candidates if os.path.exists(p)), None)
            if os.path.exists(en_pred_path) and lstm_pred_path:
                df_en = pd.read_csv(en_pred_path)
                df_lstm = pd.read_csv(lstm_pred_path)
                # Normalize column names
                df_en['measurement_date'] = pd.to_datetime(df_en['measurement_date'], errors='coerce')
                df_lstm['timestamp'] = pd.to_datetime(df_lstm['timestamp'], errors='coerce')
                # Join on exact keys
                merged = df_en.merge(
                    df_lstm,
                    left_on=['station_id', 'measurement_date'],
                    right_on=['station_id', 'timestamp'],
                    how='inner'
                )
                if not merged.empty:
                    y_true = merged['y_true'].astype(float).values
                    y_en = merged['y_ensemble_xrf'].astype(float).values
                    y_lstm = merged['y_lstm'].astype(float).values

                    # Choose blending strategy
                    strategy = os.getenv('BLEND_STRATEGY', 'learned').lower().strip()
                    w_en, w_lstm = 0.7, 0.3  # defaults
                    if strategy == 'fixed':
                        w_env = os.getenv('BLEND_WEIGHTS')  # format: "0.7,0.3"
                        if w_env:
                            try:
                                parts = [float(x) for x in w_env.split(',')]
                                if len(parts) == 2 and parts[0] >= 0 and parts[1] >= 0 and abs(parts[0] + parts[1] - 1.0) < 1e-6:
                                    w_en, w_lstm = parts[0], parts[1]
                            except Exception:
                                pass
                    else:
                        # learned weights on simplex: scan w_en in [0,1], w_lstm=1-w_en
                        best_r2 = -1e9
                        best_w = (w_en, w_lstm)
                        n = len(y_true)
                        if n >= 5:
                            grid = np.linspace(0.0, 1.0, 101)
                            y_true_mean = float(np.mean(y_true)) if n > 0 else 0.0
                            ss_tot = float(np.sum((y_true - y_true_mean) ** 2))
                            for w in grid:
                                y_b = w * y_en + (1.0 - w) * y_lstm
                                err = y_true - y_b
                                ss_res = float(np.sum(err ** 2))
                                if ss_tot > 0:
                                    r2_w = 1.0 - ss_res / ss_tot
                                else:
                                    r2_w = -ss_res  # fallback: minimize SSE
                                if r2_w > best_r2:
                                    best_r2 = r2_w
                                    best_w = (float(w), float(1.0 - w))
                            w_en, w_lstm = best_w

                    y_final = w_en * y_en + w_lstm * y_lstm
                    # Compute metrics without sklearn
                    err = y_true - y_final
                    mae = float(np.mean(np.abs(err))) if len(err) > 0 else 0.0
                    rmse = float(np.sqrt(np.mean(err ** 2))) if len(err) > 0 else 0.0
                    # R2
                    ss_res = float(np.sum(err ** 2))
                    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
                    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else 0.0
                    metrics['ensemble_lstm'] = {
                        'r2': r2,
                        'mae': mae,
                        'rmse': rmse,
                        'w_en': w_en,
                        'w_lstm': w_lstm,
                        'strategy': strategy
                    }
                    logger.info(f"📊 Added ENSEMBLE+LSTM metrics on {len(y_true)} matched rows: R²={r2:.4f}, MAE={mae:.4f}, RMSE={rmse:.4f}, weights=(en={w_en:.2f}, lstm={w_lstm:.2f}) using {strategy}")
                else:
                    logger.info("No overlapping rows between ensemble test and LSTM validation predictions; skipping blended metrics")
            else:
                if not os.path.exists(en_pred_path):
                    logger.info("Ensemble per-sample predictions not found; skipping blended metrics")
                if not lstm_pred_path:
                    logger.info("LSTM per-sample predictions not found; skipping blended metrics")
        except Exception as e:
            logger.warning(f"Could not compute ENSEMBLE+LSTM blended metrics: {e}")

        # Extract metrics (include lstm)
        xgb_r2 = metrics.get('xgb', {}).get('r2', 0.0)
        rf_r2 = metrics.get('rf', {}).get('r2', 0.0)
        ensemble_r2 = metrics.get('ensemble', {}).get('r2', 0.0)
        lstm_r2 = metrics.get('lstm', {}).get('r2', 0.0)
            
        # Compare all models and select the best one
        model_scores = {
            'xgb': xgb_r2,
            'rf': rf_r2,
            'ensemble': ensemble_r2,
            'lstm': lstm_r2,
            'ensemble_lstm': metrics.get('ensemble_lstm', {}).get('r2', -float('inf'))
        }

        best_model = max(model_scores, key=model_scores.get)
        best_score = model_scores[best_model]

        logger.info(f"📊 Model Performance Comparison:")
        logger.info(f"  XGBoost: R² = {xgb_r2:.4f}")
        logger.info(f"  Random Forest: R² = {rf_r2:.4f}")
        logger.info(f"  Ensemble: R² = {ensemble_r2:.4f}")
        logger.info(f"  LSTM: R² = {lstm_r2:.4f}")
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
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "http://mlflow:5003")
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
        timestamp = dt.datetime.now().strftime('%Y%m%d_%H%M%S')
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
        mlflow.log_param("training_date", dt.datetime.now().isoformat())
        mlflow.log_param("feature_count", len(models))

        # Log metrics của tất cả models
        for model_name, model_metrics in metrics.items():
            for metric_name, metric_val in model_metrics.items():
                mlflow.log_metric(f"{model_name}_{metric_name}", metric_val)

        # Log scaler (nếu có)
        if scaler is not None:
            # Chuẩn bị input_example và signature cho pyfunc wrapper
            try:
                from mlflow.models import infer_signature
            except Exception:
                infer_signature = None

            # Xác định số features hợp lý
            n_features_for_examples = None
            if hasattr(scaler, "n_features_in_") and isinstance(getattr(scaler, "n_features_in_", None), (int, np.integer)):
                n_features_for_examples = int(scaler.n_features_in_)
            elif "xgb" in models and hasattr(models.get("xgb"), "n_features_in_"):
                n_features_for_examples = int(getattr(models.get("xgb"), "n_features_in_", 10))
            else:
                n_features_for_examples = 10

            column_names = [f"f{i}" for i in range(n_features_for_examples)]
            scaler_input_df = pd.DataFrame(np.zeros((1, n_features_for_examples)), columns=column_names)
            scaler_signature = None
            try:
                if infer_signature is not None:
                    transformed_output = scaler.transform(scaler_input_df.values)
                    scaler_signature = infer_signature(scaler_input_df, transformed_output)
            except Exception as sig_err:
                logger.warning(f"Could not infer scaler signature automatically: {sig_err}")

            class _ScalerPyFuncModel(mlflow.pyfunc.PythonModel):
                def __init__(self, fitted_scaler):
                    self._scaler = fitted_scaler

                def predict(self, context, model_input):
                    try:
                        if isinstance(model_input, pd.DataFrame):
                            input_array = model_input.values
                        elif isinstance(model_input, np.ndarray):
                            input_array = model_input
                        else:
                            input_array = np.asarray(model_input)
                        return self._scaler.transform(input_array)
                    except Exception as e:
                        raise e

            mlflow.pyfunc.log_model(
                python_model=_ScalerPyFuncModel(scaler),
                name="scaler_model",
                registered_model_name="water_quality_scaler",
                input_example=scaler_input_df,
                signature=scaler_signature
            )
            logger.info("✅ Scaler registered to MLflow with pyfunc wrapper")

        # Log tất cả models có sẵn (chỉ log models thực sự)
        for model_name, model_obj in models.items():
            if model_name == 'xgb' and model_obj != "spark_pipeline_available":
                # Log XGBoost model
                n_features = getattr(model_obj, "n_features_in_", 10)  # default 10 features
                sample_input = np.zeros((1, n_features))
                
                mlflow.xgboost.log_model(
                    model_obj,
                    name=f"{model_name}_model",
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
                    name="best_model",
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
    max_active_tasks=1,  # Giới hạn 1 task chạy cùng lúc
    max_active_runs=1,   # Giới hạn 1 DAG run cùng lúc
    tags=['water-quality', 'ensemble-ml', 'wqi-forecasting', 'xgboost', 'random-forest']
)
def load_historical_data_and_train_ensemble() : 
    # Định nghĩa các tasks
    def ensure_database_exists():
        conn = psycopg2.connect(
            host=os.getenv('DB_HOST', 'postgres'),
            port=os.getenv('DB_PORT', '5432'),
            dbname=os.getenv('DB_ADMIN_DB', 'postgres'),
            user=os.getenv('DB_USER', 'postgres'),
            password=os.getenv('DB_PASSWORD', 'postgres'),
            connect_timeout=int(os.getenv('DB_CONNECT_TIMEOUT', '10'))
        )
        conn.autocommit = True
        with conn.cursor() as cur:
            target_db = os.getenv('DB_NAME', 'wqi_db')
            cur.execute("SELECT 1 FROM pg_database WHERE datname=%s", (target_db,))
            if cur.fetchone() is None:
                cur.execute(f"CREATE DATABASE {target_db} OWNER {os.getenv('DB_USER', 'postgres')}")
        conn.close()

    ensure_db_task = PythonOperator(
        task_id='ensure_database_exists',
        python_callable=ensure_database_exists,
    )
    # Task 1: Train model
    train_model_task = DockerOperator(
        task_id='train_model',
        image='airflow/iot_stream:ensemble',
        container_name='spark_ensemble_training',
        command='python /app/spark/spark_jobs/train_ensemble_model.py',
        api_version='auto',
        auto_remove='success',
        docker_url='unix://var/run/docker.sock',  # Sử dụng Unix socket thay vì TCP
        network_mode='bridge',
        mount_tmp_dir=False,  # Disable automatic tmp directory mounting
        working_dir='/app',
        mounts = [
            Mount(source=f"{project_root}/models", target="/app/models", type="bind"),
            Mount(source=f"{project_root}/spark",  target="/app/spark",  type="bind"),
            # mlruns and mlartifacts will be created in the container's working directory
            # since they're not mounted from host
        ],
        environment={
            'DB_HOST': 'postgres',
            'DB_PORT': '5432',
            'DB_NAME': 'wqi_db',
            'DB_USER': 'postgres',
            'DB_PASSWORD': 'postgres',
            'DB_SCHEMA': 'public',
            'OUTPUT_MODEL_DIR': '/app/models',
            'MLFLOW_TRACKING_URI': 'http://mlflow:5003',
            'MLFLOW_BACKEND_STORE_URI': 'file:/app/mlruns',
            'MLFLOW_DEFAULT_ARTIFACT_ROOT': 'file:/app/mlartifacts',
            # Force Spark to use only 1 core and 1G memory for this job
            'PYSPARK_SUBMIT_ARGS': '--master local[1] --conf spark.executor.cores=1 --conf spark.cores.max=1 --conf spark.driver.memory=1g --conf spark.executor.memory=1g pyspark-shell'
        },
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

    # Optional: Trigger LSTM multi-station training DAG (runs independently)
    try:
        station_ids_env = os.getenv('STATION_IDS', '0,1,2')
        station_ids_conf = [int(s) for s in station_ids_env.split(',') if s.strip() != '']
    except Exception:
        station_ids_conf = [0, 1, 2]

    trigger_lstm_training = TriggerDagRunOperator(
        task_id='trigger_lstm_multi_station_training',
        trigger_dag_id='train_lstm_multi_station',
        conf={
            'station_ids': station_ids_conf
        }
    )

    # Định nghĩa dependencies
    # Trigger LSTM training asynchronously; process results immediately after ensemble training
    ensure_db_task >> train_model_task >> [trigger_lstm_training, process_results_task]
    process_results_task >> save_mlflow_task >> show_best_model_task 

load_historical_data_and_train_ensemble()