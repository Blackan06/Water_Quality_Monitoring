from airflow.operators.python import PythonOperator
from airflow.decorators import dag
from pendulum import datetime
import logging
import os
import pandas as pd
import psycopg2
import json

# Avoid heavy imports at module import time to prevent DagBag timeouts.
# Import LSTMTrainingService lazily inside task functions.


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
}


def _get_db_conn():
    return psycopg2.connect(
        host=os.getenv('DB_HOST', '194.238.16.14'),
        port=os.getenv('DB_PORT', '5432'),
        database=os.getenv('DB_NAME', 'wqi_db'),
        user=os.getenv('DB_USER', 'postgres'),
        password=os.getenv('DB_PASSWORD', 'postgres1234'),
        connect_timeout=int(os.getenv('DB_CONNECT_TIMEOUT', '10'))
    )


def load_lstm_data(**context):
    """Load dữ liệu historical cho nhiều trạm trong khoảng 2003-01-15 → 2023-12-15 (mặc định).
    Cho phép override qua dag_run.conf: station_ids, start_date, end_date.
    Lưu DataFrame ra file để downstream tasks đọc.
    """
    conf = (context.get('dag_run') and context['dag_run'].conf) or {}
    # Prefer env override if dag_run.conf not provided
    station_ids = conf.get('station_ids') or [int(s) for s in os.getenv('STATION_IDS', '0,1,2').split(',')]
    start_date = conf.get('start_date', '2003-01-15')
    end_date = conf.get('end_date', '2023-12-15')

    logger.info(f"Loading historical data for stations={station_ids}, range={start_date}→{end_date}")

    query = (
        """
        SELECT station_id, measurement_date, ph, temperature, "do", wqi
        FROM historical_wqi_data
        WHERE station_id = ANY(%s)
          AND measurement_date BETWEEN %s AND %s
        ORDER BY station_id, measurement_date
        """
    )

    output_dir = os.getenv('AIRFLOW_MODELS_DIR', '/usr/local/airflow/models')
    os.makedirs(output_dir, exist_ok=True)
    data_path = os.path.join(output_dir, 'lstm_training_data.csv')
    # Remove existing file to avoid mixing old data
    try:
        if os.path.exists(data_path):
            os.remove(data_path)
    except Exception:
        pass

    record_count = 0
    station_set = set()

    with _get_db_conn() as conn:
        # Set statement timeout to avoid long-running queries
        try:
            with conn.cursor() as cset:
                # 5 minutes timeout
                cset.execute("SET statement_timeout TO '300000'")
        except Exception:
            logger.warning("Could not set statement_timeout; proceeding with defaults")

        # Validate requested stations against DB within date range
        try:
            st_df = pd.read_sql(
                """
                SELECT DISTINCT station_id 
                FROM historical_wqi_data 
                WHERE measurement_date BETWEEN %s AND %s
                ORDER BY station_id
                """,
                conn,
                params=(start_date, end_date)
            )
            available = st_df['station_id'].astype(int).tolist()
        except Exception as e:
            logger.warning(f"Failed to fetch available stations: {e}")
            available = []

        if available:
            filtered_station_ids = [int(s) for s in station_ids if int(s) in available]
            if not filtered_station_ids:
                filtered_station_ids = available
            if filtered_station_ids != station_ids:
                logger.info(f"Adjusted stations to available set in range: {filtered_station_ids}")
        else:
            filtered_station_ids = station_ids

        # Stream data in chunks to avoid memory pressure and long stalls
        chunksize = int(os.getenv('DB_CHUNKSIZE', '50000'))
        try:
            chunk_iter = pd.read_sql(
                query,
                conn,
                params=(filtered_station_ids, start_date, end_date),
                chunksize=chunksize
            )
        except Exception as e:
            logger.error(f"Failed to start chunked read: {e}")
            # Fallback to single read (may still fail)
            chunk_iter = [pd.read_sql(query, conn, params=(filtered_station_ids, start_date, end_date))]

        written_any = False
        for idx, chunk in enumerate(chunk_iter, start=1):
            if chunk is None or chunk.empty:
                continue
            # Ensure datetime and timestamp column
            try:
                chunk['measurement_date'] = pd.to_datetime(chunk['measurement_date'])
            except Exception:
                pass
            chunk['timestamp'] = chunk['measurement_date']
            # Append to CSV incrementally
            header = not written_any
            try:
                chunk.to_csv(data_path, mode='a', header=header, index=False)
                written_any = True
            except Exception as e:
                logger.error(f"Failed to write chunk {idx} to CSV: {e}")
                raise
            # Update counters
            record_count += len(chunk)
            try:
                station_set.update(chunk['station_id'].astype(int).unique().tolist())
            except Exception:
                pass
            if idx % 5 == 0:
                logger.info(f"Streamed {record_count} rows so far... (chunk {idx})")

    station_list = sorted(list(station_set)) if record_count > 0 else []
    if record_count == 0:
        logger.warning("No historical data returned for the specified criteria")
    logger.info(f"Loaded {record_count} records for stations: {station_list}")

    # Push XCom
    ti = context['task_instance']
    ti.xcom_push(key='data_path', value=data_path)
    ti.xcom_push(key='record_count', value=record_count)
    ti.xcom_push(key='stations', value=station_list)

    # Thông tin cho logs
    return json.dumps({
        'data_path': data_path,
        'records': record_count,
        'stations': station_list
    })


def train_lstm_model(**context):
    """Train LSTM global model trên dữ liệu nhiều trạm."""
    ti = context['task_instance']
    data_path = ti.xcom_pull(task_ids='load_lstm_data', key='data_path')
    if not data_path or not os.path.exists(data_path):
        logger.error(f"Training data not found at {data_path}")
        return "No data to train"

    df = pd.read_csv(data_path, parse_dates=['measurement_date', 'timestamp'])
    logger.info(f"Training LSTM on {len(df)} records, stations={sorted(df['station_id'].unique().tolist())}")

    # Lấy hyperparams từ dag_run.conf nếu có
    conf = (context.get('dag_run') and context['dag_run'].conf) or {}
    params = {
        'sequence_length': conf.get('sequence_length', 12),
        'forecast_horizon': conf.get('forecast_horizon', 1),
        'lstm_units': conf.get('lstm_units', 64),
        'dropout_rate': conf.get('dropout_rate', 0.2),
        'learning_rate': conf.get('learning_rate', 0.001),
        'epochs': conf.get('epochs', 300),
        'batch_size': conf.get('batch_size', 32),
        'validation_split': conf.get('validation_split', 0.2),
        'l2_weight': conf.get('l2_weight', 1e-4),
        'conv_filters': conf.get('conv_filters', 16),
        'conv_kernel_size': conf.get('conv_kernel_size', 3),
        'gamma_shrink': conf.get('gamma_shrink', 0.8),
    }

    # Khởi tạo service và train (lazy import to avoid heavy DagBag import)
    from include.iot_streaming.lstm_training_service import LSTMTrainingService
    lstm_service = LSTMTrainingService()
    results = lstm_service.train_global_lstm(df, **params)

    if not results or results.get('error'):
        err = results.get('error') if isinstance(results, dict) else 'Unknown error'
        logger.error(f"LSTM training failed: {err}")
        return f"Failed: {err}"

    # Lưu model
    model = results.get('model')
    output_dir = os.getenv('AIRFLOW_MODELS_DIR', '/usr/local/airflow/models')
    os.makedirs(output_dir, exist_ok=True)
    model_path = os.path.join(output_dir, 'lstm_global_model.h5')
    try:
        model.save(model_path)
        logger.info(f"✅ Saved LSTM model to {model_path}")
    except Exception as e:
        logger.warning(f"Could not save model to {model_path}: {e}")

    # Push metrics (train/test and per-sample horizon-1 outputs for downstream blending)
    ti.xcom_push(key='train_metrics', value=results.get('train_metrics'))
    ti.xcom_push(key='test_metrics', value=results.get('test_metrics'))
    ti.xcom_push(key='model_path', value=model_path)
    ti.xcom_push(key='baseline_metrics', value=results.get('baseline_metrics'))
    ti.xcom_push(key='per_station_test', value=results.get('per_station_test'))
    # Persist per-sample TEST predictions for H=1 to CSV for downstream blending
    try:
        keys = results.get('test_keys_h1')
        y_true = results.get('y_true_test_h1')
        y_pred = results.get('y_pred_test_h1')
        if keys and y_true and y_pred and len(keys) == len(y_true) == len(y_pred):
            out_dir = os.getenv('AIRFLOW_MODELS_DIR', '/usr/local/airflow/models')
            os.makedirs(out_dir, exist_ok=True)
            out_path = os.path.join(out_dir, 'lstm_test_predictions_h1.csv')
            import csv
            with open(out_path, 'w', newline='') as f:
                w = csv.writer(f)
                w.writerow(['station_id', 'timestamp', 'y_true', 'y_lstm'])
                for k, yt, yp in zip(keys, y_true, y_pred):
                    w.writerow([k.get('station_id'), k.get('timestamp'), yt, yp])
            logger.info(f"✅ Wrote LSTM validation predictions to {out_path}")
        else:
            logger.info("No per-sample validation predictions available for export")
    except Exception as e:
        logger.warning(f"Could not persist LSTM per-sample predictions: {e}")

    # Persist LSTM test metrics to models for downstream DAGs
    try:
        output_dir = os.getenv('AIRFLOW_MODELS_DIR', '/usr/local/airflow/models')
        os.makedirs(output_dir, exist_ok=True)
        lstm_metrics_path = os.path.join(output_dir, 'lstm_metrics.json')
        test_m = results.get('test_metrics') or {}
        # Derive RMSE from loss if available (loss ~ MSE average)
        rmse = None
        try:
            if isinstance(test_m.get('loss'), (int, float)):
                import math
                rmse = math.sqrt(float(test_m['loss']))
        except Exception:
            rmse = None
        payload = {
            'lstm': {
                'r2': float(test_m.get('r2')) if test_m.get('r2') is not None else None,
                'mae': float(test_m.get('mae')) if test_m.get('mae') is not None else None,
                'rmse': float(rmse) if rmse is not None else None
            }
        }
        with open(lstm_metrics_path, 'w') as f:
            json.dump(payload, f, indent=2)
        logger.info(f"✅ Wrote LSTM metrics to {lstm_metrics_path}")
    except Exception as e:
        logger.warning(f"Could not persist LSTM metrics: {e}")

    return json.dumps({
        'model_path': model_path,
        'train_metrics': results.get('train_metrics'),
        'test_metrics': results.get('test_metrics'),
        'baseline_metrics': results.get('baseline_metrics')
    })


def show_training_summary(**context):
    ti = context['task_instance']
    record_count = ti.xcom_pull(task_ids='load_lstm_data', key='record_count')
    stations = ti.xcom_pull(task_ids='load_lstm_data', key='stations')
    train_metrics = ti.xcom_pull(task_ids='train_lstm_model', key='train_metrics')
    val_metrics = ti.xcom_pull(task_ids='train_lstm_model', key='test_metrics')
    per_station_val = ti.xcom_pull(task_ids='train_lstm_model', key='per_station_test')
    baseline_metrics = ti.xcom_pull(task_ids='train_lstm_model', key='baseline_metrics')
    model_path = ti.xcom_pull(task_ids='train_lstm_model', key='model_path')

    logger.info("=== LSTM TRAINING SUMMARY ===")
    logger.info(f"Records: {record_count}")
    logger.info(f"Stations: {stations}")
    logger.info(f"Model saved: {model_path}")
    logger.info(f"Train metrics: {train_metrics}")
    logger.info(f"Test metrics: {val_metrics}")
    if baseline_metrics:
        logger.info(f"Baseline metrics: {baseline_metrics}")
    if per_station_val:
        logger.info("Per-station validation metrics:")
        logger.info(per_station_val)

    return json.dumps({
        'records': record_count,
        'stations': stations,
        'model_path': model_path,
        'train_metrics': train_metrics,
        'test_metrics': val_metrics,
        'baseline_metrics': baseline_metrics,
        'per_station_test': per_station_val
    })


@dag(
    dag_id='train_lstm_multi_station',
    default_args=default_args,
    description='Train global LSTM model on multiple stations within a fixed date range',
    schedule=None,
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=['water-quality', 'lstm', 'historical', 'multi-station']
)
def train_lstm_multi_station():
    load_task = PythonOperator(
        task_id='load_lstm_data',
        python_callable=load_lstm_data,
    )

    train_task = PythonOperator(
        task_id='train_lstm_model',
        python_callable=train_lstm_model,
    )

    summary_task = PythonOperator(
        task_id='show_training_summary',
        python_callable=show_training_summary,
    )

    load_task >> train_task >> summary_task


train_lstm_multi_station()


