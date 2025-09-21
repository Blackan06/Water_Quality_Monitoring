# /usr/local/airflow/dags/streaming_process_dag.py

from datetime import datetime, timezone
import logging

from airflow.decorators import dag
from airflow.providers.apache.kafka.sensors.kafka import AwaitMessageTriggerFunctionSensor
from airflow.operators.trigger_dagrun import TriggerDagRunOperator
import json
from include.iot_streaming.database_manager import db_manager

logger = logging.getLogger(__name__)

# ===== Callback khi có message: TRIGGER DAG B ngay tại đây =====
def on_kafka_event(event=None, **kwargs):
    """
    Với AwaitMessageTriggerFunctionSensor:
    - `event` là GIÁ TRỊ do apply_function trả về (payload chuỗi).
    - Sensor sẽ tiếp tục DEFER để nghe tiếp; vì vậy ta TRIGGER DAG B NGAY TẠI ĐÂY.
    """
    dag = kwargs.get("dag")
    ts = kwargs.get("ts")  # Airflow logical time
    ti = kwargs.get("ti")

    payload = event  # đã là string do extract_value trả về
    logger.info("📥 Kafka payload (from apply_function): %s", payload)
    
    # Parse JSON payload
    try:
        data = json.loads(payload) if payload else {}
    except Exception:
        logger.warning("Payload is not valid JSON; skipping parse")
        return True

    # (tuỳ chọn) lưu để debug
    if ti:
        ti.xcom_push(key="kafka_message", value=payload)
        ti.xcom_push(key="trigger_time", value=ts or datetime.now(timezone.utc).isoformat())

    # Handle both single object and array of objects
    if isinstance(data, list):
        # Array of objects - process each one
        logger.info(f"📊 Processing {len(data)} records from array")
        for i, item in enumerate(data):
            try:
                _process_single_record(item)
            except Exception as e:
                logger.error(f"❌ Error processing record {i}: {e}")
    else:
        # Single object
        try:
            _process_single_record(data)
        except Exception as e:
            logger.error(f"❌ Error processing single record: {e}")

    # Tạo task TriggerDagRunOperator runtime + execute ngay
    unique_suffix = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f")
    trigger_task = TriggerDagRunOperator(
        task_id=f"trigger_ml_pipeline_inline_{unique_suffix}",
        trigger_dag_id="streaming_data_processor",   # <-- đổi nếu DAG B tên khác
        conf={
            "kafka_msg": payload,
            "trigger_time": ts or datetime.now(timezone.utc).isoformat(),
            "source_dag": "streaming_process_dag",
        },
        wait_for_completion=False,
        reset_dag_run=True,
    )

    trigger_task.dag = dag
    trigger_task.execute(context=kwargs)

    logger.info("✅ Triggered external DAG 'streaming_data_processor' with conf.")
    return True  # callback xong; sensor quay lại DEFER để nghe tiếp

def _process_single_record(data):
    """Process a single record"""
    # Map fields for DB insert
    station_id = int(data.get("station_id")) if data.get("station_id") is not None else 0
    ph = float(data.get("ph")) if data.get("ph") is not None else None
    temperature = float(data.get("temperature")) if data.get("temperature") is not None else None
    do_val = float(data.get("do")) if data.get("do") is not None else None

    mt = data.get("measurement_time")
    if isinstance(mt, str):
        # Handle ISO strings including with timezone or trailing Z
        mt_str = mt[:-1] if mt.endswith("Z") else mt
        try:
            measurement_time = datetime.fromisoformat(mt_str)
        except ValueError:
            measurement_time = datetime.now(timezone.utc)
    elif isinstance(mt, (int, float)):
        measurement_time = datetime.fromtimestamp(mt, tz=timezone.utc)
    else:
        measurement_time = datetime.now(timezone.utc)

    raw_record = {
        "station_id": station_id,
        "measurement_time": measurement_time,
        "ph": ph if ph is not None else 7.0,
        "temperature": temperature if temperature is not None else 25.0,
        "do": do_val if do_val is not None else 8.0,
    }

    if db_manager.insert_raw_data(raw_record):
        logger.info("✅ Raw data inserted for station %s at %s", station_id, measurement_time)
    else:
        logger.warning("⚠️ Failed to insert raw data for station %s", station_id)

# ===== Tham số DAG =====
default_args = {
    "owner": "airflow",
    "depends_on_past": False,
    "email_on_failure": False,
    "email_on_retry": False,
}

@dag(
    default_args=default_args,
    description="Continuous IoT streaming (Kafka -> trigger external DAG inline)",
    start_date=datetime(2024, 1, 1),
    schedule=None,
    catchup=False,
    tags=["iot_pipeline_continuous"],
    max_active_runs=1,
)
def streaming_process_dag():
    """
    Flow:
    - Sensor lắng nghe Kafka liên tục (DEFERRED).
    - Mỗi khi có message: iothandlers.kafka.extract_value trả payload -> on_kafka_event được gọi -> trigger DAG B.
    - Không có downstream tasks vì sensor không success; nó sẽ quay lại nghe tiếp.
    """
    wait_for_kafka = AwaitMessageTriggerFunctionSensor(
        task_id="wait_for_kafka",
        kafka_config_id="kafka_default",                # đảm bảo Connection này tồn tại
        topics=["water-quality-data"],                  # đổi theo topic của bạn
        apply_function="include.iot_streaming.kafka_handlers.extract_value",  # <— STRING dotted-path (import được ở Triggerer)
        event_triggered_function=on_kafka_event,           # <— CALLABLE
        poll_timeout=1,
        poll_interval=10,
    )

    # Không cần return task khác; callback tự trigger DAG B

# Instantiate DAG
streaming_process_dag()
