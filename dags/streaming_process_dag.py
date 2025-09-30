# /usr/local/airflow/dags/streaming_process_dag.py

from datetime import datetime, timezone
import logging
import requests
import subprocess
import threading
import time

from airflow.decorators import dag
from airflow.providers.apache.kafka.sensors.kafka import AwaitMessageTriggerFunctionSensor
from airflow.operators.trigger_dagrun import TriggerDagRunOperator
from airflow.operators.python import PythonOperator
import json
from include.iot_streaming.database_manager import db_manager

logger = logging.getLogger(__name__)

# ===== Callback khi có message: SPARK CONSUMER + TRIGGER DAG =====
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

    # ===== SPARK CONSUMER PROCESSING =====
    spark_result = {"success": False}  # Initialize default result
    
    try:
        logger.info("🚀 Starting Spark consumer processing...")
        
        # Import Spark consumer
        from include.iot_streaming.spark_consumer import process_kafka_message_with_spark
        
        # Process message with Spark
        spark_result = process_kafka_message_with_spark(payload)
        
        if spark_result.get("success", False):
            logger.info("✅ Spark consumer processed message successfully")
            
            # Extract processed data from Spark result
            processed_data = spark_result.get("data", {})
            
            # Handle both single object and array of objects
            if isinstance(processed_data, list):
                # Array of objects - process each one
                logger.info(f"📊 Processing {len(processed_data)} records from Spark")
                for i, item in enumerate(processed_data):
                    try:
                        _process_single_record(item)
                    except Exception as e:
                        logger.error(f"❌ Error processing Spark record {i}: {e}")
            else:
                # Single object
                try:
                    _process_single_record(processed_data)
                except Exception as e:
                    logger.error(f"❌ Error processing single Spark record: {e}")
        else:
            logger.warning("⚠️ Spark consumer processing failed, falling back to direct processing")
            # Fallback to direct processing
            if isinstance(data, list):
                for i, item in enumerate(data):
                    try:
                        _process_single_record(item)
                    except Exception as e:
                        logger.error(f"❌ Error processing record {i}: {e}")
            else:
                try:
                    _process_single_record(data)
                except Exception as e:
                    logger.error(f"❌ Error processing single record: {e}")
                    
    except Exception as e:
        logger.error(f"❌ Spark consumer error: {e}")
        # Fallback to direct processing
        if isinstance(data, list):
            for i, item in enumerate(data):
                try:
                    _process_single_record(item)
                except Exception as e:
                    logger.error(f"❌ Error processing record {i}: {e}")
        else:
            try:
                _process_single_record(data)
            except Exception as e:
                logger.error(f"❌ Error processing single record: {e}")

    # ===== TRIGGER ML PIPELINE DAG =====
    try:
        # Tạo task TriggerDagRunOperator runtime + execute ngay
        unique_suffix = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f")
        trigger_task = TriggerDagRunOperator(
            task_id=f"trigger_ml_pipeline_inline_{unique_suffix}",
            trigger_dag_id="streaming_data_processor",   # <-- đổi nếu DAG B tên khác
            conf={
                "kafka_msg": payload,
                "trigger_time": ts or datetime.now(timezone.utc).isoformat(),
                "source_dag": "streaming_process_dag",
                "spark_processed": spark_result.get("success", False),
            },
            wait_for_completion=False,
            reset_dag_run=True,
        )

        trigger_task.dag = dag
        trigger_task.execute(context=kwargs)

        logger.info("✅ Triggered external DAG 'streaming_data_processor' with conf.")
    except Exception as e:
        logger.error(f"❌ Error triggering ML pipeline: {e}")

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
        
        # Calculate WQI and send notification
        wqi_value = _calculate_wqi_simple(ph if ph is not None else 7.0, 
                                        temperature if temperature is not None else 25.0, 
                                        do_val if do_val is not None else 8.0)
        
        _send_wqi_notification(station_id, wqi_value)
    else:
        logger.warning("⚠️ Failed to insert raw data for station %s", station_id)

def _calculate_wqi_simple(ph, temperature, do):
    """Calculate simple WQI based on sensor values"""
    try:
        # Simple WQI calculation (adjust weights as needed)
        ph_score = max(0, min(100, (ph - 6.0) * 25))  # pH 6-10 range
        temp_score = max(0, min(100, 100 - abs(temperature - 22) * 2))  # Optimal at 22°C
        do_score = max(0, min(100, do * 10))  # DO 0-10 mg/L range
        
        # Weighted average
        wqi = (ph_score * 0.3 + temp_score * 0.3 + do_score * 0.4)
        return round(wqi, 2)
    except Exception as e:
        logger.error(f"Error calculating WQI: {e}")
        return 50.0  # Default value

def _send_wqi_notification(station_id, wqi_value):
    """Send push notification for WQI values"""
    try:
        # Determine notification status based on WQI
        if wqi_value < 50:
            status = "critical"
            title = f"🚨 Critical Water Quality Alert - Station {station_id}"
            message = f"Current WQI is {wqi_value:.1f}. Immediate action required!"
        elif wqi_value < 60:
            status = "warning"
            title = f"⚠️ Water Quality Warning - Station {station_id}"
            message = f"Current WQI is {wqi_value:.1f}. Monitor closely."
        elif wqi_value > 80:
            status = "excellent"
            title = f"✅ Excellent Water Quality - Station {station_id}"
            message = f"Current WQI is {wqi_value:.1f}. Great water quality!"
        else:
            # Normal range, no notification needed
            return
        
        # Send notification
        _push_notification(
            account_id=station_id,
            title=title,
            message=message,
            status=status
        )
        
    except Exception as e:
        logger.error(f"Error sending WQI notification: {e}")

def _push_notification(account_id, title, message, status):
    """Send push notification to API"""
    url = "https://datamanagerment.anhkiet.xyz/notifications/send-notification"
    payload = {
        "account_id": str(account_id),
        "title": title,
        "message": message,
        "status": status
    }
    try:
        response = requests.post(url, json=payload, timeout=10)
        if response.status_code == 200:
            logger.info(f"✅ Push notification sent successfully for station {account_id}")
            return True
        else:
            logger.error(f"❌ Push notification failed: {response.text}")
            return False
    except Exception as e:
        logger.error(f"❌ Error sending push notification: {e}")
        return False

# ===== Tham số DAG =====
default_args = {
    "owner": "airflow",
    "depends_on_past": False,
    "email_on_failure": False,
    "email_on_retry": False,
}


@dag(
    default_args=default_args,
    description="Continuous IoT streaming with Spark consumer (Kafka -> trigger external DAG inline)",
    start_date=datetime(2024, 1, 1),
    schedule=None,
    catchup=False,
    tags=["iot_pipeline_continuous"],
    max_active_runs=1,
)
def streaming_process_dag():
    """
    Flow:
    - Test Spark consumer setup
    - Sensor lắng nghe Kafka liên tục (DEFERRED) for processing
    - Mỗi khi có message: Spark consumer xử lý -> lưu DB -> gửi thông báo -> trigger ML DAG
    - Spark consumer handles the main processing with fallback
    """
    
    # Airflow sensor for Kafka processing
    wait_for_kafka = AwaitMessageTriggerFunctionSensor(
        task_id="wait_for_kafka",
        kafka_config_id="kafka_default",                # đảm bảo Connection này tồn tại
        topics=["water-quality-data"],                  # đổi theo topic của bạn
        apply_function="include.iot_streaming.kafka_handlers.extract_value",  # <— STRING dotted-path (import được ở Triggerer)
        event_triggered_function=on_kafka_event,           # <— CALLABLE
        poll_timeout=1,
        poll_interval=10,
    )

# Instantiate DAG
streaming_process_dag()
