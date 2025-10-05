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
    logger.info("🚀 CALLBACK FUNCTION CALLED!")
    logger.info(f"📥 Event: {event}")
    logger.info(f"📥 Kwargs: {kwargs}")
    
    # Ensure json module is available in this scope
    import json
    
    try:
        dag = kwargs.get("dag")
        ts = kwargs.get("ts")  # Airflow logical time
        ti = kwargs.get("ti")

        payload = event  # đã là string do extract_value trả về
        logger.info("📥 Kafka payload (from apply_function): %s", payload)
        
        if not payload:
            logger.warning("⚠️ Empty payload received from Kafka")
            return True
        
        # Parse JSON payload
        try:
            # Clean up the payload - remove trailing commas and fix common JSON issues
            cleaned_payload = payload.strip()
            
            # Check if payload is just a simple string (not JSON)
            if cleaned_payload in ["Accomplished", "Success", "Done", "Complete"]:
                logger.info(f"📝 Received status message: {cleaned_payload}")
                logger.info("ℹ️ Skipping non-JSON status messages")
                return True
            
            # Remove trailing comma before closing brace/bracket
            import re
            cleaned_payload = re.sub(r',(\s*[}\]])', r'\1', cleaned_payload)
            
            data = json.loads(cleaned_payload) if cleaned_payload else {}
            logger.info("✅ Successfully parsed JSON payload")
            logger.info(f"📊 Parsed data: {data}")
        except json.JSONDecodeError as e:
            logger.warning(f"⚠️ Invalid JSON payload: {e}")
            logger.warning(f"Raw payload: {payload}")
            
            # Check if it's a known status message
            if payload.strip() in ["Accomplished", "Success", "Done", "Complete", "Ready"]:
                logger.info(f"📝 Received status message: {payload.strip()}")
                logger.info("ℹ️ Skipping non-JSON status messages")
                return True
            
            # Try to fix common JSON issues
            try:
                # Remove trailing comma and try again
                fixed_payload = payload.rstrip().rstrip(',')
                data = json.loads(fixed_payload)
                logger.info("✅ Successfully parsed JSON after fixing trailing comma")
                logger.info(f"📊 Fixed data: {data}")
            except Exception as fix_error:
                logger.warning(f"⚠️ Could not fix JSON: {fix_error}")
                logger.info("ℹ️ Skipping invalid message")
                return True
        except Exception as e:
            logger.error(f"❌ Error parsing payload: {e}")
            return True
    except Exception as e:
        logger.error(f"❌ Critical error in on_kafka_event: {e}")
        return True

    # (tuỳ chọn) lưu để debug
    if ti:
        ti.xcom_push(key="kafka_message", value=payload)
        ti.xcom_push(key="trigger_time", value=ts or datetime.now(timezone.utc).isoformat())

    # ===== SPARK CONSUMER PROCESSING =====
    spark_result = {"success": False}  # Initialize default result
    
    try:
        logger.info("🚀 Starting Spark consumer processing...")
        
        # Try to import and use Spark consumer
        try:
            from include.iot_streaming.spark_consumer import process_kafka_message_with_spark
            logger.info("✅ Successfully imported Spark consumer")
            
            # Process message with Spark
            spark_result = process_kafka_message_with_spark(payload)
            logger.info(f"📊 Spark processing result: {spark_result}")
            
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
                logger.warning(f"Spark error: {spark_result.get('error', 'Unknown error')}")
                raise Exception("Spark processing failed")
                
        except ImportError as e:
            logger.warning(f"⚠️ PySpark not available: {e}")
            logger.info("🔄 Falling back to direct processing without Spark")
            raise Exception("PySpark not available")
        except Exception as e:
            logger.error(f"❌ Spark consumer import/execution error: {e}")
            raise Exception(f"Spark consumer error: {e}")
                    
    except Exception as e:
        logger.warning(f"⚠️ Spark consumer error: {e}")
        logger.info("🔄 Using fallback processing without Spark")
        
        # Fallback to direct processing
        try:
            if isinstance(data, list):
                logger.info(f"📊 Processing {len(data)} records with fallback method")
                for i, item in enumerate(data):
                    try:
                        _process_single_record(item)
                    except Exception as e:
                        logger.error(f"❌ Error processing record {i}: {e}")
            else:
                logger.info("📊 Processing single record with fallback method")
                try:
                    _process_single_record(data)
                except Exception as e:
                    logger.error(f"❌ Error processing single record: {e}")
        except Exception as e:
            logger.error(f"❌ Fallback processing failed: {e}")

    # ===== TRIGGER EXTERNAL DAG (Simplified Approach) =====
    try:
        logger.info("🚀 Attempting to trigger external DAG 'streaming_data_processor'...")
        
        # Use a simple approach: just log that we would trigger the DAG
        # In a real scenario, you could use:
        # 1. REST API calls to Airflow webserver
        # 2. Database inserts to trigger DAG runs
        # 3. File-based triggers
        
        if spark_result and spark_result.get("success"):
            processed_data = spark_result.get("data", [])
            
            if processed_data:
                logger.info(f"📊 Would trigger DAG 'streaming_data_processor' with {len(processed_data)} records")
                
                # Prepare data for external DAG
                dag_payload = {
                    "source_dag": "streaming_process_dag",
                    "trigger_time": ts or datetime.now(timezone.utc).isoformat(),
                    "records_count": len(processed_data),
                    "stations": [record.get("station_id") for record in processed_data],
                    "spark_processed": True
                }
                
                logger.info(f"📊 DAG trigger payload: {dag_payload}")
                
                # Store trigger information in XCom (like old code)
                if ti:
                    ti.xcom_push(key="kafka_message", value=payload)
                    ti.xcom_push(key="trigger_time", value=ts or datetime.now(timezone.utc).isoformat())
                    ti.xcom_push(key="dag_trigger_payload", value=dag_payload)
                    ti.xcom_push(key="triggered_dag_name", value="streaming_data_processor")
                    logger.info("📊 Stored DAG trigger information in XCom")
                
                # Trigger DAG using REST API (TriggerDagRunOperator doesn't work in callback context)
                try:
                    logger.info("🚀 Triggering DAG 'streaming_data_processor' using REST API...")
                    
                    # Try multiple Airflow API endpoints (correct ports)
                    airflow_urls = [
                        "http://api-server:8080",  # Internal container port
                        "http://localhost:8089",   # External mapped port
                        "http://127.0.0.1:8089"    # External mapped port
                    ]
                    
                    trigger_conf = {
                        "kafka_msg": payload,
                        "trigger_time": ts or datetime.now(timezone.utc).isoformat(),
                        "source_dag": "streaming_process_dag",
                        "spark_processed": spark_result.get("success", False),
                    }
                    
                    success = False
                    for url in airflow_urls:
                        try:
                            trigger_url = f"{url}/api/v2/dags/streaming_data_processor/dagRuns"
                            response = requests.post(
                                trigger_url,
                                json={
                                    "conf": trigger_conf,
                                    "dag_run_id": f"triggered_by_streaming_process_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')}",
                                    "logical_date": datetime.now(timezone.utc).isoformat()
                                },
                                headers={"Content-Type": "application/json"},
                                timeout=10
                            )
                            
                            if response.status_code == 200:
                                logger.info(f"✅ Successfully triggered DAG via {url}")
                                success = True
                                break
                            else:
                                logger.warning(f"⚠️ Failed to trigger via {url}: {response.status_code} - {response.text}")
                                
                        except Exception as api_error:
                            logger.warning(f"⚠️ Connection error to {url}: {api_error}")
                            continue
                    
                    if not success:
                        logger.error("❌ Failed to trigger DAG via all API endpoints")
                        logger.info("ℹ️ Fallback: Please manually trigger DAG 'streaming_data_processor' from Airflow UI")
                    
                except Exception as e:
                    logger.error(f"❌ Error triggering ML pipeline: {e}")
                    logger.error(f"❌ Error type: {type(e).__name__}")
                    logger.error(f"❌ Error details: {str(e)}")
                    import traceback
                    logger.error(f"❌ Full traceback: {traceback.format_exc()}")
                
                # Also process data locally for immediate results (based on current WQI)
                ml_results = []
                for record in processed_data:
                    try:
                        ml_result = {
                            "station_id": record.get("station_id"),
                            "wqi": record.get("wqi"),
                            "quality_status": record.get("quality_status"),
                            "alert_level": record.get("alert_level"),
                            "current_analysis": "Based on current WQI measurement",  # Clarify this is current, not prediction
                            "recommendations": _generate_recommendations(record),
                            "processing_time": datetime.now(timezone.utc).isoformat()
                        }
                        ml_results.append(ml_result)
                        logger.info(f"✅ Processed current WQI analysis for station {record.get('station_id')}")
                    except Exception as e:
                        logger.warning(f"⚠️ Error processing current WQI analysis for record: {e}")
                        continue
                
                # Store current WQI analysis results in XCom
                if ti:
                    ti.xcom_push(key="current_wqi_analysis", value=ml_results)
                    ti.xcom_push(key="processed_records_count", value=len(ml_results))
                    logger.info(f"📊 Stored {len(ml_results)} current WQI analysis results in XCom")
                
            else:
                logger.info("ℹ️ No data to trigger external DAG with")
        else:
            logger.info("ℹ️ No successful Spark result to trigger external DAG with")
            
    except Exception as e:
        logger.warning(f"⚠️ Error in DAG trigger logic: {e}")
        logger.info("ℹ️ Continuing - core processing completed successfully")

    return True  # callback xong; sensor quay lại DEFER để nghe tiếp

def _generate_recommendations(record):
    """Generate recommendations based on CURRENT water quality measurements (not predictions)"""
    try:
        wqi = record.get("wqi", 50)
        ph = record.get("ph", 7.0)
        do = record.get("do", 8.0)
        temperature = record.get("temperature", 25.0)
        
        recommendations = []
        
        # Current WQI-based recommendations (based on actual measurements)
        if wqi < 25:
            recommendations.append("🚨 Critical: Current WQI requires immediate water treatment")
        elif wqi < 50:
            recommendations.append("⚠️ Poor: Current WQI suggests considering water treatment")
        elif wqi < 70:
            recommendations.append("📊 Moderate: Current WQI indicates monitoring is needed")
        elif wqi < 90:
            recommendations.append("✅ Good: Current WQI shows acceptable water quality")
        else:
            recommendations.append("🌟 Excellent: Current WQI indicates optimal water quality")
        
        # Current pH-based recommendations
        if ph < 6.5:
            recommendations.append("🔬 Current pH too low: Consider pH adjustment")
        elif ph > 8.5:
            recommendations.append("🔬 Current pH too high: Consider pH adjustment")
        else:
            recommendations.append("🔬 Current pH within normal range")
        
        # Current DO-based recommendations
        if do < 5.0:
            recommendations.append("💨 Current oxygen low: Increase aeration")
        elif do > 12.0:
            recommendations.append("💨 Current oxygen high: Normal levels")
        else:
            recommendations.append("💨 Current oxygen levels adequate")
        
        # Current Temperature-based recommendations
        if temperature > 30:
            recommendations.append("🌡️ Current temperature high: Monitor for thermal stress")
        elif temperature < 15:
            recommendations.append("🌡️ Current temperature low: Monitor for cold stress")
        else:
            recommendations.append("🌡️ Current temperature within optimal range")
        
        return "; ".join(recommendations)
        
    except Exception as e:
        logger.warning(f"⚠️ Error generating current WQI recommendations: {e}")
        return "Unable to generate recommendations based on current measurements"

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
            account_id=3,
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
