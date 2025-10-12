from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import logging
import requests
import os
from openai import OpenAI
from airflow.models import Variable
from airflow.decorators import dag, task
from pendulum import datetime
from include.iot_streaming.database_manager import db_manager
from include.iot_streaming.pipeline_processor import PipelineProcessor
from include.iot_streaming.prediction_service import PredictionService

try:
    openai_key = os.getenv("OPENAI_API_KEY") or Variable.get("openai_api_key", default_var=None)
except Exception:
    openai_key = None
# Cấu hình logging
logger = logging.getLogger(__name__)

# Default arguments cho DAG
default_args = {
    'owner': 'water_quality_team',
    'depends_on_past': False,
    'start_date': datetime(2025, 1, 1),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

# Định nghĩa DAG
 
# ============================================================================
# TASK FUNCTIONS - CHỈ ORCHESTRATION, KHÔNG BUSINESS LOGIC
# ============================================================================

def initialize_database_connection(**context):
    """Khởi tạo kết nối database - chỉ orchestration"""
    logger.info("Initializing database connection...")
    
    try:
       
        status = db_manager.check_database_status()
        
        logger.info("✅ Database connection initialized successfully")
        return status
        
    except Exception as e:
        logger.error(f"❌ Error initializing database connection: {e}")
        return f"Error: {e}"

def process_streaming_data(**context):
    """Orchestrate streaming data processing - lấy raw data và lưu vào processed table"""
    logger.info("Starting streaming data processing orchestration")
    
    try:
      
      
        
        # Lấy raw data từ database
        pipeline_processor = PipelineProcessor()
        raw_data = pipeline_processor.get_unprocessed_raw_data()
        
        if not raw_data:
            logger.info("No unprocessed raw data found")
            context['task_instance'].xcom_push(key='unprocessed_count', value=0)
            context['task_instance'].xcom_push(key='stations_with_models', value=[])
            context['task_instance'].xcom_push(key='processed_count', value=0)
            return "No unprocessed raw data found"
        
        # Process raw data thành processed data và lưu vào bảng processed_water_quality_data
        processed_count = pipeline_processor.process_raw_data(raw_data)
        
        logger.info(f"✅ Successfully processed {processed_count} raw records into processed_water_quality_data")
        
        # Lấy thông tin sau khi process
        unprocessed_count = db_manager.get_unprocessed_raw_count()
        stations_with_models = db_manager.get_stations_with_models()
        
        # Lưu thông tin cho các task tiếp theo
        context['task_instance'].xcom_push(key='unprocessed_count', value=unprocessed_count)
        context['task_instance'].xcom_push(key='stations_with_models', value=stations_with_models)
        context['task_instance'].xcom_push(key='processed_count', value=processed_count)
        
        logger.info(f"📊 Summary: Processed {processed_count} records, remaining {unprocessed_count} unprocessed, {len(stations_with_models)} stations with models")
        return f"✅ Processed {processed_count} raw records into processed table, {len(stations_with_models)} stations available"
        
    except Exception as e:
        logger.error(f"❌ Error processing streaming data: {e}")
        return f"Error: {e}"

def predict_existing_stations(**context):
    """Orchestrate predictions for existing stations - lấy records is_processed=FALSE, predict, update"""
    logger.info("Starting prediction orchestration")
    
    try:
     

        # Lấy stations có unprocessed data
        conn = db_manager.get_connection()
        if not conn:
            logger.error("Cannot connect to database")
            return "Cannot connect to database"
        
        cur = conn.cursor()
        cur.execute("""
            SELECT DISTINCT station_id 
            FROM raw_sensor_data 
            WHERE is_processed = FALSE
        """)
        
        stations_with_data = [row[0] for row in cur.fetchall()]
        cur.close()
        conn.close()
        
        if not stations_with_data:
            logger.info("No stations with unprocessed data found")
            return "No stations with unprocessed data found"
        
        logger.info(f"Found {len(stations_with_data)} stations with unprocessed data")
        
        # Gọi prediction service
        prediction_service = PredictionService()
        prediction_results = prediction_service.process_stations_predictions(stations_with_data)
        
        # Update is_processed = TRUE cho records đã predict
        if prediction_results:
            conn = db_manager.get_connection()
            if conn:
                cur = conn.cursor()
                for station_id in stations_with_data:
                    cur.execute("""
                        UPDATE raw_sensor_data 
                        SET is_processed = TRUE
                        WHERE station_id = %s AND is_processed = FALSE
                    """, (station_id,))
                
                conn.commit()
                cur.close()
                conn.close()
                logger.info(f"✅ Updated is_processed = TRUE for {len(stations_with_data)} stations")
        
        # Lưu kết quả
        if prediction_results is None:
            prediction_results = []
        
        context['task_instance'].xcom_push(key='prediction_results', value=prediction_results)
        
        successful_count = len([r for r in prediction_results if r and r.get('success')])
        total_count = len(prediction_results) if prediction_results else 0
        logger.info(f"Predictions completed: {successful_count}/{total_count} successful")
        
        return f"✅ Predictions completed: {successful_count}/{total_count} successful, updated {len(stations_with_data)} stations"
        
    except Exception as e:
        logger.error(f"Error orchestrating predictions: {e}")
        return f"Error: {e}"

def update_database_metrics(**context):
    """Orchestrate database metrics update"""
    logger.info("Updating database metrics")
    
    try:
     
        
        # Gọi service để cập nhật metrics
        metrics_result = db_manager.update_metrics()
        
        logger.info("Database metrics updated successfully")
        return f"Metrics updated: {metrics_result}"
            
    except Exception as e:
        logger.error(f"Error updating database metrics: {e}")
        return f"Error: {e}"

def generate_alerts_and_notifications(**context):
    """Orchestrate alerts generation and push notifications"""
    logger.info("Generating alerts and sending push notifications")
    try:
        
        # Lấy kết quả predictions
        prediction_results = context['task_instance'].xcom_pull(
            task_ids='predict_existing_stations', key='prediction_results'
        )
        
        if not prediction_results:
            logger.info("No prediction results for alerts")
            context['task_instance'].xcom_push(key='alerts_result', value='No prediction results for alerts')
            context['task_instance'].xcom_push(key='notifications_sent', value=0)
            return "No prediction results for alerts"
        
        # Gọi alert service
        prediction_service = PredictionService()
        alerts_result = prediction_service.generate_alerts(prediction_results)
        
        # Gửi push notifications cho từng station
        notifications_sent = 0
        
        for prediction in prediction_results:
            if not prediction or not prediction.get('success'):
                continue
                
            station_id = prediction.get('station_id')
            future_predictions = prediction.get('future_predictions', {})
            
            if not future_predictions:
                continue
            
            # Lấy dự đoán cho tháng đầu tiên (1 tháng)
            first_prediction = future_predictions.get('1month', {})
            if not first_prediction:
                continue
                
            wqi_prediction = first_prediction.get('wqi_prediction', 50.0)
            confidence_score = first_prediction.get('confidence_score', 0.5)
            
            # Lấy dữ liệu hiện tại từ database để phân tích
           
            conn = db_manager.get_connection()
            current_data = None
            
            if conn:
                cur = conn.cursor()
              
                # Nếu không có, thử tìm trong raw_sensor_data
                if not current_data:
                    cur.execute("""
                        SELECT ph, temperature, "do" 
                        FROM raw_sensor_data 
                        WHERE station_id = %s 
                        ORDER BY measurement_time DESC 
                        LIMIT 1
                    """, (station_id,))
                    current_data = cur.fetchone()
                
                cur.close()
                conn.close()
                
                if current_data:
                    ph, temperature, do = current_data
                    logger.info(f"📊 Found current data for station {station_id}: pH={ph}, Temp={temperature}, DO={do}")
                else:
                    logger.warning(f"⚠️ No current data found for station {station_id} in both processed and raw tables")
                    # Sử dụng giá trị mặc định nếu không có dữ liệu
                    ph, temperature, do = 7.0, 25.0, 8.0
                
                # Phân tích chất lượng nước
                analysis = analyze_water_quality(wqi_prediction, ph, do, temperature)
                
                # Xác định status dựa trên WQI
                status = "good" if wqi_prediction > 50 else "danger" 
                
                # Lấy tên trạm để đưa vào tiêu đề thông báo
                station_name = None
                try:
                    conn2 = db_manager.get_connection()
                    if conn2:
                        cur2 = conn2.cursor()
                        cur2.execute("""
                            SELECT station_name FROM monitoring_stations WHERE station_id = %s
                        """, (station_id,))
                        row = cur2.fetchone()
                        if row:
                            station_name = row[0]
                        cur2.close()
                        conn2.close()
                except Exception:
                    station_name = None

                # Gửi thông báo
                account_id = 3  # Sử dụng station_id làm account_id
                message = analysis
                title = f"Kết quả WQI - {station_name}" if station_name else "Kết quả WQI"
                
                if push_notification(account_id, title, message, status):
                    notifications_sent += 1
                    logger.info(f"✅ Notification sent for station {station_id}: WQI={wqi_prediction}, Status={status}")
                else:
                    logger.warning(f"⚠️ Failed to send notification for station {station_id}")
            else:
                logger.error(f"❌ Cannot connect to database for station {station_id}")
        
        context['task_instance'].xcom_push(key='alerts_result', value=alerts_result)
        context['task_instance'].xcom_push(key='notifications_sent', value=notifications_sent)
        
        logger.info(f"✅ Alerts generated and {notifications_sent} notifications sent successfully")
        return f"✅ Alerts generated: {alerts_result}, Notifications sent: {notifications_sent}"
                        
    except Exception as e:
        logger.error(f"Error generating alerts and notifications: {e}")
        return f"Error: {e}"

def mark_records_as_processed(**context):
    """Orchestrate marking raw records as processed (already done in predict_existing_stations)"""
    logger.info("Checking processed records status")
    
    try:
        # Lấy thông tin từ task trước
        prediction_results = context['task_instance'].xcom_pull(
            task_ids='predict_existing_stations', key='prediction_results'
        )
        
        if not prediction_results:
            logger.info("No prediction results found")
            context['task_instance'].xcom_push(key='processed_count', value=0)
            return "No prediction results found"
        
        successful_count = len([r for r in prediction_results if r and r.get('success')])
        context['task_instance'].xcom_push(key='processed_count', value=successful_count)
        
        logger.info(f"Confirmed {successful_count} predictions were successful")
        return f"Confirmed {successful_count} predictions were successful"
        
    except Exception as e:
        logger.error(f"Error checking processed records: {e}")
        return f"Error: {e}"

def summarize_pipeline_execution(**context):
    """Orchestrate pipeline summary"""
    logger.info("Summarizing pipeline execution")
    
    try:
        
        # Lấy dữ liệu từ các task trước
        prediction_results = context['task_instance'].xcom_pull(
            task_ids='predict_existing_stations', key='prediction_results'
        )
        alerts_result = context['task_instance'].xcom_pull(
            task_ids='generate_alerts_and_notifications', key='alerts_result'
        )
        processed_count = context['task_instance'].xcom_pull(
            task_ids='mark_records_as_processed', key='processed_count'
        )
        notifications_sent = context['task_instance'].xcom_pull(
            task_ids='generate_alerts_and_notifications', key='notifications_sent'
        )
        
        # Handle None values from XCom
        if prediction_results is None:
            prediction_results = []
        if alerts_result is None:
            alerts_result = ''
        if processed_count is None:
            processed_count = 0
        if notifications_sent is None:
            notifications_sent = 0
        
        # Gọi summary service
        pipeline_processor = PipelineProcessor()
        summary = pipeline_processor.create_summary({
            'prediction_results': prediction_results,
            'alerts_result': alerts_result,
            'processed_count': processed_count,
            'notifications_sent': notifications_sent
        })
        
        context['task_instance'].xcom_push(key='pipeline_summary', value=summary)
        
        logger.info("Pipeline summary created successfully")
        return f"Pipeline summary: {summary}"
        
    except Exception as e:
        logger.error(f"Error summarizing pipeline: {e}")
        return f"Error: {e}"

def push_notification(account_id, title, message, status):
    """Gửi push notification đến API"""
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
            logger.info("Push notification sent successfully.")
            return True
        else:
            logger.error(f"Push notification failed: {response.text}")
            return False
    except Exception as e:
        logger.error(f"Error sending push notification: {e}")
        return False

def analyze_water_quality(wqi, ph, do, temperature):
    """Phân tích chất lượng nước và đưa ra đánh giá"""
    try:
        
        # Khởi tạo OpenAI client
        client = OpenAI(api_key=os.getenv('OPENAI_API_KEY', openai_key))
        
        prompt = (
            "Đây là nước nuôi cá. "
            f"WQI dự đoán: {wqi}, pH={ph}, DO={do} mg/L, nhiệt độ={temperature}°C. "
            "Hãy đánh giá khách quan và đề xuất biện pháp, chỉ viết đúng 4 câu, "
            "mỗi câu kết thúc bằng dấu chấm, đầy đủ nghĩa, không bỏ thiếu chữ, không xuống dòng."
        )
        
        resp = client.chat.completions.create(
            model="gpt-4o-mini",  # Sử dụng model mới hơn
            messages=[{"role": "user", "content": prompt}],
            max_tokens=150,
            temperature=0.5
        )
        
        text = resp.choices[0].message.content.strip()
        # Gộp và đảm bảo không có newline
        text = " ".join(text.splitlines())
        return f"WQI tháng sau: {wqi}. {text}"
        
    except Exception as e:
        logger.error(f"Error analyzing water quality: {e}")
        return f"WQI tháng sau: {wqi}. Không thể phân tích chi tiết do lỗi hệ thống."

@dag(
    'streaming_data_processor',
    default_args=default_args,
    description='Orchestration pipeline for streaming water quality data processing',
    catchup=False,
    tags=['water-quality', 'streaming', 'orchestration', 'postgresql']
)
def streaming_data_processor():

    # ============================================================================
    # TASK DEFINITIONS
    # ============================================================================

    # Task để khởi tạo database connection
    initialize_db_task = PythonOperator(
        task_id='initialize_database_connection',
        python_callable=initialize_database_connection,
    )

    # Task để xử lý streaming data (đã bỏ, chuyển logic vào predict_existing_stations)

    # Task để dự đoán cho existing stations
    predict_existing_task = PythonOperator(
        task_id='predict_existing_stations',
        python_callable=predict_existing_stations,
    )

    # Task để cập nhật database metrics
    update_metrics_task = PythonOperator(
        task_id='update_database_metrics',
        python_callable=update_database_metrics,
    )

    # Task để tạo alerts và notifications
    alerts_task = PythonOperator(
        task_id='generate_alerts_and_notifications',
        python_callable=generate_alerts_and_notifications,
    )

    # Task để mark records as processed
    mark_processed_task = PythonOperator(
        task_id='mark_records_as_processed',
        python_callable=mark_records_as_processed,
    )

    # Task để tóm tắt pipeline execution
    summarize_task = PythonOperator(
        task_id='summarize_pipeline_execution',
        python_callable=summarize_pipeline_execution,
    )

    # ============================================================================
    # DAG DEPENDENCIES
    # ============================================================================

    # Định nghĩa dependencies - chỉ orchestration
    initialize_db_task >> predict_existing_task
    predict_existing_task >> update_metrics_task >> alerts_task >> mark_processed_task >> summarize_task 

streaming_data_processor()