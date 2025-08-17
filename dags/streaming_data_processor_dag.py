"""
## Streaming Data Processor DAG

This DAG orchestrates the streaming water quality data processing pipeline.
It handles real-time data processing, predictions, and database operations.

The pipeline consists of several tasks:
1. Initialize database connections
2. Process streaming data from raw to processed format
3. Make predictions for existing stations
4. Generate comprehensive summaries
5. Handle new station data

For more information about the water quality monitoring system, see the project documentation.

![Streaming Data Processing](https://www.databricks.com/wp-content/uploads/2020/04/streaming-analytics.png)
"""

from airflow.decorators import dag, task
from airflow.operators.python import PythonOperator
from pendulum import datetime
import logging
import requests
import os
from openai import OpenAI
from airflow.models import Variable

# Setup logging
logger = logging.getLogger(__name__)

# Get OpenAI API key from Airflow variables
openai_key = Variable.get("openai_api_key")

# Define the basic parameters of the DAG
@dag(
    start_date=datetime(2025, 1, 1),
    schedule="@hourly",
    catchup=False,
    doc_md=__doc__,
    default_args={"owner": "water_quality_team", "retries": 1},
    tags=["water-quality", "streaming", "orchestration", "postgresql"],
)
def streaming_data_processor():
    """
    Streaming water quality data processing orchestration pipeline.
    """
    
    @task
    def initialize_database_connection(**context) -> str:
        """
        Initialize database connection for streaming data processing.
        This task sets up the database connection and verifies connectivity.
        """
        logger.info("Initializing database connection...")
        
        try:
            from include.iot_streaming.database_manager import db_manager
            status = db_manager.check_database_status()
            
            logger.info("✅ Database connection initialized successfully")
            return status
            
        except Exception as e:
            logger.error(f"❌ Error initializing database connection: {e}")
            return f"Error: {e}"
    
    @task
    def process_streaming_data(**context) -> str:
        """
        Process streaming data from raw format to processed format.
        This task orchestrates the data processing pipeline.
        """
        logger.info("Starting streaming data processing orchestration")
        
        try:
            from include.iot_streaming.database_manager import db_manager
            from include.iot_streaming.pipeline_processor import PipelineProcessor
            
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
    
    @task
    def predict_existing_stations(**context) -> str:
        """
        Make predictions for existing stations with trained models.
        This task processes unprocessed records and generates predictions.
        """
        logger.info("Starting prediction orchestration")
        
        try:
            from include.iot_streaming.database_manager import db_manager
            from include.iot_streaming.prediction_service import PredictionService
            
            # Lấy thông tin từ task trước
            stations_with_models = context['task_instance'].xcom_pull(key='stations_with_models', task_ids='process_streaming_data')
            
            if not stations_with_models:
                logger.info("No stations with models available for prediction")
                return "No stations with models available"
            
            # Khởi tạo prediction service
            prediction_service = PredictionService()
            
            # Lấy records chưa được predict (is_processed=FALSE)
            unprocessed_records = db_manager.get_unprocessed_records_for_prediction()
            
            if not unprocessed_records:
                logger.info("No unprocessed records found for prediction")
                return "No unprocessed records for prediction"
            
            logger.info(f"📊 Found {len(unprocessed_records)} unprocessed records for prediction")
            
            # Thực hiện prediction cho từng record
            prediction_results = []
            for record in unprocessed_records:
                try:
                    station_id = record['station_id']
                    if station_id in stations_with_models:
                        # Có model cho station này, thực hiện prediction
                        prediction = prediction_service.predict_single_record(record)
                        if prediction:
                            prediction_results.append(prediction)
                            logger.info(f"✅ Predicted for station {station_id}: {prediction}")
                        else:
                            logger.warning(f"⚠️ No prediction generated for station {station_id}")
                    else:
                        logger.info(f"ℹ️ No model available for station {station_id}")
                except Exception as e:
                    logger.error(f"❌ Error predicting for record {record.get('id', 'unknown')}: {e}")
            
            # Cập nhật database với kết quả prediction
            if prediction_results:
                updated_count = db_manager.update_predictions(prediction_results)
                logger.info(f"✅ Updated {updated_count} predictions in database")
            else:
                updated_count = 0
                logger.info("ℹ️ No predictions to update")
            
            # Lưu thông tin cho task tiếp theo
            context['task_instance'].xcom_push(key='prediction_count', value=updated_count)
            context['task_instance'].xcom_push(key='total_records', value=len(unprocessed_records))
            
            return f"✅ Prediction completed: {updated_count}/{len(unprocessed_records)} records updated"
            
        except Exception as e:
            logger.error(f"❌ Error in prediction orchestration: {e}")
            return f"Error: {e}"
    
    @task
    def generate_comprehensive_summary(**context) -> str:
        """
        Generate comprehensive summary of streaming data processing.
        This task creates detailed reports and summaries.
        """
        logger.info("Starting comprehensive summary generation")
        
        try:
            from include.iot_streaming.comprehensive_summary_service import ComprehensiveSummaryService
            
            # Lấy thông tin từ các task trước
            processed_count = context['task_instance'].xcom_pull(key='processed_count', task_ids='process_streaming_data')
            prediction_count = context['task_instance'].xcom_pull(key='prediction_count', task_ids='predict_existing_stations')
            total_records = context['task_instance'].xcom_pull(key='total_records', task_ids='predict_existing_stations')
            
            # Khởi tạo summary service
            summary_service = ComprehensiveSummaryService()
            
            # Tạo comprehensive summary
            summary_data = {
                'processed_count': processed_count or 0,
                'prediction_count': prediction_count or 0,
                'total_records': total_records or 0,
                'timestamp': datetime.now().isoformat()
            }
            
            summary_result = summary_service.generate_summary(summary_data)
            
            logger.info("✅ Comprehensive summary generated successfully")
            return f"✅ Summary generated: {summary_result}"
            
        except Exception as e:
            logger.error(f"❌ Error generating comprehensive summary: {e}")
            return f"Error: {e}"
    
    # Define task dependencies using TaskFlow API
    db_init = initialize_database_connection()
    process_data = process_streaming_data()
    predict_data = predict_existing_stations()
    generate_summary = generate_comprehensive_summary()
    
    # Set up the pipeline flow
    db_init >> process_data >> predict_data >> generate_summary


# Instantiate the DAG
streaming_data_processor() 