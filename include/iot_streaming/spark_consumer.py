"""
Spark Consumer for Water Quality Monitoring
Xử lý Kafka messages với Apache Spark
"""

import logging
import json
import pandas as pd
import numpy as np
import requests
from datetime import datetime, timezone
from typing import Dict, Any, Optional
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when, isnan, isnull, avg, stddev, lag, window, count
from pyspark.sql.types import StructType, StructField, StringType, DoubleType, TimestampType, IntegerType

# Import database manager
from .database_manager import db_manager

logger = logging.getLogger(__name__)

class SparkKafkaConsumer:
    """Spark-based Kafka consumer for water quality data processing"""
    
    def __init__(self):
        self.spark = None
        self._initialize_spark()
    
    def _initialize_spark(self):
        """Initialize Spark session"""
        try:
            self.spark = SparkSession.builder \
                .appName("WaterQualitySparkConsumer") \
                .config("spark.sql.adaptive.enabled", "true") \
                .config("spark.sql.adaptive.coalescePartitions.enabled", "true") \
                .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer") \
                .getOrCreate()
            
            self.spark.sparkContext.setLogLevel("WARN")
            logger.info("✅ Spark session initialized successfully")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize Spark session: {e}")
            self.spark = None
    
    def process_kafka_message(self, message: str) -> Dict[str, Any]:
        """
        Process Kafka message with Spark, database ingestion, and notifications
        
        Args:
            message: JSON string from Kafka
            
        Returns:
            Dict with processing results
        """
        if not self.spark:
            logger.error("❌ Spark session not initialized")
            return {"success": False, "error": "Spark session not initialized"}
        
        try:
            # Parse JSON message
            data = json.loads(message) if message else {}
            
            if not data:
                return {"success": False, "error": "Empty message"}
            
            # Convert to DataFrame
            df = self._create_dataframe(data)
            
            # Process with Spark
            processed_df = self._process_with_spark(df)
            
            # Convert back to dict/list
            result_data = self._dataframe_to_dict(processed_df)
            
            # Database ingestion
            db_success = self._ingest_to_database(result_data)
            
            # Send notifications
            notification_success = self._send_notifications(result_data)
            
            logger.info(f"✅ Spark processed {len(result_data)} records successfully")
            logger.info(f"📊 Database ingestion: {'✅' if db_success else '❌'}")
            logger.info(f"📱 Notifications: {'✅' if notification_success else '❌'}")
            
            return {
                "success": True,
                "data": result_data,
                "processed_count": len(result_data),
                "processing_time": datetime.now(timezone.utc).isoformat(),
                "database_ingested": db_success,
                "notifications_sent": notification_success
            }
            
        except Exception as e:
            logger.error(f"❌ Spark processing error: {e}")
            return {"success": False, "error": str(e)}
    
    def _create_dataframe(self, data: Dict[str, Any]) -> 'DataFrame':
        """Create Spark DataFrame from data"""
        try:
            # Handle both single object and array
            if isinstance(data, list):
                records = data
            else:
                records = [data]
            
            # Create DataFrame
            df = self.spark.createDataFrame(records)
            
            # Add processing timestamp
            from pyspark.sql.functions import current_timestamp
            df = df.withColumn("processing_time", current_timestamp())
            
            return df
            
        except Exception as e:
            logger.error(f"❌ Error creating DataFrame: {e}")
            raise
    
    def _process_with_spark(self, df: 'DataFrame') -> 'DataFrame':
        """Process DataFrame with Spark transformations"""
        try:
            # 1. Data cleaning and validation
            df_cleaned = self._clean_data(df)
            
            # 2. Feature engineering
            df_features = self._add_features(df_cleaned)
            
            # 3. Calculate WQI
            df_wqi = self._calculate_wqi_spark(df_features)
            
            # 4. Add quality indicators
            df_final = self._add_quality_indicators(df_wqi)
            
            return df_final
            
        except Exception as e:
            logger.error(f"❌ Error in Spark processing: {e}")
            raise
    
    def _clean_data(self, df: 'DataFrame') -> 'DataFrame':
        """Clean and validate data"""
        try:
            # Handle null values
            df_cleaned = df.fillna({
                'ph': 7.0,
                'temperature': 25.0,
                'do': 8.0,
                'station_id': 0
            })
            
            # Validate ranges
            df_cleaned = df_cleaned.withColumn(
                'ph', 
                when((col('ph') < 0) | (col('ph') > 14), 7.0).otherwise(col('ph'))
            )
            
            df_cleaned = df_cleaned.withColumn(
                'temperature',
                when((col('temperature') < -10) | (col('temperature') > 50), 25.0)
                .otherwise(col('temperature'))
            )
            
            df_cleaned = df_cleaned.withColumn(
                'do',
                when((col('do') < 0) | (col('do') > 20), 8.0)
                .otherwise(col('do'))
            )
            
            return df_cleaned
            
        except Exception as e:
            logger.error(f"❌ Error cleaning data: {e}")
            raise
    
    def _add_features(self, df: 'DataFrame') -> 'DataFrame':
        """Add engineered features"""
        try:
            # Add temporal features
            from pyspark.sql.functions import hour, dayofweek, month, year
            
            df_features = df.withColumn('hour', hour('measurement_time')) \
                           .withColumn('day_of_week', dayofweek('measurement_time')) \
                           .withColumn('month', month('measurement_time')) \
                           .withColumn('year', year('measurement_time'))
            
            # Add interaction features
            df_features = df_features.withColumn(
                'ph_temp_interaction', 
                col('ph') * col('temperature')
            )
            
            df_features = df_features.withColumn(
                'ph_do_interaction',
                col('ph') * col('do')
            )
            
            df_features = df_features.withColumn(
                'temp_do_interaction',
                col('temperature') * col('do')
            )
            
            return df_features
            
        except Exception as e:
            logger.error(f"❌ Error adding features: {e}")
            raise
    
    def _calculate_wqi_spark(self, df: 'DataFrame') -> 'DataFrame':
        """Calculate WQI using Spark"""
        try:
            from pyspark.sql.functions import when, round
            
            # Calculate sub-indices
            df_wqi = df.withColumn(
                'ph_score',
                when((col('ph') >= 6.0) & (col('ph') <= 10.0), 
                     (col('ph') - 6.0) * 25)
                .otherwise(0)
            )
            
            df_wqi = df_wqi.withColumn(
                'temp_score',
                when((col('temperature') >= 15) & (col('temperature') <= 30),
                     100 - abs(col('temperature') - 22) * 2)
                .otherwise(0)
            )
            
            df_wqi = df_wqi.withColumn(
                'do_score',
                when(col('do') >= 0, col('do') * 10)
                .otherwise(0)
            )
            
            # Calculate final WQI
            df_wqi = df_wqi.withColumn(
                'wqi',
                round(
                    col('ph_score') * 0.3 + 
                    col('temp_score') * 0.3 + 
                    col('do_score') * 0.4, 2
                )
            )
            
            return df_wqi
            
        except Exception as e:
            logger.error(f"❌ Error calculating WQI: {e}")
            raise
    
    def _add_quality_indicators(self, df: 'DataFrame') -> 'DataFrame':
        """Add quality indicators and alerts"""
        try:
            from pyspark.sql.functions import when, lit
            
            # Add quality status
            df_indicators = df.withColumn(
                'quality_status',
                when(col('wqi') >= 90, 'excellent')
                .when(col('wqi') >= 70, 'good')
                .when(col('wqi') >= 50, 'moderate')
                .when(col('wqi') >= 25, 'poor')
                .otherwise('very_poor')
            )
            
            # Add alert level
            df_indicators = df_indicators.withColumn(
                'alert_level',
                when(col('wqi') < 50, 'critical')
                .when(col('wqi') < 60, 'warning')
                .when(col('wqi') > 80, 'excellent')
                .otherwise('normal')
            )
            
            # Add processing metadata
            df_indicators = df_indicators.withColumn(
                'spark_processed', lit(True)
            )
            
            return df_indicators
            
        except Exception as e:
            logger.error(f"❌ Error adding quality indicators: {e}")
            raise
    
    def _dataframe_to_dict(self, df: 'DataFrame') -> list:
        """Convert DataFrame to list of dictionaries"""
        try:
            # Collect data
            rows = df.collect()
            
            # Convert to list of dicts
            result = []
            for row in rows:
                row_dict = row.asDict()
                # Convert datetime objects to strings
                for key, value in row_dict.items():
                    if isinstance(value, datetime):
                        row_dict[key] = value.isoformat()
                result.append(row_dict)
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Error converting DataFrame to dict: {e}")
            raise
    
    def _ingest_to_database(self, processed_data: list) -> bool:
        """
        Ingest processed data to database
        
        Args:
            processed_data: List of processed records
            
        Returns:
            bool: Success status
        """
        try:
            logger.info(f"📊 Ingesting {len(processed_data)} records to database...")
            
            success_count = 0
            for record in processed_data:
                try:
                    # Prepare raw data record
                    raw_record = {
                        "station_id": int(record.get("station_id", 0)),
                        "measurement_time": self._parse_measurement_time(record.get("measurement_time")),
                        "ph": float(record.get("ph", 7.0)),
                        "temperature": float(record.get("temperature", 25.0)),
                        "do": float(record.get("do", 8.0)),
                    }
                    
                    # Insert raw data
                    if db_manager.insert_raw_data(raw_record):
                        success_count += 1
                        logger.debug(f"✅ Inserted raw data for station {raw_record['station_id']}")
                    else:
                        logger.warning(f"⚠️ Failed to insert raw data for station {raw_record['station_id']}")
                        
                except Exception as e:
                    logger.error(f"❌ Error inserting record: {e}")
                    continue
            
            success_rate = success_count / len(processed_data) if processed_data else 0
            logger.info(f"📊 Database ingestion: {success_count}/{len(processed_data)} records ({success_rate:.1%})")
            
            return success_count > 0
            
        except Exception as e:
            logger.error(f"❌ Database ingestion error: {e}")
            return False
    
    def _parse_measurement_time(self, mt) -> datetime:
        """Parse measurement time from various formats"""
        try:
            if isinstance(mt, str):
                # Handle ISO strings including with timezone or trailing Z
                mt_str = mt[:-1] if mt.endswith("Z") else mt
                return datetime.fromisoformat(mt_str)
            elif isinstance(mt, (int, float)):
                return datetime.fromtimestamp(mt, tz=timezone.utc)
            else:
                return datetime.now(timezone.utc)
        except Exception:
            return datetime.now(timezone.utc)
    
    def _send_notifications(self, processed_data: list) -> bool:
        """
        Send notifications for processed data
        
        Args:
            processed_data: List of processed records
            
        Returns:
            bool: Success status
        """
        try:
            logger.info(f"📱 Sending notifications for {len(processed_data)} records...")
            
            success_count = 0
            for record in processed_data:
                try:
                    station_id = int(record.get("station_id", 0))
                    wqi_value = float(record.get("wqi", 50.0))
                    alert_level = record.get("alert_level", "normal")
                    
                    # Only send notifications for critical/warning/excellent levels
                    if alert_level in ["critical", "warning", "excellent"]:
                        notification_sent = self._send_single_notification(
                            station_id, wqi_value, alert_level
                        )
                        if notification_sent:
                            success_count += 1
                            logger.debug(f"✅ Sent notification for station {station_id}")
                        else:
                            logger.warning(f"⚠️ Failed to send notification for station {station_id}")
                    
                except Exception as e:
                    logger.error(f"❌ Error sending notification: {e}")
                    continue
            
            logger.info(f"📱 Notifications: {success_count} sent successfully")
            return success_count > 0
            
        except Exception as e:
            logger.error(f"❌ Notification error: {e}")
            return False
    
    def _send_single_notification(self, station_id: int, wqi_value: float, alert_level: str) -> bool:
        """
        Send single notification
        
        Args:
            station_id: Station ID
            wqi_value: WQI value
            alert_level: Alert level
            
        Returns:
            bool: Success status
        """
        try:
            # Determine notification content based on alert level
            if alert_level == "critical":
                title = f"🚨 Critical Water Quality Alert - Station {station_id}"
                message = f"Current WQI is {wqi_value:.1f}. Immediate action required!"
                status = "critical"
            elif alert_level == "warning":
                title = f"⚠️ Water Quality Warning - Station {station_id}"
                message = f"Current WQI is {wqi_value:.1f}. Monitor closely."
                status = "warning"
            elif alert_level == "excellent":
                title = f"✅ Excellent Water Quality - Station {station_id}"
                message = f"Current WQI is {wqi_value:.1f}. Great water quality!"
                status = "excellent"
            else:
                return True  # No notification needed for normal level
            
            # Send notification via API
            return self._push_notification(
                account_id=station_id,
                title=title,
                message=message,
                status=status
            )
            
        except Exception as e:
            logger.error(f"❌ Error in single notification: {e}")
            return False
    
    def _push_notification(self, account_id: int, title: str, message: str, status: str) -> bool:
        """
        Send push notification to API
        
        Args:
            account_id: Account/Station ID
            title: Notification title
            message: Notification message
            status: Notification status
            
        Returns:
            bool: Success status
        """
        try:
            url = "https://datamanagerment.anhkiet.xyz/notifications/send-notification"
            payload = {
                "account_id": str(account_id),
                "title": title,
                "message": message,
                "status": status
            }
            
            response = requests.post(url, json=payload, timeout=10)
            if response.status_code == 200:
                logger.debug(f"✅ Push notification sent successfully for station {account_id}")
                return True
            else:
                logger.error(f"❌ Push notification failed: {response.text}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Error sending push notification: {e}")
            return False

    def close(self):
        """Close Spark session"""
        if self.spark:
            self.spark.stop()
            logger.info("✅ Spark session closed")

# Global consumer instance
_consumer = None

def get_spark_consumer():
    """Get or create Spark consumer instance"""
    global _consumer
    if _consumer is None:
        _consumer = SparkKafkaConsumer()
    return _consumer

def process_kafka_message_with_spark(message: str) -> Dict[str, Any]:
    """
    Process Kafka message with Spark (main function)
    
    Args:
        message: JSON string from Kafka
        
    Returns:
        Dict with processing results
    """
    try:
        consumer = get_spark_consumer()
        result = consumer.process_kafka_message(message)
        
        logger.info(f"📊 Spark processing result: {result.get('success', False)}")
        return result
        
    except Exception as e:
        logger.error(f"❌ Error in process_kafka_message_with_spark: {e}")
        return {"success": False, "error": str(e)}

def run_spark_kafka_consumer():
    """Run Spark Kafka consumer (for background processing)"""
    try:
        logger.info("🚀 Starting Spark Kafka consumer...")
        
        consumer = get_spark_consumer()
        if not consumer.spark:
            logger.error("❌ Failed to initialize Spark session")
            return
        
        # This would be used for continuous Kafka consumption
        # For now, just log that consumer is ready
        logger.info("✅ Spark Kafka consumer is ready for processing")
        
    except Exception as e:
        logger.error(f"❌ Error starting Spark consumer: {e}")

if __name__ == "__main__":
    # Test the consumer
    test_message = '{"station_id": 1, "ph": 7.2, "temperature": 25.5, "do": 8.1, "measurement_time": "2024-01-15T10:30:00Z"}'
    result = process_kafka_message_with_spark(test_message)
    print(f"Test result: {result}")
