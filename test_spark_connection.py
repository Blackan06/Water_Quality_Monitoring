#!/usr/bin/env python3
"""
Test Spark connection and processing
"""

import sys
import os
import logging
from datetime import datetime, timezone

# Add the project root to Python path
sys.path.append('/Users/kiethuynhanh/Documents/Water_Quality_Monitoring')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_spark_connection():
    """Test Spark connection"""
    try:
        logger.info("🚀 Testing Spark connection...")
        
        # Import Spark consumer
        from include.iot_streaming.spark_consumer import SparkKafkaConsumer
        
        # Create consumer instance
        consumer = SparkKafkaConsumer()
        
        if consumer.spark is None:
            logger.error("❌ Spark session is None")
            return False
            
        # Test basic Spark operations
        logger.info("📊 Testing Spark operations...")
        
        # Create test DataFrame
        test_data = [
            {"station_id": 1, "ph": 7.2, "temperature": 25.5, "do": 8.2, "measurement_time": datetime.now(timezone.utc)},
            {"station_id": 2, "ph": 6.8, "temperature": 24.0, "do": 7.5, "measurement_time": datetime.now(timezone.utc)},
        ]
        
        df = consumer.spark.createDataFrame(test_data)
        logger.info(f"✅ Created DataFrame with {df.count()} rows")
        
        # Test data cleaning
        cleaned_df = consumer._clean_data(df)
        logger.info(f"✅ Data cleaning successful: {cleaned_df.count()} valid records")
        
        # Test feature engineering
        features_df = consumer._add_features(cleaned_df)
        logger.info(f"✅ Feature engineering successful: {features_df.count()} records with features")
        
        # Test WQI calculation
        wqi_df = consumer._calculate_wqi_spark(features_df)
        logger.info(f"✅ WQI calculation successful: {wqi_df.count()} records with WQI")
        
        # Show sample results
        logger.info("📋 Sample results:")
        wqi_df.select("station_id", "ph", "temperature", "do", "wqi", "quality_status").show(5)
        
        logger.info("✅ All Spark tests passed!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Spark test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_kafka_message_processing():
    """Test processing a Kafka message"""
    try:
        logger.info("📨 Testing Kafka message processing...")
        
        from include.iot_streaming.spark_consumer import SparkKafkaConsumer
        
        consumer = SparkKafkaConsumer()
        
        # Create test message
        test_message = {
            "station_id": 1,
            "ph": 7.2,
            "temperature": 25.5,
            "do": 8.2,
            "measurement_time": datetime.now(timezone.utc).isoformat()
        }
        
        import json
        message_str = json.dumps(test_message)
        
        # Process message
        result = consumer.process_kafka_message(message_str)
        
        if result.get("success"):
            logger.info("✅ Kafka message processing successful")
            logger.info(f"📊 Processed data: {result.get('data', {})}")
            return True
        else:
            logger.error(f"❌ Kafka message processing failed: {result.get('error')}")
            return False
            
    except Exception as e:
        logger.error(f"❌ Kafka message test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    logger.info("🧪 Starting Spark connection tests...")
    
    # Test 1: Basic Spark connection
    spark_ok = test_spark_connection()
    
    if spark_ok:
        # Test 2: Kafka message processing
        kafka_ok = test_kafka_message_processing()
        
        if kafka_ok:
            logger.info("🎉 All tests passed! Spark is working correctly.")
        else:
            logger.error("❌ Kafka message processing failed")
    else:
        logger.error("❌ Spark connection failed")
    
    logger.info("🏁 Test completed")
