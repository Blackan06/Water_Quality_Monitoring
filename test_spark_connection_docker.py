#!/usr/bin/env python3
"""
Test Spark connection from Docker environment
"""

import logging
import sys
import os

# Add the include directory to Python path
sys.path.append('/usr/local/airflow/include')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_spark_connection():
    """Test Spark connection from Docker environment"""
    try:
        logger.info("🚀 Testing Spark connection from Docker...")
        
        # Import Spark consumer
        from iot_streaming.spark_consumer import get_spark_consumer
        
        # Get consumer instance
        consumer = get_spark_consumer()
        
        if consumer.spark is None:
            logger.error("❌ Spark session is None")
            return False
        
        # Test basic Spark operations
        logger.info("📊 Testing basic Spark operations...")
        
        # Create a simple DataFrame
        test_data = [
            {"station_id": 1, "ph": 7.2, "temperature": 25.5, "do": 8.1},
            {"station_id": 2, "ph": 6.8, "temperature": 24.0, "do": 7.5}
        ]
        
        df = consumer.spark.createDataFrame(test_data)
        logger.info(f"✅ Created DataFrame with {df.count()} rows")
        
        # Test WQI calculation
        from pyspark.sql.functions import when, col, round
        
        df_wqi = df.withColumn(
            'wqi',
            round(
                when((col('ph') >= 6.0) & (col('ph') <= 10.0), (col('ph') - 6.0) * 25).otherwise(0) * 0.3 +
                when((col('temperature') >= 15) & (col('temperature') <= 30), 100 - abs(col('temperature') - 22) * 2).otherwise(0) * 0.3 +
                when(col('do') >= 0, col('do') * 10).otherwise(0) * 0.4, 2
            )
        )
        
        result = df_wqi.collect()
        logger.info(f"✅ WQI calculation successful: {len(result)} results")
        
        for row in result:
            logger.info(f"   Station {row['station_id']}: WQI = {row['wqi']}")
        
        # Test message processing
        test_message = '{"station_id": 1, "ph": 7.2, "temperature": 25.5, "do": 8.1, "measurement_time": "2024-01-15T10:30:00Z"}'
        
        logger.info("📨 Testing message processing...")
        result = consumer.process_kafka_message(test_message)
        
        if result.get("success", False):
            logger.info("✅ Message processing successful!")
            logger.info(f"   Processed {result.get('processed_count', 0)} records")
        else:
            logger.error(f"❌ Message processing failed: {result.get('error', 'Unknown error')}")
            return False
        
        logger.info("🎉 All Spark tests passed!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Spark connection test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_spark_connection()
    sys.exit(0 if success else 1)
