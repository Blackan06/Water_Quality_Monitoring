from pyspark.sql import SparkSession
from pyspark.sql.functions import col, from_json
from pyspark.sql.types import StructType, StructField, StringType, FloatType, TimestampType,IntegerType
import psycopg2
from psycopg2 import sql
from datetime import datetime
from psycopg2.extras import execute_batch
from pyspark.sql import functions as F
from data.db_utils import save_to_postgresql_table

postgresql_table_name = "iot_sensor"

def run_spark_job():

# Tạo SparkSession với cấu hình để tải Kafka dependencies
    spark = SparkSession.builder \
        .appName("IoT Stream Analysis") \
        .master("local[*]") \
        .config("spark.jars", "file:///opt/bitnami/spark/jars/spark-sql-kafka-0-10_2.12-3.5.3.jar,file:///opt/bitnami/spark/jars/kafka-clients-3.4.1.jar,file:///opt/bitnami/spark/jars/spark-streaming-kafka-0-10_2.12-3.5.3.jar") \
        .config("spark.executor.extraClassPath", "file:///opt/bitnami/spark/jars/spark-sql-kafka-0-10_2.12-3.5.3.jar:file:///opt/bitnami/spark/jars/kafka-clients-3.4.1.jar:file:///opt/bitnami/spark/jars/spark-streaming-kafka-0-10_2.12-3.5.3.jar") \
        .config("spark.driver.extraJavaOptions", "-Djava.library.path=$JAVA_HOME/lib/server") \
        .getOrCreate()
    spark.conf.set("spark.sql.execution.arrow.pyspark.enabled", "true")


    # Định nghĩa schema cho dữ liệu JSON
    schema = StructType([
        StructField("id", IntegerType(), True),           # event_id sẽ được ánh xạ vào trường id
        StructField("ph", FloatType(), True),             # event_pH sẽ được ánh xạ vào trường ph
        StructField("turbidity", FloatType(), True),      # event_turbidity sẽ được ánh xạ vào trường turbidity
        StructField("temperature", FloatType(), True),   # event_temperature sẽ được ánh xạ vào trường temperature
        StructField("measurement_time", TimestampType(), True),  # event_datetime sẽ được ánh xạ vào trường measurement_time
        StructField("create_at", TimestampType(), True)  # Tạo thêm trường create_at nếu cần thiết
    ])


    # Đọc dữ liệu từ Kafka topic "iot_data"
    kafka_df = spark.readStream \
        .format("kafka") \
        .option("kafka.bootstrap.servers", "kafka:9092") \
        .option("subscribe", "water-quality-data") \
        .option("startingOffsets", "earliest") \
        .load()

    iot_df = kafka_df.selectExpr("CAST(key AS STRING) as id", "CAST(value AS STRING) as value")
   
    event_message_detail_df_2 = iot_df.select(from_json(col("value"), schema).alias("event_message_detail"))
    event_message_detail_df_3 = event_message_detail_df_2.select("event_message_detail.*")
   
    # Ghi dữ liệu vào PostgreSQL

    query = event_message_detail_df_3 \
                .writeStream \
                .trigger(processingTime='20 seconds') \
                .outputMode("update") \
                .foreachBatch(lambda current_df, epoc_id: save_to_postgresql_table(current_df,spark, epoc_id)) \
                .start()
    

    # Chạy Spark Streaming cho đến khi dừng
    query.awaitTermination()
    
run_spark_job()