import logging
import os
import math
from datetime import datetime
from dateutil.relativedelta import relativedelta

from pyspark.sql import SparkSession, functions as F, Row
from pyspark.sql.functions import col, expr, from_json, lit
from pyspark.sql.types import StructField, StructType, FloatType, TimestampType, DoubleType
from pyspark.sql.window import Window
from pyspark.ml import PipelineModel
from xgboost.spark import SparkXGBRegressorModel

# ——— Thiết lập logging ———
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("iot_stream")

def _no_op_launch(self, *args, **kwargs):
    return None

_no_op_launch.typeConverter = lambda x: x
SparkXGBRegressorModel.launch_tracker_on_driver = _no_op_launch

# ——— Cấu hình ———
MODEL_PATH       = os.getenv('MODEL_PATH', '/opt/bitnami/spark/models/best_xgb_pipeline')
ES_RESOURCE      = os.getenv('ES_RESOURCE', 'water_quality')
BOOTSTRAP_SERVERS= os.getenv('BOOTSTRAP_SERVERS_CONS', '77.37.44.237:9092')  # VPS Kafka address
TOPIC_NAME       = os.getenv('TOPIC_NAME_CONS', 'water-quality-data')
CHECKPOINT_LOC   = os.getenv('CHECKPOINT_LOCATION', '/tmp/spark/checkpoint_predict')
FORECAST_HORIZON = int(os.getenv('FORECAST_HORIZON', '12'))  # tháng

# ——— Khởi tạo Spark session ———
def get_spark_session():
    jars = [
        "file:///opt/bitnami/spark/jars/spark-sql-kafka-0-10_2.12-3.2.0.jar",
        "file:///opt/bitnami/spark/jars/kafka-clients-2.8.0.jar",
        "file:///opt/bitnami/spark/jars/elasticsearch-spark-30_2.12-8.17.3.jar"
    ]
    spark = (SparkSession.builder
             .appName("IoT Water Quality Monitoring")
             .master("local[*]")
             .config("spark.driver.memory", "2g")
             .config("spark.executor.memory", "2g")
             .config("spark.jars", ",".join(jars))
             # Kafka security configs
             .config("spark.kafka.consumer.group.id", "water_quality_spark_consumer")
             .config("spark.kafka.socket.connection.setup.timeout.ms", "10000")
             .config("spark.kafka.socket.connection.setup.max.retries", "3")
             .getOrCreate())
    spark.conf.set("spark.sql.execution.arrow.pyspark.enabled", "true")
    return spark

# ——— Tính chỉ số WQI đơn giản ———
def calculate_wqi(df):
    pH_wqi = F.when((df.ph >= 6.5) & (df.ph <= 8.5), 100).otherwise(0)
    temp_wqi = F.when((df.temperature >= 20) & (df.temperature <= 30), 100).otherwise(0)
    wqi = pH_wqi * 0.6 + temp_wqi * 0.4
    return df.withColumn("wqi", wqi)

# ——— Thêm UUID cho mỗi record ———
def add_uuid(df):
    return df.withColumn("id", expr("uuid()"))

# ——— Ghi DataFrame vào Elasticsearch ———
def save_to_elasticsearch(df):
    ES_HOST = os.getenv('ES_HOST', 'elasticsearch.anhkiet.xyz')
    ES_PORT = os.getenv('ES_PORT', '80')
    ES_USER = os.getenv('ES_USER', 'elastic')
    ES_PASSWORD = os.getenv('ES_PASSWORD', '6F2A0Ib+Tqm9Lti9Fpfl')
    try:
        df.write \
          .format("org.elasticsearch.spark.sql") \
          .option("es.nodes", ES_HOST) \
          .option("es.port", ES_PORT) \
          .option("es.net.ssl", "false") \
          .option("es.net.http.auth.user", ES_USER) \
          .option("es.net.http.auth.pass", ES_PASSWORD) \
          .option("es.resource", 'water_quality') \
          .option("es.index.auto.create", "true") \
          .option("es.nodes.wan.only", "true") \
          .option("es.scheme", "http") \
          .mode("append") \
          .save()
        logger.info("Saved to Elasticsearch")
    except Exception as e:
        logger.error(f"Error writing to Elasticsearch: {e}")

# ——— Batch predictor ———
def process_batch_predict(batch_df, batch_id):
    if batch_df.rdd.isEmpty():
        logger.info(f"Batch {batch_id}: Empty batch, skipping processing")
        return

    spark = get_spark_session()

    # 1) History WQI + UUID
    hist = calculate_wqi(batch_df).transform(add_uuid)

    # Check if there are enough rows for window-based features
    row_count = hist.count()
    if row_count < 12:
        logger.warning(f"Batch {batch_id}: Only {row_count} rows, need at least 12 for MA12. Skipping forecasting.")
        save_to_elasticsearch(hist)  # Save historical data only
        return

    # 2) Feature engineering on history
    hist = (hist
        .withColumn("day_of_year", F.dayofyear("measurement_time"))
        .withColumn("year_sin", F.sin(F.col("day_of_year") * 2 * math.pi / 365.25))
        .withColumn("year_cos", F.cos(F.col("day_of_year") * 2 * math.pi / 365.25))
        .withColumn("time_idx", F.unix_timestamp("measurement_time").cast(DoubleType()))
    )

    win3 = Window.orderBy("measurement_time").rowsBetween(-2, 0)
    win6 = Window.orderBy("measurement_time").rowsBetween(-5, 0)
    win12 = Window.orderBy("measurement_time").rowsBetween(-11, 0)
    lagw = Window.orderBy("measurement_time")

    hist = (hist
        .withColumn("MA3", F.avg("wqi").over(win3))
        .withColumn("MA6", F.avg("wqi").over(win6))
        .withColumn("MA12", F.avg("wqi").over(win12))
        .withColumn("trend3", F.col("wqi") - F.col("MA3"))
        .withColumn("trend6", F.col("wqi") - F.col("MA6"))
        .withColumn("trend12", F.col("wqi") - F.col("MA12"))
        .withColumn("ROC3", (F.col("wqi") - F.lag("wqi", 3).over(lagw)) / 3)
        .withColumn("ROC6", (F.col("wqi") - F.lag("wqi", 6).over(lagw)) / 6)
    )

    # Fill null values with 0.0
    hist = hist.fillna({
        "MA3": 0.0, "MA6": 0.0, "MA12": 0.0,
        "trend3": 0.0, "trend6": 0.0, "trend12": 0.0,
        "ROC3": 0.0, "ROC6": 0.0
    })

    # Debug: Show hist
    logger.info(f"Batch {batch_id}: Historical data")
    hist.show(truncate=False)

    # 3) Get last 12 records for wqi_history and last record for state
    last_12 = hist.orderBy(F.desc("measurement_time")).limit(12).collect()
    last = last_12[0]  # Most recent record
    ts = last['measurement_time'].replace(day=15, hour=0, minute=0, second=0, microsecond=0)  # Set to 15th
    state = {
        'ph': float(last['ph']) if last['ph'] is not None else 0.0,
        'temperature': float(last['temperature']) if last['temperature'] is not None else 0.0,
        'do': float(last['do']) if last['do'] is not None else 0.0,
        'wqi': float(last['wqi']) if last['wqi'] is not None else 0.0,
        'MA3': float(last['MA3']) if last['MA3'] is not None else 0.0,
        'MA6': float(last['MA6']) if last['MA6'] is not None else 0.0,
        'MA12': float(last['MA12']) if last['MA12'] is not None else 0.0,
        'trend3': float(last['trend3']) if last['trend3'] is not None else 0.0,
        'trend6': float(last['trend6']) if last['trend6'] is not None else 0.0,
        'trend12': float(last['trend12']) if last['trend12'] is not None else 0.0,
        'ROC3': float(last['ROC3']) if last['ROC3'] is not None else 0.0,
        'ROC6': float(last['ROC6']) if last['ROC6'] is not None else 0.0
    }

    # Initialize wqi_history (chronological order)
    wqi_history = [float(row['wqi']) for row in reversed(last_12)]
    if len(wqi_history) < 12:
        wqi_history = [state['wqi']] * (12 - len(wqi_history)) + wqi_history

    # Compute trends (stronger amplification)
    last_3 = hist.orderBy(F.desc("measurement_time")).limit(3).select("ph", "temperature", "do", "wqi").collect()
    ph_trend = (last_3[0]['ph'] - last_3[-1]['ph']) / 3 * 15 if len(last_3) >= 3 else 0.0  # Stronger trend
    temp_trend = (last_3[0]['temperature'] - last_3[-1]['temperature']) / 3 * 15 if len(last_3) >= 3 else 0.0
    do_trend = (last_3[0]['do'] - last_3[-1]['do']) / 3 * 15 if len(last_3) >= 3 else 0.0
    wqi_trend = (last_3[0]['wqi'] - last_3[-1]['wqi']) / 3 * 15 if len(last_3) >= 3 else 0.0

    # Debug: Log initial state and trends
    logger.info(f"Batch {batch_id}: Initial state = {state}, wqi_history = {wqi_history}")
    logger.info(f"Batch {batch_id}: Trends - ph: {ph_trend}, temp: {temp_trend}, do: {do_trend}, wqi: {wqi_trend}")

    # 4) Load pipeline model
    model = PipelineModel.load(MODEL_PATH)
    logger.info(f"Batch {batch_id}: Pipeline stages: {model.stages}")

   # 5) Iterative forecasting - tập trung vào dự đoán WQI
    input_schema = StructType([
        StructField("measurement_time", TimestampType()),
        StructField("ph", FloatType()),
        StructField("temperature", FloatType()),
        StructField("do", FloatType()),
        StructField("wqi", FloatType()),
        StructField("MA3", DoubleType()),
        StructField("MA6", DoubleType()),
        StructField("MA12", DoubleType()),
        StructField("trend3", DoubleType()),
        StructField("trend6", DoubleType()),
        StructField("trend12", DoubleType()),
        StructField("ROC3", DoubleType()),
        StructField("ROC6", DoubleType()),
    ])

    forecasts = []
    # Lấy giá trị WQI hiện tại
    current_wqi = state['wqi']
    # Tính độ tăng/giảm WQI theo xu hướng gần đây
    wqi_increment = wqi_trend / 15  # Điều chỉnh độ tăng/giảm theo tháng

    logger.info(f"Batch {batch_id}: Initial WQI = {current_wqi}, WQI monthly increment = {wqi_increment}")

    for i in range(FORECAST_HORIZON):
        # Update timestamp (15th of next month)
        new_ts = ts + relativedelta(months=i + 1)
        new_ts = new_ts.replace(day=15, hour=0, minute=0, second=0, microsecond=0)
        logger.info(f"Iteration {i}: Using timestamp {new_ts}, current WQI = {current_wqi}")
        
        # Tính toán thành phần mùa (seasonal component) cho WQI
        month_of_year = new_ts.month
        # Tăng biên độ dao động theo mùa (season amplitude)
        seasonal_effect = 10.0 * math.sin(2 * math.pi * (month_of_year / 12))
        
        # Giữ nguyên các thông số ph, temperature, do từ lần đọc cuối
        # Chỉ cập nhật WQI theo xu hướng và yếu tố mùa
        current_wqi = max(0.0, min(100.0, current_wqi + wqi_increment + seasonal_effect))
        
        # Cập nhật trạng thái hiện tại - chỉ cập nhật WQI
        state['wqi'] = float(current_wqi)
        
        # Tạo DataFrame đầu vào cho dự đoán
        single = spark.createDataFrame([
            (new_ts, state['ph'], state['temperature'], state['do'], state['wqi'],
            state['MA3'], state['MA6'], state['MA12'],
            state['trend3'], state['trend6'], state['trend12'],
            state['ROC3'], state['ROC6'])
        ], schema=input_schema)
        
        # Tính toán đặc trưng theo mùa
        single = (single
            .withColumn("day_of_year", F.dayofyear("measurement_time"))
            .withColumn("year_sin", F.sin(F.col("day_of_year") * 2 * math.pi / 365.25))
            .withColumn("year_cos", F.cos(F.col("day_of_year") * 2 * math.pi / 365.25))
            .withColumn("time_idx", F.unix_timestamp("measurement_time").cast(DoubleType()))
        )
        
        # Debug: Hiển thị trạng thái và lịch sử WQI
        logger.info(f"Iteration {i}: WQI State = {state['wqi']}, wqi_history = {wqi_history}")
        
        # Dự đoán với model
        pred = model.transform(single).collect()[0]["prediction"]
        logger.info(f"Iteration {i}: WQI Prediction = {pred}")
        
        # Lưu kết quả dự đoán
        forecasts.append(Row(measurement_time=new_ts, wqi_forecast=float(pred)))
        
        # Cập nhật lịch sử WQI và các đặc trưng dựa trên kết quả dự đoán
        wqi_history.append(float(pred))
        wqi_history.pop(0)  # Giữ 12 giá trị gần nhất
        
        # Cập nhật các đặc trưng MA và trend dựa trên kết quả dự đoán
        state['MA3'] = sum(wqi_history[-3:]) / 3
        state['MA6'] = sum(wqi_history[-6:]) / 6
        state['MA12'] = sum(wqi_history) / len(wqi_history)
        state['trend3'] = pred - state['MA3']
        state['trend6'] = pred - state['MA6']
        state['trend12'] = pred - state['MA12']
        state['ROC3'] = (pred - wqi_history[-4]) / 3 if len(wqi_history) >= 4 else 0.0
        state['ROC6'] = (pred - wqi_history[-7]) / 6 if len(wqi_history) >= 7 else 0.0
        
        # Cập nhật WQI hiện tại từ kết quả dự đoán để sử dụng cho lần dự đoán tiếp theo
        current_wqi = float(pred)
        
        # Điều chỉnh increment theo xu hướng mới
        if i > 0 and i % 3 == 0:  # Điều chỉnh lại xu hướng sau mỗi 3 tháng
            recent_wqis = wqi_history[-3:]
            if len(recent_wqis) >= 3:
                new_trend = (recent_wqis[-1] - recent_wqis[0]) / 3
                wqi_increment = new_trend / 5  # Điều chỉnh độ tăng/giảm theo xu hướng mới
                logger.info(f"Iteration {i}: Adjusted WQI increment to {wqi_increment} based on recent predictions")

    # 6) Write out with explicit schema
    schema_out = StructType([
        StructField("measurement_time", TimestampType()),
        StructField("wqi_forecast", DoubleType())
    ])
    out = spark.createDataFrame(forecasts, schema=schema_out)
    out = out.withColumn("measurement_time", F.date_format("measurement_time", "yyyy-MM-dd'T'00:00:00'Z'"))
    logger.info(f"Batch {batch_id}: Final forecasts")
    out.show(truncate=False)
    save_to_elasticsearch(out)
    logger.info(f"Batch {batch_id}: Iterative forecast completed")

# ——— Chạy Structured Streaming ———
def run_spark_job():
    spark = get_spark_session()

    # Định nghĩa schema cho JSON message
    schema = StructType([
        StructField("measurement_time", TimestampType()),
        StructField("ph", FloatType()),
        StructField("temperature", FloatType()),
        StructField("do", FloatType()),
        StructField("wqi", FloatType())
    ])

    df_raw = (spark.readStream
              .format("kafka")
              .option("kafka.bootstrap.servers", BOOTSTRAP_SERVERS)
              .option("subscribe", TOPIC_NAME)
              .option("startingOffsets", "earliest")
              .load())

    df = (df_raw.selectExpr("CAST(value AS STRING) AS json")
          .select(from_json(col("json"), schema).alias("data"))
          .select("data.*"))

    stream_df = df.transform(add_uuid)
    (stream_df.writeStream
      .trigger(processingTime="30 seconds")
      .outputMode("append")
      .foreachBatch(process_batch_predict)
      .option("checkpointLocation", CHECKPOINT_LOC)
      .start()
      .awaitTermination())

if __name__ == "__main__":
    run_spark_job()