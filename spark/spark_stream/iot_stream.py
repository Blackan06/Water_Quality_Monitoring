import logging
import os
import math
from datetime import datetime
from dateutil.relativedelta import relativedelta
import requests
import psycopg2
from psycopg2 import sql
from openai import OpenAI

from pyspark.sql import SparkSession, functions as F, Row
from pyspark.sql.functions import col, expr, from_json, lit
from pyspark.sql.types import StructField, StructType, FloatType, TimestampType, DoubleType
from pyspark.sql.window import Window
from pyspark.ml import PipelineModel
from xgboost.spark import SparkXGBRegressorModel
from pyspark.ml.evaluation import RegressionEvaluator

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

# Cấu hình kết nối đến PostgreSQL
DB_CONFIG = {
    'host': os.getenv('DB_HOST', '149.28.145.56'),
    'port': os.getenv('DB_PORT', '5432'),
    'database': os.getenv('DB_NAME', 'wqi_db'),
    'user': os.getenv('DB_USER', 'postgres'),
    'password': os.getenv('DB_PASSWORD', 'postgres1234')
}

# Cấu hình OpenAI API
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))


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

def save_monitoring_data_to_postgres(data):
    conn = None
    try:
        conn = psycopg2.connect(
            host=DB_CONFIG['host'],
            port=DB_CONFIG['port'],
            dbname=DB_CONFIG['database'],
            user=DB_CONFIG['user'],
            password=DB_CONFIG['password']
        )
        cur = conn.cursor()
        insert_query = sql.SQL("""
            INSERT INTO wqi_monitoring_data (temperature, "do", ph, wqi, wq_date)
            VALUES (%s, %s, %s, %s, %s)
        """)
        cur.execute(insert_query, (
            data['temperature'],
            data['do'],
            data['ph'],
            data['wqi'],
            data['wq_date']
        ))
        conn.commit()
        cur.close()
        logger.info("Saved monitoring data to wqi_monitoring_data.")
    except Exception as e:
        logger.error(f"Error saving monitoring data: {e}")
    finally:
        if conn:
            conn.close()

def push_notification(account_id, title, message, status):
    url = "https://dm.anhkiet.xyz/notifications/send-notification"
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
        else:
            logger.error(f"Push notification failed: {response.text}")
    except Exception as e:
        logger.error(f"Error sending push notification: {e}")

def analyze_water_quality(wqi, ph, do, temperature):
    prompt = (
        "Đây là nước nuôi cá. "
        f"WQI dự đoán: {wqi}, pH={ph}, DO={do} mg/L, nhiệt độ={temperature}°C. "
        "Hãy đánh giá khách quan và đề xuất biện pháp, chỉ viết đúng 4 câu, "
        "mỗi câu kết thúc bằng dấu chấm, đầy đủ nghĩa, không bỏ thiếu chữ, không xuống dòng."
    )
    resp = client.chat.completions.create(
        model="gpt-4.1-nano",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=150,
        temperature=0.5
    )
    text = resp.choices[0].message.content.strip()
    # Gộp và đảm bảo không có newline
    text = " ".join(text.splitlines())
    return f"WQI tháng sau: {wqi}. {text}"

# ——— Batch predictor ———
def process_batch_predict(batch_df, batch_id):
    logger.info(f"Processing batch {batch_id} with {batch_df.count()} records.")
    if batch_df.rdd.isEmpty():
        logger.info(f"Batch {batch_id}: Empty batch, skipping processing")
        return

    spark = get_spark_session()

    # 1) Tính WQI + UUID và lưu monitoring
    hist = calculate_wqi(batch_df).transform(add_uuid)
    logger.info(f"Batch {batch_id}: Calculated WQI and added UUID.")
    for row in hist.collect():
        save_monitoring_data_to_postgres({
            'temperature': row['temperature'],
            'do': row['do'],
            'ph': row['ph'],
            'wqi': row['wqi'],
            'wq_date': str(row['measurement_time'])
        })
    logger.info(f"Batch {batch_id}: Saved monitoring data to PostgreSQL.")

    # 2) Tính window-features trên toàn bộ history (ví dụ MA3, MA6, MA12, trend, ROC)
    win3  = Window.orderBy("measurement_time").rowsBetween(-2, 0)
    win6  = Window.orderBy("measurement_time").rowsBetween(-5, 0)
    win12 = Window.orderBy("measurement_time").rowsBetween(-11, 0)
    lagw  = Window.orderBy("measurement_time")

    hist = (hist
        .withColumn("MA3",    F.avg("wqi").over(win3))
        .withColumn("MA6",    F.avg("wqi").over(win6))
        .withColumn("MA12",   F.avg("wqi").over(win12))
        .withColumn("trend3",  F.col("wqi") - F.col("MA3"))
        .withColumn("trend6",  F.col("wqi") - F.col("MA6"))
        .withColumn("trend12", F.col("wqi") - F.col("MA12"))
        .withColumn("ROC3",   (F.col("wqi") - F.lag("wqi", 3).over(lagw)) / 3)
        .withColumn("ROC6",   (F.col("wqi") - F.lag("wqi", 6).over(lagw)) / 6)
        .fillna(0.0, ["MA3","MA6","MA12","trend3","trend6","trend12","ROC3","ROC6"])
    )
    logger.info(f"Batch {batch_id}: Calculated window features.")

    # 3) Lấy record cuối cùng và các feature vừa tính
    last = hist.orderBy(F.desc("measurement_time")).first()
    ts   = last['measurement_time'].replace(day=15, hour=0, minute=0, second=0, microsecond=0)
    state = {
        'ph': float(last['ph']),
        'temperature': float(last['temperature']),
        'do': float(last['do']),
        'wqi': float(last['wqi'])
    }
    feat = { f: float(last[f]) for f in
             ["MA3","MA6","MA12","trend3","trend6","trend12","ROC3","ROC6"] }
    logger.info(f"Batch {batch_id}: Retrieved last record and features.")

    # 4) Tạo DataFrame chỉ cho tháng tiếp theo, nhồi đúng các giá trị feature
    next_ts = ts + relativedelta(months=1)
    next_ts = next_ts.replace(day=15)

    input_schema = StructType([
        StructField("measurement_time", TimestampType()),
        StructField("ph", FloatType()),
        StructField("temperature", FloatType()),
        StructField("do", FloatType()),
        StructField("wqi", FloatType()),
        StructField("MA3", DoubleType()),    StructField("MA6", DoubleType()),
        StructField("MA12", DoubleType()),   StructField("trend3", DoubleType()),
        StructField("trend6", DoubleType()), StructField("trend12", DoubleType()),
        StructField("ROC3", DoubleType()),   StructField("ROC6", DoubleType())
    ])

    single = spark.createDataFrame([(
        next_ts,
        state['ph'], state['temperature'], state['do'], state['wqi'],
        feat['MA3'], feat['MA6'], feat['MA12'],
        feat['trend3'], feat['trend6'], feat['trend12'],
        feat['ROC3'], feat['ROC6']
    )], schema=input_schema)
    logger.info(f"Batch {batch_id}: Created DataFrame for next month prediction.")

    # 5) Nếu pipeline có stage seasonal/time_idx thì thêm vào
    single = (single
        .withColumn("day_of_year", F.dayofyear("measurement_time"))
        .withColumn("year_sin",   F.sin(F.col("day_of_year") * 2 * math.pi / 365.25))
        .withColumn("year_cos",   F.cos(F.col("day_of_year") * 2 * math.pi / 365.25))
        .withColumn("time_idx",   F.unix_timestamp("measurement_time").cast(DoubleType()))
    )
    logger.info(f"Batch {batch_id}: Added seasonal features.")

    # 6) Load model và predict
    model = PipelineModel.load(MODEL_PATH)
    pred = model.transform(single).first()["prediction"]
    logger.info(f"Batch {batch_id}: Forecast next month ({next_ts.date()}): {pred:.2f}")

    # 7) Ghi ES và gửi notification
    out = spark.createDataFrame(
        [Row(measurement_time=next_ts, wqi_forecast=float(pred))],
        StructType([
            StructField("measurement_time", TimestampType()),
            StructField("wqi_forecast", DoubleType())
        ])
    )
    save_to_elasticsearch(out)
    logger.info(f"Batch {batch_id}: Saved forecast to Elasticsearch.")

    # Gửi thông báo sau khi train xong
    status = "good" if pred > 50 else "danger"  # Sử dụng giá trị WQI dự đoán
    analysis = analyze_water_quality(pred, state['ph'], state['do'], state['temperature'])
    push_notification(
        account_id=3,
        title="Kết quả WQI tháng sau",
        message=analysis,
        status=status
        
    )

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