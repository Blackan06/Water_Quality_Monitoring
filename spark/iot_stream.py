from pyspark.sql import SparkSession
from pyspark.sql.functions import col, from_json, expr, when
from pyspark.sql.types import StructType, StructField, FloatType, TimestampType
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression

def get_spark_session():
    """
    Khởi tạo SparkSession với cấu hình cần thiết cho Kafka, Elasticsearch và tối ưu bộ nhớ.
    """
    spark = SparkSession.builder \
        .appName("IoT Water Quality Monitoring") \
        .master("local[*]") \
        .config("spark.jars", "file:///opt/bitnami/spark/jars/spark-sql-kafka-0-10_2.12-3.2.0.jar,"
                                "file:///opt/bitnami/spark/jars/kafka-clients-2.8.0.jar,"
                                "file:///opt/bitnami/spark/jars/spark-streaming-kafka-0-10_2.12-3.2.0.jar,"
                                "file:///opt/bitnami/spark/jars/elasticsearch-spark-30_2.12-8.17.3.jar") \
        .config("spark.driver.memory", "2g") \
        .config("spark.executor.memory", "2g") \
        .getOrCreate()
    spark.conf.set("spark.sql.execution.arrow.pyspark.enabled", "true")
    return spark

def add_uuid(df):
    """
    Thêm cột id với giá trị UUID tự động cho mỗi record.
    """
    return df.withColumn("id", expr("uuid()"))

def train_model(batch_df):
    """
    Training mô hình Linear Regression trên dữ liệu batch.
    - Sử dụng cột "temperature" làm feature và "ph_value" làm label.
    Trả về mô hình đã training.
    """
    if batch_df.rdd.isEmpty():
        print("Batch rỗng, bỏ qua training.")
        return None

    # Lọc các bản ghi có dữ liệu hợp lệ
    training_df = batch_df.filter(col("temperature").isNotNull() & col("ph_value").isNotNull())
    if training_df.rdd.isEmpty():
        print("Không có dữ liệu hợp lệ để training trong batch này.")
        return None

    # Phân chia lại partition cho phù hợp
    training_df = training_df.repartition(4)

    # Chuẩn bị dữ liệu training với VectorAssembler
    assembler = VectorAssembler(inputCols=["temperature"], outputCol="features")
    training_data = assembler.transform(training_df).select("features", col("ph_value").alias("label"))
    training_data.cache()

    # Khởi tạo và training mô hình Linear Regression
    lr = LinearRegression(maxIter=10, regParam=0.3, elasticNetParam=0.8)
    lr_model = lr.fit(training_data)

    # Lấy ra các chỉ số của mô hình training
    training_summary = lr_model.summary
    print(f"Training batch completed: RMSE = {training_summary.rootMeanSquaredError}, R2 = {training_summary.r2}")

    training_data.unpersist()
    return lr_model

def forecast_model(lr_model, df):
    """
    Sử dụng mô hình đã training để dự báo trên DataFrame đầu vào.
    Dữ liệu đầu vào cần có cột "temperature". Hàm này sẽ thêm cột "prediction".
    """
    if lr_model is None:
        print("Không có mô hình, bỏ qua dự báo.")
        return

    # Chuẩn bị cột features từ cột "temperature" cho dự báo
    assembler = VectorAssembler(inputCols=["temperature"], outputCol="features")
    forecast_data = assembler.transform(df)
    predictions = lr_model.transform(forecast_data)
    
    # In một vài kết quả dự báo
    predictions.select("temperature", "ph_value", "prediction").show(5, truncate=False)

def process_batch(batch_df, batch_id):
    """
    Hàm xử lý mỗi micro-batch:
    - Ghi dữ liệu vào Elasticsearch.
    - Training mô hình trên batch dữ liệu.
    - Sử dụng mô hình đã training để dự báo trên batch dữ liệu.
    """
    print(f"Processing batch id: {batch_id}")

    # Ví dụ sử dụng broadcast variable để truyền ngưỡng pH (nếu cần)
    ph_threshold_lookup = {"ph_threshold": 7.0}
    broadcast_threshold = batch_df.sparkSession.sparkContext.broadcast(ph_threshold_lookup)
    batch_df = batch_df.withColumn(
        "ph_ok",
        when(col("ph_value") >= broadcast_threshold.value["ph_threshold"], True).otherwise(False)
    )

    # Ghi dữ liệu vào Elasticsearch với domain HTTPS
    batch_df.write.format("org.elasticsearch.spark.sql") \
        .option("es.resource", "iot-sensors/data") \
        .option("es.nodes", "elasticsearch.anhkiet.xyz") \
        .option("es.port", "443") \
        .option("es.net.ssl", "true") \
        .option("es.nodes.wan.only", "true") \
        .option("es.net.http.auth.user", "elastic") \
        .option("es.net.http.auth.pass", "elasticpassword") \
        .option("es.mapping.id", "id") \
        .mode("append") \
        .save()

    # Training mô hình trên batch dữ liệu hiện tại
    lr_model = train_model(batch_df)
    
    # Sau khi training xong, sử dụng mô hình để dự báo trên dữ liệu của batch này
    forecast_model(lr_model, batch_df)

def run_spark_job(**kwargs):
    """
    Hàm chính để chạy Spark Structured Streaming:
    - Đọc dữ liệu từ Kafka.
    - Parse dữ liệu JSON theo schema.
    - Thêm cột id.
    - Sử dụng foreachBatch để xử lý mỗi micro-batch: ghi vào ES, training và dự báo.
    - Sau khi job hoàn thành, thông báo cho Airflow.
    """
    ti = kwargs.get('ti', None)
    data = ti.xcom_pull(task_ids='kafka_consumer_task', key='kafka_data') if ti else None
    print(f"Running Spark job with data: {data}")

    spark = get_spark_session()

    schema = StructType([
        StructField("measurement_time", TimestampType(), True),
        StructField("ph_value", FloatType(), True),
        StructField("temperature", FloatType(), True),
    ])

    kafka_df = spark.readStream \
        .format("kafka") \
        .option("kafka.bootstrap.servers", "kafka:9092") \
        .option("subscribe", "water-quality-data") \
        .option("startingOffsets", "earliest") \
        .load()

    iot_df = kafka_df.selectExpr("CAST(value AS STRING) as value")
    json_df = iot_df.select(from_json(col("value"), schema).alias("data")).select("data.*")
    json_df = add_uuid(json_df)
    json_df.cache()

    query = json_df.writeStream \
        .trigger(processingTime='30 seconds') \
        .outputMode("append") \
        .foreachBatch(process_batch) \
        .option("checkpointLocation", "/tmp/spark/checkpoint") \
        .start()

    query.awaitTermination()

    if ti:
        ti.xcom_push(key='status', value='completed')
    print("Spark job has completed and notified Airflow.")

if __name__ == "__main__":
    run_spark_job()
