from pyspark.sql import SparkSession
from pyspark.sql.functions import col, from_json, expr, when
from pyspark.sql.types import StructType, StructField, FloatType, TimestampType
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression
from pyspark.ml import Pipeline
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.evaluation import RegressionEvaluator

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

def build_pipeline():
    """
    Xây dựng một Pipeline gồm VectorAssembler và LinearRegression.
    """
    assembler = VectorAssembler(inputCols=["temperature", "ph_value"], outputCol="features")
    lr = LinearRegression(labelCol="WQI", featuresCol="features")
    
    # Tạo pipeline
    pipeline = Pipeline(stages=[assembler, lr])
    return pipeline

def cross_validate_model(training_data, pipeline):
    """
    Sử dụng CrossValidator để tối ưu mô hình.
    """
    paramGrid = ParamGridBuilder() \
        .addGrid(lr.regParam, [0.1, 0.01]) \
        .addGrid(lr.maxIter, [5, 10]) \
        .build()

    evaluator = RegressionEvaluator(labelCol="WQI", predictionCol="prediction", metricName="rmse")
    
    crossval = CrossValidator(estimator=pipeline,
                              estimatorParamMaps=paramGrid,
                              evaluator=evaluator,
                              numFolds=3)  # Chạy 3 lần cross-validation
    
    cv_model = crossval.fit(training_data)
    return cv_model

def train_model(batch_df):
    """
    Huấn luyện mô hình Linear Regression và sử dụng CrossValidation.
    """
    if batch_df.rdd.isEmpty():
        print("Batch rỗng, bỏ qua training.")
        return None
    batch_df.show(5, truncate=False)
    # Tiền xử lý dữ liệu
    training_df = batch_df.filter(col("temperature").isNotNull() & col("ph_value").isNotNull() & col("WQI").isNotNull())
    if training_df.rdd.isEmpty():
        print("Không có dữ liệu hợp lệ để training trong batch này.")
        return None

    # Xây dựng pipeline và thực hiện cross-validation
    pipeline = build_pipeline()
    cv_model = cross_validate_model(training_df, pipeline)
    
    print(f"Training completed with best model.")
    return cv_model.bestModel  # Trả về mô hình tốt nhất sau khi cross-validation

def forecast_model(model, df):
    """
    Sử dụng mô hình đã training để dự báo trên DataFrame đầu vào.
    Dữ liệu đầu vào cần có cột "temperature" và "ph_value".
    """
    if model is None:
        print("Không có mô hình, bỏ qua dự báo.")
        return

    assembler = VectorAssembler(inputCols=["temperature", "ph_value"], outputCol="features")
    forecast_data = assembler.transform(df)
    predictions = model.transform(forecast_data)
    predictions.select("temperature", "ph_value", "WQI", "prediction").show(5, truncate=False)

def process_batch_with_session(spark):
    """
    Trả về hàm process_batch sử dụng biến spark từ ngoài để tạo broadcast variable.
    """
    def process_batch(batch_df, batch_id):
        print(f"Processing batch id: {batch_id}")
                
        # Huấn luyện mô hình và dự báo
        model = train_model(batch_df)
        forecast_model(model, batch_df)
    return process_batch

def run_spark_job(**kwargs):
    """
    Hàm chính để chạy Spark Structured Streaming:
    - Đọc dữ liệu từ Kafka.
    - Parse dữ liệu JSON theo schema.
    - Thêm cột id.
    - Sử dụng foreachBatch với hàm process_batch đã được gắn spark session.
    """
    ti = kwargs.get('ti', None)
    data = ti.xcom_pull(task_ids='kafka_consumer_task', key='kafka_data') if ti else None
    print(f"Running Spark job with data: {data}")

    spark = get_spark_session()
    schema = StructType([
        StructField("measurement_time", TimestampType(), True),
        StructField("ph_value", FloatType(), True),
        StructField("temperature", FloatType(), True),
        StructField("WQI", FloatType(), True)  # Thêm cột WQI cho chất lượng nước
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

    query = json_df.writeStream \
        .trigger(processingTime='30 seconds') \
        .outputMode("append") \
        .foreachBatch(process_batch_with_session(spark)) \
        .option("checkpointLocation", "/tmp/spark/checkpoint") \
        .start()

    query.awaitTermination()

    if ti:
        ti.xcom_push(key='status', value='completed')
    print("Spark job has completed and notified Airflow.")

if __name__ == "__main__":
    run_spark_job()
