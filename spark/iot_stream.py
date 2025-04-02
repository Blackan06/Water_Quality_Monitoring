from pyspark.sql import SparkSession
from pyspark.sql.functions import col, from_json, expr, when
from pyspark.sql.types import StructType, StructField, FloatType, TimestampType
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression
from pyspark.ml import Pipeline
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.sql import functions as F

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
    assembler = VectorAssembler(inputCols=["temperature", "ph"], outputCol="features")
    lr = LinearRegression(labelCol="wqi", featuresCol="features")
    
    # Tạo pipeline
    pipeline = Pipeline(stages=[assembler, lr])
    return pipeline, lr

def cross_validate_model(training_data, pipeline , lr):
    """
    Sử dụng CrossValidator để tối ưu mô hình.
    """
    paramGrid = ParamGridBuilder() \
        .addGrid(lr.regParam, [0.1, 0.01]) \
        .addGrid(lr.maxIter, [5, 10]) \
        .build()

    evaluator = RegressionEvaluator(labelCol="wqi", predictionCol="prediction", metricName="rmse")
    
    crossval = CrossValidator(estimator=pipeline,
                              estimatorParamMaps=paramGrid,
                              evaluator=evaluator,
                              numFolds=3)  # Chạy 3 lần cross-validation
    
    cv_model = crossval.fit(training_data)
    return cv_model
def calculate_wqi(df):
    """
    Hàm tính Water Quality Index (WQI) dựa trên các giá trị pH và Temperature.
    Công thức tính WQI có thể thay đổi tùy theo yêu cầu cụ thể.
    """

    # Công thức giả định: WQI tính từ pH và Temperature
    # Giả sử pH tối ưu là trong phạm vi [6.5, 8.5] và Temperature tối ưu là trong phạm vi [20, 30]

    # Tính toán WQI dựa trên pH (một ví dụ cơ bản)
    pH_wqi = F.when((df['ph'] >= 6.5) & (df['ph'] <= 8.5), 100).otherwise(0)

    # Tính toán WQI dựa trên Temperature (một ví dụ cơ bản)
    temp_wqi = F.when((df['temperature'] >= 20) & (df['temperature'] <= 30), 100).otherwise(0)

    # Kết hợp các yếu tố để tính WQI tổng thể (có thể điều chỉnh tùy theo yêu cầu)
    wqi = pH_wqi * 0.6 + temp_wqi * 0.4  # Giả sử pH có trọng số 60% và Temperature có trọng số 40%

    return df.withColumn("wqi", wqi)
def train_model(batch_df):
    """
    Huấn luyện mô hình Linear Regression và sử dụng CrossValidation.
    """
    if batch_df.rdd.isEmpty():
        print("Batch rỗng, bỏ qua training.")
        return None
    batch_df.show(5, truncate=False)
    
    # Tiền xử lý dữ liệu
    training_df = batch_df.filter(col("temperature").isNotNull() & col("ph").isNotNull() & col("wqi").isNotNull())
    if training_df.rdd.isEmpty():
        print("Không có dữ liệu hợp lệ để training trong batch này.")
        return None

    # Xây dựng pipeline và thực hiện cross-validation
    pipeline, lr = build_pipeline()
    cv_model = cross_validate_model(training_df, pipeline, lr)
    
    print(f"Training completed with best model.")
    return cv_model.bestModel  

def forecast_model(model, df):
    """
    Sử dụng mô hình đã training để dự báo trên DataFrame đầu vào.
    Dữ liệu đầu vào cần có cột "temperature" và "ph_value".
    """
    if model is None:
        print("Không có mô hình, bỏ qua dự báo.")
        return

    # Đảm bảo rằng cột "features" không tồn tại trong DataFrame trước khi sử dụng VectorAssembler
    if "features" in df.columns:
        df = df.drop("features")  # Loại bỏ cột "features" nếu đã tồn tại

    
    # Dự báo
    predictions = model.transform(df)
    predictions.select("temperature", "ph", "wqi", "prediction").show(5, truncate=False)



def process_batch_with_session(spark):
    """
    Trả về hàm process_batch sử dụng biến spark từ ngoài để tạo broadcast variable.
    """
    def process_batch(batch_df, batch_id):
        print(f"Processing batch id: {batch_id}")
        batch_df_with_wqi = calculate_wqi(batch_df)
         
        # Huấn luyện mô hình và dự báo
        model = train_model(batch_df_with_wqi)
        forecast_model(model, batch_df_with_wqi)
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
        StructField("ph", FloatType(), True),
        StructField("temperature", FloatType(), True),
        StructField("wqi", FloatType(), True)  
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
