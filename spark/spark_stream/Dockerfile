# Sử dụng image PySpark chính thức làm base image
FROM bitnami/spark:3.2.0

USER root

RUN apt-get update && \
    apt-get install -y wget curl && \
    apt-get clean

# Cài đặt các thư viện Python cần thiết (pip và pandas)
RUN pip install --upgrade pip 

# Download necessary Spark dependencies (Kafka, PostgreSQL, etc.)
RUN wget https://repo1.maven.org/maven2/org/apache/spark/spark-sql-kafka-0-10_2.12/3.2.0/spark-sql-kafka-0-10_2.12-3.2.0.jar \
    && wget https://repo1.maven.org/maven2/org/apache/spark/spark-streaming-kafka-0-10_2.12/3.2.0/spark-streaming-kafka-0-10_2.12-3.2.0.jar \
    && wget https://repo1.maven.org/maven2/org/apache/spark/spark-token-provider-kafka-0-10_2.12/3.2.0/spark-token-provider-kafka-0-10_2.12-3.2.0.jar \
    && wget https://repo1.maven.org/maven2/org/apache/kafka/kafka-clients/2.8.0/kafka-clients-2.8.0.jar \
    && wget https://repo1.maven.org/maven2/org/apache/commons/commons-pool2/2.6.2/commons-pool2-2.6.2.jar \
    && wget https://repo1.maven.org/maven2/org/elasticsearch/elasticsearch-spark-30_2.12/8.17.3/elasticsearch-spark-30_2.12-8.17.3.jar \
    && mv spark-sql-kafka-0-10_2.12-3.2.0.jar /opt/bitnami/spark/jars/ \
    && mv spark-streaming-kafka-0-10_2.12-3.2.0.jar /opt/bitnami/spark/jars/ \
    && mv spark-token-provider-kafka-0-10_2.12-3.2.0.jar /opt/bitnami/spark/jars/ \
    && mv kafka-clients-2.8.0.jar /opt/bitnami/spark/jars/ \
    && mv commons-pool2-2.6.2.jar /opt/bitnami/spark/jars/ \
    && mv elasticsearch-spark-30_2.12-8.17.3.jar /opt/bitnami/spark/jars/

RUN wget https://jdbc.postgresql.org/download/postgresql-42.6.0.jar \
    && mv postgresql-42.6.0.jar /opt/bitnami/spark/jars/

# Thiết lập thư mục làm việc trong container
WORKDIR /app

# Tạo các thư mục cần thiết
COPY ./models /app/models

# Copy requirements.txt vào container
COPY ./spark/spark_stream/requirements.txt /app/requirements.txt

# Install Python dependencies từ requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Sao chép mã nguồn vào container
COPY spark/spark_stream/iot_stream.py /app/iot_stream.py

# Thiết lập môi trường Spark
ENV SPARK_HOME=/opt/bitnami/spark
ENV PATH=$SPARK_HOME/bin:$PATH
ENV SPARK_JARS_DIR=$SPARK_HOME/jars
ENV MLFLOW_TRACKING_URI=http://mlflow:5003
ENV MLFLOW_ARTIFACT_ROOT=file:/app/mlartifacts
ENV OUTPUT_MODEL_DIR=/app/models
ENV REGISTERED_MODEL_NAME=water_quality_xgb
ENV ES_PORT=80
ENV ES_INDEX=water_quality
ENV MODEL_PATH=/app/models/best_xgb_pipeline
# Lệnh để chạy Spark consumer app
COPY ./spark/spark_stream/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh
ENTRYPOINT ["/entrypoint.sh"]