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
    &&  mv elasticsearch-spark-30_2.12-8.17.3.jar /opt/bitnami/spark/jars/

RUN wget https://jdbc.postgresql.org/download/postgresql-42.6.0.jar \
    && mv postgresql-42.6.0.jar /opt/bitnami/spark/jars/

# Thiết lập thư mục làm việc trong container
WORKDIR /app

# Copy requirements.txt vào container
COPY requirements.txt /app/requirements.txt


# Install Python dependencies từ requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Sao chép mã nguồn vào container
COPY ./spark/iot_stream.py /app/iot_stream.py

# Thiết lập môi trường Spark
ENV SPARK_HOME=/opt/bitnami/spark
ENV PATH=$SPARK_HOME/bin:$PATH
ENV SPARK_JARS_DIR=$SPARK_HOME/jars


# Lệnh để chạy Spark consumer app (tệp consumer.py hoặc tương tự)
CMD ["spark-submit", "--master", "local[*]", "--jars", "/opt/bitnami/spark/jars/*", "iot_stream.py"]
