FROM --platform=linux/amd64 apache/spark:3.5.3

USER root

# 1) Cài wget, curl, python3 và các dependencies
RUN apt-get update && apt-get install -y wget curl python3 python3-pip && apt-get clean

# 2) Thiết lập JAVA_HOME và Spark environment (Apache Spark paths)
ENV JAVA_HOME=/opt/java/openjdk
ENV SPARK_HOME=/opt/spark
ENV PATH=$JAVA_HOME/bin:$SPARK_HOME/bin:$PATH
ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_DEFAULT_TIMEOUT=180 \
    PIP_NO_CACHE_DIR=1

# 3) Thêm thư viện JDBC PostgreSQL
RUN mkdir -p /opt/spark/jars \
 && wget https://jdbc.postgresql.org/download/postgresql-42.6.0.jar \
      -O /opt/spark/jars/postgresql-42.6.0.jar \
 && chmod 644 /opt/spark/jars/postgresql-42.6.0.jar

ENV CLASSPATH=/opt/spark/jars/postgresql-42.6.0.jar:$CLASSPATH

# 4) Copy và cài Python deps
WORKDIR /app
COPY requirements.txt /app/
RUN pip3 install --no-cache-dir --prefer-binary -r requirements.txt

# 5) Copy code của bạn
COPY spark_jobs/ /app/spark_jobs/

# 6) Tạo thư mục lưu model
RUN mkdir -p /app/models && chmod 777 /app/models

# 7) Thiết đặt MLflow URI và Spark config
ENV MLFLOW_TRACKING_URI=http://mlflow:5003
ENV SPARK_MODE=client
ENV SPARK_MASTER=spark://spark-master:7077

# 8) Verify Java installation
RUN java -version && echo "JAVA_HOME: $JAVA_HOME"

# 9) Chạy train ensemble model
CMD ["python3", "/app/spark_jobs/train_ensemble_model.py"]
