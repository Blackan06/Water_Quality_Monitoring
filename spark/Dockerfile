FROM bitnami/spark:latest

USER root

# 1) Cài wget, curl
RUN apt-get update && apt-get install -y wget curl

# 2) Thêm thư viện JDBC PostgreSQL
RUN mkdir -p /opt/bitnami/spark/jars \
 && wget https://jdbc.postgresql.org/download/postgresql-42.6.0.jar \
      -O /opt/bitnami/spark/jars/postgresql-42.6.0.jar \
 && chmod 644 /opt/bitnami/spark/jars/postgresql-42.6.0.jar

ENV CLASSPATH=/opt/bitnami/spark/jars/postgresql-42.6.0.jar:$CLASSPATH

# 3) Copy và cài Python deps (bao gồm mlflow[spark])
WORKDIR /app
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# 4) Copy code của bạn
COPY spark_jobs/ /app/spark_jobs/
COPY run_pipeline.py /app/

# 5) Tạo thư mục lưu model
RUN mkdir -p /app/models && chmod 777 /app/models

# 6) Thiết đặt MLflow URI và Spark config
ENV MLFLOW_TRACKING_URI=http://mlflow:5003
ENV SPARK_MODE=client
ENV SPARK_MASTER=spark://spark-master:7077

# 7) Chạy pipeline
CMD ["python3", "run_pipeline.py"]
