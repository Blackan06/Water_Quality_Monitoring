# Dockerfile (không đổi gì thêm)
FROM bitnami/spark:3.5.5-debian-12-r3
USER root

RUN apt-get update && apt-get install -y \
      python3-pip python3-dev build-essential libpq-dev curl \
    && rm -rf /var/lib/apt/lists/*

COPY ./spark/spark_batch/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

RUN curl -sSL \
      https://jdbc.postgresql.org/download/postgresql-42.6.0.jar \
      -o /opt/bitnami/spark/jars/postgresql-42.6.0.jar && \
    curl -sSL \
      https://repo1.maven.org/maven2/ml/dmlc/xgboost4j_2.12/1.6.2/xgboost4j_2.12-1.6.2.jar \
      -o /opt/bitnami/spark/jars/xgboost4j_2.12-1.6.2.jar && \
    curl -sSL \
      https://repo1.maven.org/maven2/ml/dmlc/xgboost4j-spark_2.12/1.6.2/xgboost4j-spark_2.12-1.6.2.jar \
      -o /opt/bitnami/spark/jars/xgboost4j-spark_2.12-1.6.2.jar

COPY ./spark/spark_batch/spark_job.py /app/spark_job.py
COPY ./data/WQI_data.csv /app/data/WQI_data.csv

WORKDIR /app
RUN mkdir -p models && chown 1001:1001 models
USER 1001

CMD ["spark-submit", \
     "--master","local[1]", \
     "--conf","spark.driver.memory=1g", \
     "--conf","spark.executor.memory=1g", \
     "--conf","spark.memory.fraction=0.6", \
     "--conf","spark.jars=/opt/bitnami/spark/jars/postgresql-42.6.0.jar,/opt/bitnami/spark/jars/xgboost4j_2.12-1.6.2.jar,/opt/bitnami/spark/jars/xgboost4j-spark_2.12-1.6.2.jar", \
     "spark_job.py"]
