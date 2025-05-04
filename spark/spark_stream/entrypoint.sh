#!/bin/bash
# Xoá checkpoint cũ (nếu có)
echo "[entrypoint] Clearing checkpoint at ${CHECKPOINT_LOC:-/tmp/spark/checkpoint_predict}"
rm -rf "${CHECKPOINT_LOC:-/tmp/spark/checkpoint_predict}"

# Cuối cùng: chạy Spark
exec spark-submit \
  --master local[*] \
  --jars /opt/bitnami/spark/jars/spark-sql-kafka-0-10_2.12-3.2.0.jar,/opt/bitnami/spark/jars/kafka-clients-2.8.0.jar,/opt/bitnami/spark/jars/elasticsearch-spark-30_2.12-8.17.3.jar \
  /app/iot_stream.py
