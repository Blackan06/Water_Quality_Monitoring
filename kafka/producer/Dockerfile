# Sử dụng Python base image phù hợp
FROM python:3.9-slim

WORKDIR /app

# Cài đặt các thư viện cần thiết

COPY requirements.txt /app/requirements.txt

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt


# Thêm mã nguồn Producer vào container
COPY ./kafka/producer/kafka_producer.py /app/kafka_producer.py

COPY ./logs /app/logs

# Thiết lập entrypoint để khởi chạy Producer
CMD ["python3", "kafka_producer.py"]
