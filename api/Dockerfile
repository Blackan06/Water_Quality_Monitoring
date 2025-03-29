# Sử dụng image Python chính thức từ Docker Hub
FROM python:3.9-slim

COPY . /app

COPY firebase-adminsdk.json /app/api/firebase-adminsdk.json

# Thiết lập thư mục làm việc
WORKDIR /app


# Cài đặt các thư viện cần thiết
RUN pip install google-auth requests fastapi uvicorn psycopg2-binary 

# Mở cổng mà ứng dụng sẽ lắng nghe
EXPOSE 8018

# Lệnh để chạy ứng dụng Flask (hoặc FastAPI)
CMD ["uvicorn", "api.api_wqi:app", "--host", "0.0.0.0", "--port", "8018", "--reload"]
