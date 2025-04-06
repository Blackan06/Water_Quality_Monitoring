# Sử dụng image Python chính thức từ Docker Hub
FROM python:3.9-slim

# Copy toàn bộ source code vào thư mục /app
COPY . /app

# Copy file firebase-adminsdk.json vào thư mục tương ứng
COPY api/firebase-adminsdk.json /app/api/firebase-adminsdk.json

# Thiết lập thư mục làm việc
WORKDIR /app

# Cài đặt các gói hệ thống cần thiết cho pyodbc và MS ODBC Driver cho SQL Server
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    unixodbc-dev \
    curl \
    gnupg2 \
    apt-transport-https && \
    curl https://packages.microsoft.com/keys/microsoft.asc | apt-key add - && \
    curl https://packages.microsoft.com/config/debian/10/prod.list > /etc/apt/sources.list.d/mssql-release.list && \
    apt-get update && ACCEPT_EULA=Y apt-get install -y msodbcsql17 && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Cài đặt các thư viện Python cần thiết
RUN pip install google-auth requests fastapi uvicorn pyodbc firebase-admin 'passlib[bcrypt]'

# Mở cổng mà ứng dụng sẽ lắng nghe
EXPOSE 8018

# Lệnh để chạy ứng dụng FastAPI
CMD ["uvicorn", "api.api_wqi:app", "--host", "0.0.0.0", "--port", "8018", "--reload"]
