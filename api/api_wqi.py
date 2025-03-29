from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import psycopg2
import requests
import google.auth
from google.auth.transport.requests import Request
from google.oauth2 import service_account
import json
from typing import Optional
import uuid
from datetime import datetime
import firebase_admin
from firebase_admin import credentials, messaging
app = FastAPI()


# Khai báo cấu hình PostgreSQL
DB_CONFIG = {
    'dbname': 'device_tokens',
    'user': 'postgres',
    'password': 'postgres',
    'host': '149.28.145.56',
    'port': '5432'
}

# Cấu trúc để lưu trữ token của thiết bị
class DeviceToken(BaseModel):
    id: Optional[int] = None   
    device_token: str
# Kết nối đến PostgreSQL
def get_db_connection():
    conn = psycopg2.connect(**DB_CONFIG)
    return conn

@app.get("/")
def read_root():
    # Kết nối đến cơ sở dữ liệu
    conn = get_db_connection()

    try:
        # Tạo cursor để thực hiện câu truy vấn
        with conn.cursor() as cursor:
            # Truy vấn tất cả bản ghi trong bảng device_tokens
            cursor.execute('SELECT * FROM device_tokens')
            list_device = cursor.fetchall()  # Sử dụng fetchall để lấy tất cả bản ghi

        return {"device_tokens": list_device}

    except Exception as e:
        # Xử lý lỗi nếu có
        return {"error": str(e)}

    finally:
        # Đảm bảo rằng kết nối được đóng
        conn.close()

@app.post("/register-token")
def register_token(device_token: DeviceToken):
    """
    API để nhận và lưu device token
    """
    # Lấy device_token và device_id từ dữ liệu nhận vào
    device_token_value = device_token.device_token
    device_id_value = device_token.id

    conn = get_db_connection()
    cursor = conn.cursor()

    # Kiểm tra nếu device_id đã có trong cơ sở dữ liệu
    cursor.execute('SELECT * FROM device_tokens WHERE id = %s', (device_id_value,))
    existing_token = cursor.fetchone()

    if existing_token:
        # Nếu đã có device_id thì update device token và cập nhật `updated_at`
        cursor.execute('UPDATE device_tokens SET device_token = %s WHERE id = %s RETURNING *',
                       (device_token_value, device_id_value))
        updated_token = cursor.fetchone()
        conn.commit()
        cursor.close()
        conn.close()
        return {
            'message': 'Device token updated successfully',
            'data': {'id': updated_token[1], 'device_token': updated_token[2]}
        }
    else:
        # Nếu không có device_id (thêm mới)
        current_time = datetime.now()

        cursor.execute('INSERT INTO device_tokens (device_token, created_at, updated_at) VALUES (%s, %s, %s) RETURNING *',
                    (device_token_value, current_time, current_time))

        new_token = cursor.fetchone()
        conn.commit()
        cursor.close()
        conn.close()

        return {
            'message': 'Device token registered successfully',
            'data': {'id': new_token[1], 'device_token': new_token[2]}
        }
# Định nghĩa cấu trúc dữ liệu cho yêu cầu gửi thông báo
class NotificationRequest(BaseModel):
    device_token: str
    title: str
    message: str

# Đường dẫn đến tệp Service Account JSON
SERVICE_ACCOUNT_FILE = '/app/api/firebase-adminsdk.json'
SCOPES = ['https://www.googleapis.com/auth/firebase.messaging']

def _get_access_token():
    """Retrieve a valid access token that can be used to authorize requests.

    :return: Access token.
    """
    credentials = service_account.Credentials.from_service_account_file(
        SERVICE_ACCOUNT_FILE, scopes=SCOPES)
    request = google.auth.transport.requests.Request()
    credentials.refresh(request)
    return credentials.token

@app.post("/send-notification")
def send_notification(request: NotificationRequest):
    """
    Gửi thông báo đẩy tới thiết bị với device_token
    """
    try:
        # Lấy token truy cập mới
        access_token = _get_access_token()
        
        # Lấy thông tin từ request
        device_token = request.device_token
        title = request.title
        message = request.message

        # API endpoint FCM V1
        url = "https://fcm.googleapis.com/v1/projects/watermonitoring-aaf32/messages:send"

        # Dữ liệu thông báo
        message_data = {
            "message": {
                "token": device_token,
                "notification": {
                    "title": title,
                    "body": message
                }
            }
        }

        # Gửi yêu cầu POST tới FCM API
        headers = {
            "Authorization": f"Bearer {access_token}",
            "Content-Type": "application/json"
        }

        response = requests.post(url, headers=headers, json=message_data)

        # Kiểm tra kết quả
        if response.status_code == 200:
            return {"message": "Notification sent successfully", "result": response.json()}
        else:
            print(f"FCM Error: {response.text}")
            return {"message": "Failed to send notification", "error": response.text}
            
    except Exception as e:
        print(f"Exception: {str(e)}")
        return {"message": "Failed to send notification", "error": str(e)}