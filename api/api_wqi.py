from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel
import pyodbc
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
from passlib.context import CryptContext
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],    # Cho phép tất cả các domain
    allow_credentials=True, # Nếu cần gửi kèm credentials (cookie, auth headers,...)
    allow_methods=["*"],    # Cho phép tất cả các phương thức (GET, POST, PUT, DELETE, v.v.)
    allow_headers=["*"],    # Cho phép tất cả các header
)
# Khởi tạo pwd_context cho việc mã hóa mật khẩu
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

DB_CONFIG = {
    'server': 'SQL9001.site4now.net',
    'database': 'db_aaeae7_admin',
    'username': 'db_aaeae7_admin_admin',
    'password': 'Ak0901281010@'
}

def get_db_connection():
    conn_str = (
        "DRIVER={ODBC Driver 17 for SQL Server};"
        f"SERVER={DB_CONFIG['server']};"
        f"DATABASE={DB_CONFIG['database']};"
        f"UID={DB_CONFIG['username']};"
        f"PWD={DB_CONFIG['password']};"
    )
    conn = pyodbc.connect(conn_str)
    return conn

# Cấu trúc để lưu trữ token của thiết bị
class DeviceToken(BaseModel):
    device_token: str
    user_id: int

class User(BaseModel):
    username: str
    password: str

# Cấu trúc trả về khi đăng ký người dùng
class UserOut(BaseModel):
    username: str
    id: int

@app.post("/register")
def register_user(user: User):
    """
    API đăng ký tài khoản người dùng
    """
    conn = get_db_connection()
    cursor = conn.cursor()

    try:
        cursor.execute("SELECT * FROM accounts WHERE username = ?", (user.username,))
        existing_user = cursor.fetchone()

        if existing_user:
            raise HTTPException(status_code=400, detail="Username already exists")

        # Sử dụng OUTPUT để trả về id và username sau khi INSERT
        cursor.execute(
            "INSERT INTO accounts (username, password) OUTPUT inserted.id, inserted.username VALUES (?, ?)",
            (user.username, user.password)
        )
        new_user = cursor.fetchone()
        conn.commit()
        return {"message": "User created successfully", "user": UserOut(username=new_user[1], id=new_user[0])}
    except Exception as e:
        conn.rollback()
        raise HTTPException(status_code=500, detail=f"Error occurred: {str(e)}")
    finally:
        cursor.close()
        conn.close()

@app.post("/login")
def login_user(user: User):
    """
    API đăng nhập người dùng
    """
    conn = get_db_connection()
    cursor = conn.cursor()

    try:
        cursor.execute("SELECT * FROM accounts WHERE username = ?", (user.username,))
        existing_user = cursor.fetchone()

        if not existing_user:
            raise HTTPException(status_code=400, detail="Invalid username or password")

        stored_password = existing_user[2]  # Giả sử mật khẩu nằm ở cột thứ 3

        # Kiểm tra mật khẩu
        if not pwd_context.verify(user.password, stored_password):
            raise HTTPException(status_code=400, detail="Invalid username or password")

        return {"message": "Login successful", "accounts": {"username": existing_user[1], "id": existing_user[0]}}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error occurred: {str(e)}")
    finally:
        cursor.close()
        conn.close()

@app.get("/list_tokens")
def read_root():
    conn = get_db_connection()
    try:
        with conn.cursor() as cursor:
            cursor.execute("SELECT id, user_id, device_token, created_at, updated_at FROM device_tokens")
            rows = cursor.fetchall()
            
            list_device = [
                {
                    "id": row[0],
                    "user_id": row[1],
                    "device_token": row[2],
                    "created_at": row[3].isoformat() if row[3] else None,
                    "updated_at": row[4].isoformat() if row[4] else None
                } for row in rows
            ]

        return {"device_tokens": list_device}
    except Exception as e:
        return {"error": str(e)}
    finally:
        conn.close()

@app.post("/register-token")
def register_token(device_token: DeviceToken):
    """
    API để nhận và lưu device token
    """
    device_token_value = device_token.device_token
    device_user_id_value = device_token.user_id

    conn = get_db_connection()
    cursor = conn.cursor()

    try:
        cursor.execute("SELECT * FROM device_tokens WHERE user_id = ?", (device_user_id_value,))
        existing_token = cursor.fetchone()

        current_time = datetime.now()

        if existing_token:
            cursor.execute('''
                UPDATE device_tokens
                SET device_token = ?, updated_at = ?
                OUTPUT inserted.id, inserted.user_id, inserted.device_token, inserted.created_at, inserted.updated_at
                WHERE user_id = ?
            ''', (device_token_value, current_time, device_user_id_value))

            updated_token = cursor.fetchone()
            conn.commit()

            return {
                'message': 'Device token updated successfully',
                'data': {
                    'id': updated_token[0],
                    'user_id': updated_token[1],
                    'device_token': updated_token[2],
                    'created_at': updated_token[3].isoformat() if updated_token[3] else None,
                    'updated_at': updated_token[4].isoformat() if updated_token[4] else None
                }
            }
        else:
            cursor.execute('''
                INSERT INTO device_tokens (device_token, user_id, created_at, updated_at)
                OUTPUT inserted.id, inserted.user_id, inserted.device_token, inserted.created_at, inserted.updated_at
                VALUES (?, ?, ?, ?)
            ''', (device_token_value, device_user_id_value, current_time, current_time))

            new_token = cursor.fetchone()
            conn.commit()

            return {
                'message': 'Device token registered successfully',
                'data': {
                    'id': new_token[0],
                    'user_id': new_token[1],
                    'device_token': new_token[2],
                    'created_at': new_token[3].isoformat() if new_token[3] else None,
                    'updated_at': new_token[4].isoformat() if new_token[4] else None
                }
            }

    except Exception as e:
        conn.rollback()
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")
    finally:
        cursor.close()
        conn.close()

# Định nghĩa cấu trúc dữ liệu cho yêu cầu gửi thông báo
class NotificationRequest(BaseModel):
    device_token: str
    title: str
    message: str

# Đường dẫn đến tệp Service Account JSON của Firebase
SERVICE_ACCOUNT_FILE = '/app/api/firebase-adminsdk.json'
SCOPES = ['https://www.googleapis.com/auth/firebase.messaging']

def _get_access_token():
    """Lấy access token để xác thực gửi thông báo tới Firebase"""
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
        # Lấy access token mới
        access_token = _get_access_token()
        
        device_token_value = request.device_token
        title = request.title
        message = request.message

        # Đường dẫn API của FCM V1
        url = "https://fcm.googleapis.com/v1/projects/watermonitoring-aaf32/messages:send"

        # Dữ liệu thông báo
        message_data = {
            "message": {
                "token": device_token_value,
                "notification": {
                    "title": title,
                    "body": message
                }
            }
        }

        headers = {
            "Authorization": f"Bearer {access_token}",
            "Content-Type": "application/json"
        }

        response = requests.post(url, headers=headers, json=message_data)

        if response.status_code == 200:
            return {"message": "Notification sent successfully", "result": response.json()}
        else:
            print(f"FCM Error: {response.text}")
            return {"message": "Failed to send notification", "error": response.text}
            
    except Exception as e:
        print(f"Exception: {str(e)}")
        return {"message": "Failed to send notification", "error": str(e)}
