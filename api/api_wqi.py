from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer
from pydantic import BaseModel
from datetime import datetime, timedelta
import pyodbc
import requests
import google.auth
from google.auth.transport.requests import Request
from google.oauth2 import service_account
import json
from typing import Optional
import uuid
import firebase_admin
from firebase_admin import credentials, messaging
from passlib.context import CryptContext
import jwt  # Cần cài đặt thư viện PyJWT: pip install PyJWT

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],    
    allow_credentials=True,
    allow_methods=["*"],  
    allow_headers=["*"],   
)

# Khởi tạo pwd_context cho việc mã hóa mật khẩu
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# Cấu hình JWT
SECRET_KEY = "your_secret_key"  # Thay đổi bằng secret phù hợp với môi trường của bạn
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="login")

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

class DeviceToken(BaseModel):
    device_token: str
    user_id: int

class User(BaseModel):
    username: str
    password: str

class UserOut(BaseModel):
    username: str
    id: int

class Token(BaseModel):
    access_token: str
    token_type: str

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    """
    Tạo JWT access token với dữ liệu người dùng và thời gian hết hạn
    """
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

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
        # Lưu ý: Nên mã hóa password trước khi lưu
        hashed_password = pwd_context.hash(user.password)
        cursor.execute(
            "INSERT INTO accounts (username, password) OUTPUT inserted.id, inserted.username VALUES (?, ?)",
            (user.username, hashed_password)
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

@app.post("/login", response_model=Token)
def login_user(user: User):
    """
    API đăng nhập người dùng và trả về access token
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

        # Tạo JWT token
        access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
        access_token = create_access_token(
            data={"sub": user.username, "id": existing_user[0]},
            expires_delta=access_token_expires
        )
        return {"access_token": access_token, "token_type": "bearer"}
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

class NotificationRequest(BaseModel):
    device_token: str
    title: str
    message: str

SERVICE_ACCOUNT_FILE = '/app/api/firebase-adminsdk.json'
SCOPES = ['https://www.googleapis.com/auth/firebase.messaging']

def _get_access_token():
    """Lấy access token để xác thực gửi thông báo tới Firebase"""
    credentials = service_account.Credentials.from_service_account_file(
        SERVICE_ACCOUNT_FILE, scopes=SCOPES)
    request_obj = google.auth.transport.requests.Request()
    credentials.refresh(request_obj)
    return credentials.token

@app.post("/send-notification")
def send_notification(request: NotificationRequest):
    """
    Gửi thông báo đẩy tới thiết bị với device_token
    """
    try:
        access_token = _get_access_token()
        
        device_token_value = request.device_token
        title = request.title
        message = request.message

        url = "https://fcm.googleapis.com/v1/projects/watermonitoring-aaf32/messages:send"

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
