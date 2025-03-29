from fastapi import FastAPI,HTTPException
from pydantic import BaseModel
import psycopg2
import requests
import google.auth
from google.auth.transport.requests import Request
from google.oauth2 import service_account
import json
from typing import Optional

app = FastAPI()


# Định nghĩa cấu trúc dữ liệu cho yêu cầu gửi thông báo
class NotificationRequest(BaseModel):
    device_token: str
    title: str
    message: str

# Khai báo cấu hình PostgreSQL
DB_CONFIG = {
    'dbname': 'wqi',
    'user': 'admin',
    'password': 'admin1234',
    'host': '149.28.145.56',
    'port': '5433'
}

# Đường dẫn đến tệp Service Account JSON của bạn
SERVICE_ACCOUNT_FILE = '/app/api/firebase-adminsdk.json'

# Xác thực và lấy token truy cập
credentials = service_account.Credentials.from_service_account_file(
    SERVICE_ACCOUNT_FILE,
    scopes=["https://www.googleapis.com/auth/firebase.messaging"]
)

# Refresh token nếu cần thiết
if credentials.expired and credentials.refresh_token:
    credentials.refresh(Request())

# Lấy token truy cập từ credentials
access_token = credentials.token

# Cấu trúc để lưu trữ token của thiết bị
class DeviceToken(BaseModel):
    id: Optional[int] = None  # id có thể có hoặc không
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
    Nếu đã có id thì cập nhật device token, nếu chưa thì thêm mới
    """
    # Lấy device_token từ dữ liệu nhận vào
    device_token_value = device_token.device_token
    id_value = device_token.id
    current_time = 'CURRENT_TIMESTAMP'  # Sử dụng thời gian hiện tại của DB

    conn = get_db_connection()
    cursor = conn.cursor()

    if id_value is not None:  # Kiểm tra nếu có id
        # Nếu có id thì tìm bản ghi theo id
        cursor.execute('SELECT * FROM device_tokens WHERE id = %s', (id_value,))
        existing_token = cursor.fetchone()

        if existing_token:
            # Nếu đã có id thì update device token và cập nhật `updated_at`
            cursor.execute('UPDATE device_tokens SET device_token = %s, updated_at = ' + current_time + ' WHERE id = %s RETURNING *',
                           (device_token_value, id_value))
            updated_token = cursor.fetchone()
            conn.commit()
            cursor.close()
            conn.close()
            return {
                'message': 'Device token updated successfully',
                'data': {'id': updated_token[0], 'device_token': updated_token[1]}
            }
        else:
            # Nếu id không tồn tại, thêm mới bản ghi
            cursor.execute('INSERT INTO device_tokens (device_token, created_at, updated_at) VALUES (%s, ' + current_time + ', ' + current_time + ') RETURNING *', 
                           (device_token_value,))
            new_token = cursor.fetchone()
            conn.commit()
            cursor.close()
            conn.close()

            return {
                'message': 'Device token registered successfully',
                'data': {'id': new_token[0], 'device_token': new_token[1]}
            }

    else:
        # Nếu không có id (thêm mới)
        cursor.execute('INSERT INTO device_tokens (device_token, created_at, updated_at) VALUES (%s, ' + current_time + ', ' + current_time + ') RETURNING *', 
                       (device_token_value,))
        new_token = cursor.fetchone()
        conn.commit()
        cursor.close()
        conn.close()

        return {
            'message': 'Device token registered successfully',
            'data': {'id': new_token[0], 'device_token': new_token[1]}
        }


@app.post("/send-notification")
def send_notification(request: NotificationRequest):
    """
    Gửi thông báo đẩy tới thiết bị với device_token
    """
    # Lấy thông tin từ request
    device_token = request.device_token
    title = request.title
    message = request.message

    # API endpoint FCM V1
    url = "https://fcm.googleapis.com/v1/projects/watermonitoring-aaf32/messages:send"  # Thay YOUR_PROJECT_ID bằng ID dự án Firebase của bạn

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

    response = requests.post(url, headers=headers, data=json.dumps(message_data))

    # Kiểm tra kết quả
    if response.status_code == 200:
        return {"message": "Notification sent successfully", "result": response.json()}
    else:
        return {"message": "Failed to send notification", "error": response.text}
