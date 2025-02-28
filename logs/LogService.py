import logging
import psycopg2
from datetime import datetime
import os
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Hàm để ghi log vào cơ sở dữ liệu PostgreSQL
def write_log_to_postgres(level, log_message, logger_name):
    try:
        # Kết nối PostgreSQL
        db_name = os.getenv("DB_NAME")
        db_user = os.getenv("DB_USER")
        db_password = os.getenv("DB_PASSWORD")
        db_host = os.getenv("DB_HOST")
        db_port = os.getenv("DB_PORT")

        # Kết nối PostgreSQL
        connection = psycopg2.connect(
            database=db_name,
            user=db_user,
            password=db_password,
            host=db_host,
            port=db_port
        )
        cursor = connection.cursor()

        # Thực hiện truy vấn ghi log
        insert_query = """
        INSERT INTO logs (log_time, level, message, logger_name) 
        VALUES (%s, %s, %s, %s)
        """
        cursor.execute(insert_query, (datetime.now(), level, log_message, logger_name))

        # Commit và đóng kết nối
        connection.commit()
        cursor.close()
        connection.close()

    except Exception as e:
        logger.error(f"Error writing log to PostgreSQL: {e}")
