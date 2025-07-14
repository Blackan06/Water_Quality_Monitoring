import logging
import psycopg2
from psycopg2 import sql
from datetime import datetime
import json
import os
import pandas as pd

logger = logging.getLogger(__name__)

class DatabaseManager:
    def __init__(self):
        self.db_config = {
            'host': os.getenv('DB_HOST', '194.238.16.14'),
            'port': os.getenv('DB_PORT', '5432'),
            'database': os.getenv('DB_NAME', 'wqi_db'),
            'user': os.getenv('DB_USER', 'postgres'),
            'password': os.getenv('DB_PASSWORD', 'postgres1234')
        }
        self.init_database()

    def get_connection(self):
        """Tạo kết nối đến database"""
        try:
            conn = psycopg2.connect(**self.db_config)
            return conn
        except Exception as e:
            logger.error(f"Error connecting to database: {e}")
            return None

    def init_database(self):
        """Khởi tạo các bảng cần thiết"""
        try:
            conn = self.get_connection()
            if not conn:
                return
            
            cur = conn.cursor()
            
            # Bảng lưu thông tin trạm quan trắc (station_id là int)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS monitoring_stations (
                    station_id INTEGER PRIMARY KEY,
                    station_name VARCHAR(255) NOT NULL,
                    location VARCHAR(500),
                    latitude DECIMAL(10, 8),
                    longitude DECIMAL(11, 8),
                    description TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    is_active BOOLEAN DEFAULT TRUE
                )
            """)
            
            # Bảng lưu dữ liệu thô từ sensors (WQI sẽ được tính sau)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS raw_sensor_data (
                    id SERIAL PRIMARY KEY,
                    station_id INTEGER NOT NULL,
                    measurement_time TIMESTAMP NOT NULL,
                    ph DECIMAL(5, 2),
                    temperature DECIMAL(5, 2),
                    "do" DECIMAL(5, 2),
                    wqi DECIMAL(6, 2) NULL,
                    is_processed BOOLEAN DEFAULT FALSE,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (station_id) REFERENCES monitoring_stations(station_id)
                )
            """)
            
            # Thêm cột is_processed nếu chưa tồn tại
            try:
                cur.execute("""
                    ALTER TABLE raw_sensor_data 
                    ADD COLUMN IF NOT EXISTS is_processed BOOLEAN DEFAULT FALSE
                """)
                logger.info("Added is_processed column to raw_sensor_data table")
            except Exception as e:
                logger.debug(f"is_processed column may already exist: {e}")
            
            # Bảng lưu dữ liệu đã xử lý và WQI
            cur.execute("""
                CREATE TABLE IF NOT EXISTS processed_water_quality_data (
                    id SERIAL PRIMARY KEY,
                    station_id INTEGER NOT NULL,
                    measurement_time TIMESTAMP NOT NULL,
                    ph DECIMAL(5, 2),
                    temperature DECIMAL(5, 2),
                    "do" DECIMAL(5, 2),
                    wqi DECIMAL(6, 2),
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (station_id) REFERENCES monitoring_stations(station_id),
                    UNIQUE(station_id, measurement_time)
                )
            """)
            
            # Thêm unique constraint cho processed_water_quality_data nếu chưa tồn tại
            try:
                cur.execute("""
                    ALTER TABLE processed_water_quality_data 
                    ADD CONSTRAINT processed_water_quality_data_station_time_unique 
                    UNIQUE (station_id, measurement_time)
                """)
                logger.info("Added unique constraint to processed_water_quality_data table")
            except Exception as e:
                # Constraint đã tồn tại hoặc có lỗi khác
                logger.debug(f"Unique constraint may already exist: {e}")
            
            # Bảng lưu kết quả dự đoán
            cur.execute("""
                CREATE TABLE IF NOT EXISTS prediction_results (
                    id SERIAL PRIMARY KEY,
                    station_id INTEGER NOT NULL,
                    prediction_time TIMESTAMP NOT NULL,
                    model_type VARCHAR(50) NOT NULL, -- 'xgboost' or 'lstm'
                    wqi_prediction DECIMAL(6, 2),
                    confidence_score DECIMAL(5, 4),
                    processing_time DECIMAL(8, 3),
                    model_version VARCHAR(100),
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (station_id) REFERENCES monitoring_stations(station_id)
                )
            """)
            
            # Bảng lưu thông tin model
            cur.execute("""
                CREATE TABLE IF NOT EXISTS model_registry (
                    id SERIAL PRIMARY KEY,
                    station_id INTEGER NOT NULL,
                    model_type VARCHAR(50) NOT NULL, -- 'xgboost' or 'lstm'
                    model_version VARCHAR(100) NOT NULL,
                    model_path VARCHAR(500),
                    accuracy DECIMAL(5, 4),
                    mae DECIMAL(6, 4),
                    r2_score DECIMAL(5, 4),
                    training_date TIMESTAMP,
                    is_active BOOLEAN DEFAULT TRUE,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (station_id) REFERENCES monitoring_stations(station_id)
                )
            """)
            
            # Bảng lưu lịch sử training
            cur.execute("""
                CREATE TABLE IF NOT EXISTS training_history (
                    id SERIAL PRIMARY KEY,
                    station_id INTEGER NOT NULL,
                    model_type VARCHAR(50) NOT NULL,
                    training_start TIMESTAMP,
                    training_end TIMESTAMP,
                    training_duration DECIMAL(8, 3),
                    records_used INTEGER,
                    accuracy DECIMAL(5, 4),
                    mae DECIMAL(6, 4),
                    r2_score DECIMAL(5, 4),
                    status VARCHAR(50), -- 'success', 'failed', 'in_progress'
                    error_message TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (station_id) REFERENCES monitoring_stations(station_id)
                )
            """)
            
            # Bảng lưu so sánh model performance
            cur.execute("""
                CREATE TABLE IF NOT EXISTS model_comparison (
                    id SERIAL PRIMARY KEY,
                    station_id INTEGER NOT NULL,
                    comparison_date TIMESTAMP NOT NULL,
                    xgboost_accuracy DECIMAL(5, 4),
                    lstm_accuracy DECIMAL(5, 4),
                    xgboost_processing_time DECIMAL(8, 3),
                    lstm_processing_time DECIMAL(8, 3),
                    xgboost_wqi_prediction DECIMAL(6, 2),
                    lstm_wqi_prediction DECIMAL(6, 2),
                    best_model VARCHAR(50), -- 'xgboost' or 'lstm'
                    accuracy_improvement DECIMAL(6, 4),
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (station_id) REFERENCES monitoring_stations(station_id)
                )
            """)
            
            # Bảng lưu alerts và notifications
            cur.execute("""
                CREATE TABLE IF NOT EXISTS alerts (
                    id SERIAL PRIMARY KEY,
                    station_id INTEGER NOT NULL,
                    alert_type VARCHAR(100) NOT NULL,
                    severity VARCHAR(20) NOT NULL, -- 'low', 'medium', 'high', 'critical'
                    message TEXT NOT NULL,
                    wqi_value DECIMAL(6, 2),
                    threshold_value DECIMAL(6, 2),
                    is_resolved BOOLEAN DEFAULT FALSE,
                    resolved_at TIMESTAMP,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (station_id) REFERENCES monitoring_stations(station_id)
                )
            """)
            
            # Bảng lưu dữ liệu lịch sử từ WQI_data.csv
            cur.execute("""
                CREATE TABLE IF NOT EXISTS historical_wqi_data (
                    id SERIAL PRIMARY KEY,
                    station_id INTEGER NOT NULL,
                    measurement_date DATE NOT NULL,
                    temperature DECIMAL(5, 2),
                    ph DECIMAL(5, 2),
                    "do" DECIMAL(5, 2),
                    wqi DECIMAL(6, 2),
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (station_id) REFERENCES monitoring_stations(station_id),
                    UNIQUE(station_id, measurement_date)
                )
            """)
            
            # Thêm unique constraint nếu chưa tồn tại (cho trường hợp table đã có sẵn)
            try:
                cur.execute("""
                    ALTER TABLE historical_wqi_data 
                    ADD CONSTRAINT historical_wqi_data_station_date_unique 
                    UNIQUE (station_id, measurement_date)
                """)
                logger.info("Added unique constraint to historical_wqi_data table")
            except Exception as e:
                # Constraint đã tồn tại hoặc có lỗi khác
                logger.debug(f"Unique constraint may already exist: {e}")
            
            conn.commit()
            cur.close()
            conn.close()
            
            logger.info("Database tables initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing database: {e}")

    def insert_station(self, station_data):
        """Thêm trạm quan trắc mới"""
        try:
            conn = self.get_connection()
            if not conn:
                return False
            
            cur = conn.cursor()
            
            insert_query = sql.SQL("""
                INSERT INTO monitoring_stations (station_id, station_name, location, latitude, longitude, description)
                VALUES (%s, %s, %s, %s, %s, %s)
                ON CONFLICT (station_id) 
                DO UPDATE SET 
                    station_name = EXCLUDED.station_name,
                    location = EXCLUDED.location,
                    latitude = EXCLUDED.latitude,
                    longitude = EXCLUDED.longitude,
                    description = EXCLUDED.description,
                    updated_at = CURRENT_TIMESTAMP
            """)
            
            cur.execute(insert_query, (
                station_data['station_id'],
                station_data['station_name'],
                station_data.get('location', ''),
                station_data.get('latitude', 0),
                station_data.get('longitude', 0),
                station_data.get('description', '')
            ))
            
            conn.commit()
            cur.close()
            conn.close()
            
            logger.info(f"Station {station_data['station_id']} inserted/updated successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error inserting station: {e}")
            return False

    def insert_historical_data(self, historical_data):
        """Lưu dữ liệu lịch sử từ WQI_data.csv"""
        try:
            conn = self.get_connection()
            if not conn:
                return False
            
            cur = conn.cursor()
            
            insert_query = sql.SQL("""
                INSERT INTO historical_wqi_data (station_id, measurement_date, temperature, ph, "do", wqi)
                VALUES (%s, %s, %s, %s, %s, %s)
                ON CONFLICT (station_id, measurement_date) 
                DO UPDATE SET 
                    temperature = EXCLUDED.temperature,
                    ph = EXCLUDED.ph,
                    "do" = EXCLUDED."do",
                    wqi = EXCLUDED.wqi
            """)
            
            # Log the data being inserted for debugging
            logger.debug(f"Inserting historical data: {historical_data}")
            
            cur.execute(insert_query, (
                historical_data['station_id'],
                historical_data['measurement_date'],
                historical_data.get('temperature', 0),
                historical_data.get('ph', 0),
                historical_data.get('do', 0),
                historical_data.get('wqi', 0)
            ))
            
            conn.commit()
            cur.close()
            conn.close()
            
            logger.info(f"Historical data inserted for station {historical_data['station_id']}")
            return True
            
        except Exception as e:
            logger.error(f"Error inserting historical data: {e}")
            logger.error(f"Data that failed to insert: {historical_data}")
            return False

    def insert_raw_data(self, raw_data):
        """Lưu dữ liệu thô từ sensors (không bao gồm WQI - WQI sẽ được tính sau)"""
        try:
            conn = self.get_connection()
            if not conn:
                return False
            
            cur = conn.cursor()
            
            insert_query = sql.SQL("""
                INSERT INTO raw_sensor_data 
                (station_id, measurement_time, ph, temperature, "do")
                VALUES (%s, %s, %s, %s, %s)
            """)
            
            cur.execute(insert_query, (
                raw_data['station_id'],
                raw_data['measurement_time'],
                raw_data.get('ph', 0),
                raw_data.get('temperature', 0),
                raw_data.get('do', 0)
            ))
            
            conn.commit()
            cur.close()
            conn.close()
            
            logger.info(f"Raw data inserted for station {raw_data['station_id']}")
            return True
            
        except Exception as e:
            logger.error(f"Error inserting raw data: {e}")
            return False

    def insert_monitoring_data(self, monitoring_data):
        """Lưu dữ liệu monitoring từ streaming (alias cho insert_raw_data để tương thích ngược)"""
        # Chuyển đổi format từ wq_date sang measurement_time nếu cần
        if 'wq_date' in monitoring_data:
            monitoring_data['measurement_time'] = monitoring_data.pop('wq_date')
        
        return self.insert_raw_data(monitoring_data)

    def get_monitoring_data(self, station_id, limit=1000):
        """Lấy dữ liệu monitoring từ raw_sensor_data (thay thế cho wqi_monitoring_data)"""
        try:
            conn = self.get_connection()
            if not conn:
                return []
            
            cur = conn.cursor()
            
            cur.execute("""
                SELECT measurement_time, ph, temperature, "do", wqi
                FROM raw_sensor_data 
                WHERE station_id = %s 
                ORDER BY measurement_time DESC 
                LIMIT %s
            """, (station_id, limit))
            
            data = cur.fetchall()
            
            cur.close()
            conn.close()
            
            return data
            
        except Exception as e:
            logger.error(f"Error getting monitoring data: {e}")
            return []

    def insert_processed_data(self, processed_data):
        """Lưu dữ liệu đã xử lý và WQI"""
        try:
            conn = self.get_connection()
            if not conn:
                return False
            
            cur = conn.cursor()
            
            insert_query = sql.SQL("""
                INSERT INTO processed_water_quality_data 
                (station_id, measurement_time, ph, temperature, "do", wqi)
                VALUES (%s, %s, %s, %s, %s, %s)
            """)
            
            cur.execute(insert_query, (
                processed_data['station_id'],
                processed_data['measurement_time'],
                processed_data.get('ph', 0),
                processed_data.get('temperature', 0),
                processed_data.get('do', 0),
                processed_data.get('wqi', 0)
            ))
            
            conn.commit()
            cur.close()
            conn.close()
            
            logger.info(f"Processed data inserted for station {processed_data['station_id']}")
            return True
            
        except Exception as e:
            logger.error(f"Error inserting processed data: {e}")
            return False

    def insert_prediction_result(self, prediction_data):
        """Lưu kết quả dự đoán"""
        try:
            conn = self.get_connection()
            if not conn:
                return False
            
            cur = conn.cursor()
            
            insert_query = sql.SQL("""
                INSERT INTO prediction_results 
                (station_id, prediction_time, model_type, wqi_prediction, confidence_score, processing_time, model_version)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
            """)
            
            cur.execute(insert_query, (
                prediction_data['station_id'],
                prediction_data['prediction_time'],
                prediction_data['model_type'],
                prediction_data.get('wqi_prediction', 0),
                prediction_data.get('confidence_score', 0),
                prediction_data.get('processing_time', 0),
                prediction_data.get('model_version', 'unknown')
            ))
            
            conn.commit()
            cur.close()
            conn.close()
            
            logger.info(f"Prediction result inserted for station {prediction_data['station_id']}")
            return True
            
        except Exception as e:
            logger.error(f"Error inserting prediction result: {e}")
            return False

    def check_station_exists(self, station_id):
        """Kiểm tra trạm đã tồn tại chưa"""
        try:
            conn = self.get_connection()
            if not conn:
                return False
            
            cur = conn.cursor()
            
            cur.execute("""
                SELECT COUNT(*) FROM monitoring_stations 
                WHERE station_id = %s AND is_active = TRUE
            """, (station_id,))
            
            count = cur.fetchone()[0]
            
            cur.close()
            conn.close()
            
            return count > 0
            
        except Exception as e:
            logger.error(f"Error checking station existence: {e}")
            return False

    def get_station_info(self, station_id):
        """Lấy thông tin trạm"""
        try:
            conn = self.get_connection()
            if not conn:
                return None
            
            cur = conn.cursor()
            
            cur.execute("""
                SELECT station_id, station_name, location, latitude, longitude, description
                FROM monitoring_stations 
                WHERE station_id = %s AND is_active = TRUE
            """, (station_id,))
            
            result = cur.fetchone()
            
            cur.close()
            conn.close()
            
            if result:
                return {
                    'station_id': result[0],
                    'station_name': result[1],
                    'location': result[2],
                    'latitude': result[3],
                    'longitude': result[4],
                    'description': result[5]
                }
            
            return None
            
        except Exception as e:
            logger.error(f"Error getting station info: {e}")
            return None

    def get_station_data(self, station_id, limit=1000):
        """Lấy dữ liệu của một trạm"""
        try:
            conn = self.get_connection()
            if not conn:
                return []
            
            cur = conn.cursor()
            
            cur.execute("""
                SELECT measurement_time, ph, temperature, "do", wqi
                FROM processed_water_quality_data 
                WHERE station_id = %s 
                ORDER BY measurement_time DESC 
                LIMIT %s
            """, (station_id, limit))
            
            data = cur.fetchall()
            
            cur.close()
            conn.close()
            
            return data
            
        except Exception as e:
            logger.error(f"Error getting station data: {e}")
            return []

    def get_historical_data(self, station_id, limit=1000):
        """Lấy dữ liệu lịch sử của một trạm"""
        try:
            conn = self.get_connection()
            if not conn:
                return pd.DataFrame()
            
            cur = conn.cursor()
            
            cur.execute("""
                SELECT measurement_date, temperature, ph, "do", wqi
                FROM historical_wqi_data 
                WHERE station_id = %s 
                ORDER BY measurement_date DESC 
                LIMIT %s
            """, (station_id, limit))
            
            data = cur.fetchall()
            
            cur.close()
            conn.close()
            
            if not data:
                logger.warning(f"No historical data found for station {station_id}")
                return pd.DataFrame()
            
            # Chuyển thành DataFrame với tên cột đúng
            df = pd.DataFrame(data, columns=['measurement_date', 'temperature', 'ph', 'do', 'wqi'])
            
            # Thêm station_id column
            df['station_id'] = station_id
            
            logger.info(f"Retrieved {len(df)} historical records for station {station_id}")
            return df
            
        except Exception as e:
            logger.error(f"Error getting historical data: {e}")
            return pd.DataFrame()

    def get_latest_prediction(self, station_id, model_type='xgboost'):
        """Lấy dự đoán mới nhất của một trạm"""
        try:
            conn = self.get_connection()
            if not conn:
                return None
            
            cur = conn.cursor()
            
            cur.execute("""
                SELECT prediction_time, wqi_prediction, confidence_score, processing_time, model_version
                FROM prediction_results 
                WHERE station_id = %s AND model_type = %s
                ORDER BY prediction_time DESC 
                LIMIT 1
            """, (station_id, model_type))
            
            result = cur.fetchone()
            
            cur.close()
            conn.close()
            
            if result:
                return {
                    'prediction_time': result[0],
                    'wqi_prediction': result[1],
                    'confidence_score': result[2],
                    'processing_time': result[3],
                    'model_version': result[4]
                }
            
            return None
            
        except Exception as e:
            logger.error(f"Error getting latest prediction: {e}")
            return None

    def get_all_stations(self):
        """Lấy danh sách tất cả các trạm"""
        try:
            conn = self.get_connection()
            if not conn:
                return []
            
            cur = conn.cursor()
            
            cur.execute("""
                SELECT station_id, station_name, location, latitude, longitude, description, is_active
                FROM monitoring_stations 
                ORDER BY station_id
            """)
            
            results = cur.fetchall()
            
            cur.close()
            conn.close()
            
            stations = []
            for result in results:
                stations.append({
                    'station_id': result[0],
                    'station_name': result[1],
                    'location': result[2],
                    'latitude': result[3],
                    'longitude': result[4],
                    'description': result[5],
                    'is_active': result[6]
                })
            
            return stations
            
        except Exception as e:
            logger.error(f"Error getting all stations: {e}")
            return []

    def insert_model_registry(self, model_data):
        """Lưu thông tin model vào registry"""
        try:
            conn = self.get_connection()
            if not conn:
                return False
            
            cur = conn.cursor()
            
            # Deactivate old models for this station and type
            cur.execute("""
                UPDATE model_registry 
                SET is_active = FALSE 
                WHERE station_id = %s AND model_type = %s
            """, (model_data['station_id'], model_data['model_type']))
            
            # Insert new model
            insert_query = sql.SQL("""
                INSERT INTO model_registry 
                (station_id, model_type, model_version, model_path, accuracy, mae, r2_score, training_date)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            """)
            
            cur.execute(insert_query, (
                model_data['station_id'],
                model_data['model_type'],
                model_data['model_version'],
                model_data.get('model_path', ''),
                model_data.get('accuracy', 0),
                model_data.get('mae', 0),
                model_data.get('r2_score', 0),
                model_data.get('training_date', datetime.now().isoformat())
            ))
            
            conn.commit()
            cur.close()
            conn.close()
            
            logger.info(f"Model registry updated for station {model_data['station_id']}")
            return True
            
        except Exception as e:
            logger.error(f"Error inserting model registry: {e}")
            return False

    def insert_training_history(self, training_data):
        """Lưu lịch sử training"""
        try:
            conn = self.get_connection()
            if not conn:
                return False
            
            cur = conn.cursor()
            
            insert_query = sql.SQL("""
                INSERT INTO training_history 
                (station_id, model_type, training_start, training_end, training_duration, 
                 records_used, accuracy, mae, r2_score, status, error_message)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """)
            
            cur.execute(insert_query, (
                training_data['station_id'],
                training_data['model_type'],
                training_data.get('training_start'),
                training_data.get('training_end'),
                training_data.get('training_duration', 0),
                training_data.get('records_used', 0),
                training_data.get('accuracy', 0),
                training_data.get('mae', 0),
                training_data.get('r2_score', 0),
                training_data.get('status', 'unknown'),
                training_data.get('error_message', '')
            ))
            
            conn.commit()
            cur.close()
            conn.close()
            
            logger.info(f"Training history inserted for station {training_data['station_id']}")
            return True
            
        except Exception as e:
            logger.error(f"Error inserting training history: {e}")
            return False

    def insert_model_comparison(self, comparison_data):
        """Lưu kết quả so sánh model"""
        try:
            conn = self.get_connection()
            if not conn:
                return False
            
            cur = conn.cursor()
            
            insert_query = sql.SQL("""
                INSERT INTO model_comparison 
                (station_id, comparison_date, xgboost_accuracy, lstm_accuracy, 
                 xgboost_processing_time, lstm_processing_time, xgboost_wqi_prediction, 
                 lstm_wqi_prediction, best_model, accuracy_improvement)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """)
            
            cur.execute(insert_query, (
                comparison_data['station_id'],
                comparison_data.get('comparison_date', datetime.now().isoformat()),
                comparison_data.get('xgboost_accuracy', 0),
                comparison_data.get('lstm_accuracy', 0),
                comparison_data.get('xgboost_processing_time', 0),
                comparison_data.get('lstm_processing_time', 0),
                comparison_data.get('xgboost_wqi_prediction', 0),
                comparison_data.get('lstm_wqi_prediction', 0),
                comparison_data.get('best_model', 'unknown'),
                comparison_data.get('accuracy_improvement', 0)
            ))
            
            conn.commit()
            cur.close()
            conn.close()
            
            logger.info(f"Model comparison inserted for station {comparison_data['station_id']}")
            return True
            
        except Exception as e:
            logger.error(f"Error inserting model comparison: {e}")
            return False

    def insert_alert(self, alert_data):
        """Lưu alert"""
        try:
            conn = self.get_connection()
            if not conn:
                return False
            
            cur = conn.cursor()
            
            insert_query = sql.SQL("""
                INSERT INTO alerts 
                (station_id, alert_type, severity, message, wqi_value, threshold_value)
                VALUES (%s, %s, %s, %s, %s, %s)
            """)
            
            cur.execute(insert_query, (
                alert_data['station_id'],
                alert_data['alert_type'],
                alert_data['severity'],
                alert_data['message'],
                alert_data.get('wqi_value', 0),
                alert_data.get('threshold_value', 0)
            ))
            
            conn.commit()
            cur.close()
            conn.close()
            
            logger.info(f"Alert inserted for station {alert_data['station_id']}")
            return True
            
        except Exception as e:
            logger.error(f"Error inserting alert: {e}")
            return False

    def load_wqi_data_to_db(self):
        """Load dữ liệu từ WQI_data.csv vào database"""
        try:
            import pandas as pd
            
            # Đường dẫn đến file WQI_data.csv
            csv_path = 'data/WQI_data.csv'
            if not os.path.exists(csv_path):
                logger.warning(f"WQI_data.csv not found at {csv_path}")
                return False
            
            # Đọc dữ liệu
            df = pd.read_csv(csv_path)
            logger.info(f"Loaded {len(df)} records from WQI_data.csv")
            logger.info(f"CSV columns: {list(df.columns)}")
            
            # Thêm station_id nếu chưa có
            if 'station_id' not in df.columns:
                df['station_id'] = 0  # Mặc định station_id = 0
                logger.info("Added station_id column with default value 0")
            
            # Tạo default station nếu chưa tồn tại
            default_station = {
                'station_id': 0,
                'station_name': 'Default Station',
                'location': 'Default Location',
                'latitude': 0.0,
                'longitude': 0.0,
                'description': 'Default station for historical data'
            }
            self.insert_station(default_station)
            
            # Kiểm tra và xử lý các cột cần thiết
            required_columns = ['Date', 'Temperature', 'PH', 'DO', 'WQI', 'station_id']
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                logger.error(f"Missing required columns: {missing_columns}")
                return False
            
            # Chuyển đổi Date thành datetime
            df['Date'] = pd.to_datetime(df['Date'])
            
            # Lưu từng record vào database
            success_count = 0
            for index, row in df.iterrows():
                try:
                    historical_data = {
                        'station_id': int(row['station_id']),
                        'measurement_date': row['Date'].date(),
                        'temperature': float(row['Temperature']),
                        'ph': float(row['PH']),
                        'do': float(row['DO']),
                        'wqi': float(row['WQI'])
                    }
                    
                    if self.insert_historical_data(historical_data):
                        success_count += 1
                    else:
                        logger.warning(f"Failed to insert record at index {index}")
                        
                except Exception as e:
                    logger.error(f"Error processing record at index {index}: {e}")
                    logger.error(f"Row data: {row.to_dict()}")
                    continue
            
            logger.info(f"Successfully loaded {success_count}/{len(df)} records to database")
            return success_count > 0
            
        except Exception as e:
            logger.error(f"Error loading WQI data to database: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return False

    def trigger_ml_pipeline(self):
        """Trigger ML pipeline DAG sau khi lưu dữ liệu streaming"""
        try:
            from airflow.api.client.local_client import Client
            
            # Tạo Airflow client
            client = Client(None, None)
            
            # Trigger streaming_data_processor DAG
            result = client.trigger_dag(
                dag_id='streaming_data_processor',
                conf={},
                execution_date=None
            )
            
            logger.info(f"Triggered streaming_data_processor DAG: {result}")
            return True
            
        except Exception as e:
            logger.error(f"Error triggering ML pipeline: {e}")
            return False

# Global instance
db_manager = DatabaseManager() 