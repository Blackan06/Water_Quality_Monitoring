import sys
sys.path.append('/opt/airflow/include')

from airflow.operators.python import PythonOperator
from airflow.decorators import dag
from pendulum import datetime
import logging
import os
import psycopg2
import pandas as pd


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
}


def _get_db_conn():
    return psycopg2.connect(
        host=os.getenv('DB_HOST', '194.238.16.14'),
        port=os.getenv('DB_PORT', '5432'),
        database=os.getenv('DB_NAME', 'wqi_db'),
        user=os.getenv('DB_USER', 'postgres'),
        password=os.getenv('DB_PASSWORD', 'postgres1234'),
        connect_timeout=int(os.getenv('DB_CONNECT_TIMEOUT', '10'))
    )


def create_all_tables_explicit():
    """Create/ensure all tables required by the project explicitly with SQL."""
    db_schema = os.getenv('DB_SCHEMA', 'public')
    statements = [
        # 1) monitoring_stations
        """
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
        """,
        # 2) raw_sensor_data
        """
        CREATE TABLE IF NOT EXISTS raw_sensor_data (
            id SERIAL PRIMARY KEY,
            station_id INTEGER NOT NULL,
            measurement_time TIMESTAMP NOT NULL,
            ph DECIMAL(5, 2),
            temperature DECIMAL(5, 2),
            "do" DECIMAL(5, 2),
            wqi DECIMAL(6, 2),
            is_processed BOOLEAN DEFAULT FALSE,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (station_id) REFERENCES monitoring_stations(station_id)
        )
        """,
        # Ensure is_processed column exists (older DBs may miss it)
        """
        ALTER TABLE raw_sensor_data
        ADD COLUMN IF NOT EXISTS is_processed BOOLEAN DEFAULT FALSE
        """,
        # 3) processed_water_quality_data
        """
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
        """,
        # 4) prediction_results
        """
        CREATE TABLE IF NOT EXISTS prediction_results (
            id SERIAL PRIMARY KEY,
            station_id INTEGER NOT NULL,
            prediction_time TIMESTAMP NOT NULL,
            model_type VARCHAR(50) NOT NULL,
            wqi_prediction DECIMAL(6, 2),
            confidence_score DECIMAL(5, 4),
            processing_time DECIMAL(8, 3),
            model_version VARCHAR(100),
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (station_id) REFERENCES monitoring_stations(station_id)
        )
        """,
        # 5) model_registry
        """
        CREATE TABLE IF NOT EXISTS model_registry (
            id SERIAL PRIMARY KEY,
            station_id INTEGER NOT NULL,
            model_type VARCHAR(50) NOT NULL,
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
        """,
        # 6) training_history
        """
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
            status VARCHAR(50),
            error_message TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (station_id) REFERENCES monitoring_stations(station_id)
        )
        """,
        # 7) model_comparison
        """
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
            best_model VARCHAR(50),
            accuracy_improvement DECIMAL(6, 4),
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (station_id) REFERENCES monitoring_stations(station_id)
        )
        """,
        # 8) alerts
        """
        CREATE TABLE IF NOT EXISTS alerts (
            id SERIAL PRIMARY KEY,
            station_id INTEGER NOT NULL,
            alert_type VARCHAR(100) NOT NULL,
            severity VARCHAR(20) NOT NULL,
            message TEXT NOT NULL,
            wqi_value DECIMAL(6, 2),
            threshold_value DECIMAL(6, 2),
            is_resolved BOOLEAN DEFAULT FALSE,
            resolved_at TIMESTAMP,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (station_id) REFERENCES monitoring_stations(station_id)
        )
        """,
        # 9) historical_wqi_data
        """
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
        """,
        # 10) wqi_predictions (used by dashboard summary APIs)
        """
        CREATE TABLE IF NOT EXISTS wqi_predictions (
            id SERIAL PRIMARY KEY,
            station_id INTEGER NOT NULL,
            prediction_time TIMESTAMP,
            prediction_date TIMESTAMP NOT NULL,
            prediction_horizon_months INTEGER NOT NULL,
            wqi_prediction DECIMAL(6,2),
            confidence_score DECIMAL(5,3),
            model_type VARCHAR(50),
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(station_id, prediction_date, prediction_horizon_months)
        )
        """,
        # 11) water_quality (public schema fallback)
        """
        CREATE TABLE IF NOT EXISTS water_quality (
            id SERIAL PRIMARY KEY,
            wq_date TIMESTAMP NOT NULL,
            temperature DOUBLE PRECISION,
            "DO" DOUBLE PRECISION,
            ph DOUBLE PRECISION,
            wqi DOUBLE PRECISION
        )
        """,
    ]

    # Additional ALTERs that may fail if already exist; we'll ignore duplicate errors
    additional_alters = [
        (
            "ALTER TABLE processed_water_quality_data\nADD CONSTRAINT processed_water_quality_data_station_time_unique UNIQUE (station_id, measurement_time)",
            "processed_water_quality_data unique"
        ),
        (
            "ALTER TABLE historical_wqi_data\nADD CONSTRAINT historical_wqi_data_station_date_unique UNIQUE (station_id, measurement_date)",
            "historical_wqi_data unique"
        ),
    ]

    with _get_db_conn() as conn:
        with conn.cursor() as cur:
            for stmt in statements:
                cur.execute(stmt)
            # Try optional constraints
            for stmt, name in additional_alters:
                try:
                    cur.execute(stmt)
                except Exception as e:
                    # Likely already exists; log at debug level
                    logger.debug(f"Skip adding {name} constraint: {e}")
            # Ensure schema-specific water_quality if DB_SCHEMA provided
            if db_schema and db_schema != 'public':
                try:
                    cur.execute(f"CREATE SCHEMA IF NOT EXISTS {db_schema}")
                except Exception as e:
                    logger.debug(f"Schema ensure failed or exists: {e}")
                cur.execute(f"""
                    CREATE TABLE IF NOT EXISTS {db_schema}.water_quality (
                        id SERIAL PRIMARY KEY,
                        wq_date TIMESTAMP NOT NULL,
                        temperature DOUBLE PRECISION,
                        "DO" DOUBLE PRECISION,
                        ph DOUBLE PRECISION,
                        wqi DOUBLE PRECISION
                    )
                """)
        conn.commit()
    return "All tables created/ensured"


@dag(
    dag_id='create_all_tables',
    default_args=default_args,
    description='Create or ensure all required PostgreSQL tables for WQI project',
    schedule=None,
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=['db', 'bootstrap']
)
def create_all_tables():
    create = PythonOperator(
        task_id='create_all_tables_explicit',
        python_callable=create_all_tables_explicit,
    )

    def load_historical_from_csv():
        """Load /opt/airflow/data/balanced_wqi_data.csv into historical_wqi_data."""
        csv_path_container = 'D:\WQI\Water_Quality_Monitoring\data\balanced_wqi_data.csv'
        if not os.path.exists(csv_path_container):
            raise FileNotFoundError(f"CSV not found at {csv_path_container}. Ensure ./data is mounted.")

        # Prepare insert with upsert semantics
        insert_sql = (
            """
            INSERT INTO historical_wqi_data (station_id, measurement_date, temperature, ph, "do", wqi)
            VALUES (%s, %s, %s, %s, %s, %s)
            ON CONFLICT (station_id, measurement_date) DO UPDATE SET
                temperature = EXCLUDED.temperature,
                ph = EXCLUDED.ph,
                "do" = EXCLUDED."do",
                wqi = EXCLUDED.wqi
            """
        )

        total = 0
        with _get_db_conn() as conn:
            with conn.cursor() as cur:
                for chunk in pd.read_csv(csv_path_container, chunksize=5000):
                    # Normalize and validate columns
                    if not {'Date','Temperature','PH','DO','WQI','station_id'}.issubset(chunk.columns):
                        raise ValueError('CSV must contain columns: Date, Temperature, PH, DO, WQI, station_id')
                    chunk['Date'] = pd.to_datetime(chunk['Date'], errors='coerce')
                    rows = []
                    for _, r in chunk.iterrows():
                        if pd.isna(r['Date']):
                            continue
                        rows.append((
                            int(r['station_id']) if pd.notnull(r['station_id']) else 0,
                            r['Date'].date(),
                            float(r['Temperature']) if pd.notnull(r['Temperature']) else None,
                            float(r['PH']) if pd.notnull(r['PH']) else None,
                            float(r['DO']) if pd.notnull(r['DO']) else None,
                            float(r['WQI']) if pd.notnull(r['WQI']) else None,
                        ))
                    if not rows:
                        continue
                    # Use execute_values for bulk
                    from psycopg2.extras import execute_values
                    execute_values(cur, insert_sql.replace('VALUES (%s, %s, %s, %s, %s, %s)', 'VALUES %s'), rows, page_size=1000)
                    total += len(rows)
            conn.commit()

        logger.info(f"Loaded/updated {total} rows into historical_wqi_data from CSV")
        return f"historical_wqi_data upserted: {total} rows"

    load_hist = PythonOperator(
        task_id='load_historical_wqi_from_csv',
        python_callable=load_historical_from_csv,
    )

    create >> load_hist


create_all_tables()


