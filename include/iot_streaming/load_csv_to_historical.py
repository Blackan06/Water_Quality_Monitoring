import os
import pandas as pd
import psycopg2
from psycopg2.extras import execute_values


def main():
    csv_path = os.getenv('CSV_PATH', '/opt/airflow/data/balanced_wqi_data.csv')
    host = os.getenv('DB_HOST', '194.238.16.14')
    port = os.getenv('DB_PORT', '5432')
    database = os.getenv('DB_NAME', 'wqi_db')
    user = os.getenv('DB_USER', 'postgres')
    password = os.getenv('DB_PASSWORD', 'postgres1234')

    if not os.path.exists(csv_path):
        print(f"CSV not found: {csv_path}")
        return 1

    conn = psycopg2.connect(host=host, port=port, database=database, user=user, password=password)
    cur = conn.cursor()

    insert_sql = (
        """
        INSERT INTO historical_wqi_data (station_id, measurement_date, temperature, ph, "do", wqi)
        VALUES %s
        ON CONFLICT (station_id, measurement_date) DO UPDATE SET
            temperature = EXCLUDED.temperature,
            ph = EXCLUDED.ph,
            "do" = EXCLUDED."do",
            wqi = EXCLUDED.wqi
        """
    )

    total = 0
    for chunk in pd.read_csv(csv_path, chunksize=5000):
        required = {'Date','Temperature','PH','DO','WQI','station_id'}
        if not required.issubset(chunk.columns):
            print(f"Missing required columns. Found: {list(chunk.columns)}")
            return 2
            
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
        execute_values(cur, insert_sql, rows, page_size=1000)
        total += len(rows)

    conn.commit()
    cur.close()
    conn.close()
    print(f"UPSERTED ROWS: {total}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())


