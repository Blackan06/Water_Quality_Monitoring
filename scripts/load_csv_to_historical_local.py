import argparse
import csv
from datetime import datetime
from typing import List, Tuple

import psycopg2
from psycopg2.extras import execute_values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Load balanced WQI CSV into historical_wqi_data (standalone, no Docker required)"
    )
    parser.add_argument(
        "--csv",
        required=True,
        help="Path to balanced_wqi_data.csv (e.g. D:\\WQI\\Water_Quality_Monitoring\\data\\balanced_wqi_data.csv)",
    )
    parser.add_argument("--db-host", default="postgres")
    parser.add_argument("--db-port", type=int, default=5432)
    parser.add_argument("--db-name", default="wqi_db")
    parser.add_argument("--db-user", default="postgres")
    parser.add_argument("--db-password", default="postgres1234")
    parser.add_argument("--batch-size", type=int, default=10000)
    return parser.parse_args()


def read_rows(csv_path: str, batch_size: int) -> List[Tuple]:
    rows: List[Tuple] = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        required = {"Date", "Temperature", "PH", "DO", "WQI", "station_id"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"CSV missing columns: {sorted(missing)}")

        for rec in reader:
            date_raw = rec.get("Date")
            if not date_raw:
                continue
            try:
                measurement_date = datetime.strptime(date_raw.strip(), "%Y-%m-%d").date()
            except Exception:
                # Skip malformed date rows
                continue

            def to_float(val: str):
                try:
                    return float(val) if val not in (None, "", "NaN", "nan") else None
                except Exception:
                    return None

            try:
                station_id = int(float(rec.get("station_id", 0)))
            except Exception:
                station_id = 0

            temperature = to_float(rec.get("Temperature"))
            ph = to_float(rec.get("PH"))
            do_val = to_float(rec.get("DO"))
            wqi = to_float(rec.get("WQI"))

            rows.append((station_id, measurement_date, temperature, ph, do_val, wqi))

            if len(rows) >= batch_size:
                yield rows
                rows = []

    if rows:
        yield rows


def ensure_table(conn) -> None:
    create_sql = """
    CREATE TABLE IF NOT EXISTS historical_wqi_data (
        id SERIAL PRIMARY KEY,
        station_id INTEGER NOT NULL,
        measurement_date DATE NOT NULL,
        temperature DECIMAL(5, 2),
        ph DECIMAL(5, 2),
        "do" DECIMAL(5, 2),
        wqi DECIMAL(6, 2),
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        UNIQUE(station_id, measurement_date)
    )
    """
    with conn.cursor() as cur:
        cur.execute(create_sql)
    conn.commit()


def upsert_batches(conn, batches: List[List[Tuple]]) -> int:
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
    with conn.cursor() as cur:
        for rows in batches:
            if not rows:
                continue
            execute_values(cur, insert_sql, rows, page_size=max(1000, len(rows)))
            total += len(rows)
    conn.commit()
    return total


def main() -> int:
    args = parse_args()

    conn = psycopg2.connect(
        host=args.db_host,
        port=args.db_port,
        database=args.db_name,
        user=args.db_user,
        password=args.db_password,
    )

    ensure_table(conn)

    total = 0
    for batch in read_rows(args.csv, args.batch_size):
        total += upsert_batches(conn, [batch])

    conn.close()
    print(f"UPSERTED ROWS: {total}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


