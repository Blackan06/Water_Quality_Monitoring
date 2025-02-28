CREATE DATABASE WQI
CREATE TABLE IF NOT EXISTS logs (
    log_time TIMESTAMP NOT NULL,
    level VARCHAR(50) NOT NULL,
    message TEXT NOT NULL,
    logger_name VARCHAR(255) NOT NULL
);

CREATE TABLE IF NOT EXISTS iot_sensor (
    id SERIAL PRIMARY KEY,          -- Tạo cột id là khóa chính và tự động tăng
    ph DOUBLE PRECISION,            -- Cột ph kiểu số thực (double precision)
    turbidity DOUBLE PRECISION,     -- Cột turbidity kiểu số thực (double precision)
    temperature DOUBLE PRECISION,  -- Cột temperature kiểu số thực (double precision)
    wqi DOUBLE PRECISION,           -- Cột wqi kiểu số thực (double precision)
    measurement_time TIMESTAMP,     -- Cột measurement_time kiểu timestamp
    create_at TIMESTAMP             -- Cột create_at kiểu timestamp
);
