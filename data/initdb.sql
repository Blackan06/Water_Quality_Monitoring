CREATE DATABASE WQI
CREATE TABLE IF NOT EXISTS logs (
    log_time TIMESTAMP  NULL,
    level VARCHAR(50)  NULL,
    message TEXT  NULL,
    logger_name VARCHAR(255)  NULL
);

CREATE TABLE IF NOT EXISTS iot_sensor (
    id SERIAL PRIMARY KEY,          -- Tạo cột id là khóa chính và tự động tăng
    ph DOUBLE PRECISION NULL,            -- Cột ph kiểu số thực (double precision)
    turbidity DOUBLE PRECISION NULL,     -- Cột turbidity kiểu số thực (double precision)
    temperature DOUBLE PRECISION NULL,  -- Cột temperature kiểu số thực (double precision)
    wqi DOUBLE PRECISION NULL,           -- Cột wqi kiểu số thực (double precision)
    measurement_time TIMESTAMP NULL,     -- Cột measurement_time kiểu timestamp
    create_at TIMESTAMP NULL             -- Cột create_at kiểu timestamp
);
