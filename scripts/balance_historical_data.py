#!/usr/bin/env python3
"""
Script để cân bằng dữ liệu historical cho 3 stations
- Phân tích dữ liệu hiện tại
- Bổ sung dữ liệu thiếu cho Station 2
- Loại bỏ dữ liệu thừa cho Station 0
- Lưu vào bảng historical_wqi_data
"""

import pandas as pd
import numpy as np
import psycopg2
import logging
from datetime import datetime, timedelta
import os
from typing import Dict, List, Tuple

# Cấu hình logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class HistoricalDataBalancer:
    def __init__(self, csv_file: str = "data/WQI_data.csv"):
        self.csv_file = csv_file
        self.data = None
        self.balanced_data = None
        
        # Cấu hình database
        self.DB_HOST = "postgres"
        self.DB_NAME = "wqi_db"
        self.DB_USER = "postgres"
        self.DB_PASSWORD = "postgres"
        self.DB_PORT = "5432"
    
    def load_data(self) -> pd.DataFrame:
        """Tải dữ liệu từ CSV"""
        try:
            self.data = pd.read_csv(self.csv_file)
            self.data['Date'] = pd.to_datetime(self.data['Date'])
            logger.info(f"Đã tải {len(self.data)} records từ {self.csv_file}")
            return self.data
        except Exception as e:
            logger.error(f"Lỗi khi tải dữ liệu: {e}")
            raise
    
    def analyze_current_data(self) -> Dict:
        """Phân tích dữ liệu hiện tại"""
        if self.data is None:
            self.load_data()
        
        # Phân tích theo station
        stations_analysis = {}
        
        for station_id in self.data['station_id'].unique():
            station_data = self.data[self.data['station_id'] == station_id].copy()
            station_data = station_data.sort_values('Date')
            
            # Tìm ngày bắt đầu và kết thúc
            min_date = station_data['Date'].min()
            max_date = station_data['Date'].max()
            
            # Tính số tháng mong đợi (từ 2003-01 đến 2023-12 = 252 tháng)
            expected_months = 252
            
            # Tìm các tháng thiếu
            date_range = pd.date_range(start='2003-01-15', end='2023-12-15', freq='MS')
            expected_dates = [d.replace(day=15) for d in date_range]
            
            existing_months = station_data['Date'].dt.to_period('M').unique()
            expected_months_period = [pd.Period(d, freq='M') for d in expected_dates]
            
            missing_months = []
            for period in expected_months_period:
                if period not in existing_months:
                    missing_months.append(period.strftime('%Y-%m'))
            
            # Tìm records trùng tháng (cho Station 0)
            duplicate_months = station_data.groupby(station_data['Date'].dt.to_period('M')).size()
            duplicate_months = duplicate_months[duplicate_months > 1]
            
            # Kiểm tra ngày không phải 15
            non_15th_dates = station_data[station_data['Date'].dt.day != 15]
            
            stations_analysis[station_id] = {
                'total_records': len(station_data),
                'min_date': min_date,
                'max_date': max_date,
                'expected_months': expected_months,
                'missing_months': missing_months,
                'duplicate_months': list(duplicate_months.index) if len(duplicate_months) > 0 else [],
                'non_15th_dates': len(non_15th_dates),
                'data': station_data
            }
            
            logger.info(f"Station {station_id}: {len(station_data)} records, "
                       f"thiếu {len(missing_months)} tháng, "
                       f"trùng {len(duplicate_months)} tháng, "
                       f"{len(non_15th_dates)} records không phải ngày 15")
        
        return stations_analysis
    
    def clean_station_0_data(self, station_data: pd.DataFrame) -> pd.DataFrame:
        """Làm sạch dữ liệu Station 0: loại bỏ trùng tháng và đảm bảo ngày 15"""
        logger.info("Làm sạch dữ liệu Station 0...")
        
        # 1. Loại bỏ records trùng tháng, giữ lại record mới nhất
        station_data = station_data.sort_values('Date', ascending=False)
        station_data['month_year'] = station_data['Date'].dt.to_period('M')
        station_data = station_data.drop_duplicates(subset=['month_year'], keep='first')
        station_data = station_data.drop('month_year', axis=1)
        
        # 2. Đảm bảo tất cả ngày đều là ngày 15
        station_data['Date'] = station_data['Date'].dt.to_period('M').dt.start_time + pd.Timedelta(days=14)
        
        # 3. Sắp xếp lại theo thứ tự thời gian
        station_data = station_data.sort_values('Date')
        
        logger.info(f"Sau khi làm sạch: {len(station_data)} records")
        return station_data
    
    def generate_missing_data_for_station_2(self, missing_months: List[str], 
                                          reference_data: pd.DataFrame) -> pd.DataFrame:
        """Tạo dữ liệu thiếu cho Station 2"""
        logger.info(f"Tạo dữ liệu cho {len(missing_months)} tháng thiếu...")
        
        missing_records = []
        
        for month_str in missing_months:
            # Chuyển đổi string thành datetime
            date_obj = datetime.strptime(month_str + '-15', '%Y-%m-%d')
            
            # Lấy dữ liệu tham chiếu từ cùng tháng của các năm khác
            month = date_obj.month
            reference_months = reference_data[reference_data['Date'].dt.month == month]
            
            if len(reference_months) > 0:
                # Tính trung bình của các tháng cùng mùa
                avg_temp = reference_months['Temperature'].mean()
                avg_ph = reference_months['PH'].mean()
                avg_do = reference_months['DO'].mean()
                avg_wqi = reference_months['WQI'].mean()
                
                # Thêm biến động ngẫu nhiên nhỏ
                temp = avg_temp + np.random.normal(0, 1)
                ph = avg_ph + np.random.normal(0, 0.2)
                do = avg_do + np.random.normal(0, 0.3)
                wqi = avg_wqi + np.random.normal(0, 2)
                
                # Đảm bảo giá trị trong khoảng hợp lý
                temp = max(20, min(35, temp))
                ph = max(5, min(9, ph))
                do = max(2, min(10, do))
                wqi = max(30, min(100, wqi))
                
                missing_records.append({
                    'station_id': 2,
                    'Date': date_obj,
                    'Temperature': round(temp, 2),
                    'PH': round(ph, 2),
                    'DO': round(do, 2),
                    'WQI': round(wqi, 2)
                })
                
                logger.info(f"Tạo dữ liệu cho {month_str}: Temp={temp:.2f}, PH={ph:.2f}, DO={do:.2f}, WQI={wqi:.2f}")
        
        return pd.DataFrame(missing_records)
    
    def create_balanced_dataset(self) -> pd.DataFrame:
        """Tạo dataset cân bằng cho tất cả stations"""
        logger.info("Bắt đầu cân bằng dữ liệu...")
        
        # Phân tích dữ liệu hiện tại
        analysis = self.analyze_current_data()
        
        balanced_records = []
        
        for station_id in sorted(analysis.keys()):
            station_info = analysis[station_id]
            station_data = station_info['data'].copy()
            
            if station_id == 0:
                # Xử lý Station 0: loại bỏ trùng tháng và đảm bảo ngày 15
                station_data = self.clean_station_0_data(station_data)
                logger.info(f"Station 0: Sau khi làm sạch có {len(station_data)} records")
                
            elif station_id == 2:
                # Xử lý Station 2: thêm dữ liệu thiếu
                missing_data = self.generate_missing_data_for_station_2(
                    station_info['missing_months'], 
                    station_data
                )
                station_data = pd.concat([station_data, missing_data], ignore_index=True)
                station_data = station_data.sort_values('Date')
                logger.info(f"Station 2: Sau khi thêm dữ liệu thiếu có {len(station_data)} records")
            
            # Thêm vào danh sách tổng hợp
            balanced_records.append(station_data)
        
        # Tổng hợp tất cả stations
        self.balanced_data = pd.concat(balanced_records, ignore_index=True)
        self.balanced_data = self.balanced_data.sort_values(['station_id', 'Date'])
        
        # Thống kê kết quả
        for station_id in self.balanced_data['station_id'].unique():
            station_count = len(self.balanced_data[self.balanced_data['station_id'] == station_id])
            logger.info(f"Station {station_id}: {station_count} records")
        
        total_records = len(self.balanced_data)
        logger.info(f"Tổng cộng: {total_records} records")
        
        return self.balanced_data
    
    def save_to_database(self, table_name: str = "historical_wqi_data"):
        """Lưu dữ liệu cân bằng vào database"""
        if self.balanced_data is None:
            logger.error("Chưa có dữ liệu cân bằng để lưu!")
            return
        
        try:
            # Kết nối database
            conn = psycopg2.connect(
                host=self.DB_HOST,
                database=self.DB_NAME,
                user=self.DB_USER,
                password=self.DB_PASSWORD,
                port=self.DB_PORT
            )
            cur = conn.cursor()
            
            # Tạo bảng nếu chưa tồn tại
            create_table_sql = f"""
            CREATE TABLE IF NOT EXISTS {table_name} (
                id SERIAL PRIMARY KEY,
                station INTEGER,
                date DATE,
                temperature DECIMAL(5,2),
                ph DECIMAL(4,2),
                "do" DECIMAL(4,2),
                wqi DECIMAL(5,2)
            );
            """
            cur.execute(create_table_sql)
            
            # Reset sequence về 1
            cur.execute(f"ALTER SEQUENCE {table_name}_id_seq RESTART WITH 1")
            
            # Xóa dữ liệu cũ
            cur.execute(f"DELETE FROM {table_name}")
            logger.info(f"Đã xóa dữ liệu cũ từ bảng {table_name}")
            
            # Chèn dữ liệu mới
            for _, row in self.balanced_data.iterrows():
                insert_sql = f"""
                INSERT INTO {table_name} (station_id, measurement_date, temperature, ph, "do", wqi)
                VALUES (%s, %s, %s, %s, %s, %s)
                """
                cur.execute(insert_sql, (
                    int(row['station_id']),
                    row['Date'].date(),
                    float(row['Temperature']),
                    float(row['PH']),
                    float(row['DO']),
                    float(row['WQI'])
                ))
            
            conn.commit()
            logger.info(f"Đã lưu {len(self.balanced_data)} records vào bảng {table_name}")
            
        except Exception as e:
            logger.error(f"Lỗi khi lưu vào database: {e}")
            if conn:
                conn.rollback()
        finally:
            if cur:
                cur.close()
            if conn:
                conn.close()
    
    def save_to_csv(self, output_file: str = "data/balanced_wqi_data.csv"):
        """Lưu dữ liệu cân bằng vào CSV"""
        if self.balanced_data is None:
            logger.error("Chưa có dữ liệu cân bằng để lưu!")
            return
        
        try:
            self.balanced_data.to_csv(output_file, index=False)
            logger.info(f"Đã lưu dữ liệu cân bằng vào {output_file}")
        except Exception as e:
            logger.error(f"Lỗi khi lưu CSV: {e}")

def main():
    """Hàm chính để chạy script"""
    try:
        # Khởi tạo balancer
        balancer = HistoricalDataBalancer()
        
        # Tạo dataset cân bằng
        balanced_data = balancer.create_balanced_dataset()
        
        # Lưu vào database
        balancer.save_to_database()
        
        # Lưu vào CSV
        balancer.save_to_csv()
        
        logger.info("Hoàn thành cân bằng dữ liệu!")
        
    except Exception as e:
        logger.error(f"Lỗi trong quá trình xử lý: {e}")

if __name__ == "__main__":
    main() 