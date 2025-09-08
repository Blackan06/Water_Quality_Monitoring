#!/usr/bin/env python3
"""
Resource Monitoring Script for Water Quality Monitoring System
Theo dõi CPU, Memory và Docker container usage để tối ưu hóa hiệu suất
"""

import psutil
import docker
import time
import json
import logging
from datetime import datetime
import argparse

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ResourceMonitor:
    def __init__(self):
        self.docker_client = None
        try:
            self.docker_client = docker.from_env()
        except Exception as e:
            logger.warning(f"Không thể kết nối Docker: {e}")
    
    def get_system_stats(self):
        """Lấy thống kê hệ thống"""
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        return {
            'timestamp': datetime.now().isoformat(),
            'cpu': {
                'percent': cpu_percent,
                'count': psutil.cpu_count(),
                'load_avg': psutil.getloadavg() if hasattr(psutil, 'getloadavg') else None
            },
            'memory': {
                'total': memory.total,
                'available': memory.available,
                'percent': memory.percent,
                'used': memory.used
            },
            'disk': {
                'total': disk.total,
                'used': disk.used,
                'free': disk.free,
                'percent': (disk.used / disk.total) * 100
            }
        }
    
    def get_docker_stats(self):
        """Lấy thống kê Docker containers"""
        if not self.docker_client:
            return {}
        
        containers = self.docker_client.containers.list()
        container_stats = {}
        
        for container in containers:
            try:
                stats = container.stats(stream=False)
                name = container.name
                
                # Tính CPU usage
                cpu_delta = stats['cpu_stats']['cpu_usage']['total_usage'] - \
                           stats['precpu_stats']['cpu_usage']['total_usage']
                system_delta = stats['cpu_stats']['system_cpu_usage'] - \
                              stats['precpu_stats']['system_cpu_usage']
                cpu_percent = 0.0
                if system_delta > 0 and cpu_delta > 0:
                    cpu_percent = (cpu_delta / system_delta) * len(stats['cpu_stats']['cpu_usage']['percpu_usage']) * 100.0
                
                # Tính Memory usage
                memory_usage = stats['memory_stats']['usage']
                memory_limit = stats['memory_stats']['limit']
                memory_percent = (memory_usage / memory_limit) * 100.0 if memory_limit > 0 else 0
                
                container_stats[name] = {
                    'cpu_percent': round(cpu_percent, 2),
                    'memory_usage': memory_usage,
                    'memory_limit': memory_limit,
                    'memory_percent': round(memory_percent, 2),
                    'status': container.status
                }
                
            except Exception as e:
                logger.warning(f"Không thể lấy stats cho container {container.name}: {e}")
        
        return container_stats
    
    def get_airflow_processes(self):
        """Lấy thông tin các process Airflow"""
        airflow_processes = []
        
        for proc in psutil.process_iter(['pid', 'name', 'cpu_percent', 'memory_percent', 'cmdline']):
            try:
                if 'airflow' in proc.info['name'].lower():
                    airflow_processes.append({
                        'pid': proc.info['pid'],
                        'name': proc.info['name'],
                        'cpu_percent': proc.info['cpu_percent'],
                        'memory_percent': proc.info['memory_percent'],
                        'cmdline': ' '.join(proc.info['cmdline']) if proc.info['cmdline'] else ''
                    })
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        
        return airflow_processes
    
    def check_high_cpu_usage(self, threshold=80):
        """Kiểm tra CPU usage cao"""
        system_stats = self.get_system_stats()
        high_cpu_containers = []
        high_cpu_processes = []
        
        # Kiểm tra system CPU
        if system_stats['cpu']['percent'] > threshold:
            logger.warning(f"⚠️  System CPU usage cao: {system_stats['cpu']['percent']:.1f}%")
        
        # Kiểm tra Docker containers
        docker_stats = self.get_docker_stats()
        for name, stats in docker_stats.items():
            if stats['cpu_percent'] > threshold:
                high_cpu_containers.append((name, stats['cpu_percent']))
                logger.warning(f"⚠️  Container {name} CPU usage cao: {stats['cpu_percent']:.1f}%")
        
        # Kiểm tra Airflow processes
        airflow_processes = self.get_airflow_processes()
        for proc in airflow_processes:
            if proc['cpu_percent'] > threshold:
                high_cpu_processes.append((proc['name'], proc['cpu_percent']))
                logger.warning(f"⚠️  Process {proc['name']} (PID: {proc['pid']}) CPU usage cao: {proc['cpu_percent']:.1f}%")
        
        return {
            'system_cpu': system_stats['cpu']['percent'],
            'high_cpu_containers': high_cpu_containers,
            'high_cpu_processes': high_cpu_processes
        }
    
    def generate_report(self):
        """Tạo báo cáo tổng hợp"""
        system_stats = self.get_system_stats()
        docker_stats = self.get_docker_stats()
        airflow_processes = self.get_airflow_processes()
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'system': system_stats,
            'docker_containers': docker_stats,
            'airflow_processes': airflow_processes,
            'summary': {
                'total_containers': len(docker_stats),
                'total_airflow_processes': len(airflow_processes),
                'system_cpu_usage': system_stats['cpu']['percent'],
                'system_memory_usage': system_stats['memory']['percent']
            }
        }
        
        return report
    
    def monitor_continuous(self, interval=30, duration=None):
        """Monitor liên tục"""
        logger.info(f"Bắt đầu monitor resources với interval {interval}s")
        if duration:
            logger.info(f"Sẽ chạy trong {duration}s")
        
        start_time = time.time()
        while True:
            try:
                report = self.generate_report()
                
                # In summary
                print(f"\n{'='*60}")
                print(f"📊 Resource Report - {report['timestamp']}")
                print(f"{'='*60}")
                print(f"🖥️  System CPU: {report['summary']['system_cpu_usage']:.1f}%")
                print(f"💾 System Memory: {report['summary']['system_memory_usage']:.1f}%")
                print(f"🐳 Docker Containers: {report['summary']['total_containers']}")
                print(f"🔄 Airflow Processes: {report['summary']['total_airflow_processes']}")
                
                # In chi tiết containers
                if report['docker_containers']:
                    print(f"\n🐳 Docker Containers:")
                    for name, stats in report['docker_containers'].items():
                        print(f"  {name}: CPU {stats['cpu_percent']:.1f}%, Memory {stats['memory_percent']:.1f}%")
                
                # In chi tiết Airflow processes
                if report['airflow_processes']:
                    print(f"\n🔄 Airflow Processes:")
                    for proc in report['airflow_processes']:
                        print(f"  {proc['name']} (PID: {proc['pid']}): CPU {proc['cpu_percent']:.1f}%, Memory {proc['memory_percent']:.1f}%")
                
                # Kiểm tra high CPU usage
                high_cpu = self.check_high_cpu_usage()
                
                # Kiểm tra thời gian
                if duration and (time.time() - start_time) >= duration:
                    logger.info("Đã đạt thời gian monitor")
                    break
                
                time.sleep(interval)
                
            except KeyboardInterrupt:
                logger.info("Dừng monitor do user interrupt")
                break
            except Exception as e:
                logger.error(f"Lỗi trong quá trình monitor: {e}")
                time.sleep(interval)

def main():
    parser = argparse.ArgumentParser(description='Monitor system resources')
    parser.add_argument('--interval', type=int, default=30, help='Monitor interval in seconds')
    parser.add_argument('--duration', type=int, help='Monitor duration in seconds')
    parser.add_argument('--once', action='store_true', help='Run once and exit')
    parser.add_argument('--threshold', type=int, default=80, help='High CPU usage threshold')
    
    args = parser.parse_args()
    
    monitor = ResourceMonitor()
    
    if args.once:
        # Chạy một lần
        report = monitor.generate_report()
        print(json.dumps(report, indent=2))
    else:
        # Monitor liên tục
        monitor.monitor_continuous(args.interval, args.duration)

if __name__ == "__main__":
    main()
