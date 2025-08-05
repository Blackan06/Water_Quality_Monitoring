#!/usr/bin/env python3
"""
Resource Monitor for Water Quality Monitoring System
Tự động monitor và điều chỉnh resource usage
"""

import psutil
import os
import time
import logging
import yaml
import json
from datetime import datetime
from typing import Dict, Any, Optional

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ResourceMonitor:
    def __init__(self, config_path: str = "config/resource_config.yaml"):
        """Initialize resource monitor with configuration"""
        self.config_path = config_path
        self.config = self.load_config()
        self.monitoring_active = True
        self.resource_history = []
        
    def load_config(self) -> Dict[str, Any]:
        """Load resource configuration"""
        try:
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
            logger.info("Resource configuration loaded successfully")
            return config
        except Exception as e:
            logger.error(f"Error loading config: {e}")
            return {}
    
    def get_system_resources(self) -> Dict[str, float]:
        """Get current system resource usage"""
        try:
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            
            # Memory usage
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            memory_used_gb = memory.used / (1024**3)
            
            # Disk usage
            disk = psutil.disk_usage('/')
            disk_percent = disk.percent
            
            # Network I/O
            network = psutil.net_io_counters()
            network_bytes_sent = network.bytes_sent
            network_bytes_recv = network.bytes_recv
            
            return {
                'cpu_percent': cpu_percent,
                'memory_percent': memory_percent,
                'memory_used_gb': memory_used_gb,
                'disk_percent': disk_percent,
                'network_bytes_sent': network_bytes_sent,
                'network_bytes_recv': network_bytes_recv,
                'timestamp': datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Error getting system resources: {e}")
            return {}
    
    def check_resource_limits(self, resources: Dict[str, float]) -> Dict[str, bool]:
        """Check if resources are within limits"""
        limits = {
            'cpu_high': resources.get('cpu_percent', 0) > 80,
            'memory_high': resources.get('memory_percent', 0) > 85,
            'disk_high': resources.get('disk_percent', 0) > 90,
            'memory_gb_high': resources.get('memory_used_gb', 0) > 8.0  # 8GB limit
        }
        return limits
    
    def get_process_resources(self) -> Dict[str, Any]:
        """Get resource usage for specific processes"""
        try:
            processes = {}
            
            # Check for Python processes (likely our training)
            for proc in psutil.process_iter(['pid', 'name', 'cpu_percent', 'memory_percent']):
                try:
                    if 'python' in proc.info['name'].lower():
                        processes[proc.info['pid']] = {
                            'name': proc.info['name'],
                            'cpu_percent': proc.info['cpu_percent'],
                            'memory_percent': proc.info['memory_percent']
                        }
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
            
            return processes
        except Exception as e:
            logger.error(f"Error getting process resources: {e}")
            return {}
    
    def log_resource_usage(self, resources: Dict[str, float], limits: Dict[str, bool]):
        """Log current resource usage"""
        logger.info(f"Resource Usage - CPU: {resources.get('cpu_percent', 0):.1f}%, "
                   f"Memory: {resources.get('memory_percent', 0):.1f}% "
                   f"({resources.get('memory_used_gb', 0):.2f}GB), "
                   f"Disk: {resources.get('disk_percent', 0):.1f}%")
        
        # Log warnings for high usage
        if any(limits.values()):
            warnings = []
            if limits['cpu_high']:
                warnings.append("CPU usage high")
            if limits['memory_high']:
                warnings.append("Memory usage high")
            if limits['disk_high']:
                warnings.append("Disk usage high")
            if limits['memory_gb_high']:
                warnings.append("Memory usage > 8GB")
            
            logger.warning(f"Resource warnings: {', '.join(warnings)}")
    
    def save_resource_history(self, resources: Dict[str, float]):
        """Save resource usage history"""
        self.resource_history.append(resources)
        
        # Keep only last 1000 entries
        if len(self.resource_history) > 1000:
            self.resource_history = self.resource_history[-1000:]
    
    def get_resource_recommendations(self, resources: Dict[str, float]) -> Dict[str, str]:
        """Get recommendations based on current resource usage"""
        recommendations = {}
        
        cpu_percent = resources.get('cpu_percent', 0)
        memory_percent = resources.get('memory_percent', 0)
        memory_gb = resources.get('memory_used_gb', 0)
        
        if cpu_percent > 90:
            recommendations['cpu'] = "Consider reducing parallel processing or batch size"
        elif cpu_percent > 70:
            recommendations['cpu'] = "Monitor CPU usage, consider optimization"
        
        if memory_percent > 90:
            recommendations['memory'] = "Critical memory usage - reduce model complexity"
        elif memory_percent > 80:
            recommendations['memory'] = "High memory usage - consider smaller models"
        elif memory_gb > 6:
            recommendations['memory'] = "Memory usage > 6GB - optimize data processing"
        
        return recommendations
    
    def cleanup_resources(self):
        """Cleanup resources when needed"""
        try:
            # Clear Python cache
            import gc
            gc.collect()
            
            # Clear TensorFlow memory
            try:
                import tensorflow as tf
                tf.keras.backend.clear_session()
                logger.info("Cleared TensorFlow session")
            except ImportError:
                pass
            
            # Clear MLflow cache if needed
            try:
                import mlflow
                # MLflow doesn't have direct cache clearing, but we can log this
                logger.info("MLflow session active")
            except ImportError:
                pass
                
        except Exception as e:
            logger.error(f"Error during resource cleanup: {e}")
    
    def monitor_resources(self, interval: int = 30):
        """Main monitoring loop"""
        logger.info(f"Starting resource monitoring with {interval}s intervals")
        
        while self.monitoring_active:
            try:
                # Get current resources
                resources = self.get_system_resources()
                limits = self.check_resource_limits(resources)
                
                # Log current usage
                self.log_resource_usage(resources, limits)
                
                # Save to history
                self.save_resource_history(resources)
                
                # Get recommendations
                recommendations = self.get_resource_recommendations(resources)
                if recommendations:
                    logger.info(f"Recommendations: {recommendations}")
                
                # Check if cleanup is needed
                if limits['memory_high'] or limits['memory_gb_high']:
                    logger.warning("High memory usage detected, performing cleanup")
                    self.cleanup_resources()
                
                # Sleep for interval
                time.sleep(interval)
                
            except KeyboardInterrupt:
                logger.info("Resource monitoring stopped by user")
                break
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                time.sleep(interval)
    
    def get_resource_summary(self) -> Dict[str, Any]:
        """Get resource usage summary"""
        if not self.resource_history:
            return {}
        
        # Calculate averages
        cpu_values = [r.get('cpu_percent', 0) for r in self.resource_history]
        memory_values = [r.get('memory_percent', 0) for r in self.resource_history]
        memory_gb_values = [r.get('memory_used_gb', 0) for r in self.resource_history]
        
        summary = {
            'total_samples': len(self.resource_history),
            'time_period': {
                'start': self.resource_history[0].get('timestamp'),
                'end': self.resource_history[-1].get('timestamp')
            },
            'cpu': {
                'average': sum(cpu_values) / len(cpu_values),
                'max': max(cpu_values),
                'min': min(cpu_values)
            },
            'memory': {
                'average_percent': sum(memory_values) / len(memory_values),
                'max_percent': max(memory_values),
                'average_gb': sum(memory_gb_values) / len(memory_gb_values),
                'max_gb': max(memory_gb_values)
            }
        }
        
        return summary
    
    def stop_monitoring(self):
        """Stop resource monitoring"""
        self.monitoring_active = False
        logger.info("Resource monitoring stopped")

def main():
    """Main function to run resource monitor"""
    monitor = ResourceMonitor()
    
    try:
        # Run monitoring for 1 hour by default
        monitor.monitor_resources(interval=30)
    except KeyboardInterrupt:
        logger.info("Monitoring stopped by user")
    finally:
        # Print summary
        summary = monitor.get_resource_summary()
        if summary:
            logger.info(f"Resource monitoring summary: {json.dumps(summary, indent=2)}")

if __name__ == "__main__":
    main() 