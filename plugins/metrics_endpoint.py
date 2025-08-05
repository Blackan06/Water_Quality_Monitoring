from flask import Blueprint, Response
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
from include.iot_streaming.prometheus_exporter import get_metrics

# Create blueprint
metrics_bp = Blueprint('metrics', __name__)

@metrics_bp.route('/metrics')
def metrics():
    """Expose Prometheus metrics"""
    try:
        # Get metrics and update with real data
        metrics = get_metrics()
        
        # Load data from database
        from include.iot_streaming.database_manager import db_manager
        
        conn = db_manager.get_connection()
        if conn:
            cur = conn.cursor()
            
            # Get recent data
            cur.execute("""
                SELECT station_id, measurement_time, ph, temperature, "do", wqi
                FROM processed_water_quality_data
                ORDER BY measurement_time DESC
                LIMIT 50
            """)
            
            recent_data = cur.fetchall()
            
            # Get station info
            cur.execute("SELECT station_id, station_name FROM monitoring_stations WHERE is_active = TRUE")
            stations = {row[0]: row[1] for row in cur.fetchall()}
            
            cur.close()
            conn.close()
            
            if recent_data:
                for row in recent_data:
                    station_id, measurement_time, ph, temp, do_val, wqi = row
                    station_name = stations.get(station_id, f"Station_{station_id}")
                    
                    # Update metrics
                    metrics.update_wqi(station_id, station_name, wqi or 0)
                    metrics.update_ph(station_id, station_name, ph or 0)
                    metrics.update_temperature(station_id, station_name, temp or 0)
                    metrics.update_do(station_id, station_name, do_val or 0)
        
        return Response(generate_latest(), mimetype=CONTENT_TYPE_LATEST)
    except Exception as e:
        return Response(f"Error: {str(e)}", status=500) 