import logging
import json
import os
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

class PredictionService:
    """Service for managing predictions and saving to database"""
    
    def __init__(self):
        self.db_manager = None
        try:
            from include.iot_streaming.database_manager import db_manager
            self.db_manager = db_manager
        except ImportError:
            logger.warning("Database manager not available")
    
    def save_prediction_results(self, prediction_results):
        """Save prediction results to database for dashboard"""
        try:
            if not self.db_manager:
                logger.error("Database manager not available")
                return False
            
            conn = self.db_manager.get_connection()
            if not conn:
                logger.error("Cannot connect to database")
                return False
            
            cur = conn.cursor()
            
            # Bảng đã có sẵn, chỉ cần lưu data
            logger.info("Using existing wqi_predictions table")
            
            # Save predictions
            saved_count = 0
            for station_id, result in prediction_results.items():
                if result.get('success', False):
                    future_predictions = result.get('future_predictions', {})
                    model_type = result.get('model_type', 'ensemble')
                    
                    for horizon_key, prediction in future_predictions.items():
                        try:
                            # Extract horizon months from key (e.g., "1month" -> 1)
                            horizon_months = int(horizon_key.replace('month', ''))
                            wqi_prediction = prediction.get('wqi_prediction', 0)
                            confidence_score = prediction.get('confidence_score', 0)
                            prediction_time = prediction.get('prediction_time')
                            
                            if prediction_time:
                                cur.execute("""
                                    INSERT INTO wqi_predictions 
                                    (station_id, prediction_date, prediction_horizon_months, 
                                     wqi_prediction, confidence_score, model_type)
                                    VALUES (%s, %s, %s, %s, %s, %s)
                                    ON CONFLICT (station_id, prediction_date, prediction_horizon_months) 
                                    DO UPDATE SET 
                                        wqi_prediction = EXCLUDED.wqi_prediction,
                                        confidence_score = EXCLUDED.confidence_score,
                                        model_type = EXCLUDED.model_type
                                """, (
                                    station_id,
                                    prediction_time,
                                    horizon_months,
                                    wqi_prediction,
                                    confidence_score,
                                    model_type
                                ))
                                saved_count += 1
                                
                        except Exception as e:
                            logger.error(f"Error saving prediction for station {station_id}, horizon {horizon_key}: {e}")
            
            conn.commit()
            cur.close()
            conn.close()
            
            logger.info(f"✅ Saved {saved_count} predictions to database")
            
            # Lưu predictions vào historical_wqi_data để dashboard hiển thị
            logger.info(f"📊 Prediction results to save: {prediction_results}")
            historical_saved = self._save_predictions_to_historical(prediction_results)
            if historical_saved:
                logger.info(f"✅ Saved {historical_saved} predictions to historical_wqi_data")
            else:
                logger.warning("⚠️ No predictions saved to historical_wqi_data")
            
            return True
            
        except Exception as e:
            logger.error(f"Error saving predictions to database: {e}")
            return False

    def _save_predictions_to_historical(self, prediction_results):
        """Lưu predictions vào bảng historical_wqi_data cho dashboard"""
        try:
            conn = self.db_manager.get_connection()
            if not conn:
                logger.error("Cannot connect to database for historical save")
                return 0
            
            cur = conn.cursor()
            
            # Tạo bảng historical_wqi_data nếu chưa có
            cur.execute("""
                CREATE TABLE IF NOT EXISTS historical_wqi_data (
                    id SERIAL PRIMARY KEY,
                    station_id INTEGER NOT NULL,
                    measurement_date TIMESTAMP NOT NULL,
                    wqi DECIMAL(10,2),
                    ph DECIMAL(5,2),
                    temperature DECIMAL(5,2),
                    "do" DECIMAL(5,2),
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(station_id, measurement_date)
                )
            """)
            
            saved_count = 0
            for station_id, result in prediction_results.items():
                if result.get('success', False):
                    future_predictions = result.get('future_predictions', {})
                    
                    for horizon_key, prediction in future_predictions.items():
                        try:
                            prediction_date = prediction.get('prediction_time')
                            wqi_prediction = prediction.get('wqi_prediction', 0)
                            
                            if prediction_date:
                                # Lưu vào historical_wqi_data với dummy values cho ph, temperature, do
                                cur.execute("""
                                    INSERT INTO historical_wqi_data 
                                    (station_id, measurement_date, wqi, ph, temperature, "do")
                                    VALUES (%s, %s, %s, %s, %s, %s)
                                    ON CONFLICT (station_id, measurement_date) 
                                    DO UPDATE SET 
                                        wqi = EXCLUDED.wqi,
                                        ph = EXCLUDED.ph,
                                        temperature = EXCLUDED.temperature,
                                        "do" = EXCLUDED."do"
                                """, (
                                    station_id,
                                    prediction_date,
                                    wqi_prediction,
                                    7.0,  # Default pH
                                    25.0, # Default temperature
                                    8.0   # Default DO
                                ))
                                saved_count += 1
                                
                        except Exception as e:
                            logger.error(f"Error saving to historical for station {station_id}, horizon {horizon_key}: {e}")
            
            conn.commit()
            cur.close()
            conn.close()
            
            return saved_count
            
        except Exception as e:
            logger.error(f"Error saving predictions to historical: {e}")
            return 0

    def make_time_series_predictions(self, station_id, historical_data, model_type='rf', horizons=[1, 3, 6, 12]):
        """Thực hiện time-series predictions cho một station"""
        try:
            logger.info(f"Making time-series predictions for station {station_id} using {model_type} model")
            
            # Load best model
            model = self._load_model(model_type)
            if not model:
                logger.error(f"Failed to load {model_type} model")
                return None
            
            # Prepare historical data
            if not historical_data or len(historical_data) < 12:
                logger.warning(f"Insufficient historical data for station {station_id}")
                return None
            
            # Convert historical data to features
            features = self._prepare_features(historical_data)
            if not features:
                logger.error(f"Failed to prepare features for station {station_id}")
                return None
            
            # Make predictions for each horizon
            predictions = {}
            latest_time = historical_data[0][4] if historical_data else datetime.now()  # measurement_time
            
            for horizon_months in horizons:
                try:
                    # Predict WQI for this horizon
                    wqi_prediction = self._predict_wqi(model, features, horizon_months)
                    confidence_score = self._calculate_confidence(features)
                    
                    # Calculate prediction time
                    prediction_time = latest_time + relativedelta(months=horizon_months)
                    
                    predictions[f'{horizon_months}month'] = {
                        'wqi_prediction': round(wqi_prediction, 2),
                        'confidence_score': round(confidence_score, 3),
                        'prediction_time': prediction_time,
                        'model_type': model_type,
                        'horizon_months': horizon_months
                    }
                    
                    logger.info(f"Station {station_id}: {horizon_months} month(s) ahead WQI = {wqi_prediction:.2f}, Confidence = {confidence_score:.3f}")
                    
                except Exception as e:
                    logger.error(f"Error predicting {horizon_months} months ahead for station {station_id}: {e}")
            
            if predictions:
                result = {
                    'station_id': station_id,
                    'success': True,
                    'model_type': model_type,
                    'current_time': latest_time,
                    'future_predictions': predictions,
                    'prediction_horizons': horizons
                }
                
                # Save to database
                self.save_prediction_results({station_id: result})
                
                return result
            else:
                logger.warning(f"No predictions generated for station {station_id}")
                return None
                
        except Exception as e:
            logger.error(f"Error making time-series predictions for station {station_id}: {e}")
            return None

    def _load_model(self, model_type):
        """Load trained model from MLflow registry or files"""
        try:
            import mlflow
            import mlflow.sklearn
            import mlflow.pyspark.ml
            import pickle
            import os
            
            # Try MLflow first
            try:
                mlflow.set_tracking_uri("http://77.37.44.237:5003")
                
                if model_type == 'xgb':
                    # Load best model from MLflow registry
                    model = mlflow.sklearn.load_model("models:/water_quality_best_model/Production")
                    logger.info("✅ Loaded best model from MLflow registry")
                    return model
                elif model_type == 'scaler':
                    # Load scaler from MLflow registry
                    scaler = mlflow.sklearn.load_model("models:/water_quality_scaler/Production")
                    logger.info("✅ Loaded scaler from MLflow registry")
                    return scaler
                elif model_type == 'rf':
                    # Spark RF pipeline not available in MLflow (PySpark dependency)
                    logger.warning("⚠️ Spark RF pipeline not available in MLflow, trying files")
                    return None
                else:
                    logger.warning(f"Model type {model_type} not available in MLflow, trying files")
                    
            except Exception as e:
                logger.warning(f"MLflow loading failed: {e}, trying files")
            
            # Fallback to file loading
            if model_type == 'xgb':
                # Try best model first, then XGBoost
                best_model_path = 'models/best_model.pkl'
                xgb_path = 'models/xgb.pkl'
                
                if os.path.exists(best_model_path):
                    with open(best_model_path, 'rb') as f:
                        model = pickle.load(f)
                    logger.info("✅ Loaded best model from files")
                    return model
                elif os.path.exists(xgb_path):
                    with open(xgb_path, 'rb') as f:
                        model = pickle.load(f)
                    logger.info("✅ Loaded XGBoost model from files")
                    return model
                else:
                    logger.error(f"❌ Best model not found at {best_model_path} or {xgb_path}")
                    return None
            elif model_type == 'scaler':
                scaler_path = 'models/scaler.pkl'
                if os.path.exists(scaler_path):
                    with open(scaler_path, 'rb') as f:
                        scaler = pickle.load(f)
                    logger.info("✅ Loaded scaler from files")
                    return scaler
                else:
                    logger.error(f"❌ Scaler not found at {scaler_path}")
                    return None
            elif model_type == 'rf':
                model_path = 'models/rf_pipeline'
                if os.path.exists(model_path):
                    from pyspark.ml import PipelineModel
                    model = PipelineModel.load(model_path)
                    logger.info("✅ Loaded Spark RF pipeline from files")
                    return model
                else:
                    logger.error(f"❌ Spark RF pipeline not found at {model_path}")
                    return None
            else:
                logger.error(f"❌ Unknown model type: {model_type}")
                return None
                
        except Exception as e:
            logger.error(f"❌ Error loading model {model_type}: {e}")
            return None

    def _prepare_features(self, historical_data):
        """Prepare features from historical data"""
        try:
            # Convert historical data to DataFrame
            df = pd.DataFrame(historical_data, columns=['ph', 'temperature', 'do', 'wqi', 'measurement_time'])
            
            # Calculate time-series features
            features = []
            for i in range(len(df)):
                if i >= 11:  # Need at least 12 months of data
                    # Recent features (last 12 months)
                    recent_data = df.iloc[i-11:i+1]
                    
                    feature_vector = [
                        recent_data['ph'].mean(),
                        recent_data['ph'].std(),
                        recent_data['temperature'].mean(),
                        recent_data['temperature'].std(),
                        recent_data['do'].mean(),
                        recent_data['do'].std(),
                        recent_data['wqi'].mean(),
                        recent_data['wqi'].std(),
                        recent_data['wqi'].iloc[-1],  # Latest WQI
                        i  # Time index
                    ]
                    features.append(feature_vector)
            
            if features:
                return np.array(features)
            else:
                return None
                
        except Exception as e:
            logger.error(f"Error preparing features: {e}")
            return None

    def _predict_wqi(self, model, features, horizon_months):
        """Predict WQI using loaded model"""
        try:
            if len(features) == 0:
                return 50.0  # Default value
            
            # Use the most recent feature vector
            latest_features = features[-1]
            
            # Make prediction
            prediction = model.predict([latest_features])[0]
            
            # Ensure prediction is within valid range (0-100)
            prediction = max(0, min(100, prediction))
            
            return prediction
            
        except Exception as e:
            logger.error(f"Error predicting WQI: {e}")
            return 50.0  # Default value

    def _calculate_confidence(self, features):
        """Calculate confidence score based on data stability"""
        try:
            if len(features) == 0:
                return 0.5  # Default confidence
            
            # Calculate confidence based on feature stability
            wqi_values = [f[6] for f in features]  # WQI mean values
            wqi_std = np.std(wqi_values)
            
            # Higher stability (lower std) = higher confidence
            confidence = max(0.3, min(0.95, 1.0 - (wqi_std / 50.0)))
            
            return confidence
            
        except Exception as e:
            logger.error(f"Error calculating confidence: {e}")
            return 0.5  # Default confidence
    
    def get_recent_predictions(self, station_id=None, days=30):
        """Get recent predictions for dashboard"""
        try:
            if not self.db_manager:
                logger.error("Database manager not available")
                return []
            
            conn = self.db_manager.get_connection()
            if not conn:
                logger.error("Cannot connect to database")
                return []
            
            cur = conn.cursor()
            
            # Get recent predictions
            if station_id:
                cur.execute("""
                    SELECT station_id, prediction_time, prediction_horizon_months,
                           wqi_prediction, confidence_score, model_type
                    FROM wqi_predictions
                    WHERE station_id = %s AND prediction_time >= %s
                    ORDER BY prediction_time DESC, prediction_horizon_months ASC
                """, (station_id, datetime.now() - timedelta(days=days)))
            else:
                cur.execute("""
                    SELECT station_id, prediction_time, prediction_horizon_months,
                           wqi_prediction, confidence_score, model_type
                    FROM wqi_predictions
                    WHERE prediction_time >= %s
                    ORDER BY prediction_time DESC, prediction_horizon_months ASC
                """, (datetime.now() - timedelta(days=days),))
            
            predictions = cur.fetchall()
            cur.close()
            conn.close()
            
            # Convert to list of dictionaries
            prediction_list = []
            for row in predictions:
                prediction_list.append({
                    'station_id': row[0],
                    'prediction_time': row[1].isoformat() if row[1] else None,
                    'horizon_months': row[2],
                    'wqi_prediction': float(row[3]) if row[3] else 0,
                    'confidence_score': float(row[4]) if row[4] else 0,
                    'model_type': row[5]
                })
            
            logger.info(f"Retrieved {len(prediction_list)} recent predictions")
            return prediction_list
            
        except Exception as e:
            logger.error(f"Error getting recent predictions: {e}")
            return []
    
    def get_prediction_summary(self):
        """Get prediction summary for dashboard"""
        try:
            if not self.db_manager:
                logger.error("Database manager not available")
                return {}
            
            conn = self.db_manager.get_connection()
            if not conn:
                logger.error("Cannot connect to database")
                return {}
            
            cur = conn.cursor()
            
            # Get summary statistics
            cur.execute("""
                SELECT 
                    COUNT(*) as total_predictions,
                    COUNT(DISTINCT station_id) as total_stations,
                    AVG(wqi_prediction) as avg_wqi,
                    AVG(confidence_score) as avg_confidence,
                    model_type,
                    prediction_horizon_months
                FROM wqi_predictions
                WHERE prediction_time >= %s
                GROUP BY model_type, prediction_horizon_months
                ORDER BY model_type, prediction_horizon_months
            """, (datetime.now() - timedelta(days=30),))
            
            summary_data = cur.fetchall()
            cur.close()
            conn.close()
            
            # Convert to summary dictionary
            summary = {
                'total_predictions': 0,
                'total_stations': 0,
                'avg_wqi': 0,
                'avg_confidence': 0,
                'by_model_type': {},
                'by_horizon': {}
            }
            
            for row in summary_data:
                total_pred, total_stations, avg_wqi, avg_conf, model_type, horizon = row
                summary['total_predictions'] += total_pred
                summary['total_stations'] = max(summary['total_stations'], total_stations)
                summary['avg_wqi'] = float(avg_wqi) if avg_wqi else 0
                summary['avg_confidence'] = float(avg_conf) if avg_conf else 0
                
                # Group by model type
                if model_type not in summary['by_model_type']:
                    summary['by_model_type'][model_type] = {
                        'total_predictions': 0,
                        'avg_wqi': 0,
                        'avg_confidence': 0
                    }
                summary['by_model_type'][model_type]['total_predictions'] += total_pred
                summary['by_model_type'][model_type]['avg_wqi'] = float(avg_wqi) if avg_wqi else 0
                summary['by_model_type'][model_type]['avg_confidence'] = float(avg_conf) if avg_conf else 0
                
                # Group by horizon
                if horizon not in summary['by_horizon']:
                    summary['by_horizon'][horizon] = {
                        'total_predictions': 0,
                        'avg_wqi': 0,
                        'avg_confidence': 0
                    }
                summary['by_horizon'][horizon]['total_predictions'] += total_pred
                summary['by_horizon'][horizon]['avg_wqi'] = float(avg_wqi) if avg_wqi else 0
                summary['by_horizon'][horizon]['avg_confidence'] = float(avg_conf) if avg_conf else 0
            
            logger.info(f"Generated prediction summary: {summary['total_predictions']} predictions")
            return summary
            
        except Exception as e:
            logger.error(f"Error getting prediction summary: {e}")
            return {}
    
    def cleanup_old_predictions(self, days=90):
        """Clean up old predictions"""
        try:
            if not self.db_manager:
                logger.error("Database manager not available")
                return False
            
            conn = self.db_manager.get_connection()
            if not conn:
                logger.error("Cannot connect to database")
                return False
            
            cur = conn.cursor()
            
            # Delete old predictions
            cur.execute("""
                DELETE FROM wqi_predictions
                WHERE prediction_time < %s
            """, (datetime.now() - timedelta(days=days),))
            
            deleted_count = cur.rowcount
            conn.commit()
            cur.close()
            conn.close()
            
            logger.info(f"✅ Cleaned up {deleted_count} old predictions")
            return True
            
        except Exception as e:
            logger.error(f"Error cleaning up old predictions: {e}")
            return False

    def process_stations_predictions(self, stations_with_data):
        """Process predictions for multiple stations - business logic moved from DAG"""
        logger.info(f"Processing predictions for {len(stations_with_data)} stations")
        
        try:
            if not stations_with_data:
                logger.warning("No stations with data found for predictions")
                return []
            
            logger.info(f"Found {len(stations_with_data)} stations with data")
            
            # Determine best model from metrics
            best_model_type = self._determine_best_model()
            
            # Process predictions for each station
            prediction_results = []
            successful_predictions = 0
            
            for station_id in stations_with_data:
                try:
                    # Get historical data for station
                    historical_data = self.db_manager.get_station_historical_data(station_id, limit=48)
                    
                    if historical_data and len(historical_data) >= 1:
                        data_count = len(historical_data)
                        
                        if data_count >= 12:
                            # Enough data for real ML predictions
                            horizons = [1, 3, 6, 12]
                            predictions = self.make_time_series_predictions(
                                station_id=station_id,
                                historical_data=historical_data,
                                model_type=best_model_type,
                                horizons=horizons
                            )
                        elif data_count >= 3:
                            # 3-11 records: predict 3 months
                            horizons = [1, 2, 3]
                            predictions = self._create_simple_predictions(station_id, historical_data, best_model_type, horizons)
                        else:
                            # 1-2 records: predict 3 months for testing
                            horizons = [1, 2, 3]
                            predictions = self._create_simple_predictions(station_id, historical_data, best_model_type, horizons)
                        
                        if predictions:
                            prediction_results.append(predictions)
                            successful_predictions += 1
                            logger.info(f"✅ Predictions created for station {station_id} ({data_count} records, {len(horizons)} months)")
                        else:
                            logger.warning(f"⚠️ Failed predictions for station {station_id}")
                    else:
                        logger.warning(f"⚠️ No historical data for station {station_id}")
                        
                except Exception as e:
                    logger.error(f"Error predicting for station {station_id}: {e}")
            
            logger.info(f"Completed predictions: {successful_predictions}/{len(stations_with_data)} successful using {best_model_type} model")
            return prediction_results
            
        except Exception as e:
            logger.error(f"Error processing stations predictions: {e}")
            return []

    def _determine_best_model(self):
        """Determine best model from MLflow registry or files"""
        try:
            import mlflow
            import os
            
            # Connect to MLflow
            mlflow_tracking_uri = os.getenv('MLFLOW_TRACKING_URI', 'http://77.37.44.237:5003')
            mlflow.set_tracking_uri(mlflow_tracking_uri)
            
            try:
                client = mlflow.tracking.MlflowClient()
                
                # Look for the best model in experiments
                best_model_type = 'xgb'  # default
                best_score = 0.0
                
                experiments = client.list_experiments()
                logger.info(f"Found {len(experiments)} experiments in MLflow")
                
                # Look for experiments with water quality models
                valid_experiments = []
                for exp in experiments:
                    if exp.name in ["water_quality_models", "water_quality_predictions"] or exp.name.startswith("water_quality_models_"):
                        valid_experiments.append(exp)
                        logger.info(f"Found valid experiment: {exp.name}")
                
                if not valid_experiments:
                    logger.warning("No water quality experiments found in MLflow")
                    return self._determine_best_model_from_files()
                
                # Check all valid experiments for the best model
                best_model_type = 'xgb'  # default
                best_score = 0.0
                best_experiment = None
                
                for exp in valid_experiments:
                    try:
                        # Get the latest run with best metrics
                        runs = client.search_runs(
                            exp.experiment_id, 
                            order_by=["metrics.r2 DESC"], 
                            max_results=1
                        )
                        
                        if runs:
                            run = runs[0]
                            metrics = run.data.metrics
                            
                            # Compare model performances (only XGBoost and Ensemble)
                            xgb_r2 = metrics.get('xgb_r2', 0.0)
                            ensemble_r2 = metrics.get('ensemble_r2', 0.0)
                            
                            model_scores = {
                                'xgb': xgb_r2,
                                'ensemble': ensemble_r2
                            }
                            
                            current_best = max(model_scores, key=model_scores.get)
                            current_score = model_scores[current_best]
                            
                            logger.info(f"📊 MLflow Model Performance in {exp.name}:")
                            logger.info(f"  XGBoost: R² = {xgb_r2:.4f}")
                            logger.info(f"  Ensemble: R² = {ensemble_r2:.4f}")
                            
                            # Update best if this experiment has better scores
                            if current_score > best_score:
                                best_model_type = current_best
                                best_score = current_score
                                best_experiment = exp.name
                                
                    except Exception as exp_error:
                        logger.warning(f"Error checking experiment {exp.name}: {exp_error}")
                        continue
                
                if best_experiment:
                    logger.info(f"🏆 Best model from MLflow: {best_model_type.upper()} (R²: {best_score:.4f}) from experiment: {best_experiment}")
                    return best_model_type
                
                # Fallback to file-based check if MLflow fails
                logger.warning("No models found in MLflow, falling back to file-based check")
                return self._determine_best_model_from_files()
                
            except Exception as e:
                logger.warning(f"MLflow connection failed: {e}, falling back to file-based check")
                return self._determine_best_model_from_files()
                
        except Exception as e:
            logger.error(f"Error determining best model: {e}")
            return 'xgb'

    def _determine_best_model_from_files(self):
        """Fallback method to determine best model from files"""
        try:
            import json
            import os
            
            # Check for new comprehensive metrics first
            metrics_file = './models/metrics.json'
            metadata_file = './models/enhanced_metadata.json'
            
            if os.path.exists(metrics_file):
                with open(metrics_file, 'r') as f:
                    metrics = json.load(f)
                logger.info("📊 Using comprehensive ensemble metrics (file-based)")
            elif os.path.exists(metadata_file):
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
                metrics = metadata['metrics']
                logger.info("📊 Using enhanced metadata (file-based fallback)")
            else:
                logger.warning("No metrics files found, using default XGBoost model")
                return 'xgb'
            
            # Compare all models and select the best one (only XGBoost and Ensemble)
            xgb_r2 = metrics.get('xgb', {}).get('r2', 0.0)
            ensemble_r2 = metrics.get('ensemble', {}).get('r2', 0.0)
            
            model_scores = {
                'xgb': xgb_r2,
                'ensemble': ensemble_r2
            }
            
            best_model_type = max(model_scores, key=model_scores.get)
            best_score = model_scores[best_model_type]
            
            logger.info(f"📊 File-based Model Performance:")
            logger.info(f"  XGBoost: R² = {xgb_r2:.4f}")
            logger.info(f"  Ensemble: R² = {ensemble_r2:.4f}")
            logger.info(f"🏆 Best model from files: {best_model_type.upper()} (R²: {best_score:.4f})")
            
            return best_model_type
            
        except Exception as e:
            logger.error(f"Error determining best model from files: {e}")
            return 'xgb'

    def _create_simple_predictions(self, station_id, historical_data, model_type, horizons):
        """Create simple predictions based on data count"""
        try:
            # Get latest data - format: (ph, temperature, do, measurement_time)
            latest_record = historical_data[0]
            ph = latest_record[0]
            temperature = latest_record[1]
            do_val = latest_record[2]
            latest_time = latest_record[3]  # measurement_time
            
            # Calculate WQI from raw data
            latest_wqi = self._calculate_wqi_simple(ph, temperature, do_val)
            
            # Create predictions for each horizon
            predictions = {}
            
            for horizon_months in horizons:
                # Create prediction time
                prediction_time = latest_time + relativedelta(months=horizon_months)
                
                # Create WQI prediction based on latest WQI + trend
                import random
                data_count = len(historical_data)
                
                if data_count >= 3:
                    # Has trend data - calculate trend from 3 recent records
                    recent_wqi = []
                    for record in historical_data[:3]:
                        ph, temp, do_val = record[0], record[1], record[2]
                        wqi = self._calculate_wqi_simple(ph, temp, do_val)
                        recent_wqi.append(wqi)
                    
                    trend = (recent_wqi[0] - recent_wqi[-1]) / len(recent_wqi)  # Trend per record
                    variation = trend * horizon_months + random.uniform(-2, 2)
                else:
                    # Only 1-2 records - random variation
                    variation = random.uniform(-3, 3)
                
                wqi_prediction = max(0, min(100, latest_wqi + variation))
                
                # Confidence score based on data count
                confidence_score = min(0.9, 0.4 + (data_count * 0.1))
                
                predictions[f'{horizon_months}month'] = {
                    'wqi_prediction': round(wqi_prediction, 2),
                    'confidence_score': round(confidence_score, 3),
                    'prediction_time': prediction_time,
                    'model_type': f'{model_type}_simple',
                    'horizon_months': horizon_months
                }
            
            result = {
                'station_id': station_id,
                'success': True,
                'model_type': f'{model_type}_simple',
                'current_time': latest_time,  # Use current time instead of latest_time
                'future_predictions': predictions,
                'prediction_horizons': horizons,
                'note': f'Simple predictions based on {data_count} record(s)'
            }
            
            # Save to database
            self.save_prediction_results({station_id: result})
            
            return result
            
        except Exception as e:
            logger.error(f"Error creating simple predictions for station {station_id}: {e}")
            return None

    def _calculate_wqi_simple(self, ph, temperature, do):
        """Tính WQI đơn giản từ ph, temperature, do"""
        try:
            # Convert to float
            ph = float(ph) if ph is not None else 7.0
            temperature = float(temperature) if temperature is not None else 25.0
            do = float(do) if do is not None else 8.0
            
            # pH sub-index (optimal: 7.0)
            if ph <= 7.0:
                ph_subindex = 100 - (7.0 - ph) * 20
            else:
                ph_subindex = 100 - (ph - 7.0) * 20
            ph_subindex = max(0, min(100, ph_subindex))
            
            # Temperature sub-index (optimal: 20-25°C)
            if 20 <= temperature <= 25:
                temp_subindex = 100
            elif temperature < 20:
                temp_subindex = 100 - (20 - temperature) * 5
            else:
                temp_subindex = 100 - (temperature - 25) * 5
            temp_subindex = max(0, min(100, temp_subindex))
            
            # DO sub-index (optimal: >8 mg/L)
            if do >= 8:
                do_subindex = 100
            else:
                do_subindex = do * 12.5
            do_subindex = max(0, min(100, do_subindex))
            
            # Weighted average: pH (30%), Temperature (20%), DO (50%)
            wqi = (ph_subindex * 0.3) + (temp_subindex * 0.2) + (do_subindex * 0.5)
            
            return round(wqi, 2)
            
        except Exception as e:
            logger.error(f"Error calculating WQI: {e}")
            return 50.0  # Default value

    def generate_alerts(self, prediction_results):
        """Generate alerts based on prediction results"""
        try:
            if not prediction_results:
                return "No prediction results for alerts"
            
            # Create simple alerts
            successful_predictions = len([p for p in prediction_results if p.get('success')])
            total_predictions = len(prediction_results)
            
            alerts_generated = 0
            if successful_predictions < total_predictions:
                alerts_generated += 1
                logger.warning(f"⚠️ Alert: {total_predictions - successful_predictions} predictions failed")
            
            # Check for critical WQI values
            critical_alerts = 0
            for result in prediction_results:
                if result.get('success'):
                    future_predictions = result.get('future_predictions', {})
                    for horizon_key, prediction in future_predictions.items():
                        wqi_prediction = prediction.get('wqi_prediction', 0)
                        if wqi_prediction < 60 or wqi_prediction > 80:
                            critical_alerts += 1
                            logger.warning(f"⚠️ Critical WQI alert: Station {result['station_id']}, WQI={wqi_prediction}")
            
            total_alerts = alerts_generated + critical_alerts
            logger.info(f"Generated {total_alerts} alerts for {total_predictions} predictions")
            
            return f"Generated {total_alerts} alerts: {alerts_generated} failure alerts, {critical_alerts} critical WQI alerts"
            
        except Exception as e:
            logger.error(f"Error generating alerts: {e}")
            return f"Error generating alerts: {e}"

# Global instance
prediction_service = PredictionService() 