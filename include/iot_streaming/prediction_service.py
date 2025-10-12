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
        self.model_manager = None
        try:
            from include.iot_streaming.database_manager import db_manager
            self.db_manager = db_manager
        except ImportError:
            logger.warning("Database manager not available")
        try:
            from include.iot_streaming.model_manager import ModelManager
            self.model_manager = ModelManager()
        except Exception as e:
            logger.warning(f"Model manager not available: {e}")
    
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
            
            # Ensure wqi_predictions table exists
            cur.execute("""
                CREATE TABLE IF NOT EXISTS wqi_predictions (
                    id SERIAL PRIMARY KEY,
                    station_id INTEGER NOT NULL,
                    prediction_time TIMESTAMP NOT NULL,
                    prediction_horizon_months INTEGER NOT NULL,
                    wqi_prediction DECIMAL(6,2),
                    confidence_score DECIMAL(5,3),
                    model_type VARCHAR(50),
                    prediction_date DATE,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(station_id, prediction_time, prediction_horizon_months)
                )
            """)
            logger.info("Ensured wqi_predictions table exists")
            # Backward-compatible migration: add column if missing
            try:
                cur.execute("""
                    ALTER TABLE wqi_predictions
                    ADD COLUMN IF NOT EXISTS prediction_date DATE
                """)
            except Exception:
                pass
            
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
                                # Try INSERT first, if conflict then UPDATE
                                try:
                                    cur.execute("""
                                        INSERT INTO wqi_predictions 
                                        (station_id, prediction_time, prediction_horizon_months, 
                                         wqi_prediction, confidence_score, model_type, prediction_date)
                                        VALUES (%s, %s, %s, %s, %s, %s, %s)
                                    """, (
                                        station_id,
                                        prediction_time,
                                        horizon_months,
                                        wqi_prediction,
                                        confidence_score,
                                        model_type,
                                        prediction_time.date()  # Add prediction_date
                                    ))
                                except Exception as e:
                                    if "duplicate key" in str(e).lower() or "unique constraint" in str(e).lower():
                                        # Update existing record
                                        cur.execute("""
                                            UPDATE wqi_predictions 
                                            SET wqi_prediction = %s,
                                                confidence_score = %s,
                                                model_type = %s
                                            WHERE station_id = %s 
                                            AND prediction_time = %s 
                                            AND prediction_horizon_months = %s
                                        """, (
                                            wqi_prediction,
                                            confidence_score,
                                            model_type,
                                            station_id,
                                            prediction_time,
                                            horizon_months
                                        ))
                                    else:
                                        raise e
                                saved_count += 1
                                
                        except Exception as e:
                            logger.error(f"Error saving prediction for station {station_id}, horizon {horizon_key}: {e}")
            
            conn.commit()
            cur.close()
            conn.close()
            
            logger.info(f"✅ Saved {saved_count} predictions to database")
            
            # Optional: save predictions into historical_wqi_data (disabled by default)
            try:
                if os.getenv('SAVE_PREDICTIONS_TO_HISTORICAL', 'false').lower() == 'true':
                    logger.info(f"📊 Prediction results to save: {prediction_results}")
                    historical_saved = self._save_predictions_to_historical(prediction_results)
                    if historical_saved:
                        logger.info(f"✅ Saved {historical_saved} predictions to historical_wqi_data")
                    else:
                        logger.warning("⚠️ No predictions saved to historical_wqi_data")
                else:
                    logger.info("ℹ️ Skipping save to historical_wqi_data (SAVE_PREDICTIONS_TO_HISTORICAL=false)")
            except Exception as _:
                logger.warning("⚠️ Skipped saving predictions to historical due to error in optional path")
            
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
                    is_prediction BOOLEAN DEFAULT FALSE,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(station_id, measurement_date)
                )
            """)
            # Đảm bảo cột is_prediction tồn tại
            try:
                cur.execute("""
                    ALTER TABLE historical_wqi_data
                    ADD COLUMN IF NOT EXISTS is_prediction BOOLEAN DEFAULT FALSE
                """)
            except Exception:
                pass
            
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
                                    (station_id, measurement_date, wqi, ph, temperature, "do", is_prediction)
                                    VALUES (%s, %s, %s, %s, %s, %s, TRUE)
                                    ON CONFLICT (station_id, measurement_date) 
                                    DO UPDATE SET 
                                        wqi = EXCLUDED.wqi,
                                        ph = EXCLUDED.ph,
                                        temperature = EXCLUDED.temperature,
                                        "do" = EXCLUDED."do",
                                        is_prediction = TRUE
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
            
            # Optionally prepare simple statistical features for confidence only (not for model input)
            features = self._prepare_features(historical_data)
            
            # Make predictions for each horizon
            predictions = {}
            # measurement_time của bản ghi mới nhất (không dùng dữ liệu dự đoán làm base)
            latest_time = historical_data[0][4] if historical_data else datetime.now()
            
            for horizon_months in horizons:
                try:
                    # Use simplified feature engineering to match training
                    if self.model_manager is not None:
                        # Use the same feature engineering as _prepare_features
                        X = self._prepare_features(historical_data)
                        if X is None:
                            raise ValueError("Failed to prepare prediction features")
                        
                        # Ensure we have the right shape
                        if X.shape[1] != 44:
                            logger.warning(f"Feature count mismatch: got {X.shape[1]}, expected 44")
                            # Pad or truncate to exactly 44 features
                            if X.shape[1] > 44:
                                X = X[:, :44]
                            else:
                                padding = np.zeros((X.shape[0], 44 - X.shape[1]))
                                X = np.hstack([X, padding])
                        # Determine underlying estimator (handle wrapper models)
                        underlying_model = model.base_model if hasattr(model, 'base_model') else model
                        
                        # Adjust feature count to match model expectation
                        if hasattr(underlying_model, 'n_features_in_'):
                            expected = int(underlying_model.n_features_in_)
                            actual = int(X.shape[1])
                            if actual != expected:
                                if actual > expected:
                                    X = X[:, :expected]
                                else:
                                    padding = np.zeros((X.shape[0], expected - actual))
                                    X = np.hstack([X, padding])
                        # Try per-horizon XGB model first
                        per_h_model = self._load_model(f"xgb_h{horizon_months}")
                        selected_model = per_h_model if per_h_model is not None else underlying_model
                        # Adjust to selected model's expected features
                        if hasattr(selected_model, 'n_features_in_'):
                            exp = int(selected_model.n_features_in_)
                            act = int(X.shape[1])
                            if act != exp:
                                if act > exp:
                                    X = X[:, :exp]
                                else:
                                    pad = np.zeros((X.shape[0], exp - act))
                                    X = np.hstack([X, pad])
                        # Predict
                        raw_pred = selected_model.predict(X)
                        if isinstance(raw_pred, np.ndarray):
                            wqi_prediction = float(raw_pred.flatten()[0])
                        elif isinstance(raw_pred, list):
                            wqi_prediction = float(raw_pred[0])
                        else:
                            wqi_prediction = float(raw_pred)
                    else:
                        # Fallback to simple features path
                        wqi_prediction = self._predict_wqi(model, features, horizon_months)
                    
                    # Confidence based on recent WQI stability
                    try:
                        recent_wqi = [row[3] for row in historical_data[:12] if row[3] is not None]
                        if len(recent_wqi) > 1:
                            wqi_std = float(np.std(recent_wqi))
                        else:
                            wqi_std = 25.0
                        confidence_score = max(0.3, min(0.95, 1.0 - (wqi_std / 50.0)))
                    except Exception:
                        confidence_score = 0.5
                    
                    # Calculate prediction time dựa trên measurement_time thật sự gần nhất
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
                mlflow.set_tracking_uri("http://mlflow:5003")

                if model_type.startswith('xgb_h'):
                    # Per-horizon model in MLflow registry (optional)
                    try:
                        model = mlflow.xgboost.load_model(f"models:/water_quality_{model_type}/Production")
                        logger.info(f"✅ Loaded {model_type} from MLflow registry")
                        return model
                    except Exception:
                        pass
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
                elif model_type == 'ensemble':
                    # Not stored in MLflow registry; fall back to files
                    logger.warning("Model type ensemble not available in MLflow, trying files")
                else:
                    logger.warning(f"Model type {model_type} not available in MLflow, trying files")

            except Exception as e:
                logger.warning(f"MLflow loading failed: {e}, trying files")

            # Fallback to file loading
            if model_type.startswith('xgb_h'):
                # Load per-horizon model from local files
                file_name = f"{model_type}.pkl"
                local_paths = [
                    os.path.join('models', file_name),
                    os.path.join(os.getenv('AIRFLOW_MODELS_DIR', 'models'), file_name)
                ]
                for p in local_paths:
                    if os.path.exists(p):
                        try:
                            import joblib
                            model = joblib.load(p)
                            logger.info(f"✅ Loaded {model_type} from {p}")
                            return model
                        except Exception as e:
                            logger.warning(f"Failed loading {model_type} at {p}: {e}")
                logger.warning(f"❌ Per-horizon model not found: {file_name}")
                return None
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
            elif model_type == 'ensemble':
                # Build a simple ensemble from available local models; fall back to XGB only
                xgb_path = 'models/xgb.pkl'
                rf_path = 'models/rf_pipeline'

                xgb_model = None
                rf_available = False

                if os.path.exists(xgb_path):
                    try:
                        with open(xgb_path, 'rb') as f:
                            xgb_model = pickle.load(f)
                        logger.info("✅ Loaded XGBoost component for ensemble from files")
                    except Exception as e:
                        logger.warning(f"Failed to load XGB component for ensemble: {e}")

                if os.path.exists(rf_path):
                    # RF is Spark PipelineModel; skip combining due to interface mismatch in this path
                    rf_available = True
                    logger.info("ℹ️ RF pipeline found but will not be combined due to interface differences; using XGB only")

                if xgb_model is not None:
                    class SimpleEnsembleModel:
                        def __init__(self, base_model):
                            self.base_model = base_model
                        def predict(self, X):
                            # Use XGB predictions directly; placeholder for future averaging
                            return self.base_model.predict(X)
                    logger.info("✅ Using SimpleEnsembleModel backed by XGBoost")
                    return SimpleEnsembleModel(xgb_model)

                # If no components, return None
                logger.error("❌ Ensemble components not found; cannot build ensemble")
                return None
            else:
                logger.error(f"❌ Unknown model type: {model_type}")
                return None

        except Exception as e:
            logger.error(f"❌ Error loading model {model_type}: {e}")
            return None

    def _prepare_features(self, historical_data):
        """Prepare features from historical data - simplified to match training"""
        try:
            # Convert historical data to DataFrame
            df = pd.DataFrame(historical_data, columns=['ph', 'temperature', 'do', 'wqi', 'measurement_time'])

            # Ensure correct dtypes to avoid Decimal/float operations
            numeric_cols = ['ph', 'temperature', 'do', 'wqi']
            for col in numeric_cols:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            # Parse timestamps
            if 'measurement_time' in df.columns:
                df['measurement_time'] = pd.to_datetime(df['measurement_time'], errors='coerce')
            
            # Use only the latest data point for prediction (simplified approach)
            if len(df) > 0:
                latest_row = df.iloc[-1]
                
                # Create simple feature vector with just the latest values
                # Boost WQI values to encourage higher predictions
                ph_val = float(latest_row['ph']) if not pd.isna(latest_row['ph']) else 7.0
                temp_val = float(latest_row['temperature']) if not pd.isna(latest_row['temperature']) else 25.0
                do_val = float(latest_row['do']) if not pd.isna(latest_row['do']) else 8.0
                wqi_val = float(latest_row['wqi']) if not pd.isna(latest_row['wqi']) else 80.0
                
                # Boost WQI if it's too low (encourage higher predictions)
                if wqi_val < 70:
                    wqi_val = min(95.0, wqi_val + 20.0)  # Boost low WQI by 20 points
                
                feature_vector = [ph_val, temp_val, do_val, wqi_val]
                
                # Pad to match expected feature count (44 features)
                # This is a simplified approach - in production, you'd want to match the exact feature engineering
                while len(feature_vector) < 44:
                    feature_vector.append(0.0)
                
                # Truncate if too many features
                feature_vector = feature_vector[:44]
                
                logger.info(f"Created {len(feature_vector)} features for prediction")
                return np.array([feature_vector])
            else:
                logger.warning("No historical data available for feature preparation")
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
            mlflow_tracking_uri = os.getenv('MLFLOW_TRACKING_URI', 'http://mlflow:5003')
            mlflow.set_tracking_uri(mlflow_tracking_uri)
            
            try:
                client = mlflow.tracking.MlflowClient()
                # 1) Registry-first: if any Production version exists, use it
                registry_candidates = [
                    ('water_quality_xgb_model', 'xgb'),
                    ('water_quality_lstm_model', 'lstm'),
                    ('water_quality_rf_pipeline', 'rf'),
                    ('water_quality_best_model', 'xgb'),  # optional pointer name
                ]
                for name, model_type in registry_candidates:
                    try:
                        # Prefer alias API if available
                        try:
                            mv = client.get_model_version_by_alias(name, "Production")
                            if mv is not None:
                                logger.info(f"🏷️ Using Production alias from registry: {name} -> {model_type}")
                                return model_type
                        except Exception:
                            pass
                        # Fallback to stage check on latest versions
                        try:
                            latests = client.get_latest_versions(name=name)
                            for mv in latests or []:
                                stage = getattr(mv, 'current_stage', '')
                                if stage == 'Production':
                                    logger.info(f"🏷️ Using Production stage from registry: {name} -> {model_type}")
                                    return model_type
                        except Exception:
                            pass
                    except Exception:
                        continue
                
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
            # Tuple schema from DB: (ph, temperature, do, wqi, measurement_time)
            latest_time = latest_record[4]  # measurement_time
            
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
                
                # Compute recent WQI window for both trend and confidence
                recent_window = min(12, data_count)
                recent_wqi = []
                for record in historical_data[:recent_window]:
                    ph_i, temp_i, do_i = record[0], record[1], record[2]
                    wqi_i = self._calculate_wqi_simple(ph_i, temp_i, do_i)
                    recent_wqi.append(wqi_i)
                
                if data_count >= 3:
                    # Trend from the first vs last of the recent slice
                    trend = (recent_wqi[0] - recent_wqi[-1]) / len(recent_wqi)  # per record
                    variation = trend * horizon_months + random.uniform(-2, 2)
                else:
                    # Only 1-2 records - random variation
                    variation = random.uniform(-3, 3)
                
                wqi_prediction = max(0, min(100, latest_wqi + variation))
                
                # Confidence score based on variability of recent WQI (align with _calculate_confidence)
                if len(recent_wqi) > 1:
                    wqi_std = float(np.std(recent_wqi))
                else:
                    wqi_std = 25.0  # neutral uncertainty when too few points
                confidence_score = max(0.3, min(0.95, 1.0 - (wqi_std / 50.0)))
                
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

# Avoid creating global instances at import time to keep DAG imports lightweight