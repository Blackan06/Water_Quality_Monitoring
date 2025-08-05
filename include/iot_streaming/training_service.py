"""
Training Service for Water Quality Monitoring
Handles all training logic and data processing
"""

import logging
import pandas as pd
import numpy as np
import os
from datetime import datetime
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)

class TrainingService:
    def __init__(self):
        self.model_manager = None
        self._init_model_manager()
    
    def _init_model_manager(self):
        """Initialize model manager"""
        try:
            from .model_manager import model_manager
            self.model_manager = model_manager
            logger.info("✅ Model manager initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize model manager: {e}")
            self.model_manager = None
    
    def load_historical_data(self) -> Optional[pd.DataFrame]:
        """Load historical data from CSV file"""
        try:
            csv_path = 'data/WQI_data.csv'
            if not os.path.exists(csv_path):
                logger.error(f"Historical data file not found: {csv_path}")
                return None
            
            # Load CSV data
            df = pd.read_csv(csv_path)
            logger.info(f"Loaded historical data: {len(df)} records from {csv_path}")
            
            # Convert Date column to datetime
            df['Date'] = pd.to_datetime(df['Date'])
            df['timestamp'] = df['Date']  # Add timestamp column for consistency
            
            # Rename columns to match expected format
            df = df.rename(columns={
                'PH': 'ph',
                'DO': 'do',
                'Temperature': 'temperature',
                'WQI': 'wqi'
            })
            
            # Verify data quality
            logger.info(f"Data columns: {list(df.columns)}")
            logger.info(f"Date range: {df['Date'].min()} to {df['Date'].max()}")
            logger.info(f"Stations: {sorted(df['station_id'].unique())}")
            logger.info(f"Records per station: {df.groupby('station_id').size().to_dict()}")
            
            # Check for missing values
            missing_data = df.isnull().sum()
            if missing_data.sum() > 0:
                logger.warning(f"Missing values found: {missing_data[missing_data > 0].to_dict()}")
            
            return df
            
        except Exception as e:
            logger.error(f"Error loading historical data: {e}")
            return None
    
    def create_enhanced_features(self, df: pd.DataFrame, station_id: int) -> Optional[pd.DataFrame]:
        """Create enhanced features with proper time-based engineering to avoid data leakage"""
        try:
            # Filter data for this station
            station_data = df[df['station_id'] == station_id].copy()
            station_data = station_data.sort_values('Date')
            
            # Basic temporal features (safe - no future info)
            station_data['month'] = station_data['Date'].dt.month
            station_data['quarter'] = station_data['Date'].dt.quarter
            station_data['year'] = station_data['Date'].dt.year
            station_data['day_of_year'] = station_data['Date'].dt.dayofyear
            
            # Cyclical encoding (safe)
            station_data['month_sin'] = np.sin(2 * np.pi * station_data['month'] / 12)
            station_data['month_cos'] = np.cos(2 * np.pi * station_data['month'] / 12)
            station_data['quarter_sin'] = np.sin(2 * np.pi * station_data['quarter'] / 4)
            station_data['quarter_cos'] = np.cos(2 * np.pi * station_data['quarter'] / 4)
            
            # Season indicators (safe)
            station_data['is_rainy_season'] = ((station_data['month'] >= 5) & (station_data['month'] <= 10)).astype(int)
            station_data['is_dry_season'] = ((station_data['month'] <= 4) | (station_data['month'] >= 11)).astype(int)
            
            # Lag features (safe - only past information)
            station_data['wqi_lag_1'] = station_data['wqi'].shift(1)
            station_data['wqi_lag_2'] = station_data['wqi'].shift(2)
            station_data['wqi_lag_3'] = station_data['wqi'].shift(3)
            
            # EXPANDING rolling statistics (safe - only past information)
            # Use expanding window instead of rolling to avoid future leakage
            station_data['wqi_expanding_mean'] = station_data['wqi'].expanding(min_periods=1).mean()
            station_data['wqi_expanding_std'] = station_data['wqi'].expanding(min_periods=1).std()
            
            # Limited rolling statistics with proper window (only past)
            # Use shift to ensure we only use past information
            station_data['wqi_rolling_mean_3'] = station_data['wqi'].shift(1).rolling(window=3, min_periods=1).mean()
            station_data['wqi_rolling_std_3'] = station_data['wqi'].shift(1).rolling(window=3, min_periods=1).std()
            
            # Interaction features (safe - current values only)
            station_data['ph_temp_interaction'] = station_data['ph'] * station_data['temperature']
            station_data['ph_do_interaction'] = station_data['ph'] * station_data['do']
            station_data['temp_do_interaction'] = station_data['temperature'] * station_data['do']
            
            # Delta features (safe - only past differences)
            station_data['delta_wqi'] = station_data['wqi'].diff()
            station_data['delta_ph'] = station_data['ph'].diff()
            station_data['delta_temperature'] = station_data['temperature'].diff()
            station_data['delta_do'] = station_data['do'].diff()
            
            # Anomaly detection using expanding statistics (safe)
            station_data['wqi_zscore'] = (station_data['wqi'] - station_data['wqi_expanding_mean']) / (station_data['wqi_expanding_std'] + 1e-8)
            station_data['is_anomaly'] = (abs(station_data['wqi_zscore']) > 2).astype(int)
            
            # Spatial features with proper time alignment (if multiple stations exist)
            other_stations = df[df['station_id'] != station_id]['station_id'].unique()
            if len(other_stations) > 0:
                # Use nearest station as spatial reference
                nearest_station = other_stations[0]
                nearest_data = df[df['station_id'] == nearest_station].copy()
                nearest_data = nearest_data.sort_values('Date')
                
                # Merge with nearest station data (safe - same date only)
                station_data = station_data.merge(
                    nearest_data[['Date', 'wqi', 'ph', 'temperature', 'do']].rename(columns={
                        'wqi': 'nearest_wqi',
                        'ph': 'nearest_ph',
                        'temperature': 'nearest_temperature',
                        'do': 'nearest_do'
                    }),
                    on='Date',
                    how='left'
                )
                
                # Spatial lag features (safe - only past information)
                station_data['spatial_wqi_lag_1'] = station_data['nearest_wqi'].shift(1)
                station_data['spatial_ph_lag_1'] = station_data['nearest_ph'].shift(1)
                station_data['spatial_temp_lag_1'] = station_data['nearest_temperature'].shift(1)
                station_data['spatial_do_lag_1'] = station_data['nearest_do'].shift(1)
                
                # Spatial difference features (safe - current values only)
                station_data['spatial_wqi_diff'] = station_data['wqi'] - station_data['nearest_wqi']
                station_data['spatial_ph_diff'] = station_data['ph'] - station_data['nearest_ph']
                station_data['spatial_temp_diff'] = station_data['temperature'] - station_data['nearest_temperature']
                station_data['spatial_do_diff'] = station_data['do'] - station_data['nearest_do']
            
            # Remove rows with NaN values (after feature engineering)
            station_data = station_data.dropna()
            
            logger.info(f"Enhanced features created for station {station_id}: {len(station_data)} records")
            logger.info(f"Features: {list(station_data.columns)}")
            
            return station_data
            
        except Exception as e:
            logger.error(f"Error creating enhanced features for station {station_id}: {e}")
            return None
    
    def detect_data_leakage(self, df: pd.DataFrame, target_col: str = 'wqi') -> Dict[str, Any]:
        """
        Detect potential data leakage by checking correlation with future values
        """
        try:
            import numpy as np
            from scipy.stats import pearsonr
            
            leakage_report = {
                'suspicious_features': [],
                'high_correlation_features': [],
                'recommendations': []
            }
            
            # Check for correlation with future target values
            future_target = df[target_col].shift(-1)
            
            for col in df.columns:
                if col != target_col and col not in ['Date', 'timestamp']:
                    # Remove NaN values for correlation calculation
                    valid_mask = ~(df[col].isna() | future_target.isna())
                    if valid_mask.sum() > 10:  # Need at least 10 valid pairs
                        correlation, p_value = pearsonr(df[col][valid_mask], future_target[valid_mask])
                        
                        if abs(correlation) > 0.95:
                            leakage_report['suspicious_features'].append({
                                'feature': col,
                                'correlation_with_future': correlation,
                                'p_value': p_value,
                                'risk_level': 'HIGH'
                            })
                        elif abs(correlation) > 0.8:
                            leakage_report['high_correlation_features'].append({
                                'feature': col,
                                'correlation_with_future': correlation,
                                'p_value': p_value,
                                'risk_level': 'MEDIUM'
                            })
            
            # Generate recommendations
            if leakage_report['suspicious_features']:
                leakage_report['recommendations'].append("⚠️ HIGH RISK: Features with >95% correlation to future values detected")
                leakage_report['recommendations'].append("→ Consider removing or modifying these features")
                
            if leakage_report['high_correlation_features']:
                leakage_report['recommendations'].append("⚠️ MEDIUM RISK: Features with >80% correlation to future values detected")
                leakage_report['recommendations'].append("→ Review these features for potential leakage")
            
            if not leakage_report['suspicious_features'] and not leakage_report['high_correlation_features']:
                leakage_report['recommendations'].append("✅ No significant data leakage detected")
            
            # Log results
            logger.info("=== DATA LEAKAGE DETECTION ===")
            if leakage_report['suspicious_features']:
                logger.warning(f"🚨 {len(leakage_report['suspicious_features'])} suspicious features detected:")
                for item in leakage_report['suspicious_features']:
                    logger.warning(f"  - {item['feature']}: corr={item['correlation_with_future']:.4f}")
            
            if leakage_report['high_correlation_features']:
                logger.warning(f"⚠️ {len(leakage_report['high_correlation_features'])} high correlation features:")
                for item in leakage_report['high_correlation_features']:
                    logger.warning(f"  - {item['feature']}: corr={item['correlation_with_future']:.4f}")
            
            if not leakage_report['suspicious_features'] and not leakage_report['high_correlation_features']:
                logger.info("✅ No data leakage detected")
            
            return leakage_report
            
        except Exception as e:
            logger.error(f"Data leakage detection failed: {e}")
            return {'error': str(e)}
    
    def create_enhanced_features_global(self, df: pd.DataFrame) -> Optional[pd.DataFrame]:
        """Create enhanced features for global model (all stations combined)"""
        try:
            # Sort by station_id and date to maintain temporal order per station
            df_sorted = df.sort_values(['station_id', 'Date']).copy()
            
            # Basic temporal features (safe - no future info)
            df_sorted['month'] = df_sorted['Date'].dt.month
            df_sorted['quarter'] = df_sorted['Date'].dt.quarter
            df_sorted['year'] = df_sorted['Date'].dt.year
            df_sorted['day_of_year'] = df_sorted['Date'].dt.dayofyear
            
            # Cyclical encoding (safe)
            df_sorted['month_sin'] = np.sin(2 * np.pi * df_sorted['month'] / 12)
            df_sorted['month_cos'] = np.cos(2 * np.pi * df_sorted['month'] / 12)
            df_sorted['quarter_sin'] = np.sin(2 * np.pi * df_sorted['quarter'] / 4)
            df_sorted['quarter_cos'] = np.cos(2 * np.pi * df_sorted['quarter'] / 4)
            
            # Season indicators (safe)
            df_sorted['is_rainy_season'] = ((df_sorted['month'] >= 5) & (df_sorted['month'] <= 10)).astype(int)
            df_sorted['is_dry_season'] = ((df_sorted['month'] <= 4) | (df_sorted['month'] >= 11)).astype(int)
            
            # Station-specific lag features (safe - only past information per station)
            df_sorted['wqi_lag_1'] = df_sorted.groupby('station_id')['wqi'].shift(1)
            df_sorted['wqi_lag_2'] = df_sorted.groupby('station_id')['wqi'].shift(2)
            df_sorted['wqi_lag_3'] = df_sorted.groupby('station_id')['wqi'].shift(3)
            
            # Station ID embedding (categorical encoding)
            df_sorted['station_id_embedding'] = df_sorted['station_id'].astype('category').cat.codes
            
            # Station-specific expanding statistics (safe - only past information per station)
            df_sorted['wqi_expanding_mean'] = df_sorted.groupby('station_id')['wqi'].expanding(min_periods=1).mean().reset_index(0, drop=True)
            df_sorted['wqi_expanding_std'] = df_sorted.groupby('station_id')['wqi'].expanding(min_periods=1).std().reset_index(0, drop=True)
            
            # Station-specific rolling statistics with proper window (only past per station)
            df_sorted['wqi_rolling_mean_3'] = df_sorted.groupby('station_id')['wqi'].shift(1).rolling(window=3, min_periods=1).mean().reset_index(0, drop=True)
            df_sorted['wqi_rolling_std_3'] = df_sorted.groupby('station_id')['wqi'].shift(1).rolling(window=3, min_periods=1).std().reset_index(0, drop=True)
            
            # Interaction features (safe - current values only)
            df_sorted['ph_temp_interaction'] = df_sorted['ph'] * df_sorted['temperature']
            df_sorted['ph_do_interaction'] = df_sorted['ph'] * df_sorted['do']
            df_sorted['temp_do_interaction'] = df_sorted['temperature'] * df_sorted['do']
            
            # Station-specific delta features (safe - only past differences per station)
            df_sorted['delta_wqi'] = df_sorted.groupby('station_id')['wqi'].diff()
            df_sorted['delta_ph'] = df_sorted.groupby('station_id')['ph'].diff()
            df_sorted['delta_temperature'] = df_sorted.groupby('station_id')['temperature'].diff()
            df_sorted['delta_do'] = df_sorted.groupby('station_id')['do'].diff()
            
            # Anomaly detection using expanding statistics per station (safe)
            df_sorted['wqi_zscore'] = (df_sorted['wqi'] - df_sorted['wqi_expanding_mean']) / (df_sorted['wqi_expanding_std'] + 1e-8)
            df_sorted['is_anomaly'] = (abs(df_sorted['wqi_zscore']) > 2).astype(int)
            
            # Remove rows with NaN values (after feature engineering)
            df_sorted = df_sorted.dropna()
            
            logger.info(f"Global enhanced features created: {len(df_sorted)} records")
            logger.info(f"Stations included: {sorted(df_sorted['station_id'].unique())}")
            logger.info(f"Features: {list(df_sorted.columns)}")
            
            return df_sorted
            
        except Exception as e:
            logger.error(f"Error creating global enhanced features: {e}")
            return None

    def evaluate_per_station(self, test_df: pd.DataFrame, y_true_col='y_true', y_pred_col='y_pred', station_col='station_id'):
        """
        Evaluate R², MAE, RMSE for each station_id from a test dataframe
        Assumes test_df has y_true, y_pred and station_id columns
        """
        try:
            from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
            
            station_metrics = []

            for sid in sorted(test_df[station_col].unique()):
                subset = test_df[test_df[station_col] == sid]
                y_true = subset[y_true_col].values
                y_pred = subset[y_pred_col].values

                if len(y_true) < 2:
                    logger.warning(f"Station {sid}: Insufficient data ({len(y_true)} samples)")
                    continue  # cần ít nhất 2 điểm để tính R²

                r2 = r2_score(y_true, y_pred)
                mae = mean_absolute_error(y_true, y_pred)
                rmse = np.sqrt(mean_squared_error(y_true, y_pred))

                station_metrics.append({
                    'station_id': sid,
                    'r2': r2,
                    'mae': mae,
                    'rmse': rmse,
                    'n_samples': len(subset)
                })
                
                logger.info(f"Station {sid}: R²={r2:.4f}, MAE={mae:.4f}, RMSE={rmse:.4f}, Samples={len(subset)}")

            return pd.DataFrame(station_metrics)
            
        except Exception as e:
            logger.error(f"Error in per-station evaluation: {e}")
            return pd.DataFrame()

    def train_global_model(self) -> Dict[str, Any]:
        """Train a single XGBoost model for all stations with detailed per-station evaluation"""
        try:
            logger.info("=== STARTING GLOBAL XGBOOST TRAINING PROCESS ===")
            
            if self.model_manager is None:
                logger.error("Model manager not initialized")
                return {'error': 'Model manager not initialized'}
            
            logger.info("Starting Global WQI forecasting training with XGBoost")
            
            # Load dữ liệu lịch sử từ CSV
            logger.info("Loading historical data...")
            historical_df = self.load_historical_data()
            if historical_df is None:
                logger.error("Failed to load historical data")
                return {'error': 'Failed to load historical data'}
            
            logger.info(f"✅ Historical data loaded successfully: {len(historical_df)} records")
            logger.info(f"Stations: {sorted(historical_df['station_id'].unique())}")
            
            # Create enhanced features for all stations combined
            logger.info("Creating global enhanced features...")
            global_data = self.create_enhanced_features_global(historical_df)
            if global_data is None:
                logger.error("Failed to create global enhanced features")
                return {'error': 'Failed to create global enhanced features'}
            
            # Train global model using model_manager
            logger.info("Training global XGBoost model...")
            
            # Use station_id 0 as global model identifier
            global_result = self.model_manager.train_xgboost_model(0, global_data)
            
            if 'error' not in global_result:
                # Extract metrics
                global_r2 = global_result.get('r2_score', 0)
                global_mae = global_result.get('mae', 0)
                global_rmse = global_result.get('rmse', 0)
                global_mape = global_result.get('mape', 0)
                
                logger.info(f"✅ Global XGBoost completed - R²: {global_r2:.4f}, MAE: {global_mae:.4f}, RMSE: {global_rmse:.4f}")
                
                # Check if performance is good enough
                if global_r2 > 0.8:
                    logger.info(f"🎉 Excellent Global XGBoost performance!")
                elif global_r2 > 0.6:
                    logger.info(f"✅ Good Global XGBoost performance")
                elif global_r2 > 0.4:
                    logger.info(f"⚠️ Acceptable Global XGBoost performance")
                else:
                    logger.warning(f"❌ Poor Global XGBoost performance")
                
                # Feature importance analysis
                feature_importance = global_result.get('feature_importance', {})
                if feature_importance:
                    top_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:10]
                    logger.info(f"Top 10 features: {top_features}")
                
                # Model status analysis
                model_status = global_result.get('model_status', 'unknown')
                overfitting_ratio = global_result.get('overfitting_ratio', 0)
                
                logger.info(f"Model status: {model_status}, Overfitting ratio: {overfitting_ratio:.2f}")
                
                # Prepare results for all stations
                all_stations = sorted(historical_df['station_id'].unique())
                successful_stations = [int(station_id) for station_id in all_stations]
                
                # Create results structure
                results = {
                    'best_model_type': 'xgboost_global',
                    'best_model_r2': self.convert_numpy_types(global_r2),
                    'successful_stations': successful_stations,
                    'xgb_results': {
                        'global': {
                            'r2_score': global_r2,
                            'mae': global_mae,
                            'rmse': global_rmse,
                            'mape': global_mape,
                            'feature_importance': feature_importance,
                            'train_r2': global_result.get('train_r2', 0),
                            'val_r2': global_result.get('val_r2', 0),
                            'overfitting_ratio': overfitting_ratio,
                            'model_status': model_status,
                            'model_path': global_result.get('model_path', ''),
                            'model_version': global_result.get('model_version', '')
                        }
                    },
                    'summary': f"Global XGBoost training completed: Best model (XGBoost Global) trained for {len(successful_stations)} stations with R2={global_r2:.4f}"
                }
                
                logger.info("=== GLOBAL XGBOOST TRAINING PROCESS COMPLETED SUCCESSFULLY ===")
                return results
                
            else:
                logger.error(f"❌ Global XGBoost failed: {global_result['error']}")
                return {'error': global_result['error']}
                
        except Exception as e:
            logger.error(f"=== CRITICAL ERROR IN GLOBAL XGBOOST TRAINING PROCESS ===")
            logger.error(f"Error: {e}")
            import traceback
            logger.error(f"Full traceback: {traceback.format_exc()}")
            return {'error': f"Global XGBoost training failed with error: {str(e)}"}

    def train_global_model_with_detailed_eval(self) -> Dict[str, Any]:
        """Train global model with detailed per-station evaluation and MLflow logging"""
        try:
            logger.info("=== STARTING GLOBAL XGBOOST TRAINING WITH DETAILED EVALUATION ===")
            
            if self.model_manager is None:
                logger.error("Model manager not initialized")
                return {'error': 'Model manager not initialized'}
            
            # Load and prepare data
            historical_df = self.load_historical_data()
            if historical_df is None:
                return {'error': 'Failed to load historical data'}
            
            global_data = self.create_enhanced_features_global(historical_df)
            if global_data is None:
                return {'error': 'Failed to create enhanced features'}
            
            # Detect data leakage
            leakage_report = self.detect_data_leakage(global_data)
            if 'error' not in leakage_report:
                logger.info("Data leakage detection completed")
                if leakage_report.get('suspicious_features'):
                    logger.warning("⚠️ Data leakage detected - review features before training")
                else:
                    logger.info("✅ No data leakage detected - safe to proceed")
            
            # Train global model with detailed evaluation
            global_result = self.model_manager.train_xgboost_model(0, global_data)
            
            if 'error' in global_result:
                return {'error': global_result['error']}
            
            # Extract global metrics
            global_r2 = global_result.get('r2_score', 0)
            global_mae = global_result.get('mae', 0)
            global_rmse = global_result.get('rmse', 0)
            global_mape = global_result.get('mape', 0)
            
            logger.info(f"✅ Global XGBoost completed - R²: {global_r2:.4f}, MAE: {global_mae:.4f}, RMSE: {global_rmse:.4f}")
            
            # Get test predictions for per-station evaluation
            try:
                # Load the trained model
                model_path = global_result.get('model_path', '')
                if model_path and os.path.exists(model_path):
                    import joblib
                    model = joblib.load(model_path)
                    
                    # Use the same feature processing as training
                    # Get test data from model_manager's prepare_training_data
                    X_train, X_test, y_train, y_test = self.model_manager.prepare_training_data(global_data, 'xgboost')
                    
                    if X_test is not None:
                        # Make predictions using the processed test data
                        y_pred = model.predict(X_test)
                        y_true = y_test
                        
                        # Create test dataframe for per-station evaluation
                        # We need to map back to original station_ids
                        test_df = pd.DataFrame({
                            'station_id': global_data.iloc[-len(y_test):]['station_id'].values,
                            'y_true': y_true,
                            'y_pred': y_pred
                        })
                        
                        # Evaluate per station
                        station_eval_df = self.evaluate_per_station(test_df)
                        
                        if not station_eval_df.empty:
                            logger.info("=== PER-STATION EVALUATION RESULTS ===")
                            logger.info(f"\n{station_eval_df.to_string(index=False)}")
                            
                            # Save to CSV
                            eval_csv_path = 'logs/eval_by_station.csv'
                            os.makedirs('logs', exist_ok=True)
                            station_eval_df.to_csv(eval_csv_path, index=False)
                            logger.info(f"✅ Per-station evaluation saved to: {eval_csv_path}")
                            
                            # Add per-station metrics to results
                            station_metrics = {}
                            for _, row in station_eval_df.iterrows():
                                station_id = int(row['station_id'])
                                station_metrics[station_id] = {
                                    'r2_score': float(row['r2']),
                                    'mae': float(row['mae']),
                                    'rmse': float(row['rmse']),
                                    'n_samples': int(row['n_samples'])
                                }
                            
                            # Log to MLflow if available
                            try:
                                import mlflow
                                # End any existing run first
                                try:
                                    mlflow.end_run()
                                except:
                                    pass
                                
                                mlflow.set_experiment("water_quality")
                                with mlflow.start_run() as run:
                                    # Log global metrics with proper type conversion
                                    mlflow.log_metric("global_r2", float(global_r2))
                                    mlflow.log_metric("global_mae", float(global_mae))
                                    mlflow.log_metric("global_rmse", float(global_rmse))
                                    mlflow.log_metric("global_mape", float(global_mape))
                                    
                                    # Log per-station metrics with proper type conversion
                                    for station_id, metrics in station_metrics.items():
                                        mlflow.log_metric(f"station_{station_id}_r2", float(metrics['r2_score']))
                                        mlflow.log_metric(f"station_{station_id}_mae", float(metrics['mae']))
                                        mlflow.log_metric(f"station_{station_id}_rmse", float(metrics['rmse']))
                                        mlflow.log_metric(f"station_{station_id}_n_samples", int(metrics['n_samples']))
                                    
                                    # Log CSV artifact
                                    mlflow.log_artifact(eval_csv_path)
                                    
                                    # Create and log features.csv
                                    features_info = []
                                    for i, col in enumerate(global_data.columns):
                                        if col not in ['wqi', 'Date', 'timestamp']:
                                            feature_type = 'temporal' if col in ['month', 'quarter', 'year', 'day_of_year', 'month_sin', 'month_cos', 'quarter_sin', 'quarter_cos', 'is_rainy_season', 'is_dry_season'] else \
                                                         'lag' if 'lag' in col else \
                                                         'rolling' if 'rolling' in col else \
                                                         'expanding' if 'expanding' in col else \
                                                         'interaction' if 'interaction' in col else \
                                                         'delta' if 'delta' in col else \
                                                         'embedding' if 'embedding' in col else \
                                                         'anomaly' if 'anomaly' in col else \
                                                         'spatial' if 'nearest' in col else 'base'
                                            
                                            features_info.append({
                                                'feature_name': col,
                                                'feature_type': feature_type,
                                                'feature_index': int(i),
                                                'description': f'{feature_type} feature for WQI prediction'
                                            })
                                    
                                    features_df = pd.DataFrame(features_info)
                                    features_csv_path = 'logs/features_info.csv'
                                    features_df.to_csv(features_csv_path, index=False)
                                    mlflow.log_artifact(features_csv_path)
                                    logger.info(f"✅ Features info saved to: {features_csv_path}")
                                    
                                    # Create feature importance plot if model has feature_importances_
                                    try:
                                        if hasattr(model, 'feature_importances_'):
                                            import matplotlib.pyplot as plt
                                            
                                            # Get feature importance
                                            importance = model.feature_importances_
                                            feature_names = [col for col in global_data.columns if col not in ['wqi', 'Date', 'timestamp']]
                                            
                                            # Create importance dataframe
                                            importance_df = pd.DataFrame({
                                                'feature': feature_names[:len(importance)],
                                                'importance': importance
                                            }).sort_values('importance', ascending=False)
                                            
                                            # Plot top 15 features
                                            plt.figure(figsize=(12, 8))
                                            top_features = importance_df.head(15)
                                            plt.barh(range(len(top_features)), top_features['importance'])
                                            plt.yticks(range(len(top_features)), top_features['feature'])
                                            plt.xlabel('Feature Importance')
                                            plt.title('Top 15 Feature Importance (XGBoost Global Model)')
                                            plt.gca().invert_yaxis()
                                            plt.tight_layout()
                                            
                                            # Save plot
                                            importance_plot_path = 'logs/feature_importance.png'
                                            plt.savefig(importance_plot_path, dpi=300, bbox_inches='tight')
                                            plt.close()
                                            
                                            # Log to MLflow
                                            mlflow.log_artifact(importance_plot_path)
                                            logger.info(f"✅ Feature importance plot saved to: {importance_plot_path}")
                                            
                                    except Exception as plot_error:
                                        logger.warning(f"Feature importance plot failed: {plot_error}")
                                    
                                    logger.info("✅ MLflow logging completed successfully")
                                    
                            except Exception as mlflow_error:
                                logger.warning(f"MLflow logging failed: {mlflow_error}")
                            
                            # Add station metrics to results
                            global_result['station_metrics'] = station_metrics
                            global_result['station_eval_df'] = station_eval_df.to_dict('records')
                            
                        else:
                            logger.warning("No per-station evaluation data available")
                    else:
                        logger.warning("Could not prepare test data for evaluation")
                        
            except Exception as eval_error:
                logger.warning(f"Per-station evaluation failed: {eval_error}")
                import traceback
                logger.warning(f"Evaluation traceback: {traceback.format_exc()}")
            
            # Prepare final results
            all_stations = sorted(historical_df['station_id'].unique())
            successful_stations = [int(station_id) for station_id in all_stations]
            
            # Remove model object from global_result to avoid serialization issues
            serializable_global_result = {}
            for key, value in global_result.items():
                if key != 'model':  # Skip the model object
                    serializable_global_result[key] = value
            
            results = {
                'best_model_type': 'xgboost_global',
                'best_model_r2': self.convert_numpy_types(global_r2),
                'successful_stations': successful_stations,
                'xgb_results': {
                    'global': serializable_global_result
                },
                'summary': f"Global XGBoost training completed: Best model (XGBoost Global) trained for {len(successful_stations)} stations with R2={global_r2:.4f}"
            }
            
            logger.info("=== GLOBAL XGBOOST TRAINING WITH DETAILED EVALUATION COMPLETED ===")
            return results
            
        except Exception as e:
            logger.error(f"=== CRITICAL ERROR IN GLOBAL XGBOOST TRAINING WITH DETAILED EVALUATION ===")
            logger.error(f"Error: {e}")
            import traceback
            logger.error(f"Full traceback: {traceback.format_exc()}")
            return {'error': f"Global XGBoost training failed with error: {str(e)}"}

    def train_all_stations(self) -> Dict[str, Any]:
        """Train Global XGBoost model for all stations combined with detailed evaluation"""
        return self.train_global_model_with_detailed_eval()
    
    def convert_numpy_types(self, obj):
        """Convert numpy types to Python native types for JSON serialization"""
        import numpy as np
        
        if isinstance(obj, (np.integer, np.int64, np.int32, np.int16, np.int8)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32, np.float16)):
            return float(obj)
        elif isinstance(obj, (np.bool_)):
            return bool(obj)
        elif isinstance(obj, dict):
            return {k: self.convert_numpy_types(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self.convert_numpy_types(i) for i in obj]
        elif isinstance(obj, tuple):
            return tuple(self.convert_numpy_types(i) for i in obj)
        elif isinstance(obj, np.ndarray):
            return self.convert_numpy_types(obj.tolist())
        else:
            return obj

# Global instance
training_service = TrainingService() 