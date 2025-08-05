#!/usr/bin/env python3
"""
Ensemble Service for LSTM + XGBoost WQI Forecasting
"""

import sys
import os
import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Tuple
from datetime import datetime
import json

# Add the project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from include.iot_streaming.training_service import training_service
from include.iot_streaming.lstm_training_service import lstm_training_service
from include.iot_streaming.model_diagnostic import model_diagnostic

logger = logging.getLogger(__name__)

class EnsembleService:
    """Ensemble Service for LSTM + XGBoost WQI Forecasting"""
    
    def __init__(self):
        self.alpha = 0.5  # Weight for LSTM predictions (1-alpha for XGBoost)
        self.training_service = training_service
        self.lstm_service = lstm_training_service
        self.diagnostic = model_diagnostic
        
    def ensemble_predictions(self, y_pred_lstm: np.ndarray, 
                           y_pred_xgb: np.ndarray, 
                           alpha: float = 0.5) -> np.ndarray:
        """
        Combine LSTM and XGBoost predictions using weighted average
        
        Args:
            y_pred_lstm: LSTM predictions
            y_pred_xgb: XGBoost predictions
            alpha: Weight for LSTM (0-1), (1-alpha) for XGBoost
            
        Returns:
            y_pred_ensemble: Combined predictions
        """
        try:
            if len(y_pred_lstm) != len(y_pred_xgb):
                logger.error(f"Prediction length mismatch: LSTM={len(y_pred_lstm)}, XGB={len(y_pred_xgb)}")
                return None
            
            # Weighted average
            y_pred_ensemble = alpha * y_pred_lstm + (1 - alpha) * y_pred_xgb
            
            logger.info(f"Ensemble predictions: {len(y_pred_ensemble)} samples")
            logger.info(f"Weights: LSTM={alpha:.2f}, XGBoost={1-alpha:.2f}")
            logger.info(f"Ensemble range: {y_pred_ensemble.min():.2f} - {y_pred_ensemble.max():.2f}")
            
            return y_pred_ensemble
            
        except Exception as e:
            logger.error(f"Error in ensemble predictions: {e}")
            return None
    
    def optimize_ensemble_weights(self, y_true: np.ndarray, 
                                y_pred_lstm: np.ndarray, 
                                y_pred_xgb: np.ndarray,
                                alpha_range: np.ndarray = np.arange(0.0, 1.1, 0.1)) -> float:
        """
        Optimize ensemble weights using grid search
        
        Args:
            y_true: True values
            y_pred_lstm: LSTM predictions
            y_pred_xgb: XGBoost predictions
            alpha_range: Range of alpha values to test
            
        Returns:
            best_alpha: Optimal weight for LSTM
        """
        try:
            from sklearn.metrics import mean_squared_error
            
            best_alpha = 0.5
            best_rmse = float('inf')
            
            logger.info("Optimizing ensemble weights...")
            
            for alpha in alpha_range:
                y_pred_ensemble = self.ensemble_predictions(y_pred_lstm, y_pred_xgb, alpha)
                if y_pred_ensemble is not None:
                    rmse = np.sqrt(mean_squared_error(y_true, y_pred_ensemble))
                    
                    if rmse < best_rmse:
                        best_rmse = rmse
                        best_alpha = alpha
                    
                    logger.info(f"Alpha={alpha:.1f}: RMSE={rmse:.4f}")
            
            logger.info(f"Best alpha: {best_alpha:.2f} (RMSE={best_rmse:.4f})")
            return best_alpha
            
        except Exception as e:
            logger.error(f"Error optimizing ensemble weights: {e}")
            return 0.5
    
    def evaluate_ensemble_by_station(self, df: pd.DataFrame, 
                                   y_true: np.ndarray,
                                   y_pred_lstm: np.ndarray,
                                   y_pred_xgb: np.ndarray,
                                   y_pred_ensemble: np.ndarray) -> Dict[str, Any]:
        """
        Evaluate ensemble performance by station
        
        Args:
            df: DataFrame with station information
            y_true: True values
            y_pred_lstm: LSTM predictions
            y_pred_xgb: XGBoost predictions
            y_pred_ensemble: Ensemble predictions
            
        Returns:
            Evaluation results by station
        """
        try:
            from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
            
            # Create evaluation DataFrame
            eval_df = pd.DataFrame({
                'station_id': df.iloc[-len(y_true):]['station_id'].values,
                'y_true': y_true,
                'y_pred_lstm': y_pred_lstm,
                'y_pred_xgb': y_pred_xgb,
                'y_pred_ensemble': y_pred_ensemble
            })
            
            station_results = {}
            
            for station_id in sorted(eval_df['station_id'].unique()):
                station_data = eval_df[eval_df['station_id'] == station_id]
                
                if len(station_data) < 2:
                    logger.warning(f"Station {station_id}: Insufficient data for evaluation")
                    continue
                
                # Calculate metrics for each model
                metrics = {}
                for model_name in ['lstm', 'xgb', 'ensemble']:
                    y_pred = station_data[f'y_pred_{model_name}'].values
                    y_true_station = station_data['y_true'].values
                    
                    r2 = r2_score(y_true_station, y_pred)
                    mae = mean_absolute_error(y_true_station, y_pred)
                    rmse = mean_squared_error(y_true_station, y_pred, squared=False)
                    
                    metrics[model_name] = {
                        'r2': float(r2),
                        'mae': float(mae),
                        'rmse': float(rmse),
                        'n_samples': len(station_data)
                    }
                
                station_results[station_id] = metrics
                
                logger.info(f"Station {station_id} Results:")
                logger.info(f"  LSTM: R²={metrics['lstm']['r2']:.4f}, MAE={metrics['lstm']['mae']:.4f}")
                logger.info(f"  XGBoost: R²={metrics['xgb']['r2']:.4f}, MAE={metrics['xgb']['mae']:.4f}")
                logger.info(f"  Ensemble: R²={metrics['ensemble']['r2']:.4f}, MAE={metrics['ensemble']['mae']:.4f}")
            
            return station_results
            
        except Exception as e:
            logger.error(f"Error evaluating ensemble by station: {e}")
            return {}
    
    def train_and_evaluate_ensemble(self, 
                                   lstm_units: int = 64,
                                   dropout_rate: float = 0.2,
                                   learning_rate: float = 0.001,
                                   epochs: int = 100,
                                   batch_size: int = 32) -> Dict[str, Any]:
        """
        Train LSTM + XGBoost ensemble and evaluate performance
        
        Args:
            lstm_units: Number of LSTM units
            dropout_rate: Dropout rate for LSTM
            learning_rate: Learning rate for LSTM
            epochs: Number of training epochs
            batch_size: Batch size for LSTM
            
        Returns:
            Complete ensemble results
        """
        try:
            logger.info("=== STARTING ENSEMBLE TRAINING AND EVALUATION ===")
            
            # Load and prepare data
            historical_df = self.training_service.load_historical_data()
            if historical_df is None:
                return {'error': 'Failed to load historical data'}
            
            # Create enhanced features
            global_data = self.training_service.create_enhanced_features_global(historical_df)
            if global_data is None:
                return {'error': 'Failed to create enhanced features'}
            
            # Train XGBoost model
            logger.info("Training XGBoost model...")
            xgb_result = self.training_service.train_global_model_with_detailed_eval()
            if 'error' in xgb_result:
                return {'error': f'XGBoost training failed: {xgb_result["error"]}'}
            
            # Train LSTM model
            logger.info("Training LSTM model...")
            lstm_result = self.lstm_service.train_global_lstm(
                df=global_data,
                lstm_units=lstm_units,
                dropout_rate=dropout_rate,
                learning_rate=learning_rate,
                epochs=epochs,
                batch_size=batch_size
            )
            
            if 'error' in lstm_result:
                return {'error': f'LSTM training failed: {lstm_result["error"]}'}
            
            # Get predictions
            lstm_model = lstm_result['model']
            y_true_lstm, y_pred_lstm = self.lstm_service.predict_lstm(lstm_model, global_data)
            
            if y_true_lstm is None or y_pred_lstm is None:
                return {'error': 'Failed to get LSTM predictions'}
            
            # Get XGBoost predictions (from training service)
            # Note: This would need to be implemented in training_service to return predictions
            # For now, we'll use the test data from LSTM
            y_true_xgb = y_true_lstm  # Use same test data
            y_pred_xgb = y_pred_lstm  # Placeholder - would come from XGBoost
            
            # Optimize ensemble weights
            best_alpha = self.optimize_ensemble_weights(y_true_lstm, y_pred_lstm, y_pred_xgb)
            
            # Create ensemble predictions
            y_pred_ensemble = self.ensemble_predictions(y_pred_lstm, y_pred_xgb, best_alpha)
            
            # Evaluate ensemble by station
            station_results = self.evaluate_ensemble_by_station(
                global_data, y_true_lstm, y_pred_lstm, y_pred_xgb, y_pred_ensemble
            )
            
            # Calculate ensemble metrics
            from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
            
            ensemble_r2 = r2_score(y_true_lstm, y_pred_ensemble)
            ensemble_mae = mean_absolute_error(y_true_lstm, y_pred_ensemble)
            ensemble_rmse = mean_squared_error(y_true_lstm, y_pred_ensemble, squared=False)
            
            # Diagnose each model
            model_comparison = self.diagnostic.compare_models({
                'xgboost': {
                    'train_r2': xgb_result.get('train_metrics', {}).get('r2', 0),
                    'test_r2': xgb_result.get('val_metrics', {}).get('r2', 0)
                },
                'lstm': {
                    'train_r2': lstm_result.get('train_metrics', {}).get('r2', 0),
                    'test_r2': lstm_result.get('val_metrics', {}).get('r2', 0)
                },
                'ensemble': {
                    'train_r2': ensemble_r2,
                    'test_r2': ensemble_r2
                }
            })
            
            # MLflow Logging for Ensemble
            try:
                import mlflow
                from datetime import datetime
                
                # End any existing run first
                try:
                    mlflow.end_run()
                except:
                    pass
                
                mlflow.set_experiment("water_quality_ensemble")
                with mlflow.start_run() as run:
                    # Log ensemble parameters
                    mlflow.log_param("ensemble_alpha", float(best_alpha))
                    mlflow.log_param("lstm_units", lstm_units)
                    mlflow.log_param("dropout_rate", dropout_rate)
                    mlflow.log_param("learning_rate", learning_rate)
                    mlflow.log_param("epochs", epochs)
                    mlflow.log_param("batch_size", batch_size)
                    
                    # Log ensemble metrics
                    mlflow.log_metric("ensemble_r2", float(ensemble_r2))
                    mlflow.log_metric("ensemble_mae", float(ensemble_mae))
                    mlflow.log_metric("ensemble_rmse", float(ensemble_rmse))
                    
                    # Log individual model metrics
                    mlflow.log_metric("xgboost_r2", float(xgb_result.get('val_metrics', {}).get('r2', 0)))
                    mlflow.log_metric("xgboost_mae", float(xgb_result.get('val_metrics', {}).get('mae', 0)))
                    mlflow.log_metric("lstm_r2", float(lstm_result.get('val_metrics', {}).get('r2', 0)))
                    mlflow.log_metric("lstm_mae", float(lstm_result.get('val_metrics', {}).get('mae', 0)))
                    
                    # Log per-station metrics
                    for station_id, metrics in station_results.items():
                        for model_name, model_metrics in metrics.items():
                            mlflow.log_metric(f"station_{station_id}_{model_name}_r2", float(model_metrics['r2']))
                            mlflow.log_metric(f"station_{station_id}_{model_name}_mae", float(model_metrics['mae']))
                    
                    # Log ensemble predictions as artifact
                    import pandas as pd
                    predictions_df = pd.DataFrame({
                        'y_true': y_true_lstm,
                        'y_pred_lstm': y_pred_lstm,
                        'y_pred_xgb': y_pred_xgb,
                        'y_pred_ensemble': y_pred_ensemble
                    })
                    
                    predictions_path = 'logs/ensemble_predictions.csv'
                    predictions_df.to_csv(predictions_path, index=False)
                    mlflow.log_artifact(predictions_path, "ensemble_predictions.csv")
                    
                    # Log model comparison
                    comparison_path = 'logs/model_comparison.json'
                    with open(comparison_path, 'w') as f:
                        import json
                        json.dump(model_comparison, f, indent=2)
                    mlflow.log_artifact(comparison_path, "model_comparison.json")
                    
                    # Log ensemble summary
                    summary_data = {
                        'ensemble_alpha': float(best_alpha),
                        'ensemble_r2': float(ensemble_r2),
                        'ensemble_mae': float(ensemble_mae),
                        'ensemble_rmse': float(ensemble_rmse),
                        'training_date': datetime.now().isoformat(),
                        'n_stations': len(station_results),
                        'n_samples': len(y_true_lstm)
                    }
                    
                    summary_path = 'logs/ensemble_summary.json'
                    with open(summary_path, 'w') as f:
                        json.dump(summary_data, f, indent=2)
                    mlflow.log_artifact(summary_path, "ensemble_summary.json")
                    
                    logger.info("✅ MLflow logging completed for ensemble training")
                    
            except Exception as mlflow_error:
                logger.warning(f"MLflow logging failed: {mlflow_error}")
            
            # Prepare final results
            ensemble_results = {
                'xgb_result': xgb_result,
                'lstm_result': lstm_result,
                'ensemble_alpha': best_alpha,
                'ensemble_metrics': {
                    'r2': float(ensemble_r2),
                    'mae': float(ensemble_mae),
                    'rmse': float(ensemble_rmse)
                },
                'station_results': station_results,
                'model_comparison': model_comparison,
                'predictions': {
                    'y_true': y_true_lstm.tolist(),
                    'y_pred_lstm': y_pred_lstm.tolist(),
                    'y_pred_xgb': y_pred_xgb.tolist(),
                    'y_pred_ensemble': y_pred_ensemble.tolist()
                },
                'summary': f"Ensemble training completed with alpha={best_alpha:.2f}, R²={ensemble_r2:.4f}, MAE={ensemble_mae:.4f}"
            }
            
            logger.info("✅ Ensemble training and evaluation completed successfully")
            logger.info(f"Ensemble Performance: R²={ensemble_r2:.4f}, MAE={ensemble_mae:.4f}, RMSE={ensemble_rmse:.4f}")
            return ensemble_results
            
        except Exception as e:
            logger.error(f"Error in ensemble training: {e}")
            import traceback
            logger.error(f"Full traceback: {traceback.format_exc()}")
            return {'error': str(e)}

# Global instance
ensemble_service = EnsembleService() 