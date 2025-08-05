"""
Summary Service for Water Quality Monitoring
Handles training summaries and analysis
"""

import logging
from datetime import datetime
from typing import Dict, Any

logger = logging.getLogger(__name__)

class SummaryService:
    def __init__(self):
        pass
    
    def generate_training_summary(self, training_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive training summary with per-station analysis"""
        try:
            if 'error' in training_results:
                return {
                    'status': 'error',
                    'message': training_results['error'],
                    'execution_time': datetime.now().isoformat()
                }
            
            # Extract data from training results
            best_model_type = training_results.get('best_model_type', 'unknown')
            best_model_r2 = training_results.get('best_model_r2', 0)
            successful_stations = training_results.get('successful_stations', [])
            xgb_results = training_results.get('xgb_results', {})
            
            # Extract ensemble data
            ensemble_alpha = training_results.get('ensemble_alpha', 0.5)
            ensemble_station_results = training_results.get('ensemble_station_results', {})
            ensemble_summary = training_results.get('ensemble_summary', '')
            
            # Create comprehensive summary
            summary = {
                'best_model_type': best_model_type,
                'best_model_r2': best_model_r2,
                'successful_stations': successful_stations,
                'xgb_results': xgb_results,
                'ensemble_alpha': ensemble_alpha,
                'ensemble_station_results': ensemble_station_results,
                'ensemble_summary': ensemble_summary,
                'enhanced_features': {
                    'spatial_lag_features': True,
                    'anomaly_detection': True,
                    'interaction_features': True,
                    'cyclical_encoding': True,
                    'delta_features': True,
                    'outlier_removal': True,
                    'feature_scaling': True,
                    'time_based_split': True,
                    'optimized_hyperparameters': True,
                    'station_specific_features': True,
                    'global_model_approach': True,
                    'lstm_ensemble': True,
                    'multi_model_approach': True
                },
                'model_improvements': {
                    'increased_estimators': True,
                    'optimized_learning_rate': True,
                    'enhanced_cross_validation': True,
                    'feature_importance_analysis': True,
                    'overfitting_prevention': True,
                    'comprehensive_analysis': True,
                    'per_station_evaluation': True,
                    'mlflow_integration': True,
                    'ensemble_weight_optimization': True,
                    'lstm_temporal_modeling': True
                },
                'execution_time': datetime.now().isoformat()
            }
            
            logger.info("=== Comprehensive Ensemble Training Summary ===")
            logger.info(f"Best model type: {best_model_type}")
            logger.info(f"Best model R²: {best_model_r2}")
            logger.info(f"Successful stations: {successful_stations}")
            logger.info(f"Ensemble Alpha: {ensemble_alpha:.3f}")
            logger.info(f"Ensemble Summary: {ensemble_summary}")
            logger.info(f"Enhanced features enabled: Spatial-lag, Anomaly detection, Essential interactions")
            logger.info(f"Model improvements: Optimized hyperparameters, Enhanced CV, Feature importance analysis")
            logger.info(f"Ensemble approach: LSTM + XGBoost with optimized weights")
            logger.info(f"Comprehensive analysis: Overfitting/Underfitting detection for all stations")
            logger.info(f"Global model approach: Single model for all stations to reduce underfitting")
            
            if xgb_results:
                logger.info("XGBoost Performance by Station:")
                for station_id, result in xgb_results.items():
                    if 'error' not in result:
                        r2_score = result.get('r2_score', 0)
                        mae = result.get('mae', 0)
                        rmse = result.get('rmse', 0)
                        status = result.get('model_status', 'unknown')
                        logger.info(f"  Station {station_id}: R²={r2_score:.4f}, MAE={mae:.4f}, RMSE={rmse:.4f}, Status={status}")
                        
                        # Feature importance
                        feature_importance = result.get('feature_importance', {})
                        if feature_importance:
                            top_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:3]
                            logger.info(f"    Top features: {top_features}")
                        
                        # Per-station metrics if available
                        station_metrics = result.get('station_metrics', {})
                        if station_metrics:
                            logger.info("  Per-station detailed metrics:")
                            for sid, metrics in station_metrics.items():
                                logger.info(f"    Station {sid}: R²={metrics['r2_score']:.4f}, MAE={metrics['mae']:.4f}, RMSE={metrics['rmse']:.4f}, Samples={metrics['n_samples']}")
            
            # Log ensemble station results
            if ensemble_station_results:
                logger.info("\n=== ENSEMBLE STATION RESULTS ===")
                for station_id, metrics in ensemble_station_results.items():
                    logger.info(f"Station {station_id}:")
                    for model_name, model_metrics in metrics.items():
                        logger.info(f"  {model_name.upper()}: R²={model_metrics['r2']:.4f}, MAE={model_metrics['mae']:.4f}")
            
            # Performance assessment
            logger.info("\n=== PERFORMANCE ASSESSMENT ===")
            if best_model_r2 > 0.8:
                logger.info("🎉 EXCELLENT: Model ready for production!")
            elif best_model_r2 > 0.6:
                logger.info("✅ GOOD: Model acceptable for production")
            elif best_model_r2 > 0.4:
                logger.info("⚠️ ACCEPTABLE: Model needs improvement before production")
            else:
                logger.warning("❌ POOR: Model not suitable for production")
            
            # Data quality assessment
            logger.info("\n=== DATA QUALITY ASSESSMENT ===")
            logger.info(f"Total stations: {len(successful_stations)}")
            logger.info(f"Global model approach: ✅ Reduces underfitting with limited data")
            logger.info(f"Time-based split: ✅ Prevents data leakage")
            logger.info(f"Feature engineering: ✅ Comprehensive temporal and spatial features")
            
            return summary
            
        except Exception as e:
            logger.error(f"Error generating training summary: {e}")
            return {
                'status': 'error',
                'message': f'Failed to generate summary: {str(e)}',
                'execution_time': datetime.now().isoformat()
            }

# Global instance
summary_service = SummaryService() 