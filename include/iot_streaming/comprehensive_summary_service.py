import logging

logger = logging.getLogger(__name__)

class ComprehensiveSummaryService:
    """Service for generating comprehensive training summaries"""
    
    def __init__(self):
        pass
    
    def generate_comprehensive_summary(self, training_results):
        """Generate comprehensive summary from training results"""
        try:
            logger.info("Generating comprehensive training summary...")
            
            # Extract results with safe defaults
            best_model_type = training_results.get('best_model_type', 'Unknown')
            best_model_r2 = training_results.get('best_model_r2', 0.0)
            successful_stations = training_results.get('successful_stations', [])
            xgb_results = training_results.get('xgb_results', {})
            spark_ensemble_weights = training_results.get('spark_ensemble_weights', {})
            spark_ensemble_metrics = training_results.get('spark_ensemble_metrics', {})
            spark_ensemble_summary = training_results.get('spark_ensemble_summary', {})
            
            # Build comprehensive summary
            summary = {
                'training_overview': {
                    'xgboost_training': {
                        'best_model_type': best_model_type,
                        'best_r2_score': best_model_r2,
                        'successful_stations': successful_stations,
                        'results': xgb_results
                    },
                    'spark_ensemble_training': {
                        'weights': spark_ensemble_weights,
                        'metrics': spark_ensemble_metrics,
                        'summary': spark_ensemble_summary
                    }
                },
                'model_comparison': self._compare_models(training_results),
                'recommendations': self._generate_recommendations(training_results)
            }
            
            logger.info("Comprehensive summary generated successfully")
            return summary
            
        except Exception as e:
            logger.error(f"Error generating comprehensive summary: {e}")
            return {
                'error': str(e),
                'message': f"Summary generation failed: {str(e)}"
            }
    
    def _compare_models(self, training_results):
        """Compare different model performances"""
        comparison = {}
        
        # XGBoost results
        xgb_r2 = training_results.get('best_model_r2', 0.0)
        if xgb_r2 is None:
            xgb_r2 = 0.0
        comparison['xgboost'] = {
            'r2_score': xgb_r2,
            'model_type': training_results.get('best_model_type', 'Unknown')
        }
        
        # Spark Ensemble results
        spark_metrics = training_results.get('spark_ensemble_metrics', {})
        if spark_metrics:
            comparison['spark_ensemble'] = {
                'xgb_r2': spark_metrics.get('xgb', {}).get('r2', 0.0),
                'rf_r2': spark_metrics.get('rf', {}).get('r2', 0.0),
                'ensemble_r2': spark_metrics.get('ensemble', {}).get('r2', 0.0)
            }
        
        # Determine best overall model
        if comparison:
            best_overall = max(comparison.keys(), 
                              key=lambda x: comparison[x].get('r2_score', comparison[x].get('ensemble_r2', 0.0)))
            comparison['best_overall'] = best_overall
        else:
            comparison['best_overall'] = 'none'
        
        return comparison
    
    def _generate_recommendations(self, training_results):
        """Generate recommendations based on training results"""
        recommendations = []
        
        # XGBoost recommendations
        xgb_r2 = training_results.get('best_model_r2', 0.0)
        if xgb_r2 is None:
            xgb_r2 = 0.0
            
        if xgb_r2 > 0.8:
            recommendations.append("XGBoost model shows excellent performance (R² > 0.8)")
        elif xgb_r2 > 0.6:
            recommendations.append("XGBoost model shows good performance (R² > 0.6)")
        else:
            recommendations.append("XGBoost model performance needs improvement")
        
        # Spark Ensemble recommendations
        spark_metrics = training_results.get('spark_ensemble_metrics', {})
        if spark_metrics:
            ensemble_r2 = spark_metrics.get('ensemble', {}).get('r2', 0.0)
            if ensemble_r2 is None:
                ensemble_r2 = 0.0
                
            if ensemble_r2 > 0.8:
                recommendations.append("Spark Ensemble shows excellent performance")
            elif ensemble_r2 > 0.6:
                recommendations.append("Spark Ensemble shows good performance")
            else:
                recommendations.append("Spark Ensemble performance needs improvement")
        
        # Deployment recommendations
        if training_results.get('successful_stations'):
            recommendations.append("Models trained successfully for multiple stations")
        
        return recommendations

# Global instance
comprehensive_summary_service = ComprehensiveSummaryService() 