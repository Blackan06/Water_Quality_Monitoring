#!/usr/bin/env python3
"""
Model Diagnostic Service for Underfitting/Overfitting Detection
"""

import logging
import numpy as np
from typing import Dict, Any, Tuple
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

logger = logging.getLogger(__name__)

class ModelDiagnostic:
    """Model Diagnostic Service for Underfitting/Overfitting Detection"""
    
    def __init__(self):
        self.overfitting_threshold = 0.15  # 15% difference
        self.underfitting_threshold = 0.6   # R² < 0.6
        self.good_performance_threshold = 0.7  # R² > 0.7
    
    def diagnose_model_performance(self, 
                                 train_metrics: Dict[str, float],
                                 test_metrics: Dict[str, float]) -> Dict[str, Any]:
        """
        Diagnose model performance for underfitting/overfitting
        
        Args:
            train_metrics: Training metrics (R², MAE, RMSE)
            test_metrics: Test metrics (R², MAE, RMSE)
            
        Returns:
            Diagnosis results
        """
        try:
            train_r2 = train_metrics.get('r2_score', 0)
            test_r2 = test_metrics.get('r2_score', 0)
            
            # Calculate performance difference
            r2_difference = train_r2 - test_r2
            r2_difference_percent = (r2_difference / train_r2) * 100 if train_r2 > 0 else 0
            
            # Determine diagnosis
            diagnosis = self._determine_diagnosis(train_r2, test_r2, r2_difference_percent)
            
            # Calculate confidence level
            confidence = self._calculate_confidence(train_r2, test_r2, r2_difference_percent)
            
            # Generate recommendations
            recommendations = self._generate_recommendations(diagnosis, train_r2, test_r2, r2_difference_percent)
            
            results = {
                'diagnosis': diagnosis,
                'confidence': confidence,
                'train_r2': train_r2,
                'test_r2': test_r2,
                'r2_difference': r2_difference,
                'r2_difference_percent': r2_difference_percent,
                'recommendations': recommendations,
                'is_overfitting': diagnosis == 'overfitting',
                'is_underfitting': diagnosis == 'underfitting',
                'is_good': diagnosis == 'good'
            }
            
            # Log diagnosis
            logger.info("=== MODEL DIAGNOSTIC RESULTS ===")
            logger.info(f"Train R²: {train_r2:.4f}")
            logger.info(f"Test R²: {test_r2:.4f}")
            logger.info(f"R² Difference: {r2_difference:.4f} ({r2_difference_percent:.2f}%)")
            logger.info(f"Diagnosis: {diagnosis.upper()}")
            logger.info(f"Confidence: {confidence:.1f}%")
            
            return results
            
        except Exception as e:
            logger.error(f"Error in model diagnosis: {e}")
            return {'error': str(e)}
    
    def _determine_diagnosis(self, train_r2: float, test_r2: float, r2_diff_percent: float) -> str:
        """Determine if model is overfitting, underfitting, or good"""
        
        # Check for overfitting
        if r2_diff_percent > self.overfitting_threshold * 100:
            return 'overfitting'
        
        # Check for underfitting
        if test_r2 < self.underfitting_threshold:
            return 'underfitting'
        
        # Check for good performance
        if test_r2 >= self.good_performance_threshold and r2_diff_percent < 0.05 * 100:
            return 'good'
        
        # Borderline cases
        if test_r2 >= 0.6 and r2_diff_percent < 0.15 * 100:
            return 'acceptable'
        
        return 'poor'
    
    def _calculate_confidence(self, train_r2: float, test_r2: float, r2_diff_percent: float) -> float:
        """Calculate confidence level in diagnosis"""
        
        # Higher confidence if metrics are consistent
        if r2_diff_percent < 0.05 * 100 and test_r2 > 0.8:
            return 95.0
        elif r2_diff_percent < 0.1 * 100 and test_r2 > 0.7:
            return 85.0
        elif r2_diff_percent < 0.15 * 100 and test_r2 > 0.6:
            return 75.0
        else:
            return 60.0
    
    def _generate_recommendations(self, diagnosis: str, train_r2: float, test_r2: float, r2_diff_percent: float) -> list:
        """Generate recommendations based on diagnosis"""
        
        recommendations = []
        
        if diagnosis == 'overfitting':
            recommendations.extend([
                "🔴 REDUCE MODEL COMPLEXITY:",
                "  - Decrease number of trees (XGBoost) or layers (LSTM)",
                "  - Increase regularization (alpha, lambda)",
                "  - Add more dropout layers (LSTM)",
                "  - Use early stopping",
                "  - Reduce feature count",
                "  - Increase training data if possible"
            ])
        
        elif diagnosis == 'underfitting':
            recommendations.extend([
                "🔵 INCREASE MODEL CAPACITY:",
                "  - Add more trees (XGBoost) or layers (LSTM)",
                "  - Decrease regularization",
                "  - Add more features",
                "  - Try different algorithms",
                "  - Check data quality and preprocessing"
            ])
        
        elif diagnosis == 'good':
            recommendations.extend([
                "✅ MODEL IS WELL-BALANCED:",
                "  - Ready for production deployment",
                "  - Consider ensemble methods for further improvement",
                "  - Monitor performance on new data"
            ])
        
        elif diagnosis == 'acceptable':
            recommendations.extend([
                "🟡 MODEL NEEDS MINOR IMPROVEMENTS:",
                "  - Fine-tune hyperparameters",
                "  - Try feature engineering",
                "  - Consider ensemble methods"
            ])
        
        else:  # poor
            recommendations.extend([
                "❌ MODEL NEEDS MAJOR IMPROVEMENTS:",
                "  - Check data quality and preprocessing",
                "  - Try different algorithms",
                "  - Increase training data",
                "  - Review feature engineering"
            ])
        
        return recommendations
    
    def compare_models(self, model_results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Compare multiple models and recommend the best one
        
        Args:
            model_results: Dictionary of model results
                {
                    'xgboost': {'train_r2': 0.85, 'test_r2': 0.82, ...},
                    'lstm': {'train_r2': 0.80, 'test_r2': 0.78, ...},
                    'ensemble': {'train_r2': 0.87, 'test_r2': 0.85, ...}
                }
        
        Returns:
            Comparison results with recommendations
        """
        try:
            comparison = {}
            best_model = None
            best_score = -1
            
            for model_name, metrics in model_results.items():
                # Diagnose each model
                diagnosis = self.diagnose_model_performance(
                    {'r2_score': metrics.get('train_r2', 0)},
                    {'r2_score': metrics.get('test_r2', 0)}
                )
                
                comparison[model_name] = {
                    'diagnosis': diagnosis,
                    'test_r2': metrics.get('test_r2', 0),
                    'train_r2': metrics.get('train_r2', 0),
                    'is_overfitting': diagnosis['is_overfitting'],
                    'is_underfitting': diagnosis['is_underfitting'],
                    'is_good': diagnosis['is_good']
                }
                
                # Track best model (highest test R²)
                test_r2 = metrics.get('test_r2', 0)
                if test_r2 > best_score:
                    best_score = test_r2
                    best_model = model_name
            
            # Overall recommendation
            overall_recommendation = self._generate_overall_recommendation(comparison, best_model)
            
            results = {
                'model_comparison': comparison,
                'best_model': best_model,
                'best_test_r2': best_score,
                'overall_recommendation': overall_recommendation
            }
            
            # Log comparison
            logger.info("=== MODEL COMPARISON ===")
            for model_name, result in comparison.items():
                logger.info(f"{model_name.upper()}: Test R²={result['test_r2']:.4f}, Diagnosis={result['diagnosis']['diagnosis']}")
            logger.info(f"BEST MODEL: {best_model.upper()} (Test R²={best_score:.4f})")
            
            return results
            
        except Exception as e:
            logger.error(f"Error in model comparison: {e}")
            return {'error': str(e)}
    
    def _generate_overall_recommendation(self, comparison: Dict[str, Any], best_model: str) -> list:
        """Generate overall recommendation based on model comparison"""
        
        recommendations = []
        
        # Check if ensemble is best
        if best_model == 'ensemble':
            recommendations.append("🎯 ENSEMBLE MODEL IS BEST - Continue with ensemble approach")
        elif 'ensemble' in comparison:
            recommendations.append("🔄 Consider ensemble methods to improve performance")
        
        # Check for overfitting across models
        overfitting_models = [name for name, result in comparison.items() if result['is_overfitting']]
        if overfitting_models:
            recommendations.append(f"⚠️ Overfitting detected in: {', '.join(overfitting_models)}")
        
        # Check for underfitting across models
        underfitting_models = [name for name, result in comparison.items() if result['is_underfitting']]
        if underfitting_models:
            recommendations.append(f"⚠️ Underfitting detected in: {', '.join(underfitting_models)}")
        
        # Overall performance assessment
        good_models = [name for name, result in comparison.items() if result['is_good']]
        if good_models:
            recommendations.append(f"✅ Good performance models: {', '.join(good_models)}")
        
        return recommendations

# Global instance
model_diagnostic = ModelDiagnostic() 