#!/usr/bin/env python3
"""
Configuration Manager for Water Quality Monitoring Models
Loads hyperparameters from YAML file for easy tuning
"""

import yaml
import os
import logging
from typing import Dict, Any, List, Union

logger = logging.getLogger(__name__)

class ConfigManager:
    """Manages configuration loading and access for model hyperparameters"""
    
    def __init__(self, config_path: str = "config/hyperparameters.yaml"):
        """Initialize config manager"""
        self.config_path = config_path
        self.config = self._load_config()
        
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        try:
            if os.path.exists(self.config_path):
                with open(self.config_path, 'r', encoding='utf-8') as file:
                    config = yaml.safe_load(file)
                logger.info(f"Loaded configuration from {self.config_path}")
                return config
            else:
                logger.warning(f"Config file {self.config_path} not found, using defaults")
                return self._get_default_config()
        except Exception as e:
            logger.error(f"Error loading config: {e}")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration if file not found"""
        return {
            'xgboost': {
                'n_estimators_range': [100, 500],
                'max_depth_range': [3, 12],
                'learning_rate_range': [0.001, 0.2],
                'subsample_range': [0.6, 1.0],
                'colsample_bytree_range': [0.6, 1.0],
                'gamma_range': [0.0, 1.0],
                'min_child_weight_range': [1, 10],
                'reg_alpha_range': [0.0, 1.0],
                'reg_lambda_range': [0.0, 1.0],
                'n_trials': 20,
                'timeout': 300,
                'cv_splits': 3,
                'test_size_ratio': 0.2
            },
            'lstm': {
                'units_range': [64, 128, 256, 512],
                'dropout_range': [0.1, 0.4],
                'batch_size_range': [16, 32, 64, 128],
                'sequence_length_range': [12, 24, 36, 48],
                'learning_rate_range': [0.0001, 0.01],
                'layers_range': [1, 3],
                'optimizer_range': ['adam', 'rmsprop', 'nadam'],
                'use_attention': [True, False],
                'use_bidirectional': [True, False],
                'epochs': 100,
                'patience': 10,
                'lr_patience': 5,
                'lr_factor': 0.5,
                'min_lr': 1e-7,
                'n_trials': 15,
                'timeout': 600
            },
            'stacking': {
                'enabled': True,
                'meta_learners': ['ridge', 'random_forest'],
                'ridge_alpha': 1.0,
                'rf_n_estimators': 100
            },
            'features': {
                'basic_features': ['ph', 'temperature', 'do'],
                'temporal_features': ['month_sin', 'month_cos', 'quarter_sin', 'quarter_cos'],
                'seasonal_features': ['is_rainy_season', 'is_dry_season'],
                'lag_features': ['wqi_lag_1', 'wqi_lag_2', 'wqi_lag_3'],
                'rolling_features': ['wqi_rolling_mean_3', 'wqi_rolling_std_3'],
                'interaction_features': ['ph_temp_interaction', 'ph_do_interaction', 'temp_do_interaction'],
                'lag_windows': [1, 2, 3, 6, 12, 24],
                'rolling_windows': [3, 6, 12, 24]
            },
            'data': {
                'min_samples': 20,
                'test_size': 0.2,
                'random_state': 42
            },
            'model_selection': {
                'min_r2_score': 0.5,
                'max_mape': 15.0,
                'max_mae_ratio': 0.1,
                'xgboost_weight': 1.0,
                'lstm_weight': 1.0,
                'stacking_weight': 1.2
            },
            'logging': {
                'level': 'INFO',
                'save_models': True,
                'save_predictions': True,
                'plot_results': False
            },
            'targets': {
                'r2_score': 0.8,
                'mape': 10.0,
                'mae_ratio': 0.05
            }
        }
    
    def get_xgboost_config(self) -> Dict[str, Any]:
        """Get XGBoost configuration"""
        return self.config.get('xgboost', {})
    
    def get_lstm_config(self) -> Dict[str, Any]:
        """Get LSTM configuration"""
        return self.config.get('lstm', {})
    
    def get_stacking_config(self) -> Dict[str, Any]:
        """Get stacking configuration"""
        return self.config.get('stacking', {})
    
    def get_features_config(self) -> Dict[str, Any]:
        """Get features configuration"""
        return self.config.get('features', {})
    
    def get_data_config(self) -> Dict[str, Any]:
        """Get data processing configuration"""
        return self.config.get('data', {})
    
    def get_model_selection_config(self) -> Dict[str, Any]:
        """Get model selection configuration"""
        return self.config.get('model_selection', {})
    
    def get_logging_config(self) -> Dict[str, Any]:
        """Get logging configuration"""
        return self.config.get('logging', {})
    
    def get_targets_config(self) -> Dict[str, Any]:
        """Get performance targets configuration"""
        return self.config.get('targets', {})
    
    def reload_config(self):
        """Reload configuration from file"""
        self.config = self._load_config()
        logger.info("Configuration reloaded")
    
    def update_config(self, section: str, key: str, value: Any):
        """Update a specific configuration value"""
        if section in self.config:
            self.config[section][key] = value
            logger.info(f"Updated config: {section}.{key} = {value}")
        else:
            logger.warning(f"Section {section} not found in config")
    
    def save_config(self, output_path: str = None):
        """Save current configuration to file"""
        if output_path is None:
            output_path = self.config_path
        
        try:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as file:
                yaml.dump(self.config, file, default_flow_style=False, allow_unicode=True)
            logger.info(f"Configuration saved to {output_path}")
        except Exception as e:
            logger.error(f"Error saving config: {e}")
    
    def get_all_features(self) -> List[str]:
        """Get all available features"""
        features_config = self.get_features_config()
        all_features = []
        
        # Add basic features
        all_features.extend(features_config.get('basic_features', []))
        
        # Add advanced features
        all_features.extend(features_config.get('temporal_features', []))
        all_features.extend(features_config.get('seasonal_features', []))
        all_features.extend(features_config.get('lag_features', []))
        all_features.extend(features_config.get('rolling_features', []))
        all_features.extend(features_config.get('interaction_features', []))
        
        return all_features
    
    def is_stacking_enabled(self) -> bool:
        """Check if stacking is enabled"""
        stacking_config = self.get_stacking_config()
        return stacking_config.get('enabled', True)
    
    def get_meta_learners(self) -> List[str]:
        """Get available meta learners"""
        stacking_config = self.get_stacking_config()
        return stacking_config.get('meta_learners', ['ridge', 'random_forest'])

# Global config manager instance
config_manager = ConfigManager() 