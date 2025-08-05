#!/usr/bin/env python3
"""
LSTM Training Service for Global Multi-Series WQI Forecasting
"""

import sys
import os
import logging
import numpy as np
import pandas as pd
import tensorflow as tf
from typing import Dict, Any, Optional, Tuple
from datetime import datetime
import json

# Add the project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from include.iot_streaming.training_service import training_service

logger = logging.getLogger(__name__)

class LSTMTrainingService:
    """LSTM Training Service for Global Multi-Series WQI Forecasting"""
    
    def __init__(self):
        self.sequence_length = 6  # 6 months to predict next month
        self.model = None
        self.scaler = None
        self.feature_names = None
        self.training_service = training_service
        
    def generate_lstm_sequences(self, df: pd.DataFrame, sequence_length: int = 6) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate LSTM sequences from DataFrame
        
        Args:
            df: DataFrame with features and target
            sequence_length: Number of time steps to look back
            
        Returns:
            X_seq: Shape (samples, sequence_length, features)
            y: Shape (samples,)
        """
        try:
            # Sort by station_id and timestamp to maintain temporal order
            df_sorted = df.sort_values(['station_id', 'timestamp']).reset_index(drop=True)
            
            # Get feature columns (exclude target and metadata)
            feature_cols = [col for col in df_sorted.columns 
                          if col not in ['wqi', 'Date', 'timestamp']]
            
            # Create sequences per station
            X_sequences = []
            y_targets = []
            
            for station_id in sorted(df_sorted['station_id'].unique()):
                station_data = df_sorted[df_sorted['station_id'] == station_id].copy()
                
                if len(station_data) < sequence_length + 1:
                    logger.warning(f"Station {station_id}: Insufficient data for sequence length {sequence_length}")
                    continue
                
                # Create sequences for this station
                for i in range(len(station_data) - sequence_length):
                    # Input sequence
                    seq_features = station_data[feature_cols].iloc[i:i+sequence_length].values
                    X_sequences.append(seq_features)
                    
                    # Target (next WQI value)
                    target = station_data['wqi'].iloc[i+sequence_length]
                    y_targets.append(target)
            
            if not X_sequences:
                logger.error("No valid sequences generated")
                return None, None
            
            X_seq = np.array(X_sequences)
            y = np.array(y_targets)
            
            logger.info(f"Generated LSTM sequences: {X_seq.shape[0]} samples, "
                       f"sequence_length={sequence_length}, features={X_seq.shape[2]}")
            
            return X_seq, y
            
        except Exception as e:
            logger.error(f"Error generating LSTM sequences: {e}")
            return None, None
    
    def create_lstm_model(self, input_shape: Tuple[int, int], 
                         lstm_units: int = 64, 
                         dropout_rate: float = 0.2,
                         learning_rate: float = 0.001) -> tf.keras.Model:
        """
        Create LSTM model for WQI forecasting
        
        Args:
            input_shape: (sequence_length, num_features)
            lstm_units: Number of LSTM units
            dropout_rate: Dropout rate for regularization
            learning_rate: Learning rate for optimizer
            
        Returns:
            Compiled LSTM model
        """
        try:
            model = tf.keras.Sequential([
                # First LSTM layer
                tf.keras.layers.LSTM(
                    units=lstm_units,
                    return_sequences=True,
                    input_shape=input_shape,
                    dropout=dropout_rate,
                    recurrent_dropout=dropout_rate
                ),
                
                # Second LSTM layer
                tf.keras.layers.LSTM(
                    units=lstm_units // 2,
                    return_sequences=False,
                    dropout=dropout_rate,
                    recurrent_dropout=dropout_rate
                ),
                
                # Dense layers
                tf.keras.layers.Dense(32, activation='relu'),
                tf.keras.layers.Dropout(dropout_rate),
                tf.keras.layers.Dense(16, activation='relu'),
                tf.keras.layers.Dense(1, activation='linear')  # Linear for regression
            ])
            
            # Compile model
            optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
            model.compile(
                optimizer=optimizer,
                loss='mse',
                metrics=['mae', 'mape']
            )
            
            logger.info(f"Created LSTM model: {model.summary()}")
            return model
            
        except Exception as e:
            logger.error(f"Error creating LSTM model: {e}")
            return None
    
    def train_global_lstm(self, df: pd.DataFrame, 
                          lstm_units: int = 64,
                          dropout_rate: float = 0.2,
                          learning_rate: float = 0.001,
                          epochs: int = 100,
                          batch_size: int = 32,
                          validation_split: float = 0.2) -> Dict[str, Any]:
        """
        Train global LSTM model for all stations
        
        Args:
            df: DataFrame with features and target
            lstm_units: Number of LSTM units
            dropout_rate: Dropout rate
            learning_rate: Learning rate
            epochs: Number of training epochs
            batch_size: Batch size
            validation_split: Validation split ratio
            
        Returns:
            Training results dictionary
        """
        try:
            logger.info("=== STARTING GLOBAL LSTM TRAINING ===")
            
            # Generate sequences
            X_seq, y = self.generate_lstm_sequences(df, self.sequence_length)
            if X_seq is None or y is None:
                return {'error': 'Failed to generate sequences'}
            
            # Split data (maintain temporal order)
            split_idx = int(len(X_seq) * (1 - validation_split))
            X_train, X_val = X_seq[:split_idx], X_seq[split_idx:]
            y_train, y_val = y[:split_idx], y[split_idx:]
            
            logger.info(f"Data split: Train={len(X_train)}, Val={len(X_val)}")
            logger.info(f"Input shape: {X_train.shape}")
            
            # Create and train model
            model = self.create_lstm_model(
                input_shape=(X_train.shape[1], X_train.shape[2]),
                lstm_units=lstm_units,
                dropout_rate=dropout_rate,
                learning_rate=learning_rate
            )
            
            if model is None:
                return {'error': 'Failed to create LSTM model'}
            
            # Early stopping to prevent overfitting
            early_stopping = tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True
            )
            
            # Train model
            history = model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=epochs,
                batch_size=batch_size,
                callbacks=[early_stopping],
                verbose=1
            )
            
            # Evaluate model
            train_loss, train_mae, train_mape = model.evaluate(X_train, y_train, verbose=0)
            val_loss, val_mae, val_mape = model.evaluate(X_val, y_val, verbose=0)
            
            # Calculate R²
            y_train_pred = model.predict(X_train, verbose=0).flatten()
            y_val_pred = model.predict(X_val, verbose=0).flatten()
            
            from sklearn.metrics import r2_score
            train_r2 = r2_score(y_train, y_train_pred)
            val_r2 = r2_score(y_val, y_val_pred)
            
            # Prepare results
            results = {
                'model': model,
                'train_metrics': {
                    'loss': float(train_loss),
                    'mae': float(train_mae),
                    'mape': float(train_mape),
                    'r2': float(train_r2)
                },
                'val_metrics': {
                    'loss': float(val_loss),
                    'mae': float(val_mae),
                    'mape': float(val_mape),
                    'r2': float(val_r2)
                },
                'training_history': history.history,
                'input_shape': X_train.shape,
                'sequence_length': self.sequence_length,
                'hyperparameters': {
                    'lstm_units': lstm_units,
                    'dropout_rate': dropout_rate,
                    'learning_rate': learning_rate,
                    'epochs': epochs,
                    'batch_size': batch_size
                }
            }
            
            logger.info(f"✅ LSTM Training completed:")
            logger.info(f"  Train R²: {train_r2:.4f}, MAE: {train_mae:.4f}")
            logger.info(f"  Val R²: {val_r2:.4f}, MAE: {val_mae:.4f}")
            
            return results
            
        except Exception as e:
            logger.error(f"Error training LSTM model: {e}")
            import traceback
            logger.error(f"Full traceback: {traceback.format_exc()}")
            return {'error': str(e)}
    
    def predict_lstm(self, model: tf.keras.Model, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Make predictions using trained LSTM model
        
        Args:
            model: Trained LSTM model
            df: DataFrame with features
            
        Returns:
            y_true: True values
            y_pred: Predicted values
        """
        try:
            # Generate sequences for prediction
            X_seq, y_true = self.generate_lstm_sequences(df, self.sequence_length)
            if X_seq is None or y_true is None:
                return None, None
            
            # Make predictions
            y_pred = model.predict(X_seq, verbose=0).flatten()
            
            logger.info(f"LSTM predictions: {len(y_pred)} samples")
            logger.info(f"True range: {y_true.min():.2f} - {y_true.max():.2f}")
            logger.info(f"Pred range: {y_pred.min():.2f} - {y_pred.max():.2f}")
            
            return y_true, y_pred
            
        except Exception as e:
            logger.error(f"Error making LSTM predictions: {e}")
            return None, None
    
    def save_lstm_model(self, model: tf.keras.Model, model_path: str) -> bool:
        """Save LSTM model to file"""
        try:
            model.save(model_path)
            logger.info(f"LSTM model saved to: {model_path}")
            return True
        except Exception as e:
            logger.error(f"Error saving LSTM model: {e}")
            return False
    
    def load_lstm_model(self, model_path: str) -> Optional[tf.keras.Model]:
        """Load LSTM model from file"""
        try:
            model = tf.keras.models.load_model(model_path)
            logger.info(f"LSTM model loaded from: {model_path}")
            return model
        except Exception as e:
            logger.error(f"Error loading LSTM model: {e}")
            return None

# Global instance
lstm_training_service = LSTMTrainingService() 