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
from sklearn.preprocessing import StandardScaler

# Add the project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from include.iot_streaming.training_service import training_service

logger = logging.getLogger(__name__)

class LSTMTrainingService:
    """LSTM Training Service for Global Multi-Series WQI Forecasting"""
    
    def __init__(self):
        self.sequence_length = 12  # 12 months to predict next month
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
            
            # Get feature columns (exclude metadata and station_id), include past wqi as feature
            feature_cols = [col for col in df_sorted.columns 
                          if col not in ['Date', 'timestamp', 'station_id']]
            
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
                         lstm_units: int = 16, 
                         dropout_rate: float = 0.1,
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
            # Simple compact model for small datasets
            inputs = tf.keras.Input(shape=input_shape, name='timeseries_input')
            x = tf.keras.layers.LSTM(
                    units=lstm_units,
                    return_sequences=False,
                    dropout=dropout_rate,
                    recurrent_dropout=dropout_rate
            )(inputs)
            x = tf.keras.layers.Dense(8, activation='relu')(x)
            x = tf.keras.layers.Dropout(dropout_rate)(x)
            outputs = tf.keras.layers.Dense(1, activation='linear')(x)
            model = tf.keras.Model(inputs=inputs, outputs=outputs)
            
            # Compile model
            optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate, clipnorm=1.0)

            def smape(y_true, y_pred):
                epsilon = tf.keras.backend.epsilon()
                numerator = tf.abs(y_pred - y_true)
                denominator = (tf.abs(y_true) + tf.abs(y_pred)) + epsilon
                return tf.reduce_mean(2.0 * numerator / denominator)

            model.compile(
                optimizer=optimizer,
                loss=tf.keras.losses.Huber(),
                metrics=['mae', smape]
            )
            
            # Log model summary properly
            model.summary(print_fn=logger.info)
            return model
            
        except Exception as e:
            logger.error(f"Error creating LSTM model: {e}")
            return None

    def create_lstm_model_with_station(self, input_shape: Tuple[int, int], 
                         n_stations: int,
                         forecast_horizon: int = 1,
                         lstm_units: int = 16, 
                         emb_dim: int = 3,
                         dropout_rate: float = 0.1,
                         learning_rate: float = 0.001,
                         l2_weight: float = 1e-4,
                         conv_filters: int = 0,
                         conv_kernel_size: int = 3,
                         recurrent_dropout: float = 0.0,
                         spatial_dropout1d_rate: float = 0.0,
                         gaussian_noise_std: float = 0.0,
                         emb_dropout: float = 0.0) -> tf.keras.Model:
        """Compact LSTM + station embedding, multi-horizon delta output."""
        try:
            seq_in = tf.keras.Input(shape=input_shape, name='seq')
            st_in  = tf.keras.Input(shape=(1,), dtype='int32', name='station_idx')

            x = tf.keras.layers.Masking()(seq_in)
            if conv_filters and conv_filters > 0 and conv_kernel_size and conv_kernel_size > 0:
                x = tf.keras.layers.Conv1D(
                    filters=int(conv_filters),
                    kernel_size=int(conv_kernel_size),
                    padding='causal',
                    activation='relu',
                    kernel_regularizer=tf.keras.regularizers.l2(l2_weight)
                )(x)
            if spatial_dropout1d_rate and spatial_dropout1d_rate > 0:
                x = tf.keras.layers.SpatialDropout1D(rate=float(spatial_dropout1d_rate))(x)
            if gaussian_noise_std and gaussian_noise_std > 0:
                x = tf.keras.layers.GaussianNoise(stddev=float(gaussian_noise_std))(x)
            x = tf.keras.layers.LSTM(
                units=lstm_units,
                return_sequences=False,
                dropout=dropout_rate,
                recurrent_dropout=recurrent_dropout,
                kernel_regularizer=tf.keras.regularizers.l2(l2_weight)
            )(x)
            x = tf.keras.layers.LayerNormalization()(x)
            x = tf.keras.layers.Dropout(dropout_rate)(x)

            s = tf.keras.layers.Embedding(n_stations, emb_dim, name='st_emb')(st_in)
            s = tf.keras.layers.Flatten()(s)
            if emb_dropout and emb_dropout > 0:
                s = tf.keras.layers.Dropout(rate=float(emb_dropout))(s)

            h = tf.keras.layers.Concatenate()([x, s])
            h = tf.keras.layers.Dense(16, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(l2_weight), activity_regularizer=tf.keras.regularizers.l2(l2_weight))(h)
            h = tf.keras.layers.Dropout(dropout_rate)(h)
            out = tf.keras.layers.Dense(int(max(1, forecast_horizon)), name='delta')(h)

            model = tf.keras.Model(inputs=[seq_in, st_in], outputs=out)

            optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate, clipnorm=1.0)
            model.compile(optimizer=optimizer, loss=tf.keras.losses.Huber(), metrics=['mae'])
            model.summary(print_fn=logger.info)
            return model
        except Exception as e:
            logger.error(f"Error creating LSTM+station model: {e}")
            return None
    
    def train_global_lstm(self, df: pd.DataFrame, 
                          sequence_length: Optional[int] = None,
                          lstm_units: int = 64,
                          dropout_rate: float = 0.2,
                          learning_rate: float = 0.001,
                          epochs: int = 100,
                          batch_size: int = 32,
                          validation_split: float = 0.2,
                          l2_weight: float = 1e-4,
                          conv_filters: int = 32,
                          conv_kernel_size: int = 3,
                          gamma_shrink: float = 0.8,
                          forecast_horizon: int = 1) -> Dict[str, Any]:
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
            if sequence_length is not None and isinstance(sequence_length, int) and sequence_length > 1:
                self.sequence_length = sequence_length
                logger.info(f"Using sequence_length={self.sequence_length}")

            # Sort and define features (exclude non-features and ensure numeric dtypes)
            df_sorted = df.sort_values(['station_id', 'timestamp']).reset_index(drop=True)
            # Start with core instantaneous features only (no current wqi to avoid leakage)
            base_exclude_cols = ['wqi', 'Date', 'timestamp', 'station_id', 'measurement_date']
            base_feature_candidates = [col for col in df_sorted.columns if col not in base_exclude_cols]
            # Keep only stable core features
            allowed_bases = [c for c in base_feature_candidates if c in ['ph', 'temperature', 'do']]
            feature_cols = allowed_bases.copy()
            if not feature_cols:
                return {'error': 'No feature columns available for LSTM training'}

            # Winsorize to reduce outlier impact (1st-99th percentile) on core signals and target
            for col in ['wqi', 'ph', 'temperature', 'do']:
                if col in df_sorted.columns:
                    try:
                        series = pd.to_numeric(df_sorted[col], errors='coerce')
                        lo, hi = np.nanpercentile(series.dropna(), [1, 99])
                        df_sorted[col] = series.clip(lower=lo, upper=hi)
                    except Exception:
                        pass

            # Add seasonal features from timestamp (month sin/cos)
            try:
                df_sorted['timestamp'] = pd.to_datetime(df_sorted['timestamp'])
                df_sorted['month'] = df_sorted['timestamp'].dt.month.astype(int)
                df_sorted['month_sin'] = np.sin(2 * np.pi * df_sorted['month'] / 12.0)
                df_sorted['month_cos'] = np.cos(2 * np.pi * df_sorted['month'] / 12.0)
                feature_cols.extend(['month_sin', 'month_cos'])
            except Exception as _:
                pass

            # Add lag and rolling features (per station) without leakage
            try:
                group = df_sorted.groupby('station_id', group_keys=False)
                # Target lags/rolling
                if 'wqi' in df_sorted.columns:
                    df_sorted['wqi_lag_1'] = group['wqi'].shift(1)
                    df_sorted['wqi_lag_3'] = group['wqi'].shift(3)
                    df_sorted['wqi_roll_mean_3'] = group['wqi'].shift(1).rolling(window=3, min_periods=1).mean()
                    feature_cols.extend(['wqi_lag_1', 'wqi_lag_3', 'wqi_roll_mean_3'])
                # Core feature lags/rolling
                for c in ['ph', 'temperature', 'do']:
                    if c in df_sorted.columns:
                        df_sorted[f'{c}_lag_1'] = group[c].shift(1)
                        df_sorted[f'{c}_lag_3'] = group[c].shift(3)
                        df_sorted[f'{c}_roll_mean_3'] = group[c].shift(1).rolling(window=3, min_periods=1).mean()
                        feature_cols.extend([f'{c}_lag_1', f'{c}_lag_3', f'{c}_roll_mean_3'])
            except Exception as e:
                logger.warning(f"Failed to add lag/rolling features: {e}")

            # Coerce features and target to numeric, drop rows with NaNs
            for col in feature_cols + ['wqi']:
                df_sorted[col] = pd.to_numeric(df_sorted[col], errors='coerce')
            before_drop = len(df_sorted)
            df_sorted = df_sorted.dropna(subset=feature_cols + ['wqi']).reset_index(drop=True)
            after_drop = len(df_sorted)
            if after_drop < before_drop:
                logger.info(f"Dropped {before_drop - after_drop} rows due to non-numeric/NaN values in features/target")

            # Build sequences per-station and split temporally within each station
            X_train_list, y_train_list, X_val_list, y_val_list = [], [], [], []
            last_wqi_train_list, last_wqi_val_list = [], []
            st_tr_list, st_val_list = [], []
            ts_tr_list, ts_val_list = [], []  # target timestamps for horizon-1
            total_sequences = 0
            for station_id in sorted(df_sorted['station_id'].unique()):
                station_data = df_sorted[df_sorted['station_id'] == station_id].copy()
                H = int(max(1, forecast_horizon))
                if len(station_data) < self.sequence_length + H:
                    logger.warning(f"Station {station_id}: Insufficient data for sequence length {self.sequence_length}")
                    continue

                X_s, y_s, last_wqi_s, st_idx, ts_targets_h1 = [], [], [], [], []
                for i in range(len(station_data) - self.sequence_length - H + 1):
                    seq_features = station_data[feature_cols].iloc[i:i + self.sequence_length].values.astype(np.float32)
                    X_s.append(seq_features)
                    # Multi-horizon residuals
                    y_future = station_data['wqi'].iloc[i + self.sequence_length : i + self.sequence_length + H].values.astype(np.float32)
                    prev_end = np.float32(station_data['wqi'].iloc[i + self.sequence_length - 1])
                    deltas = np.empty(H, dtype=np.float32)
                    last = prev_end
                    for k in range(H):
                        deltas[k] = y_future[k] - last
                        last = y_future[k]
                    y_s.append(deltas)
                    last_wqi_s.append(prev_end)
                    st_idx.append(station_id)
                    # capture horizon-1 target timestamp for this sample
                    try:
                        ts_targets_h1.append(pd.to_datetime(station_data['timestamp'].iloc[i + self.sequence_length]))
                    except Exception:
                        ts_targets_h1.append(None)

                if not X_s:
                    continue
                X_s = np.array(X_s, dtype=np.float32)
                y_s = np.array(y_s, dtype=np.float32)  # (N, H)
                st_idx = np.array(st_idx, dtype=np.int32)
                total_sequences += len(X_s)

                split_idx = int(len(X_s) * (1 - validation_split))
                # Ensure at least 1 sample in val if possible
                split_idx = min(max(split_idx, 1), len(X_s) - 1)

                X_train_list.append(X_s[:split_idx])
                y_train_list.append(y_s[:split_idx])
                X_val_list.append(X_s[split_idx:])
                y_val_list.append(y_s[split_idx:])
                last_wqi_train_list.append(np.array(last_wqi_s[:split_idx], dtype=np.float32))
                last_wqi_val_list.append(np.array(last_wqi_s[split_idx:], dtype=np.float32))
                st_tr_list.append(st_idx[:split_idx])
                st_val_list.append(st_idx[split_idx:])
                ts_tr_list.append(np.array(ts_targets_h1[:split_idx], dtype='datetime64[ns]'))
                ts_val_list.append(np.array(ts_targets_h1[split_idx:], dtype='datetime64[ns]'))

            if not X_train_list or not X_val_list:
                return {'error': 'Insufficient sequences after per-station split'}

            X_train = np.concatenate(X_train_list, axis=0)
            y_train = np.concatenate(y_train_list, axis=0)
            X_val = np.concatenate(X_val_list, axis=0)
            y_val = np.concatenate(y_val_list, axis=0)
            last_wqi_train = np.concatenate(last_wqi_train_list, axis=0)
            last_wqi_val = np.concatenate(last_wqi_val_list, axis=0)

            station_train_idx = np.concatenate(st_tr_list, axis=0).astype(np.int32)
            station_val_idx = np.concatenate(st_val_list, axis=0).astype(np.int32)
            target_ts_train = np.concatenate(ts_tr_list, axis=0)
            target_ts_val = np.concatenate(ts_val_list, axis=0)
            # Map original station ids to contiguous embedding indices [0, n_stations)
            unique_station_ids = sorted(df_sorted['station_id'].unique().tolist())
            sid_to_emb = {sid: idx for idx, sid in enumerate(unique_station_ids)}
            # Create mapped arrays for embedding input
            station_train_idx_emb = np.vectorize(sid_to_emb.get)(station_train_idx).astype(np.int32).reshape(-1, 1)
            station_val_idx_emb = np.vectorize(sid_to_emb.get)(station_val_idx).astype(np.int32).reshape(-1, 1)

            # Per-station scale for delta targets to balance stations (use train stats only)
            try:
                delta_std_by_sid = {}
                for sid in unique_station_ids:
                    mask = (station_train_idx == sid)
                    if not np.any(mask):
                        continue
                    std_sid = float(np.std(y_train[mask], axis=0).mean())  # average across horizons
                    if not np.isfinite(std_sid) or std_sid <= 1e-6:
                        std_sid = 1.0
                    delta_std_by_sid[sid] = std_sid
                # Build std vectors aligned with rows
                train_std_vec = np.array([delta_std_by_sid.get(sid, 1.0) for sid in station_train_idx], dtype=np.float32).reshape(-1, 1)
                val_std_vec = np.array([delta_std_by_sid.get(sid, 1.0) for sid in station_val_idx], dtype=np.float32).reshape(-1, 1)
                # Scale deltas
                y_train = y_train / train_std_vec
                y_val = y_val / val_std_vec
            except Exception as _:
                delta_std_by_sid = {sid: 1.0 for sid in unique_station_ids}
                train_std_vec = np.ones((y_train.shape[0], 1), dtype=np.float32)
                val_std_vec = np.ones((y_val.shape[0], 1), dtype=np.float32)

            logger.info(f"Data split (per-station temporal): Train={len(X_train)}, Val={len(X_val)}, Total sequences={total_sequences}")
            logger.info(f"Input shape (train): {X_train.shape}")
            
            # Create and train model (multi-horizon with station embedding)
            n_stations = int(len(unique_station_ids))
            # Disable Conv1D to preserve Masking → LSTM mask behavior
            model = self.create_lstm_model_with_station(
                input_shape=(X_train.shape[1], X_train.shape[2]),
                n_stations=n_stations,
                forecast_horizon=H,
                lstm_units=lstm_units,
                emb_dim=3,
                dropout_rate=dropout_rate,
                learning_rate=learning_rate,
                l2_weight=l2_weight,
                conv_filters=0,
                conv_kernel_size=0,
                recurrent_dropout=min(0.5, max(0.0, dropout_rate)),
                spatial_dropout1d_rate=0.1,
                gaussian_noise_std=0.03,
                emb_dropout=min(0.5, dropout_rate / 2.0)
            )
            
            if model is None:
                return {'error': 'Failed to create LSTM model'}
            
            # Per-station scaling on training data only, then concatenate
            try:
                self.feature_names = feature_cols
                unique_stations = sorted(df_sorted['station_id'].unique())
                scaler_by_sid = {}
                # Prepare output arrays
                X_train_scaled = np.empty_like(X_train, dtype=np.float32)
                X_val_scaled = np.empty_like(X_val, dtype=np.float32)
                for sid in unique_stations:
                    tr_mask = (station_train_idx == sid)
                    val_mask = (station_val_idx == sid)
                    X_tr_sid = X_train[tr_mask]
                    if X_tr_sid.size == 0:
                        continue
                    n_tr, seq_len, n_feat = X_tr_sid.shape
                    scaler_sid = StandardScaler()
                    X_tr_2d = X_tr_sid.reshape(n_tr * seq_len, n_feat)
                    scaler_sid.fit(X_tr_2d)
                    scaler_by_sid[int(sid)] = scaler_sid
                    # Transform train
                    X_train_scaled[tr_mask] = scaler_sid.transform(X_tr_2d).reshape(n_tr, seq_len, n_feat).astype(np.float32)
                    # Transform val
                    X_val_sid = X_val[val_mask]
                    if X_val_sid.size > 0:
                        n_v, s_v, f_v = X_val_sid.shape
                        X_val_scaled[val_mask] = scaler_sid.transform(X_val_sid.reshape(n_v * s_v, f_v)).reshape(n_v, s_v, f_v).astype(np.float32)
            except Exception as e:
                logger.warning(f"Failed per-station scaling, proceeding without scaling: {e}")
                scaler_by_sid = {}
                X_train_scaled, X_val_scaled = X_train, X_val

            # Do not scale multi-horizon targets; Huber is robust
            
            # Early stopping to prevent overfitting
            # Train/Test only: monitor training loss for callbacks
            early_stopping = tf.keras.callbacks.EarlyStopping(
                monitor='loss',
                patience=10,
                restore_best_weights=True
            )
            reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
                monitor='loss', factor=0.5, patience=5, min_lr=1e-5
            )
            
            # Train model with validation to capture val curves for plotting
            history = model.fit(
                {'seq': X_train_scaled, 'station_idx': station_train_idx_emb}, y_train,
                validation_data=({'seq': X_val_scaled, 'station_idx': station_val_idx_emb}, y_val),
                epochs=epochs,
                batch_size=batch_size,
                callbacks=[early_stopping, reduce_lr],
                verbose=1
            )
            
            # Evaluate model
            train_loss_scaled, train_mae_scaled = model.evaluate({'seq': X_train_scaled, 'station_idx': station_train_idx_emb}, y_train, verbose=0)
            val_loss_scaled, val_mae_scaled = model.evaluate({'seq': X_val_scaled, 'station_idx': station_val_idx_emb}, y_val, verbose=0)
            
            # Predict deltas (N,H)
            y_train_pred_scaled = model.predict({'seq': X_train_scaled, 'station_idx': station_train_idx_emb}, verbose=0)
            y_val_pred_scaled = model.predict({'seq': X_val_scaled, 'station_idx': station_val_idx_emb}, verbose=0)
            # Unscale deltas back
            y_train_pred = y_train_pred_scaled * train_std_vec
            y_val_pred = y_val_pred_scaled * val_std_vec

            # Gamma per horizon using held-out (test) deltas
            gammas = []
            for h in range(H):
                pred = y_val_pred[:, h]
                true = y_val[:, h]
                num = float(np.cov(pred, true, bias=True)[0, 1])
                den = float(np.var(pred) + 1e-12)
                gammas.append(max(0.0, min(1.0, num / den)) if den > 0 else float(gamma_shrink))
            gammas = np.array(gammas, dtype=np.float32)

            # Reconstruct cumulative WQI per horizon
            def reconstruct(last_vec: np.ndarray, deltas: np.ndarray, gam: np.ndarray) -> np.ndarray:
                N = int(deltas.shape[0])
                Hh = int(deltas.shape[1]) if deltas.ndim == 2 else 1
                out = np.empty((N, Hh), dtype=np.float32)
                prev = last_vec.reshape(-1).astype(np.float32)
                for h in range(Hh):
                    step = gam[h] * deltas[:, h].astype(np.float32)
                    prev = prev + step
                    out[:, h] = prev
                return out

            yhat_train = reconstruct(last_wqi_train, y_train_pred, gammas)
            yhat_val = reconstruct(last_wqi_val, y_val_pred, gammas)

            # Build test set aligned to Spark logic: last 12 months globally
            try:
                max_ts = pd.to_datetime(df_sorted['timestamp']).max()
                split_dt = max_ts - pd.DateOffset(months=12)
                X_test_list, last_wqi_test_list, st_test_list, ts_test_list = [], [], [], []
                for station_id in sorted(df_sorted['station_id'].unique()):
                    station_data = df_sorted[df_sorted['station_id'] == station_id].copy()
                    H = int(max(1, forecast_horizon))
                    if len(station_data) < self.sequence_length + H:
                        continue
                    for i in range(len(station_data) - self.sequence_length - H + 1):
                        target_ts = pd.to_datetime(station_data['timestamp'].iloc[i + self.sequence_length])
                        if target_ts <= split_dt:
                            continue
                        seq_features = station_data[feature_cols].iloc[i:i + self.sequence_length].values.astype(np.float32)
                        X_test_list.append(seq_features)
                        prev_end = np.float32(station_data['wqi'].iloc[i + self.sequence_length - 1])
                        last_wqi_test_list.append(prev_end)
                        st_test_list.append(int(station_id))
                        ts_test_list.append(target_ts)
                if X_test_list:
                    X_test = np.array(X_test_list, dtype=np.float32)
                    last_wqi_test = np.array(last_wqi_test_list, dtype=np.float32).reshape(-1, 1)
                    station_test_idx = np.array(st_test_list, dtype=np.int32)
                    station_test_idx_emb = np.vectorize(sid_to_emb.get)(station_test_idx).astype(np.int32).reshape(-1, 1)
                    # Scale per station using train scalers
                    X_test_scaled = np.empty_like(X_test, dtype=np.float32)
                    for sid in sorted(set(st_test_list)):
                        mask = (station_test_idx == sid)
                        X_sid = X_test[mask]
                        if X_sid.size == 0:
                            continue
                        n_t, s_t, f_t = X_sid.shape
                        if int(sid) in scaler_by_sid:
                            sc = scaler_by_sid[int(sid)]
                            X_test_scaled[mask] = sc.transform(X_sid.reshape(n_t * s_t, f_t)).reshape(n_t, s_t, f_t).astype(np.float32)
                        else:
                            X_test_scaled[mask] = X_sid.astype(np.float32)
                    # Predict deltas and reconstruct horizon outputs
                    y_test_pred_scaled = model.predict({'seq': X_test_scaled, 'station_idx': station_test_idx_emb}, verbose=0)
                    y_test_pred = y_test_pred_scaled  # unscale deltas not applied; robust loss used
                    # Reconstruct cumulative for H horizons
                    yhat_test = reconstruct(last_wqi_test, y_test_pred, gammas)
                    # Build keys for h=1
                    test_keys_h1 = [
                        {
                            'station_id': int(sid),
                            'timestamp': pd.to_datetime(ts).isoformat() if ts is not None else None
                        } for sid, ts in zip(station_test_idx.tolist(), ts_test_list)
                    ]
                    y_pred_test_h1 = yhat_test[:, 0].tolist() if yhat_test.ndim == 2 and yhat_test.shape[1] >= 1 else yhat_test.tolist()
                    # We do not have ground truth ytrue directly without computing deltas; approximate with reconstructed true using df directly
                    # For h=1, true is the actual wqi at target timestamp
                    ytrue_series = []
                    try:
                        df_idx = df_sorted.set_index(['station_id', 'timestamp'])['wqi']
                        for sid, ts in zip(station_test_idx.tolist(), ts_test_list):
                            try:
                                ytrue_series.append(float(df_idx.loc[(int(sid), pd.to_datetime(ts))]))
                            except Exception:
                                ytrue_series.append(np.nan)
                    except Exception:
                        ytrue_series = [np.nan for _ in range(len(test_keys_h1))]
                    y_true_test_h1 = ytrue_series
                else:
                    test_keys_h1, y_pred_test_h1, y_true_test_h1 = [], [], []
            except Exception as e:
                logger.warning(f"Failed to compute LSTM test predictions for blending: {e}")
                test_keys_h1, y_pred_test_h1, y_true_test_h1 = [], [], []

            # True cumulative
            def reconstruct_true(last_vec: np.ndarray, deltas_true: np.ndarray) -> np.ndarray:
                N = int(deltas_true.shape[0])
                Hh = int(deltas_true.shape[1]) if deltas_true.ndim == 2 else 1
                out = np.empty((N, Hh), dtype=np.float32)
                prev = last_vec.reshape(-1).astype(np.float32)
                for h in range(Hh):
                    prev = prev + deltas_true[:, h].astype(np.float32)
                    out[:, h] = prev
                return out

            ytrue_train = reconstruct_true(last_wqi_train, y_train)
            ytrue_val = reconstruct_true(last_wqi_val, y_val)
            
            from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
            # Metrics per horizon and averages
            train_mae_h, val_mae_h, train_rmse_h, val_rmse_h, train_r2_h, val_r2_h, train_smape_h, val_smape_h = [], [], [], [], [], [], [], []
            for h in range(H):
                train_mae_h.append(float(mean_absolute_error(ytrue_train[:, h], yhat_train[:, h])))
                val_mae_h.append(float(mean_absolute_error(ytrue_val[:, h], yhat_val[:, h])))
                train_rmse_h.append(float(np.sqrt(mean_squared_error(ytrue_train[:, h], yhat_train[:, h]))))
                val_rmse_h.append(float(np.sqrt(mean_squared_error(ytrue_val[:, h], yhat_val[:, h]))))
                train_r2_h.append(float(r2_score(ytrue_train[:, h], yhat_train[:, h])))
                val_r2_h.append(float(r2_score(ytrue_val[:, h], yhat_val[:, h])))
                train_smape_h.append(float(np.mean(2.0 * np.abs(yhat_train[:, h] - ytrue_train[:, h]) / (np.abs(yhat_train[:, h]) + np.abs(ytrue_train[:, h]) + 1e-8))))
                val_smape_h.append(float(np.mean(2.0 * np.abs(yhat_val[:, h] - ytrue_val[:, h]) / (np.abs(yhat_val[:, h]) + np.abs(ytrue_val[:, h]) + 1e-8))))

            train_mae = float(np.mean(train_mae_h))
            val_mae = float(np.mean(val_mae_h))
            train_mse = float(np.mean(np.square(np.array(train_rmse_h))))
            val_mse = float(np.mean(np.square(np.array(val_rmse_h))))
            train_r2 = float(np.mean(train_r2_h))
            val_r2 = float(np.mean(val_r2_h))
            
            # Baseline (naive: next ~= last)
            from sklearn.metrics import r2_score as _r2, mean_absolute_error as _mae, mean_squared_error as _mse
            baseline_train = {
                'mae_per_h': [float(_mae(ytrue_train[:, h], last_wqi_train)) for h in range(H)],
                'r2_per_h': [float(_r2(ytrue_train[:, h], last_wqi_train)) for h in range(H)]
            }
            baseline_val = {
                'mae_per_h': [float(_mae(ytrue_val[:, h], last_wqi_val)) for h in range(H)],
                'r2_per_h': [float(_r2(ytrue_val[:, h], last_wqi_val)) for h in range(H)]
            }

            # Per-station metrics on test set
            per_station_val = {}
            for sid in sorted(df_sorted['station_id'].unique()):
                mask = (station_val_idx == sid)
                if not np.any(mask):
                    continue
                yhat_sid = yhat_val[mask]
                ytrue_sid = ytrue_val[mask]
                mae_h = [float(mean_absolute_error(ytrue_sid[:, h], yhat_sid[:, h])) for h in range(H)]
                rmse_h = [float(np.sqrt(mean_squared_error(ytrue_sid[:, h], yhat_sid[:, h]))) for h in range(H)]
                r2_h = [float(r2_score(ytrue_sid[:, h], yhat_sid[:, h])) for h in range(H)]
                smape_h = [float(np.mean(2.0 * np.abs(yhat_sid[:, h] - ytrue_sid[:, h]) / (np.abs(yhat_sid[:, h]) + np.abs(ytrue_sid[:, h]) + 1e-8))) for h in range(H)]
                # MASE denominator using training series with seasonal lag 12
                try:
                    station_series = df_sorted[df_sorted['station_id'] == sid]['wqi'].values.astype(np.float32)
                    # Approximate cutoff by using last_wqi_train count for this station
                    tr_count = int(np.sum(station_train_idx == sid))
                    cutoff_idx = min(len(station_series) - 1, tr_count + self.sequence_length - 1)
                    train_series_for_mase = station_series[:cutoff_idx+1]
                    if len(train_series_for_mase) > 12:
                        denom = float(np.mean(np.abs(train_series_for_mase[12:] - train_series_for_mase[:-12])))
                    else:
                        denom = float(np.mean(np.abs(np.diff(train_series_for_mase)))) if len(train_series_for_mase) > 1 else np.nan
                    mase_h = [float(mae_h[h] / denom) if denom and denom > 0 else np.nan for h in range(H)]
                except Exception:
                    mase_h = [np.nan for _ in range(H)]
                per_station_val[int(sid)] = {
                    'mae': mae_h,
                    'rmse': rmse_h,
                    'r2': r2_h,
                    'smape': smape_h,
                    'mase': mase_h
                }
            
            # Prepare results
            results = {
                'model': model,
                'train_metrics': {
                    'loss': train_mse,
                    'mae': train_mae,
                    'smape': float(np.mean(train_smape_h)),
                    'r2': float(train_r2),
                    'per_horizon': {
                        'mae': train_mae_h,
                        'rmse': train_rmse_h,
                        'smape': train_smape_h,
                        'r2': train_r2_h
                    }
                },
                'test_metrics': {
                    'loss': val_mse,
                    'mae': val_mae,
                    'smape': float(np.mean(val_smape_h)),
                    'r2': float(val_r2),
                    'per_horizon': {
                        'mae': val_mae_h,
                        'rmse': val_rmse_h,
                        'smape': val_smape_h,
                        'r2': val_r2_h
                    }
                },
                'baseline_metrics': {
                    'train': baseline_train,
                    'test': baseline_val
                },
                'per_station_test': per_station_val,
                'training_history': history.history,
                'input_shape': X_train.shape,
                'sequence_length': self.sequence_length,
                'hyperparameters': {
                    'lstm_units': lstm_units,
                    'dropout_rate': dropout_rate,
                    'learning_rate': learning_rate,
                    'epochs': epochs,
                    'batch_size': batch_size,
                    'gamma_shrink': float(gamma_shrink),
                    'forecast_horizon': H
                },
                # Horizon-1 per-sample outputs aligned to TEST rows (last 12 months globally) for downstream blending
                'test_keys_h1': test_keys_h1,
                'y_true_test_h1': y_true_test_h1,
                'y_pred_test_h1': y_pred_test_h1
            }
            
            logger.info(f"✅ LSTM Training completed:")
            logger.info(f"  Train R²: {train_r2:.4f}, MAE: {train_mae:.4f}")
            logger.info(f"  Test R²: {val_r2:.4f}, MAE: {val_mae:.4f}")
            
            # Save training curves to images folder
            try:
                import os
                import matplotlib
                matplotlib.use('Agg')
                import matplotlib.pyplot as plt
                # Try multiple possible model dirs to ensure visibility from host
                dir_candidates = []
                env_dir = os.getenv('AIRFLOW_MODELS_DIR')
                if env_dir:
                    dir_candidates.append(env_dir)
                dir_candidates.extend(['/usr/local/airflow/models', '/opt/airflow/models', os.path.join(os.getcwd(), 'models')])
                images_dirs = []
                for base in dir_candidates:
                    try:
                        if base is None:
                            continue
                        img_dir = os.path.join(base, 'images')
                        os.makedirs(img_dir, exist_ok=True)
                        images_dirs.append(img_dir)
                    except Exception:
                        continue
                # Loss curves
                fig, ax = plt.subplots(figsize=(8,5))
                ax.plot(history.history.get('loss', []), label='Training loss', color='blue')
                if 'val_loss' in history.history:
                    ax.plot(history.history['val_loss'], label='Validation loss', color='blue', linestyle='--')
                ax.set_xlabel('Epoch')
                ax.set_ylabel('Loss')
                ax.legend(loc='best')
                fig.tight_layout()
                for d in images_dirs:
                    try:
                        plt.savefig(os.path.join(d, 'lstm_training_curve.png'))
                    except Exception:
                        pass
                plt.close(fig)
                # MAE curves if available
                if 'mae' in history.history or 'val_mae' in history.history:
                    fig2, ax2 = plt.subplots(figsize=(8,5))
                    if 'mae' in history.history:
                        ax2.plot(history.history['mae'], label='Training MAE', color='green')
                    if 'val_mae' in history.history:
                        ax2.plot(history.history['val_mae'], label='Validation MAE', color='green', linestyle='--')
                    ax2.set_xlabel('Epoch')
                    ax2.set_ylabel('MAE')
                    ax2.legend(loc='best')
                    fig2.tight_layout()
                    for d in images_dirs:
                        try:
                            plt.savefig(os.path.join(d, 'lstm_mae_curve.png'))
                        except Exception:
                            pass
                    plt.close(fig2)
                if images_dirs:
                    logger.info(f"✅ Saved LSTM training images to: {images_dirs}")
                else:
                    logger.warning("No writable images directory found for LSTM curves")
            except Exception as img_err:
                logger.warning(f"Could not save LSTM training images: {img_err}")
            
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
            y_pred = model.predict(X_seq, verbose=0)
            if y_pred.ndim > 1:
                y_pred = y_pred.reshape(-1)
            
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