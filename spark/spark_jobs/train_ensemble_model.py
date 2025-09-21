#!/usr/bin/env python3
import os
import math
import logging
import numpy as np
from datetime import datetime

from pyspark.sql import SparkSession, functions as F
from pyspark.sql.window import Window
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.feature import (
    StringIndexer, OneHotEncoder, VectorAssembler, Imputer
)
from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator

import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import json
# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_spark_session():
    """Tạo SparkSession với nhiều cấu hình fallback."""
    builder = SparkSession.builder.appName("Comprehensive-Monthly-TimeSeries-WQI-Training")
    for drv_mem, ex_mem, max_res in [("2g","2g","1g"), ("1g","1g","512m"), ("512m","512m",None)]:
        b = builder.config("spark.driver.memory", drv_mem)\
                   .config("spark.executor.memory", ex_mem)\
                   .master("local[*]")
        if max_res:
            b = b.config("spark.driver.maxResultSize", max_res)
        try:
            spark = b.getOrCreate()
            logger.info(f"✅ SparkSession với driver={drv_mem}, executor={ex_mem}")
            return spark
        except Exception as e:
            logger.warning(f"❌ SparkSession thất bại (driver={drv_mem}, executor={ex_mem}): {e}")
    raise RuntimeError("🚫 Không thể khởi tạo SparkSession")


def load_data_from_postgres(spark):
    """Load dữ liệu WQI lịch sử từ PostgreSQL."""
    try:
        # Read DB connection settings from environment with sensible defaults for Docker network
        db_host = os.getenv('DB_HOST', '194.238.16.14')
        db_port = os.getenv('DB_PORT', '5432')
        db_name = os.getenv('DB_NAME', 'wqi_db')
        db_user = os.getenv('DB_USER', 'postgres')
        db_password = os.getenv('DB_PASSWORD', 'postgres')
        jdbc_url = f"jdbc:postgresql://{db_host}:{db_port}/{db_name}"
        
        df = (
            spark.read.format("jdbc")
            .option("url", jdbc_url)
            .option("dbtable", "public.historical_wqi_data")
            .option("user", db_user)
            .option("password", db_password)
            .option("driver", "org.postgresql.Driver")
            .load()
        )
        logger.info(f"✅ Đã load {df.count()} dòng từ Postgres")
        return df
    except Exception as e:
        logger.error(f"❌ Load từ Postgres thất bại abc: {e} + {jdbc_url} ")
        raise


def comprehensive_feature_engineering(df):
    """Tạo feature: time, cyclical, lag, rolling, interaction, trend."""
    logger.info("🔧 Feature engineering...")
    win = Window.partitionBy("station_id").orderBy("measurement_date")

    # 1) Time features
    df = df.withColumn("year", F.year("measurement_date")) \
           .withColumn("month", F.month("measurement_date")) \
           .withColumn("day_of_year", F.dayofyear("measurement_date")) \
           .withColumn("quarter", F.quarter("measurement_date"))

    # 2) Cyclical encoding
    df = df.withColumn(
                "month_sin",
                F.sin(F.col("month") * F.lit(2 * math.pi) / F.lit(12))
           ).withColumn(
                "month_cos",
                F.cos(F.col("month") * F.lit(2 * math.pi) / F.lit(12))
           ).withColumn(
                "day_sin",
                F.sin(F.col("day_of_year") * F.lit(2 * math.pi) / F.lit(365.25))
           ).withColumn(
                "day_cos",
                F.cos(F.col("day_of_year") * F.lit(2 * math.pi) / F.lit(365.25))
           )

    # 3) Lag features
    for lag in [1, 2, 3, 6, 12]:
        df = df.withColumn(f"wqi_lag_{lag}",  F.lag("wqi", lag).over(win)) \
               .withColumn(f"ph_lag_{lag}",   F.lag("ph", lag).over(win)) \
               .withColumn(f"temp_lag_{lag}", F.lag("temperature", lag).over(win)) \
               .withColumn(f"do_lag_{lag}",   F.lag("do", lag).over(win))

    # 4) Rolling window stats
    for w in [3, 6, 12]:
        wwin = win.rowsBetween(-w + 1, 0)
        df = df.withColumn(f"wqi_ma_{w}",  F.avg("wqi").over(wwin)) \
               .withColumn(f"wqi_std_{w}", F.stddev("wqi").over(wwin)) \
               .withColumn(f"ph_ma_{w}",   F.avg("ph").over(wwin)) \
               .withColumn(f"temp_ma_{w}", F.avg("temperature").over(wwin)) \
               .withColumn(f"do_ma_{w}",   F.avg("do").over(wwin))

    # 5) Rate-of-change (safe)
    for lag in [1, 3, 6]:
        df = df.withColumn(
            f"wqi_roc_{lag}",
            F.when(
                F.col(f"wqi_lag_{lag}").isNotNull(),
                (F.col("wqi") - F.col(f"wqi_lag_{lag}")) / F.col(f"wqi_lag_{lag}")
            ).otherwise(0.0)
        )

    # 6) Interaction features
    df = df.withColumn("ph_temp_i", F.col("ph") * F.col("temperature")) \
           .withColumn("ph_do_i",   F.col("ph") * F.col("do")) \
           .withColumn("temp_do_i", F.col("temperature") * F.col("do"))

    # 7) Station/global trends
    df = df.withColumn("station_avg_wqi", F.avg("wqi").over(Window.partitionBy("station_id"))) \
           .withColumn("station_std_wqi", F.stddev("wqi").over(Window.partitionBy("station_id")))
    gwin = Window.orderBy("measurement_date").rowsBetween(-12, 0)
    df = df.withColumn("global_wqi_ma12", F.avg("wqi").over(gwin)) \
           .withColumn("global_trend", F.col("wqi") - F.col("global_wqi_ma12"))

    return df

def add_horizon_labels(df):
    """Thêm các nhãn y_h = WQI(t+h) theo từng trạm với validation để tránh trùng lặp."""
    win = Window.partitionBy("station_id").orderBy("measurement_date")
    
    # Tạo horizon labels với validation
    for h in [1, 3, 6, 12]:
        # Tạo label cho horizon h
        df = df.withColumn(f"y_{h}", F.lead("wqi", h).over(win))
        
        # Thêm validation để đảm bảo tính duy nhất
        # Kiểm tra xem có trùng lặp với các horizon khác không
        if h > 1:
            # So sánh với các horizon nhỏ hơn để tránh trùng lặp
            for prev_h in [1, 3, 6]:
                if prev_h < h:
                    df = df.withColumn(
                        f"y_{h}_unique",
                        F.when(
                            F.col(f"y_{h}").isNotNull() & 
                            (F.col(f"y_{h}") != F.col(f"y_{prev_h}")),
                            F.col(f"y_{h}")
                        ).otherwise(None)
                    )
                    # Cập nhật lại y_h với giá trị unique
                    df = df.withColumn(f"y_{h}", F.col(f"y_{h}_unique")).drop(f"y_{h}_unique")
    
    # Thêm feature để đảm bảo tính đa dạng của dự đoán
    df = df.withColumn("horizon_diversity", 
        F.when(F.col("y_3").isNotNull() & F.col("y_6").isNotNull() & F.col("y_12").isNotNull(),
               F.abs(F.col("y_3") - F.col("y_6")) + F.abs(F.col("y_6") - F.col("y_12")) + F.abs(F.col("y_3") - F.col("y_12"))
        ).otherwise(0.0)
    )
    
    return df

def create_diverse_ensemble_predictions(models, test_data, horizon_models=None, horizon_scalers=None):
    """Tạo ensemble predictions với validation đảm bảo tính đa dạng."""
    logger.info("🎯 Creating diverse ensemble predictions...")
    
    try:
        # Lấy predictions từ main models
        xgb_model = models['xgb']
        rf_pipeline = models['rf_pipeline']
        scaler = models['scaler']
        
        # Transform test data
        test_tf = rf_pipeline.transform(test_data)
        X_test = np.array([r.features.toArray() for r in test_tf.select("features").collect()])
        X_test_s = scaler.transform(X_test)
        
        # Main ensemble predictions
        y_xgb = xgb_model.predict(X_test_s)
        y_rf = np.array(rf_pipeline.transform(test_data).select("prediction").collect()).flatten()
        y_ensemble = 0.6 * y_xgb + 0.4 * y_rf
        
        # Horizon-specific predictions nếu có
        horizon_predictions = {}
        if horizon_models and horizon_scalers:
            for h in [3, 6, 12]:
                if h in horizon_models and h in horizon_scalers:
                    Xh_test = horizon_scalers[h].transform(X_test_s)
                    pred_h = horizon_models[h].predict(Xh_test)
                    horizon_predictions[f"horizon_{h}"] = pred_h
        
        # Validation: Kiểm tra tính đa dạng
        all_predictions = {
            'xgb': y_xgb,
            'rf': y_rf,
            'ensemble': y_ensemble
        }
        all_predictions.update(horizon_predictions)
        
        # Tính toán diversity matrix
        diversity_matrix = {}
        for name1, pred1 in all_predictions.items():
            for name2, pred2 in all_predictions.items():
                if name1 < name2:  # Tránh duplicate
                    diff = np.abs(pred1 - pred2)
                    diversity_matrix[f"{name1}_vs_{name2}"] = {
                        'mean_diff': float(np.mean(diff)),
                        'std_diff': float(np.std(diff)),
                        'min_diff': float(np.min(diff)),
                        'max_diff': float(np.max(diff))
                    }
        
        # Log diversity results
        logger.info("📊 Prediction Diversity Analysis:")
        for pair, stats in diversity_matrix.items():
            logger.info(f"  {pair}: mean_diff={stats['mean_diff']:.4f}, std_diff={stats['std_diff']:.4f}")
        
        # Cảnh báo nếu predictions quá giống nhau
        low_diversity_pairs = [pair for pair, stats in diversity_matrix.items() 
                             if stats['mean_diff'] < 0.1]
        if low_diversity_pairs:
            logger.warning(f"⚠️ Low diversity detected in: {low_diversity_pairs}")
        else:
            logger.info("✅ Good prediction diversity across all models")
        
        return {
            'predictions': all_predictions,
            'diversity_matrix': diversity_matrix,
            'test_data': test_data
        }
        
    except Exception as e:
        logger.error(f"❌ Error creating diverse ensemble predictions: {e}")
        return None

def split_train_test_monthly(df):
    """Split 12 tháng cuối làm test, còn lại train."""
    max_dt = df.agg(F.max("measurement_date")).first()[0]
    split_dt = max_dt - F.expr("INTERVAL 12 MONTH")
    train = df.filter(F.col("measurement_date") <= split_dt)
    test  = df.filter(F.col("measurement_date") >  split_dt)
    logger.info(f"✅ Split: train={train.count()}, test={test.count()} at {split_dt}")
    return train, test


def create_comprehensive_feature_pipeline():
    """Xây pipeline: station encoding, Imputer, VectorAssembler."""
    logger.info("🔧 Build feature pipeline with Imputer...")
    idx = StringIndexer(inputCol="station_id", outputCol="station_idx", handleInvalid="keep")
    enc = OneHotEncoder(inputCol="station_idx", outputCol="station_ohe")

    feature_cols = [
        "ph","temperature","do","wqi",
        "year","month","day_of_year","quarter",
        "month_sin","month_cos","day_sin","day_cos",
        *[f"wqi_lag_{l}" for l in [1,2,3,6,12]],
        *[f"ph_lag_{l}"  for l in [1,2,3,6,12]],
        *[f"temp_lag_{l}"for l in [1,2,3,6,12]],
        *[f"do_lag_{l}"  for l in [1,2,3,6,12]],
        *[f"{m}_{w}" for m in ["wqi_ma","wqi_std","ph_ma","temp_ma","do_ma"] for w in [3,6,12]],
        *[f"wqi_roc_{l}" for l in [1,3,6]],
        "ph_temp_i","ph_do_i","temp_do_i",
        "station_avg_wqi","station_std_wqi","global_wqi_ma12","global_trend"
    ]

    imputer = Imputer(
        inputCols=feature_cols,
        outputCols=[f"{c}_imp" for c in feature_cols]
    ).setStrategy("mean")

    assembler = VectorAssembler(
        inputCols=[f"{c}_imp" for c in feature_cols] + ["station_ohe"],
        outputCol="features",
        handleInvalid="skip"
    )

    return Pipeline(stages=[idx, enc, imputer, assembler])


def train_with_optimized_tuning(train, test, spark):
    """Train XGBoost (TS split) + Spark RF (CrossValidator)."""
    pipeline = create_comprehensive_feature_pipeline()
    model_pipe = pipeline.fit(train)

    train_tf = model_pipe.transform(train)
    test_tf  = model_pipe.transform(test)

    def to_np(df):
        arr = np.array([r.features.toArray() for r in df.select("features").collect()])
        return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)

    X_train = to_np(train_tf)
    y_train = np.array([float(r.wqi) for r in train_tf.select("wqi").collect()], dtype=np.float64)
    X_test  = to_np(test_tf)
    y_test  = np.array([float(r.wqi) for r in test_tf.select("wqi").collect()], dtype=np.float64)

    logger.info(f"📊 Training on {X_train.shape[1]} features")

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    # --- XGBoost tuning ---
    best_xgb = None
    best_score = -np.inf
    best_params = None
    grid = {
        'n_estimators':[100,200],'max_depth':[4,6],
        'learning_rate':[0.1,0.2],'subsample':[0.8],
        'reg_alpha':[0,0.1],'reg_lambda':[0,0.1]
    }
    tscv = TimeSeriesSplit(n_splits=3)
    for ne in grid['n_estimators']:
      for md in grid['max_depth']:
        for lr in grid['learning_rate']:
          for ss in grid['subsample']:
            for ra in grid['reg_alpha']:
              for rl in grid['reg_lambda']:
                m = xgb.XGBRegressor(
                    n_estimators=ne, max_depth=md,
                    learning_rate=lr, subsample=ss,
                    reg_alpha=ra, reg_lambda=rl,
                    random_state=42, n_jobs=-1
                )
                scores = []
                for ti, vi in tscv.split(X_train_s):
                    m.fit(X_train_s[ti], y_train[ti])
                    scores.append(r2_score(y_train[vi], m.predict(X_train_s[vi])))
                sc = np.mean(scores)
                if sc > best_score:
                    best_score = sc
                    best_params = dict(ne=ne, md=md, lr=lr, ss=ss, ra=ra, rl=rl)
                    best_xgb = xgb.XGBRegressor(
                        n_estimators=ne,
                        max_depth=md,
                        learning_rate=lr,
                        subsample=ss,
                        reg_alpha=ra,
                        reg_lambda=rl,
                        random_state=42,
                        n_jobs=-1
                    )
                    # Version-agnostic fit (no callbacks/early stopping to avoid API mismatch)
                    best_xgb.fit(X_train_s, y_train)
    logger.info(f"🎯 Best XGB CV R2={best_score:.4f}, params={best_params}")
    # Save XGB learning curve (approx using CV loop not trivial; plot train vs test errors after fit)
    try:
        images_dir = "/app/models/images"
        os.makedirs(images_dir, exist_ok=True)
        # After fitting best_xgb, compute learning curve proxy by evaluating on train/test subsets
        # Plot test vs a moving window of training size (simple proxy)
        sizes = np.linspace(0.1, 1.0, 10)
        train_errs, test_errs = [], []
        # Map best_params to full names
        pf = dict(
            n_estimators=best_params['ne'],
            max_depth=best_params['md'],
            learning_rate=best_params['lr'],
            subsample=best_params['ss'],
            reg_alpha=best_params['ra'],
            reg_lambda=best_params['rl']
        )
        for s in sizes:
            n = max(10, int(X_train_s.shape[0] * s))
            m = xgb.XGBRegressor(**pf, random_state=42, n_jobs=-1)
            m.fit(X_train_s[:n], y_train[:n])
            train_pred = m.predict(X_train_s[:n])
            test_pred  = m.predict(X_test_s)
            train_errs.append(float(np.sqrt(mean_squared_error(y_train[:n], train_pred))))
            test_errs.append(float(np.sqrt(mean_squared_error(y_test, test_pred))))
        fig, ax = plt.subplots(figsize=(8,5))
        ax.plot(sizes, train_errs, label='train RMSE')
        ax.plot(sizes, test_errs, label='test RMSE')
        ax.set_xlabel('Train size fraction')
        ax.set_ylabel('RMSE')
        ax.legend(loc='best')
        fig.tight_layout()
        plt.savefig(os.path.join(images_dir, 'xgb_learning_curve.png'))
        plt.close(fig)
        logger.info(f"✅ Saved XGB learning curve to {images_dir}")
    except Exception as e:
        logger.warning(f"Could not save XGB learning curve: {e}")

    # --- Spark RF tuning ---
    rf = RandomForestRegressor(labelCol="wqi", featuresCol="features", seed=42)
    paramGrid = ParamGridBuilder()\
        .addGrid(rf.maxDepth, [10,15])\
        .addGrid(rf.numTrees, [100,200])\
        .addGrid(rf.minInstancesPerNode, [1,2])\
        .build()
    cv = CrossValidator(
        estimator=Pipeline(stages=[pipeline, rf]),
        evaluator=RegressionEvaluator(labelCol="wqi", metricName="r2"),
        estimatorParamMaps=paramGrid,
        numFolds=3
    )
    cvModel = cv.fit(train)
    best_rf_pipeline = cvModel.bestModel
    best_rf_model = best_rf_pipeline.stages[-1]  # RandomForestRegressionModel
    rf_score = max(cvModel.avgMetrics)
    logger.info(f"🎯 Best RF CV R2={rf_score:.4f}")
    # Save RF learning curve proxy
    try:
        images_dir = "/app/models/images"
        os.makedirs(images_dir, exist_ok=True)
        # Use predictions on test while training set grows via pipeline fit (expensive → sample)
        # For simplicity, plot test RMSE points for different numTrees
        trees = [50, 100, 150, 200]
        rmses = []
        for nt in trees:
            rf_tmp = RandomForestRegressor(labelCol="wqi", featuresCol="features", numTrees=nt, seed=42)
            model_tmp = Pipeline(stages=[create_comprehensive_feature_pipeline(), rf_tmp]).fit(train)
            pred_tmp = model_tmp.transform(test).select("prediction").toPandas().values.flatten()
            rmse_tmp = float(np.sqrt(mean_squared_error(y_test, pred_tmp[:len(y_test)]))) if len(pred_tmp) >= len(y_test) else np.nan
            rmses.append(rmse_tmp)
        fig2, ax2 = plt.subplots(figsize=(8,5))
        ax2.plot(trees, rmses, marker='o')
        ax2.set_xlabel('numTrees')
        ax2.set_ylabel('Test RMSE')
        fig2.tight_layout()
        plt.savefig(os.path.join(images_dir, 'rf_learning_curve.png'))
        plt.close(fig2)
        logger.info(f"✅ Saved RF learning curve to {images_dir}")
    except Exception as e:
        logger.warning(f"Could not save RF learning curve: {e}")

    # --- Predict & ensemble ---
    y_xgb = best_xgb.predict(X_test_s)
    y_rf  = np.array(best_rf_pipeline.transform(test).select("prediction").collect()).flatten()

    def calc(y, yh):
        return {
            'r2':   r2_score(y, yh),
            'mae':  mean_absolute_error(y, yh),
            'rmse': np.sqrt(mean_squared_error(y, yh))
        }

    m_xgb = calc(y_test, y_xgb)
    m_rf  = calc(y_test, y_rf)
    y_en  = 0.6 * y_xgb + 0.4 * y_rf
    m_en  = calc(y_test, y_en)

    logger.info(f"📊 Results:\n  XGB: {m_xgb}\n  RF:  {m_rf}\n  ENS: {m_en}")
    
    # ---- Walk-forward backtest (time-series) for XGB & RF (sklearn) ----
    try:
        X_all_s = np.vstack([X_train_s, X_test_s])
        y_all   = np.concatenate([y_train, y_test])
        def backtest_estimator(est, X, y, n_splits=5):
            tscv = TimeSeriesSplit(n_splits=n_splits)
            rmse, mae, r2 = [], [], []
            X = np.asarray(X, dtype=np.float64)
            y = np.asarray(y, dtype=np.float64)
            for tr, va in tscv.split(X):
                est_ = est.__class__(**est.get_params())
                est_.fit(X[tr], y[tr])
                yh = est_.predict(X[va])
                err = yh - y[va]
                rmse.append(float(np.sqrt(np.mean(err**2))))
                mae.append(float(np.mean(np.abs(err))))
                r2.append(float(r2_score(y[va], yh)))
            return rmse, mae, r2

        # XGB params from best
        pf = dict(
            n_estimators=best_params['ne'],
            max_depth=best_params['md'],
            learning_rate=best_params['lr'],
            subsample=best_params['ss'],
            colsample_bytree=0.8,
            reg_alpha=best_params['ra'],
            reg_lambda=best_params['rl'],
            random_state=42,
            n_jobs=-1
        )
        xgb_bt = xgb.XGBRegressor(**pf)
        rmse_xgb_bt, mae_xgb_bt, r2_xgb_bt = backtest_estimator(xgb_bt, X_all_s, y_all, n_splits=5)

        # RF backtest (sklearn) + OOB
        from sklearn.ensemble import RandomForestRegressor as SKRF
        rf_sk = SKRF(n_estimators=400, max_depth=None, random_state=42, n_jobs=-1, oob_score=True, bootstrap=True)
        rmse_rf_bt, mae_rf_bt, r2_rf_bt = backtest_estimator(rf_sk, X_all_s, y_all, n_splits=5)
        # OOB on full train-like set (use X_train_s)
        rf_sk.fit(X_train_s, y_train)
        oob_r2 = float(getattr(rf_sk, 'oob_score_', np.nan))

        backtest_payload = {
            'xgb_backtest': {'rmse': rmse_xgb_bt, 'mae': mae_xgb_bt, 'r2': r2_xgb_bt},
            'rf_backtest':  {'rmse': rmse_rf_bt,  'mae': mae_rf_bt,  'r2': r2_rf_bt, 'oob_r2': oob_r2}
        }
        try:
            out_dir_bt = "/app/models"
            os.makedirs(out_dir_bt, exist_ok=True)
            with open(f"{out_dir_bt}/metrics_backtest.json", 'w') as f:
                json.dump(backtest_payload, f, indent=2)
            logger.info("✅ Saved walk-forward backtest metrics to metrics_backtest.json")
        except Exception as be:
            logger.warning(f"Could not save backtest metrics: {be}")
    except Exception as e:
        logger.warning(f"Backtest computation failed: {e}")

    # ---- Train per-horizon XGBoost models (xgb_h1, xgb_h3, xgb_h6, xgb_h12) ----
    try:
        import joblib
        out_dir = "/app/models"
        os.makedirs(out_dir, exist_ok=True)
        
        # Dictionary để lưu trữ các model và scaler cho validation
        horizon_models = {}
        horizon_scalers = {}
        
        # Prepare features matrix for train set (already available as X_train_s) but we need y_h labels
        for h in [1, 3, 6, 12]:
            col_y = f"y_{h}"
            # Join labels from original train DataFrame to transformed vectors order by rowid
            rows = train_tf.select("features").withColumn("row_id", F.monotonically_increasing_id())
            y_rows = train.select(col_y).withColumn("row_id", F.monotonically_increasing_id())
            joined = rows.join(y_rows, on="row_id", how="inner").where(F.col(col_y).isNotNull()).select("features", col_y).collect()
            if not joined:
                logger.warning(f"No training samples for horizon {h}")
                continue
            # Convert to numpy
            try:
                Xh = np.array([r['features'].toArray() for r in joined], dtype=np.float64)
            except Exception:
                Xh = np.array([r['features'] for r in joined], dtype=np.float64)
            yh = np.array([float(r[col_y]) for r in joined], dtype=np.float64)
            
            # Thêm regularization để đảm bảo tính đa dạng
            scaler_h = StandardScaler()
            Xh_s = scaler_h.fit_transform(Xh)
            
            # Tăng regularization cho các horizon dài hơn để tránh overfitting
            reg_alpha = 0.1 if h <= 3 else 0.2
            reg_lambda = 1.0 if h <= 3 else 2.0
            
            model_h = xgb.XGBRegressor(
                n_estimators=300, 
                max_depth=5, 
                learning_rate=0.08, 
                subsample=0.9, 
                colsample_bytree=0.9,
                reg_alpha=reg_alpha,
                reg_lambda=reg_lambda,
                random_state=42, 
                n_jobs=-1
            )
            model_h.fit(Xh_s, yh)
            
            # Lưu model và scaler
            horizon_models[h] = model_h
            horizon_scalers[h] = scaler_h
            
            joblib.dump(model_h, os.path.join(out_dir, f"xgb_h{h}.pkl"))
            joblib.dump(scaler_h, os.path.join(out_dir, f"xgb_h{h}_scaler.pkl"))
            logger.info(f"✅ Trained and saved xgb_h{h} and scaler")
        
        # Validation: Kiểm tra tính đa dạng của dự đoán
        if len(horizon_models) >= 3:  # Có ít nhất 3 model
            logger.info("🔍 Validating prediction diversity...")
            try:
                # Sử dụng hàm create_diverse_ensemble_predictions để validation
                diversity_result = create_diverse_ensemble_predictions(
                    {'xgb': best_xgb, 'rf_pipeline': best_rf_pipeline, 'scaler': scaler},
                    test,
                    horizon_models,
                    horizon_scalers
                )
                
                if diversity_result:
                    logger.info("✅ Diversity validation completed successfully")
                else:
                    logger.warning("⚠️ Diversity validation failed")
                        
            except Exception as e:
                logger.warning(f"Could not validate prediction diversity: {e}")
                
    except Exception as e:
        logger.warning(f"Per-horizon XGB training failed: {e}")
    # Persist per-sample test predictions for downstream blending
    try:
        out_dir = "/app/models"
        # Collect keys in the same order as test_tf rows
        key_rows = test_tf.select("station_id", "measurement_date").collect()
        import csv
        from datetime import datetime as _dt
        with open(f"{out_dir}/ensemble_test_predictions.csv", "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["station_id", "measurement_date", "y_true", "y_xgb", "y_rf", "y_ensemble_xrf"])
            for i, row in enumerate(key_rows):
                # measurement_date may be a datetime/date; convert to ISO string
                md = row.measurement_date
                if hasattr(md, 'isoformat'):
                    md_str = md.isoformat()
                else:
                    md_str = str(md)
                writer.writerow([row.station_id, md_str, float(y_test[i]), float(y_xgb[i]), float(y_rf[i]), float(y_en[i])])
        logger.info("✅ Wrote ensemble_test_predictions.csv for downstream blending")
    except Exception as e:
        logger.warning(f"Could not write ensemble_test_predictions.csv: {e}")
    return {'xgb': best_xgb,
            'rf_pipeline': best_rf_pipeline,  # lưu cả pipeline để apply sau
            'scaler': scaler}, \
           {'xgb': m_xgb, 'rf': m_rf, 'ensemble': m_en}


def save_comprehensive_models(models, metrics, out_dir="/app/models"):
    """Lưu XGB + Spark-RF + scaler + metrics vào files (MLflow sẽ được lưu bởi DAG)."""
    os.makedirs(out_dir, exist_ok=True)
    import pickle, json
    
    # 1) Lưu XGBoost model
    with open(f"{out_dir}/xgb.pkl","wb") as f:
        pickle.dump(models['xgb'], f)
    
    # 2) Lưu scaler
    with open(f"{out_dir}/scaler.pkl","wb") as f:
        pickle.dump(models['scaler'], f)
    
    # 3) Lưu Spark-RF pipeline
    rf_pipe = models['rf_pipeline']
    rf_pipe.write().overwrite().save(f"{out_dir}/rf_pipeline")
    
    # 4) Lưu metrics JSON (bao gồm placeholder cho lstm nếu blending sau)
    try:
        # If downstream blending adds LSTM, metrics can be merged later
        with open(f"{out_dir}/metrics.json","w") as f:
            json.dump(metrics, f, indent=2)
    except Exception as e:
        logger.warning(f"Could not write metrics.json: {e}")
    
    # 5) Tìm best model
    best_model = max(metrics.items(), key=lambda x: x[1]['r2'])
    logger.info(f"🏆 Best model: {best_model[0]} (R²: {best_model[1]['r2']:.4f})")
    
    logger.info("✅ Đã lưu XGB, Spark-RF pipeline, scaler và metrics vào files.")
    logger.info("📝 MLflow sẽ được lưu bởi DAG sau khi training hoàn tất.")


def main():
    spark = create_spark_session()
    try:
        df = load_data_from_postgres(spark)
        df = comprehensive_feature_engineering(df)
        df = add_horizon_labels(df)
        train, test = split_train_test_monthly(df)
        models, metrics = train_with_optimized_tuning(train, test, spark)
        save_comprehensive_models(models, metrics)
        logger.info("🎉 Pipeline training hoàn tất!")
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
