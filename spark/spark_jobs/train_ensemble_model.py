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
    y_train = np.array([r.wqi for r in train_tf.select("wqi").collect()])
    X_test  = to_np(test_tf)
    y_test  = np.array([r.wqi for r in test_tf.select("wqi").collect()])

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
                    best_score, best_params = sc, dict(ne=ne,md=md,lr=lr,ss=ss,ra=ra,rl=rl)
                    best_xgb = xgb.XGBRegressor(**best_params, random_state=42, n_jobs=-1)
                    best_xgb.fit(X_train_s, y_train)
    logger.info(f"🎯 Best XGB CV R2={best_score:.4f}, params={best_params}")

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
        train, test = split_train_test_monthly(df)
        models, metrics = train_with_optimized_tuning(train, test, spark)
        save_comprehensive_models(models, metrics)
        logger.info("🎉 Pipeline training hoàn tất!")
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
