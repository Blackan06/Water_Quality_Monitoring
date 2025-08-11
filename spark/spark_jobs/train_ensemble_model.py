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
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.stat import Correlation

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
        df = (
            spark.read.format("jdbc")
            .option("url", "jdbc:postgresql://194.238.16.14:5432/wqi_db")
            .option("dbtable", "public.historical_wqi_data")
            .option("user", "postgres")
            .option("password", "postgres1234")
            .option("driver", "org.postgresql.Driver")
            .load()
        )
        logger.info(f"✅ Đã load {df.count()} dòng từ Postgres")
        return df
    except Exception as e:
        logger.error(f"❌ Load từ Postgres thất bại: {e}")
        raise


def clean_time_series_monthly(df, for_ml=True):
    """
    Làm sạch dữ liệu theo tháng cho từng station:
    - Ép kiểu, bỏ trùng (station_id, measurement_date)
    - Ràng buộc miền giá trị vật lý cơ bản
    - Resample về tháng (mean)
    - Bổ sung đủ mốc tháng (lịch liên tục) cho mỗi station
    - Forward-fill (cho ML) hoặc Forward-fill + Back-fill (cho EDA)
    """
    logger.info("🧹 Clean time-series (monthly) ...")

    # 0) Chuẩn cột & bỏ trùng
    df = (df
        .withColumn("measurement_date", F.to_timestamp("measurement_date"))
        .dropDuplicates(["station_id", "measurement_date"])
        .filter(F.col("measurement_date").isNotNull())
    )

    # 1) Ràng buộc miền giá trị (tuỳ domain, có thể nới/siết)
    df = (df
        .filter((F.col("wqi").isNull()) | ((F.col("wqi") >= 0) & (F.col("wqi") <= 200)))
        .filter((F.col("ph").isNull())  | ((F.col("ph") >= 0) & (F.col("ph") <= 14)))
        .filter((F.col("temperature").isNull()) | ((F.col("temperature") > -5) & (F.col("temperature") < 80)))
        .filter((F.col("do").isNull())  | ((F.col("do") >= 0) & (F.col("do") <= 25)))
    )

    # 2) Resample về THÁNG (nếu đã là tháng thì vẫn an toàn: groupBy sẽ gộp)
    dfm = (df
        .withColumn("month_date", F.date_trunc("month", F.col("measurement_date")))
        .groupBy("station_id", "month_date")
        .agg(
            F.avg("wqi").alias("wqi"),
            F.avg("ph").alias("ph"),
            F.avg("temperature").alias("temperature"),
            F.avg("do").alias("do")
        )
    )

    # 3) Sinh lịch tháng liên tục cho từng station
    minmax = (dfm.groupBy("station_id")
        .agg(F.min("month_date").alias("min_d"), F.max("month_date").alias("max_d"))
    )
    cal = (minmax
        .select(
            "station_id",
            F.sequence(F.col("min_d"), F.col("max_d"), F.expr("INTERVAL 1 MONTH")).alias("month_seq")
        )
        .withColumn("month_date", F.explode(F.col("month_seq")))
        .select("station_id", "month_date")
    )

    # 4) Join lịch với dữ liệu → có thể null ở tháng thiếu
    df_full = cal.join(dfm, ["station_id", "month_date"], "left")

    # 5) Forward-fill (OK cho ML)
    win_ff = (Window.partitionBy("station_id")
        .orderBy("month_date")
        .rowsBetween(Window.unboundedPreceding, 0)
    )
    for c in ["wqi", "ph", "temperature", "do"]:
        df_full = df_full.withColumn(c, F.last(F.col(c), ignorenulls=True).over(win_ff))

    if not for_ml:
        # Back-fill (chỉ dùng cho EDA, KHÔNG dùng cho training)
        win_bf = (Window.partitionBy("station_id")
            .orderBy(F.col("month_date").desc())
            .rowsBetween(Window.unboundedPreceding, 0)
        )
        for c in ["wqi", "ph", "temperature", "do"]:
            df_full = df_full.withColumn(c, F.coalesce(F.col(c), F.last(F.col(c), ignorenulls=True).over(win_bf)))

    # 6) Trả về đúng schema ngày tháng bạn đang dùng
    df_clean = (df_full
        .withColumnRenamed("month_date", "measurement_date")
        .orderBy("station_id", "measurement_date")
    )

    logger.info("✅ Done clean monthly time-series.")
    return df_clean


def add_label_for_horizon(df, H: int):
    """Tạo nhãn dự báo WQI ở tương lai H tháng: label = WQI_{t+H}."""
    win = Window.partitionBy("station_id").orderBy("measurement_date")
    return df.withColumn("label", F.lead("wqi", H).over(win)).filter(F.col("label").isNotNull())


def add_time_folds(df, k: int = 3):
    """Gán fold theo mốc thời gian để CrossValidator không trộn ngẫu nhiên."""
    # chỉ cần theo trục thời gian toàn cục là đủ
    df = df.withColumn("foldCol", F.ntile(int(k)).over(Window.orderBy("measurement_date")))
    return df
    # NOTE: Với dữ liệu ít trạm (3 stations) thì fold theo thời gian toàn cục là OK
    # Nếu có nhiều trạm và phân bố khác nhau, có thể cân nhắc fold theo (station_id, time)


def split_train_test_monthly(df, test_months=12):
    """
    Chia theo mốc thời gian: lấy test_months tháng cuối làm test, còn lại là train.
    FIXED: Sử dụng add_months thay vì F.expr
    """
    split_dt = df.select(F.add_months(F.max("measurement_date"), -int(test_months)).alias("dt")).first()["dt"]
    train = df.filter(F.col("measurement_date") <= F.lit(split_dt))
    test  = df.filter(F.col("measurement_date")  > F.lit(split_dt))
    logger.info(f"✅ Split (train/test): train={train.count()}, test={test.count()}, split_dt={split_dt}")
    return train, test


def drop_cold_start_rows(df, required_lags=("wqi_lag_12", "ph_lag_12", "temp_lag_12", "do_lag_12")):
    """
    Loại các dòng chưa đủ quá khứ cho các lag lớn nhất (tránh Imputer bù mean gây nhiễu).
    Gọi sau bước feature_engineering.
    """
    cond = None
    for c in required_lags:
        expr = F.col(c).isNotNull()
        cond = expr if cond is None else (cond & expr)
    before = df.count()
    df2 = df.filter(cond)
    after = df2.count()
    logger.info(f"🧊 Drop cold-start: {before-after} rows removed (remain={after})")
    return df2


def compute_correlations(df):
    """
    Trả về:
      - corr_global: dict Pearson corr giữa WQI và DO/pH/Temp toàn cục
      - corr_by_station: DataFrame corr theo từng station
      - corr_matrix: numpy array 4x4 cho [wqi, do, ph, temperature]
    """
    # --- Global corr (WQI với DO/pH/Temp)
    row = (df.select(
        F.corr("wqi","do").alias("corr_wqi_do"),
        F.corr("wqi","ph").alias("corr_wqi_ph"),
        F.corr("wqi","temperature").alias("corr_wqi_temp")
    ).first())
    corr_global = {
        "wqi~do": float(row["corr_wqi_do"]) if row["corr_wqi_do"] is not None else None,
        "wqi~ph": float(row["corr_wqi_ph"]) if row["corr_wqi_ph"] is not None else None,
        "wqi~temp": float(row["corr_wqi_temp"]) if row["corr_wqi_temp"] is not None else None,
    }

    # --- Corr theo trạm
    corr_by_station = (df.groupBy("station_id").agg(
        F.corr("wqi","do").alias("corr_wqi_do"),
        F.corr("wqi","ph").alias("corr_wqi_ph"),
        F.corr("wqi","temperature").alias("corr_wqi_temp")
    ))

    # --- Correlation matrix cho [wqi, do, ph, temperature]
    cols = ["wqi","do","ph","temperature"]
    vec_df = VectorAssembler(inputCols=cols, outputCol="vec")\
        .transform(df.select(*cols).na.drop())
    mat = Correlation.corr(vec_df, "vec", "pearson").head()[0].toArray()  # numpy matrix 4x4

    return corr_global, corr_by_station, mat


def analyze_seasonality(df):
    """
    Tính seasonality theo tháng:
      - wqi_mean_by_month: trung bình WQI theo tháng (1..12)
      - seasonal_index: wqi_mean_by_month / overall_mean
      - wqi_mean_by_station_month: trung bình theo trạm & tháng
    """
    dfm = df.withColumn("month", F.month("measurement_date"))

    wqi_mean_by_month = (dfm.groupBy("month")
        .agg(F.avg("wqi").alias("wqi_mean"), F.stddev("wqi").alias("wqi_std"), F.count("*").alias("n")))

    overall_mean = df.agg(F.avg("wqi").alias("m")).first()["m"]
    seasonal_index = wqi_mean_by_month.withColumn(
        "seasonal_index", F.col("wqi_mean") / F.lit(overall_mean)
    )

    wqi_mean_by_station_month = (dfm.groupBy("station_id","month")
        .agg(F.avg("wqi").alias("wqi_mean")))

    return wqi_mean_by_month.orderBy("month"), seasonal_index.orderBy("month"), wqi_mean_by_station_month


def add_wqi_moving_averages(df):
    """
    Thêm các cột wqi_ma_3, wqi_ma_6, wqi_ma_12 (moving average theo trạm).
    Cửa sổ: từ quá khứ tới hiện tại (an toàn cho forecast tương lai).
    """
    win = (Window.partitionBy("station_id")
           .orderBy("measurement_date"))

    for w in [3,6,12]:
        wwin = win.rowsBetween(-w+1, 0)  # include current t
        df = df.withColumn(f"wqi_ma_{w}", F.avg("wqi").over(wwin))
    return df


def comprehensive_feature_engineering(df):
    """Tạo feature: time, cyclical, lag, rolling, interaction, trend (FIXED DATA LEAKAGE)."""
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

    # 3) Lag features (FIXED: chỉ dùng quá khứ)
    for lag in [1, 2, 3, 6, 12]:
        df = df.withColumn(f"wqi_lag_{lag}",  F.lag("wqi", lag).over(win)) \
               .withColumn(f"ph_lag_{lag}",   F.lag("ph", lag).over(win)) \
               .withColumn(f"temp_lag_{lag}", F.lag("temperature", lag).over(win)) \
               .withColumn(f"do_lag_{lag}",   F.lag("do", lag).over(win))

    # 4) Rolling window stats (FIXED: chỉ dùng quá khứ)
    for w in [3, 6, 12]:
        wwin = win.rowsBetween(-w + 1, 0)  # chỉ quá khứ đến hiện tại
        df = df.withColumn(f"wqi_ma_{w}",  F.avg("wqi").over(wwin)) \
               .withColumn(f"wqi_std_{w}", F.stddev("wqi").over(wwin)) \
               .withColumn(f"ph_ma_{w}",   F.avg("ph").over(wwin)) \
               .withColumn(f"temp_ma_{w}", F.avg("temperature").over(wwin)) \
               .withColumn(f"do_ma_{w}",   F.avg("do").over(wwin))

    # 5) Rate-of-change (safe - tránh chia cho 0)
    for lag in [1, 3, 6]:
        df = df.withColumn(
            f"wqi_roc_{lag}",
            F.when(
                (F.col(f"wqi_lag_{lag}").isNotNull()) & (F.col(f"wqi_lag_{lag}") != 0),
                (F.col("wqi") - F.col(f"wqi_lag_{lag}")) / F.col(f"wqi_lag_{lag}")
            ).otherwise(F.lit(0.0))
        )

    # 6) Interaction features
    df = df.withColumn("ph_temp_i", F.col("ph") * F.col("temperature")) \
           .withColumn("ph_do_i",   F.col("ph") * F.col("do")) \
           .withColumn("temp_do_i", F.col("temperature") * F.col("do"))

    # 7) Station/global trends (FIXED: chỉ dùng quá khứ)
    # Station statistics: expanding window (chỉ quá khứ đến hiện tại)
    wstat = Window.partitionBy("station_id").orderBy("measurement_date")\
                  .rowsBetween(Window.unboundedPreceding, 0)  # chỉ quá khứ đến hiện tại t
    df = df.withColumn("station_avg_wqi", F.avg("wqi").over(wstat)) \
           .withColumn("station_std_wqi", F.stddev("wqi").over(wstat))

    # Global trends: rolling window (chỉ quá khứ)
    gwin = Window.orderBy("measurement_date").rowsBetween(-12, 0)
    df = df.withColumn("global_wqi_ma12", F.avg("wqi").over(gwin)) \
           .withColumn("global_trend", F.col("wqi") - F.col("global_wqi_ma12"))

    return df


def create_comprehensive_feature_pipeline():
    """Xây pipeline: station encoding, Imputer, VectorAssembler."""
    logger.info("🔧 Build feature pipeline with Imputer...")
    idx = StringIndexer(inputCol="station_id", outputCol="station_idx", handleInvalid="keep")
    enc = OneHotEncoder(inputCol="station_idx", outputCol="station_ohe")

    feature_cols = [
        "ph","temperature","do","wqi",  # NOTE: wqi hiện tại - hợp lệ cho forecast nếu có WQI tại thời điểm dự báo
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

    # PRODUCTION NOTE: Nếu không có WQI hiện tại lúc dự báo, bỏ "wqi" khỏi feature_cols
    # Giữ lại: wqi_lag_*, rolling features, pH/DO/Temp là đủ

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


def train_with_optimized_tuning(train, test, spark, label_col="label"):
    """Train XGBoost (TS CV) + Spark RF (CV theo thời gian) cho FORECAST (dùng label)."""
    pipeline = create_comprehensive_feature_pipeline()
    model_pipe = pipeline.fit(train)

    train_tf = model_pipe.transform(train)
    test_tf  = model_pipe.transform(test)

    def to_np(dfv, label_col=label_col):
        # Sắp xếp tuyệt đối theo thời gian (và theo station để ổn định)
        rows = (dfv
                .select("station_id", "measurement_date", "features", label_col)
                .orderBy("measurement_date", "station_id")
                .collect())
        X = np.array([r["features"].toArray() for r in rows])
        y = np.array([r[label_col] for r in rows])
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        return X, y

    X_train, y_train = to_np(train_tf, label_col)
    X_test,  y_test  = to_np(test_tf,  label_col)

    logger.info(f"📊 Training on {X_train.shape[1]} features (label={label_col})")

    # XGBoost + StandardScaler trong pipeline (tránh CV leakage)
    from sklearn.pipeline import Pipeline as SKPipeline
    tscv = TimeSeriesSplit(n_splits=3)

    best_xgb, best_score, best_params = None, -np.inf, None
    grid = {
        'n_estimators':[100,200],'max_depth':[4,6],
        'learning_rate':[0.1,0.2],'subsample':[0.8],
        'reg_alpha':[0,0.1],'reg_lambda':[0,0.1]
    }

    for ne in grid['n_estimators']:
      for md in grid['max_depth']:
        for lr in grid['learning_rate']:
          for ss in grid['subsample']:
            for ra in grid['reg_alpha']:
              for rl in grid['reg_lambda']:
                pipe = SKPipeline([
                    ("scaler", StandardScaler()),
                    ("xgb", xgb.XGBRegressor(
                        n_estimators=ne, max_depth=md, learning_rate=lr,
                        subsample=ss, reg_alpha=ra, reg_lambda=rl,
                        random_state=42, n_jobs=-1))
                ])
                scores = []
                for ti, vi in tscv.split(X_train):
                    pipe.fit(X_train[ti], y_train[ti])
                    scores.append(r2_score(y_train[vi], pipe.predict(X_train[vi])))
                sc = float(np.mean(scores))
                if sc > best_score:
                    best_score, best_params, best_xgb = sc, dict(ne=ne,md=md,lr=lr,ss=ss,ra=ra,rl=rl), pipe
    best_xgb.fit(X_train, y_train)
    logger.info(f"🎯 Best XGB CV R2={best_score:.4f}, params={best_params}")

    # Spark RF với label_col và fold theo thời gian
    # NOTE: Pipeline được fit 2 lần (cho XGB và RF) - có thể tối ưu hiệu năng nếu cần
    rf = RandomForestRegressor(labelCol=label_col, featuresCol="features", seed=42)
    paramGrid = ParamGridBuilder()\
        .addGrid(rf.maxDepth, [10,15])\
        .addGrid(rf.numTrees, [100,200])\
        .addGrid(rf.minInstancesPerNode, [1,2])\
        .build()
    cv = CrossValidator(
        estimator=Pipeline(stages=[pipeline, rf]),
        evaluator=RegressionEvaluator(labelCol=label_col, metricName="r2"),
        estimatorParamMaps=paramGrid,
        numFolds=3
    ).setFoldCol("foldCol")   # dùng cột fold theo thời gian
    cvModel = cv.fit(train)   # train đã có foldCol

    best_rf_pipeline = cvModel.bestModel
    y_xgb = best_xgb.predict(X_test)
    y_rf  = np.array(best_rf_pipeline.transform(test).select("prediction").collect()).flatten()
    y_en  = 0.6*y_xgb + 0.4*y_rf

    def calc(y, yh):
        return {'r2': r2_score(y,yh),
                'mae': mean_absolute_error(y,yh),
                'rmse': math.sqrt(mean_squared_error(y,yh))}
    m_xgb, m_rf, m_en = calc(y_test,y_xgb), calc(y_test,y_rf), calc(y_test,y_en)

    logger.info(f"📊 Results (forecast): XGB={m_xgb} RF={m_rf} ENS={m_en}")
    return {'xgb': best_xgb, 'rf_pipeline': best_rf_pipeline,
            'scaler': best_xgb.named_steps['scaler']}, \
           {'xgb': m_xgb, 'rf': m_rf, 'ensemble': m_en}


def export_corr_matrix(df, out_dir="/app/models/eda"):
    """
    Tính correlation matrix và xuất CSV + heatmap PNG.
    """
    import os, numpy as np, pandas as pd, matplotlib.pyplot as plt
    os.makedirs(out_dir, exist_ok=True)

    _, _, mat = compute_correlations(df)
    cols = ["wqi","do","ph","temperature"]
    cm = pd.DataFrame(mat, index=cols, columns=cols)
    cm.to_csv(os.path.join(out_dir, "corr_matrix.csv"))

    # Heatmap đơn giản
    plt.figure()
    plt.imshow(cm, interpolation="nearest")
    plt.xticks(range(len(cols)), cols, rotation=45, ha="right")
    plt.yticks(range(len(cols)), cols)
    plt.colorbar()
    plt.title("Correlation Matrix (Pearson)")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "corr_matrix.png"))


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
    
    # 4) Lưu metrics JSON
    with open(f"{out_dir}/metrics.json","w") as f:
        json.dump(metrics, f, indent=2)
    
    # 5) Tìm best model
    best_model = max(metrics.items(), key=lambda x: x[1]['r2'])
    logger.info(f"🏆 Best model: {best_model[0]} (R²: {best_model[1]['r2']:.4f})")
    
    logger.info("✅ Đã lưu XGB, Spark-RF pipeline, scaler và metrics vào files.")
    logger.info("📝 MLflow sẽ được lưu bởi DAG sau khi training hoàn tất.")


def main():
    spark = create_spark_session()
    try:
        df_raw = load_data_from_postgres(spark)

        # Clean cho ML (KHÔNG back-fill để tránh leakage)
        df_clean = clean_time_series_monthly(df_raw, for_ml=True)

        # (TUỲ CHỌN) EDA Analysis - chạy khi cần báo cáo chi tiết
        # df_clean_eda = clean_time_series_monthly(df_raw, for_ml=False)  # có back-fill cho EDA
        # corr_global, corr_by_station, corr_mat = compute_correlations(df_clean_eda)
        # wqi_by_month, seasonal_idx, wqi_by_station_month = analyze_seasonality(df_clean_eda)
        # export_corr_matrix(df_clean_eda)

        # Feature engineering (đã fix windows quá khứ)
        df_feat = comprehensive_feature_engineering(df_clean)
        df_feat = drop_cold_start_rows(df_feat, required_lags=("wqi_lag_12","ph_lag_12","temp_lag_12","do_lag_12"))

        for H in [1, 3, 12]:
            logger.info(f"🚀 Forecast horizon H={H} months")
            dfH = add_label_for_horizon(df_feat, H)

            # split 12 tháng cuối làm test
            train, test = split_train_test_monthly(dfH, test_months=12)

            # gán fold thời gian cho Spark CV
            train_f = add_time_folds(train, k=3)

            models, metrics = train_with_optimized_tuning(train_f, test, spark, label_col="label")
            save_comprehensive_models(models, metrics, out_dir=f"/app/models/H{H}")
            
            logger.info(f"✅ Completed training for H={H} months")
        
        logger.info("🎉 All forecast models training completed!")
        logger.info("📁 Trained models saved in /app/models/H1, H3, H12")
        
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
