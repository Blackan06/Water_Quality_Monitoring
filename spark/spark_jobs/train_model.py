import logging
import os

from pyspark.sql import SparkSession, functions as F
from pyspark.sql.window import Window
from pyspark.sql.types import DoubleType
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml import Pipeline
from xgboost.spark import SparkXGBRegressor

# ——— Thiết lập logging ———
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("train_model")

# ——— Cấu hình database & output ———
DB_CONFIG = {
    'host':     os.getenv('DB_HOST', '194.238.16.14'),
    'port':     os.getenv('DB_PORT', '5432'),
    'database': os.getenv('DB_NAME', 'wqi_db'),
    'user':     os.getenv('DB_USER', 'postgres'),
    'password': os.getenv('DB_PASSWORD', 'postgres1234'),
    'schema':   os.getenv('DB_SCHEMA', 'public')
}
OUTPUT_MODEL_DIR = os.getenv('OUTPUT_MODEL_DIR', '/app/models')

def create_spark_session():
    return (SparkSession.builder
            .appName("WaterQuality-Training")
            .config("spark.driver.memory", "2g")
            .config("spark.executor.memory", "2g")
            .master("local[*]")
            .getOrCreate())

def calculate_wqi(temperature, do, ph):
    # Temperature sub-index
    if temperature <= 25: q1 = 100
    elif temperature <= 30: q1 = 80
    else: q1 = 60
    # DO sub-index
    if do >= 7: q2 = 100
    elif do >= 6: q2 = 80
    elif do >= 5: q2 = 60
    elif do >= 4: q2 = 40
    else: q2 = 20
    # pH sub-index
    if 6.5 <= ph <= 8.5: q3 = 100
    elif 6.0 <= ph < 6.5 or 8.5 < ph <= 9.0: q3 = 80
    elif 5.5 <= ph < 6.0 or 9.0 < ph <= 9.5: q3 = 60
    elif 5.0 <= ph < 5.5 or 9.5 < ph <= 10.0: q3 = 40
    else: q3 = 20
    return 0.25 * q1 + 0.45 * q2 + 0.30 * q3

def train_and_evaluate(spark: SparkSession, output_dir: str) -> bool:
    source = f"{DB_CONFIG['schema']}.water_quality"
    logger.info(f"Loading data from {source}")
    subquery = f"""
      (SELECT id, wq_date,
        CAST(temperature AS DOUBLE PRECISION) AS temperature,
        CAST("DO" AS DOUBLE PRECISION)       AS do,
        CAST(ph AS DOUBLE PRECISION)         AS ph,
        CAST(wqi AS DOUBLE PRECISION)        AS wqi,
        ROW_NUMBER() OVER (ORDER BY wq_date) AS time_idx
       FROM {source}
       WHERE wq_date IS NOT NULL
         AND temperature IS NOT NULL
         AND "DO" IS NOT NULL
         AND ph IS NOT NULL
       ORDER BY wq_date
      ) AS t
    """
    df = (spark.read.format("jdbc")
          .option("url", f"jdbc:postgresql://{DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['database']}")
          .option("dbtable", subquery)
          .option("user", DB_CONFIG['user'])
          .option("password", DB_CONFIG['password'])
          .option("driver", "org.postgresql.Driver")
          .load())

    # UDF tính WQI nếu NULL
    wqi_udf = F.udf(calculate_wqi, DoubleType())
    df = df.withColumn("wqi",
            F.when(F.col("wqi").isNull(),
                   wqi_udf("temperature","do","ph")
            ).otherwise(F.col("wqi"))
    )

    # Cast sang double
    for c in ["temperature","do","ph","wqi","time_idx"]:
        df = df.withColumn(c, F.col(c).cast("double"))

    count = df.count()
    if count < 60:
        raise ValueError(f"Insufficient data: only {count} rows")

    # Thêm feature thời gian & seasonal
    df = df.withColumn("day_of_year", F.dayofyear("wq_date"))
    df = df.withColumn("year_sin", F.sin(F.col("day_of_year") * 2 * 3.14159 / 365.25))
    df = df.withColumn("year_cos", F.cos(F.col("day_of_year") * 2 * 3.14159 / 365.25))

    # Window specs cho rolling
    win3 =  Window.orderBy("wq_date").rowsBetween(-2, 0)
    win6 =  Window.orderBy("wq_date").rowsBetween(-5, 0)
    win12 = Window.orderBy("wq_date").rowsBetween(-11, 0)

    # Tính Moving Averages
    df = (df
          .withColumn("MA3",  F.avg("wqi").over(win3))
          .withColumn("MA6",  F.avg("wqi").over(win6))
          .withColumn("MA12", F.avg("wqi").over(win12))
    )

    # Tính xu hướng (trend = wqi - MA)
    df = (df
          .withColumn("trend3",  F.col("wqi") - F.col("MA3"))
          .withColumn("trend6",  F.col("wqi") - F.col("MA6"))
          .withColumn("trend12", F.col("wqi") - F.col("MA12"))
    )

    # Tính Rate of Change
    df = (df
          .withColumn("ROC3", (F.col("wqi") - F.lag("wqi", 3).over(Window.orderBy("wq_date"))) / 3)
          .withColumn("ROC6", (F.col("wqi") - F.lag("wqi", 6).over(Window.orderBy("wq_date"))) / 6)
    )

    # Split train/test (last 24 rows test)
    split_date = df.orderBy("wq_date").select("wq_date") \
                   .collect()[int(count-24)][0]
    train = df.filter(F.col("wq_date") < split_date)
    test  = df.filter(F.col("wq_date") >= split_date)
    logger.info(f"Train: {train.count()} rows, Test: {test.count()} rows")

    # Pipeline & CrossValidator
    feature_cols = [
        "temperature","do","ph",
        "year_sin","year_cos","time_idx",
        "MA3","MA6","MA12",
        "trend3","trend6","trend12",
        "ROC3","ROC6"
    ]
    assembler = VectorAssembler(inputCols=feature_cols, outputCol="features", handleInvalid="keep")
    xgb = SparkXGBRegressor(features_col="features", label_col="wqi",
                             objective="reg:squarederror", max_depth=5,
                             learning_rate=0.05, n_estimators=100)
    pipe = Pipeline(stages=[assembler, xgb])

    grid = (ParamGridBuilder()
            .addGrid(xgb.max_depth,    [3,5])
            .addGrid(xgb.learning_rate,[0.03,0.05])
            .addGrid(xgb.subsample,    [0.7,0.8])
            .build())

    evaluator_rmse = RegressionEvaluator(labelCol="wqi", metricName="rmse")
    evaluator_r2   = RegressionEvaluator(labelCol="wqi", metricName="r2")
    cv = CrossValidator(estimator=pipe,
                       estimatorParamMaps=grid,
                       evaluator=evaluator_rmse,
                       numFolds=5,
                       parallelism=2,
                       seed=42)

    # Fit & chọn best model
    cv_model   = cv.fit(train)
    best_model = cv_model.bestModel

    # Đánh giá trên test
    preds = best_model.transform(test)
    rmse  = evaluator_rmse.evaluate(preds)
    r2    = evaluator_r2.evaluate(preds)
    logger.info(f"Test RMSE = {rmse:.4f}, R2 = {r2:.4f}")

    # Lưu model ra disk
    out_path = os.path.join(output_dir, "best_xgb_pipeline")
    best_model.write().overwrite().save(out_path)
    logger.info(f"Model saved locally at {out_path}")

    return True

def main():
    spark = create_spark_session()
    try:
        if train_and_evaluate(spark, OUTPUT_MODEL_DIR):
            logger.info("Training & saving model completed successfully.")
    finally:
        spark.stop()

if __name__ == "__main__":
    main()
