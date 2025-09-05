from pyspark.sql import SparkSession, Window
import pyspark.sql.functions as F
from pyspark.ml.feature import VectorAssembler, MinMaxScaler
from pyspark.ml.linalg import VectorUDT
import logging
import os
from pyspark.ml import Pipeline
from pyspark.ml.regression import RandomForestRegressor, LinearRegression
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.sql.utils import AnalysisException
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from xgboost.spark import SparkXGBRegressor
from dateutil.relativedelta import relativedelta
import math
from pyspark.sql.types import StructType, StructField, TimestampType, DoubleType
from datetime import datetime

# --- logging ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# --- DB config ---
DB_CONFIG = {
    'host': os.environ.get('DB_HOST', '194.238.16.14'),
    'port': os.environ.get('DB_PORT', '5432'),
    'database': os.environ.get('DB_NAME', 'wqi_db'),
    'user': os.environ.get('DB_USER', 'postgres'),
    'password': os.environ.get('DB_PASSWORD', 'postgres1234'),
    'schema': os.environ.get('DB_SCHEMA', 'public')
}
TABLE = f"{DB_CONFIG['schema']}.water_quality"
TABLE_FEATURE = f"{DB_CONFIG['schema']}.water_quality_feature"


def create_spark_session():
    return (SparkSession.builder
            .appName("WaterQuality-Batch")
            .config("spark.jars", "/opt/bitnami/spark/jars/postgresql-42.6.0.jar")
            .config("spark.sql.legacy.timeParserPolicy", "LEGACY")
            .config("spark.driver.memory", "2g")
            .config("spark.executor.memory", "2g")
            .master("local[*]")
            .getOrCreate())


def load_data_from_postgres(spark):
    logger.info("Loading data from PostgreSQL")
    sub = f"""
      (SELECT wq_date, temperature, "DO" AS DO, ph, wqi
       FROM {TABLE}) AS t
    """
    try:
        df = (spark.read.format("jdbc")
              .option("url", f"jdbc:postgresql://{DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['database']}")
              .option("dbtable", sub)
              .option("user", DB_CONFIG['user'])
              .option("password", DB_CONFIG['password'])
              .option("driver", "org.postgresql.Driver")
              .load())
        cnt = df.count()
        logger.info(f"Loaded {cnt} rows")
        return df
    except AnalysisException as ae:
        logger.error(f"Postgres load error: {ae}")
        raise


def analyze_data(df):
    logger.info("Analyzing data")
    total = df.count()
    if total == 0:
        logger.warning("No data!")
        return
    for c in ['temperature','DO','ph']:
        s = df.select(
            F.mean(c).alias('m'),
            F.stddev(c).alias('s'),
            F.min(c).alias('lo'),
            F.max(c).alias('hi')
        ).first()
        miss = df.filter(F.col(c).isNull()).count()
        logger.info(f"{c}: mean={s['m']:.2f}, std={s['s']:.2f}, min={s['lo']:.2f}, max={s['hi']:.2f}, missing={miss}/{total}")
    for a,b in [('temperature','DO'),('temperature','ph'),('DO','ph')]:
        corr = df.stat.corr(a,b) or 0.0
        logger.info(f"Corr {a} vs {b}: {corr:.3f}")


def preprocess_data(df):
    if df.rdd.isEmpty():
        raise ValueError("Empty input")
    logger.info("Preprocessing")

    count = df.count()
    if count == 0:
        raise ValueError("No data for preprocessing")
    logger.info(f"Initial row count: {count}")
    df.show(5)

    # parse date as timestamp
    df = df.withColumn("wq_date", F.col("wq_date").cast("timestamp"))

    # fill nulls
    df = df.na.fill({
        'temperature': 25.0,
        'DO': 8.0,
        'ph': 7.0,
        'wqi': 50.0
    })

    # days_since_start
    min_date = df.select(F.min("wq_date")).first()[0]
    df = df.withColumn("days_since_start",
        F.when(F.col("wq_date").isNotNull(), F.datediff(F.col("wq_date"), F.lit(min_date)))
         .otherwise(F.lit(0)).cast('double'))

    # recalc WQI
    df = (df
          .withColumn('temperature', F.col('temperature').cast('double'))
          .withColumn('DO', F.col('DO').cast('double'))
          .withColumn('ph', F.col('ph').cast('double'))
          .withColumn('do_sat', 14.652 - 0.41022*F.col('temperature')
                                    + 0.007991*F.pow('temperature',2)
                                    - 0.000077774*F.pow('temperature',3))
          .withColumn('do_pct', F.when(F.col('do_sat')>0,
                                       F.col('DO')/F.col('do_sat')*100)
                                .otherwise(0.0))
          .withColumn('ph_dev', F.abs(F.col('ph') - 7.0))
          .withColumn('ph_q',   100 - F.col('ph_dev')/7.0*100)
          .withColumn('wqi',    (F.col('do_pct') + F.col('ph_q'))/2.0)
    )

    # log-transform
    df = (df
          .withColumn('T_log',  F.when(F.col('temperature')>0, F.log('temperature')).otherwise(F.log(F.lit(0.1))))
          .withColumn('DO_log', F.when(F.col('DO')>0,          F.log('DO')).otherwise(F.log(F.lit(0.1))))
          .withColumn('PH_log', F.when(F.col('ph')>0,          F.log('ph')).otherwise(F.log(F.lit(0.1))))
    )

    # rolling means
    win3 = Window.orderBy('wq_date').rowsBetween(-2,0)
    win6 = Window.orderBy('wq_date').rowsBetween(-5,0)
    df = (df
          .withColumn('MA3', F.coalesce(F.avg('wqi').over(win3), F.lit(50.0)))
          .withColumn('MA6', F.coalesce(F.avg('wqi').over(win6), F.lit(50.0)))
    )

    logger.info("Feature sample:")
    df.select('T_log','DO_log','PH_log','MA3','MA6','days_since_start').show(5)
    return df


def train_and_evaluate(df, output_model_dir):
    train, test = df.randomSplit([0.8,0.2], seed=42)
    logger.info(f"Train={train.count()}, Test={test.count()}")

    assembler = VectorAssembler(
        inputCols=['T_log','DO_log','PH_log','MA3','MA6','days_since_start'],
        outputCol='features', handleInvalid='keep'
    )

    # Random Forest
    rf = RandomForestRegressor(featuresCol='features', labelCol='wqi')
    pipe_rf = Pipeline(stages=[assembler, rf])
    grid_rf = (ParamGridBuilder()
               .addGrid(rf.numTrees, [50,100])
               .addGrid(rf.maxDepth,  [5,10])
               .build())

    # XGBoost
    xgb = SparkXGBRegressor(
        features_col='features', label_col='wqi', objective='reg:squarederror'
    )
    pipe_xgb = Pipeline(stages=[assembler, xgb])
    grid_xgb = (ParamGridBuilder()
                .addGrid(xgb.max_depth,     [5,10])
                .addGrid(xgb.learning_rate, [0.1,0.2])
                .build())

    # evaluators for multiple metrics
    metrics = ['rmse', 'r2', 'mae', 'mse']
    evaluators = {m: RegressionEvaluator(labelCol='wqi', metricName=m) for m in metrics}

    cv_rf  = CrossValidator(estimator=pipe_rf,  estimatorParamMaps=grid_rf,  evaluator=evaluators['rmse'], numFolds=3)
    cv_xgb = CrossValidator(estimator=pipe_xgb, estimatorParamMaps=grid_xgb, evaluator=evaluators['rmse'], numFolds=3)

    logger.info("Tuning RF…")
    m_rf  = cv_rf.fit(train)
    logger.info("Tuning XGB…")
    m_xgb = cv_xgb.fit(train)

    # compute and log all metrics
    for name, model in (('RF', m_rf), ('XGB', m_xgb)):
        for m, ev in evaluators.items():
            val = ev.evaluate(model.transform(test))
            logger.info(f"{name} {m.upper()}={val:.4f}")

    # pick best by RMSE
    best_model = m_rf.bestModel if evaluators['rmse'].evaluate(m_rf.transform(test)) <= evaluators['rmse'].evaluate(m_xgb.transform(test)) else m_xgb.bestModel
    best_name = 'rf' if isinstance(best_model.stages[-1], RandomForestRegressor) else 'xgb'

    out = os.path.join(output_model_dir, f"best_{best_name}")
    best_model.write().overwrite().save(out)
    logger.info(f"Saved best model to {out}")
    return best_model


def save_to_postgres(df):
    jdbc = f"jdbc:postgresql://{DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['database']}"
    vecs = [f.name for f in df.schema.fields if isinstance(f.dataType, VectorUDT)]
    (df.drop(*vecs)
       .write.format("jdbc")
       .option("url", jdbc)
       .option("dbtable", TABLE_FEATURE)
       .option("user", DB_CONFIG['user'])
       .option("password", DB_CONFIG['password'])
       .option("driver", "org.postgresql.Driver")
       .mode("overwrite")
       .save())
    logger.info("Data back to Postgres")


def main():
    spark = create_spark_session()
    try:
        df0  = load_data_from_postgres(spark)
        analyze_data(df0)
        df1  = preprocess_data(df0)
        best = train_and_evaluate(df1, output_model_dir="/app/models")
        save_to_postgres(df1)
    finally:
        spark.stop()
        logger.info("Done.")

if __name__ == "__main__":
    main()
