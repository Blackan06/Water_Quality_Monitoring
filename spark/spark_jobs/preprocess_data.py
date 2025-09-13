from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window
import logging
import os

# --- logging ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# --- DB config ---
DB_CONFIG = {
    'host': os.environ.get('DB_HOST', 'postgres'),
    'port': os.environ.get('DB_PORT', '5432'),
    'database': os.environ.get('DB_NAME', 'wqi_db'),
    'user': os.environ.get('DB_USER', 'postgres'),
    'password': os.environ.get('DB_PASSWORD', 'postgres'),
    'schema': os.environ.get('DB_SCHEMA', 'public')
}

def calculate_wqi(temperature, do, ph):
    """Calculate WQI based on temperature, DO, and pH values."""
    # Temperature sub-index (q1)
    if temperature <= 25:
        q1 = 100
    elif 25 < temperature <= 30:
        q1 = 80
    else:
        q1 = 60
    
    # DO sub-index (q2)
    if do >= 7:
        q2 = 100
    elif 6 <= do < 7:
        q2 = 80
    elif 5 <= do < 6:
        q2 = 60
    elif 4 <= do < 5:
        q2 = 40
    else:
        q2 = 20
    
    # pH sub-index (q3)
    if 6.5 <= ph <= 8.5:
        q3 = 100
    elif 6.0 <= ph < 6.5 or 8.5 < ph <= 9.0:
        q3 = 80
    elif 5.5 <= ph < 6.0 or 9.0 < ph <= 9.5:
        q3 = 60
    elif 5.0 <= ph < 5.5 or 9.5 < ph <= 10.0:
        q3 = 40
    else:
        q3 = 20
    
    # Calculate WQI using weighted arithmetic mean
    wqi = (0.25 * q1 + 0.45 * q2 + 0.30 * q3)
    return float(wqi)

def create_spark_session():
    return (SparkSession.builder
            .appName("WaterQuality-Preprocessing")
            .config("spark.jars", "/opt/bitnami/spark/jars/postgresql-42.6.0.jar")
            .config("spark.driver.memory", "2g")
            .config("spark.executor.memory", "2g")
            .master("local[*]")
            .getOrCreate())

def preprocess_data(spark):
    source_table = f"{DB_CONFIG['schema']}.water_quality"
    logger.info(f"Loading data from {source_table}")
    
    try:
        # Load data with proper date filtering
        sub = f"""
          (SELECT 
            id,
            wq_date,
            temperature,
            "DO",
            ph,
            wqi
           FROM {source_table}
           WHERE wq_date IS NOT NULL 
           AND temperature IS NOT NULL 
           AND "DO" IS NOT NULL 
           AND ph IS NOT NULL
           ORDER BY wq_date) AS t
        """
        
        df = (spark.read.format("jdbc")
              .option("url", f"jdbc:postgresql://{DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['database']}")
              .option("dbtable", sub)
              .option("user", DB_CONFIG['user'])
              .option("password", DB_CONFIG['password'])
              .option("driver", "org.postgresql.Driver")
              .load())
        
        # Register UDF for WQI calculation
        calculate_wqi_udf = F.udf(calculate_wqi)
        
        # Calculate WQI for NULL values
        df = df.withColumn(
            "wqi",
            F.when(F.col("wqi").isNull(),
                  calculate_wqi_udf(
                      F.col("temperature"),
                      F.col("DO"),
                      F.col("ph")
                  )
            ).otherwise(F.col("wqi"))
        )
        
        # Add time-based features
        df = df.withColumn('year', F.year('wq_date'))
        df = df.withColumn('month', F.month('wq_date'))
        df = df.withColumn('day_of_year', F.dayofyear('wq_date'))
        
        # Calculate seasonal features
        df = df.withColumn('season', 
            F.when((F.col('month') >= 3) & (F.col('month') <= 5), 'spring')
             .when((F.col('month') >= 6) & (F.col('month') <= 8), 'summer')
             .when((F.col('month') >= 9) & (F.col('month') <= 11), 'autumn')
             .otherwise('winter'))
        
        # Create time-based features
        df = df.withColumn('year_sin', F.sin(F.col('day_of_year') * 2 * 3.14159 / 365.25))
        df = df.withColumn('year_cos', F.cos(F.col('day_of_year') * 2 * 3.14159 / 365.25))
        
        # Calculate time index and partition key
        df = df.withColumn('time_idx', 
                          F.row_number().over(Window.orderBy('wq_date')))
        
        # Create window specs with proper partitioning
        win3 = (Window.partitionBy('year')
                      .orderBy('wq_date')
                      .rowsBetween(-2, 0))
        
        win6 = (Window.partitionBy('year')
                      .orderBy('wq_date')
                      .rowsBetween(-5, 0))
        
        win12 = (Window.partitionBy('year')
                       .orderBy('wq_date')
                       .rowsBetween(-11, 0))
        
        # Calculate moving averages and other statistics
        df = (df
            .withColumn('MA3', F.avg('wqi').over(win3))
            .withColumn('MA6', F.avg('wqi').over(win6))
            .withColumn('MA12', F.avg('wqi').over(win12))
            .withColumn('STD3', F.stddev('wqi').over(win3))
            .withColumn('STD6', F.stddev('wqi').over(win6))
        )
        
        # Fill nulls with appropriate values
        df = df.na.fill({
            'MA3': df.select(F.mean('wqi')).first()[0],
            'MA6': df.select(F.mean('wqi')).first()[0],
            'MA12': df.select(F.mean('wqi')).first()[0],
            'STD3': df.select(F.stddev('wqi')).first()[0],
            'STD6': df.select(F.stddev('wqi')).first()[0]
        })
        
        # Calculate trends
        df = (df
            .withColumn('trend3', F.col('wqi') - F.col('MA3'))
            .withColumn('trend6', F.col('wqi') - F.col('MA6'))
            .withColumn('trend12', F.col('wqi') - F.col('MA12'))
            # Add rate of change
            .withColumn('ROC3', (F.col('wqi') - F.lag('wqi', 3).over(Window.orderBy('wq_date'))) / 3)
            .withColumn('ROC6', (F.col('wqi') - F.lag('wqi', 6).over(Window.orderBy('wq_date'))) / 6)
        )
        
        # Create temporary view
        df.createOrReplaceTempView('preprocessed_data')
        logger.info("Created temporary view 'preprocessed_data'")
        
        # Show sample data
        logger.info("Sample of preprocessed data:")
        df.select('id', 'wq_date', 'temperature', 'DO', 'ph', 'wqi').show(5)
        
        # Show statistics
        logger.info("Statistics of key features:")
        df.select('wqi', 'MA3', 'MA6', 'MA12', 'trend3', 'trend6').describe().show()
        
        return True
        
    except Exception as e:
        logger.error(f"Error in preprocessing: {str(e)}")
        raise

def main():
    spark = create_spark_session()
    try:
        success = preprocess_data(spark)
        if success:
            logger.info("Data preprocessing completed successfully")
    finally:
        spark.stop()

if __name__ == "__main__":
    main() 