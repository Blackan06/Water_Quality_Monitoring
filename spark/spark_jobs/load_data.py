from pyspark.sql import SparkSession
import pyspark.sql.functions as F
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
TABLE = f"{DB_CONFIG['schema']}.water_quality"

def create_spark_session():
    return (SparkSession.builder
            .appName("WaterQuality-LoadData")
            .config("spark.jars", "/opt/bitnami/spark/jars/postgresql-42.6.0.jar")
            .config("spark.driver.extraClassPath", "/opt/bitnami/spark/jars/postgresql-42.6.0.jar")
            .config("spark.executor.extraClassPath", "/opt/bitnami/spark/jars/postgresql-42.6.0.jar")
            .config("spark.sql.legacy.timeParserPolicy", "LEGACY")
            .config("spark.driver.memory", "2g")
            .config("spark.executor.memory", "2g")
            .master("local[*]")
            .getOrCreate())

def load_data_from_postgres(spark):
    logger.info("Loading data from PostgreSQL")
    try:
        # Load from source table
        source_table = f"{DB_CONFIG['schema']}.water_quality"
        logger.info(f"Loading from source table {source_table}")
        
        sub = f"""
          (SELECT wq_date, temperature, "DO" AS DO, ph, wqi
           FROM {source_table}
           WHERE wq_date IS NOT NULL) AS t
        """
        
        df = (spark.read.format("jdbc")
              .option("url", f"jdbc:postgresql://{DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['database']}")
              .option("dbtable", sub)
              .option("user", DB_CONFIG['user'])
              .option("password", DB_CONFIG['password'])
              .option("driver", "org.postgresql.Driver")
              .option("fetchsize", "10000")
              .load())
        
        cnt = df.count()
        if cnt == 0:
            raise ValueError(f"No data found in {source_table}")
        logger.info(f"Loaded {cnt} rows from source table")
        
        return df
        
    except Exception as e:
        logger.error(f"Error in load_data_from_postgres: {str(e)}")
        raise

def main():
    spark = create_spark_session()
    try:
        df = load_data_from_postgres(spark)
        if df is not None:
            logger.info("Data loading completed successfully")
            df.createOrReplaceTempView("water_quality_data")
            logger.info("Created temporary view 'water_quality_data'")
    finally:
        spark.stop()

if __name__ == "__main__":
    main() 