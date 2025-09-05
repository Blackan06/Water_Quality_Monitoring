from pyspark.sql import SparkSession
from pyspark.ml.linalg import VectorUDT
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
    'host': os.environ.get('DB_HOST', '194.238.16.14'),
    'port': os.environ.get('DB_PORT', '5432'),
    'database': os.environ.get('DB_NAME', 'wqi_db'),
    'user': os.environ.get('DB_USER', 'postgres'),
    'password': os.environ.get('DB_PASSWORD', 'postgres1234'),
    'schema': os.environ.get('DB_SCHEMA', 'public')
}
TABLE_FEATURE = f"{DB_CONFIG['schema']}.water_quality_feature"

def create_spark_session():
    return (SparkSession.builder
            .appName("WaterQuality-SaveResults")
            .config("spark.jars", "/opt/bitnami/spark/jars/postgresql-42.6.0.jar")
            .config("spark.driver.memory", "2g")
            .config("spark.executor.memory", "2g")
            .master("local[*]")
            .getOrCreate())

def save_results(spark):
    # Load features from water_quality_features table
    features_table = f"{DB_CONFIG['schema']}.water_quality_features"
    logger.info(f"Loading features from {features_table}")
    
    try:
        df = (spark.read.format("jdbc")
              .option("url", f"jdbc:postgresql://{DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['database']}")
              .option("dbtable", features_table)
              .option("user", DB_CONFIG['user'])
              .option("password", DB_CONFIG['password'])
              .option("driver", "org.postgresql.Driver")
              .load())
        
        # Log summary statistics
        logger.info("Feature statistics:")
        df.describe().show()
        
        return True
    except Exception as e:
        logger.error(f"Error loading from {features_table}: {e}")
        raise

def main():
    spark = create_spark_session()
    try:
        success = save_results(spark)
        if success:
            logger.info("Results saving completed successfully")
    finally:
        spark.stop()

if __name__ == "__main__":
    main() 