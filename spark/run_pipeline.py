import logging
from spark_jobs import load_data, preprocess_data, train_model, save_results

# --- logging ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def run_pipeline():
    try:
        # Step 1: Load Data
        logger.info("Starting data loading...")
        spark = load_data.create_spark_session()
        if not load_data.load_data_from_postgres(spark):
            raise Exception("Data loading failed")
        
        # Step 2: Preprocess
        logger.info("Starting preprocessing...")
        if not preprocess_data.preprocess_data(spark):
            raise Exception("Preprocessing failed")
            
        # Step 3: Train Model
        logger.info("Starting model training...")
        if not train_model.train_and_evaluate(spark, "/app/models"):
            raise Exception("Model training failed")
            
        # Step 4: Save Results
        logger.info("Saving results...")
        if not save_results.save_to_postgres(spark):
            raise Exception("Saving results failed")
            
        logger.info("Pipeline completed successfully!")
        
    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}")
        raise
    finally:
        spark.stop()

if __name__ == "__main__":
    run_pipeline() 