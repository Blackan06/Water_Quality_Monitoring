import psycopg2
from datetime import datetime
from logs.LogService import write_log_to_postgres
import os
from dotenv import load_dotenv
import logging

load_dotenv()

db_name = os.getenv("DB_NAME")
db_user = os.getenv("DB_USER")
db_password = os.getenv("DB_PASSWORD")
db_host = os.getenv("DB_HOST")
db_port = os.getenv("DB_PORT")
db_postgresql_driver = "org.postgresql.Driver"
db_properties = {}
db_properties['user'] = db_user
db_properties['password'] = db_password
db_properties['driver'] = db_postgresql_driver

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
def save_to_postgresql_table(spark,current_df,epoch_id):


    #write_log_to_postgres("INFO","Starting to write batch to PostgreSQL","Spark Consumer")
    #write_log_to_postgres("INFO",f"jdbc:postgresql://{db_host}:{db_port}/{db_name}","Spark Consumer")
    postgresql_jdbc_url = f"jdbc:postgresql://{db_host}:{db_port}/{db_name}"

    try:
        print(f'spark test: {spark}')
        # Log schema of the current DataFrame
        #write_log_to_postgres("INFO",f"Current DataFrame schema:{current_df.schema.simpleString()}","Spark Consumer")

        current_df.printSchema()

        # Log preview of the current DataFrame
        #write_log_to_postgres("INFO",f"Preview of the current DataFrame::{current_df.show(truncate=False)}","Spark Consumer")

        current_df.show(truncate=False)

        # Read existing data for verification
        #write_log_to_postgres("INFO",f"Reading existing data from PostgreSQL...","Spark Consumer")
        print(db_properties)
        print(f"jdbc:postgresql://{db_host}:{db_port}/{db_name}")

        try:
            print("Attempting to read existing data from PostgreSQL...")
            test_df = spark.read.jdbc(
                url=f"jdbc:postgresql://{db_host}:{db_port}/{db_name}",
                table="iot_sensor",
                properties=db_properties
            )
            print(f"1abc {test_df}")
            test_df.show(5)

        except Exception as e:
            error_message = f"Error while reading from PostgreSQL: {e}"
            write_log_to_postgres("ERROR", error_message, 'Spark Consumer')
            print(error_message)
            raise 

        print(f"1abc {test_df}")
        print(f"1abc {test_df.show(5)}")
        #write_log_to_postgres("INFO",f"Existing data in PostgreSQL:","Spark Consumer")

        test_df.show(truncate=False)
        print(f"1abc {test_df.show(truncate=False)}")
        print(f"1abc {test_df.show(5)}")

        # Write new data to the PostgreSQL table
        current_df.write.jdbc(url=postgresql_jdbc_url,
                            table="iot_sensor",
                            mode='append',
                            properties=db_properties)
        #write_log_to_postgres("INFO",f"Batch successfully written to PostgreSQL.","Spark Consumer")

    except Exception as e:
        write_log_to_postgres("ERROR",f"Error while writing to PostgreSQL: {e}",'Spark Consumer')