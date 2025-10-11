"""
Spark Consumer for Water Quality Monitoring
Xử lý Kafka messages với Apache Spark
"""

import logging
import json
from pickle import FALSE
import pandas as pd
import numpy as np
import requests
import os
import sys
import time
import shutil
import threading
from datetime import datetime, timezone
from typing import Dict, Any, Optional, List
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when, isnan, isnull, avg, stddev, lag, window, count
from pyspark.sql.types import StructType, StructField, StringType, DoubleType, TimestampType, IntegerType
from contextlib import contextmanager

# Import database manager
from .database_manager import db_manager

# Try to import findspark for better Spark initialization
try:
    import findspark
    findspark.init()
except ImportError:
    pass

logger = logging.getLogger(__name__)

# Force reload - Spark local mode enabled
class SparkKafkaConsumer:
    """Spark-based Kafka consumer for water quality data processing"""
    
    def __init__(self):
        logger.info("🚀 Initializing SparkKafkaConsumer...")
        self.spark = None
        try:
            self._initialize_spark()
            if self.spark:
                logger.info("✅ SparkKafkaConsumer initialized successfully with Spark session")
            else:
                logger.error("❌ SparkKafkaConsumer initialized but Spark session is None")
        except Exception as e:
            logger.error(f"❌ Failed to initialize SparkKafkaConsumer: {e}")
            import traceback
            logger.error(f"❌ Traceback: {traceback.format_exc()}")
            self.spark = None

    @contextmanager
    def _heartbeat(self, label: str, interval_seconds: int = 10):
        """Emit periodic logs to indicate long-running action is alive."""
        stop_event = threading.Event()

        def _beat():
            tick = 1
            while not stop_event.is_set():
                time.sleep(interval_seconds)
                if stop_event.is_set():
                    break
                try:
                    logger.info(f"⏳ {label} still running... t+{tick * interval_seconds}s")
                except Exception:
                    pass
                tick += 1

        thread = threading.Thread(target=_beat, daemon=True)
        thread.start()
        try:
            yield
        finally:
            stop_event.set()
            try:
                thread.join(timeout=1)
            except Exception:
                pass
    
    def _check_spark_context_active(self) -> bool:
        """Check if SparkContext is active and not stopped"""
        try:
            from pyspark import SparkContext
            
            if SparkContext._active_spark_context is None:
                logger.info("ℹ️ No active SparkContext found")
                return False
            
            # Try to access SparkContext to verify it's not stopped
            try:
                sc = SparkContext._active_spark_context
                # Try to get master URL - this will fail if context is stopped
                master = sc.master
                logger.info(f"✅ SparkContext is active - Master: {master}")
                return True
            except Exception as e:
                logger.warning(f"⚠️ SparkContext exists but is stopped: {e}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Error checking SparkContext: {e}")
            return False
    
    def _cleanup_spark_context(self):
        """Clean up any existing SparkContext to avoid conflicts"""
        try:
            from pyspark import SparkContext
            from pyspark.sql import SparkSession
            
            logger.info("🧹 Starting SparkContext cleanup...")
            
            # Check if SparkContext is active first
            is_active = self._check_spark_context_active()
            
            # Force clear any existing SparkContext (even if stopped)
            try:
                if SparkContext._active_spark_context is not None:
                    logger.info("🧹 Found existing SparkContext, attempting to stop...")
                    
                    # Only try to stop if it's actually active
                    if is_active:
                        try:
                            SparkContext._active_spark_context.stop()
                            logger.info("✅ SparkContext stopped successfully")
                        except Exception as stop_error:
                            logger.warning(f"⚠️ Error stopping SparkContext: {stop_error}")
                    else:
                        logger.info("ℹ️ SparkContext already stopped, just clearing reference")
                    
                    # Force clear the reference regardless
                    SparkContext._active_spark_context = None
                    logger.info("✅ SparkContext reference cleared")
                else:
                    logger.info("ℹ️ No existing SparkContext found")
            except Exception as sc_error:
                logger.warning(f"⚠️ Error accessing SparkContext: {sc_error}")
            
            # Stop any existing SparkSession
            try:
                if hasattr(SparkSession, '_instantiatedSession') and SparkSession._instantiatedSession is not None:
                    logger.info("🧹 Found existing SparkSession, attempting to stop...")
                    try:
                        SparkSession._instantiatedSession.stop()
                        logger.info("✅ SparkSession stopped successfully")
                    except Exception as stop_error:
                        logger.warning(f"⚠️ Error stopping SparkSession: {stop_error}")
                    
                    # Force clear the reference
                    SparkSession._instantiatedSession = None
                    logger.info("✅ SparkSession reference cleared")
                else:
                    logger.info("ℹ️ No existing SparkSession found")
            except Exception as ss_error:
                logger.warning(f"⚠️ Error accessing SparkSession: {ss_error}")
            
            # Wait for cleanup to complete (reduced from 3s to 1s)
            import time
            logger.info("⏳ Waiting 1 second for cleanup to complete...")
            time.sleep(1)
            logger.info("✅ Cleanup completed")
            
        except Exception as e:
            logger.error(f"❌ Critical error during SparkContext cleanup: {e}")
            import traceback
            logger.error(f"❌ Traceback: {traceback.format_exc()}")
            # Continue anyway - we'll try to create new session
    
    def _initialize_spark(self, force_local: Optional[bool] = None):
        """Initialize Spark session
        Args:
            force_local: If provided, overrides environment-based local/cluster decision
        """
        try:
            import os
            import socket
            
            # Clean up any existing SparkContext first
            self._cleanup_spark_context()
            
            # Check if running in Docker (Airflow) or locally
            is_docker = os.path.exists('/.dockerenv') or os.environ.get('AIRFLOW_HOME')
            
            # Force local mode for testing (comment out for production)
            env_force_local = os.environ.get('SPARK_FORCE_LOCAL', 'true').lower() == 'true'  # Default to true
            force_local = env_force_local if force_local is None else bool(force_local)
            
            logger.info(f"🔍 Environment check - is_docker: {is_docker}, force_local: {force_local}")
            
            # Check if Spark cluster is available (with error handling)
            try:
                spark_cluster_available = self._check_spark_cluster_availability()
                logger.info(f"🔍 Cluster availability check: {spark_cluster_available}")
            except Exception as e:
                logger.warning(f"⚠️ Cluster availability check failed: {e}")
                spark_cluster_available = False
            
            # Use cluster mode only when explicitly enabled
            enable_cluster = os.environ.get('SPARK_ENABLE_CLUSTER', 'false').lower() == 'true'
            if not force_local and spark_cluster_available and enable_cluster:
                # Running in Docker/Airflow - use Spark cluster with enhanced networking config
                logger.info("🔗 Attempting to connect to Spark cluster...")
                
                # Try cluster mode with retry logic
                max_retries = 3
                for attempt in range(max_retries):
                    try:
                        logger.info(f"🔄 Cluster connection attempt {attempt + 1}/{max_retries}")
                        
                        # Derive a reachable driver host within Docker network (if any)
                        driver_host = os.environ.get('SPARK_DRIVER_HOST')
                        if not driver_host:
                            try:
                                # Use container IP within network
                                driver_host = socket.gethostbyname(socket.gethostname())
                            except Exception:
                                # Fallback: hostname string
                                driver_host = socket.gethostname()
                        logger.info(f"🔗 Using driver host for cluster mode: {driver_host}")

                        self.spark = SparkSession.builder \
                            .appName(f"WaterQualitySparkConsumer-{attempt}") \
                            .master("spark://spark-master:7077") \
                            .config("spark.sql.adaptive.enabled", "true") \
                            .config("spark.sql.adaptive.coalescePartitions.enabled", "true") \
                            .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer") \
                            .config("spark.driver.memory", "1g") \
                            .config("spark.driver.maxResultSize", "512m") \
                            .config("spark.driver.bindAddress", "0.0.0.0") \
                            .config("spark.driver.host", driver_host) \
                            .config("spark.pyspark.driver.python", sys.executable) \
                            .config("spark.pyspark.python", "/usr/bin/python3") \
                            .config("spark.executorEnv.PYSPARK_PYTHON", "/usr/bin/python3") \
                            .config("spark.executorEnv.PATH", "/usr/bin:/usr/local/bin:${PATH}") \
                            .config("spark.python.worker.reuse", "true") \
                            .config("spark.executor.memory", "512m") \
                            .config("spark.executor.cores", "1") \
                            .config("spark.cores.max", "2") \
                            .config("spark.driver.extraJavaOptions", "-Djava.net.preferIPv4Stack=true -XX:+UseG1GC -XX:+UseStringDeduplication") \
                            .config("spark.executor.extraJavaOptions", "-XX:+UseG1GC -XX:+UseStringDeduplication") \
                            .config("spark.network.timeout", "600s") \
                            .config("spark.rpc.askTimeout", "300s") \
                            .config("spark.rpc.lookupTimeout", "300s") \
                            .config("spark.network.io.retryWait", "60s") \
                            .config("spark.network.io.maxRetries", "5") \
                            .config("spark.network.io.preferDirectBufs", "false") \
                            .config("spark.sql.adaptive.advisoryPartitionSizeInBytes", "64MB") \
                            .config("spark.sql.execution.arrow.pyspark.enabled", "true") \
                            .config("spark.ui.showConsoleProgress", "true") \
                            .config("spark.sql.ui.retainedExecutions", "200") \
                            .getOrCreate()
                        logger.info("✅ Successfully connected to Spark cluster!")
                        break  # Success, exit retry loop
                        
                    except Exception as cluster_error:
                        logger.error(f"❌ Cluster connection attempt {attempt + 1} failed: {cluster_error}")
                        
                        # Clean up failed attempt
                        if self.spark:
                            try:
                                self.spark.stop()
                                self.spark = None
                            except:
                                pass
                        
                        if attempt < max_retries - 1:
                            logger.info(f"🔄 Retrying in 5 seconds... (attempt {attempt + 2}/{max_retries})")
                            import time
                            time.sleep(5)
                        else:
                            logger.error("❌ All cluster connection attempts failed")
                            raise cluster_error  # Will trigger fallback to local mode
                

            else:
                # Running locally - use local mode with enhanced configuration
                logger.info("🔄 Initializing Spark in LOCAL mode...")
                # Force python path for executors to avoid /usr/local/bin/python
                python_executable = shutil.which("python3") or sys.executable
                
                # Set Java options for local mode to avoid networking issues
                os.environ['PYSPARK_SUBMIT_ARGS'] = '--driver-memory 2g --executor-memory 2g pyspark-shell'
                os.environ['PYSPARK_DRIVER_PYTHON'] = python_executable
                os.environ['PYSPARK_PYTHON'] = python_executable
                
                try:
                    logger.info("🔧 Building Spark session with local[2] configuration...")
                    self.spark = SparkSession.builder \
                        .appName("WaterQualitySparkConsumer-Local") \
                        .master("local[2]") \
                        .config("spark.sql.adaptive.enabled", "true") \
                        .config("spark.sql.adaptive.coalescePartitions.enabled", "true") \
                        .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer") \
                        .config("spark.driver.memory", "2g") \
                        .config("spark.driver.maxResultSize", "1g") \
                        .config("spark.executor.memory", "1g") \
                        .config("spark.driver.bindAddress", "0.0.0.0") \
                        .config("spark.driver.host", os.environ.get("SPARK_DRIVER_HOST", "api-server")) \
                        .config("spark.pyspark.python", python_executable) \
                        .config("spark.pyspark.driver.python", python_executable) \
                        .config("spark.executorEnv.PYSPARK_PYTHON", python_executable) \
                        .config("spark.driver.extraJavaOptions", "-Djava.net.preferIPv4Stack=true -XX:+UseG1GC") \
                        .config("spark.sql.execution.arrow.pyspark.enabled", "true") \
                        .config("spark.sql.adaptive.advisoryPartitionSizeInBytes", "64MB") \
                        .config("spark.sql.adaptive.coalescePartitions.minPartitionSize", "1MB") \
                        .config("spark.ui.enabled", "true") \
                        .config("spark.ui.port", os.environ.get("SPARK_UI_PORT", "4040")) \
                        .config("spark.ui.showConsoleProgress", "true") \
                        .config("spark.port.maxRetries", "100") \
                        .getOrCreate()
                    
                    logger.info("✅ Spark local mode initialized successfully")
                    logger.info(f"🔗 Spark version: {self.spark.version}")
                    logger.info(f"🔗 Spark master: {self.spark.sparkContext.master}")
                    
                except Exception as local_error:
                    logger.error(f"❌ Local mode initialization failed: {local_error}")
                    import traceback
                    logger.error(f"❌ Traceback: {traceback.format_exc()}")
                    
                    # Try minimal local mode as fallback
                    logger.info("🔄 Trying minimal local mode with local[1]...")
                    try:
                        self.spark = SparkSession.builder \
                            .appName("WaterQualitySparkConsumer-Minimal") \
                            .master("local[1]") \
                            .config("spark.driver.memory", "1g") \
                            .config("spark.ui.enabled", "false") \
                            .getOrCreate()
                        logger.info("✅ Minimal Spark local mode initialized")
                    except Exception as minimal_error:
                        logger.error(f"❌ Even minimal local mode failed: {minimal_error}")
                        self.spark = None
                        raise
                
            
            self.spark.sparkContext.setLogLevel("ERROR")  # Reduce log noise
            logger.info("✅ Spark session initialized successfully")
            logger.info(f"🔗 Spark Master: {self.spark.sparkContext.master}")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize Spark session: {e}")
            self.spark = None
    
    
    
    def _check_spark_cluster_availability(self) -> bool:
        """Check if Spark cluster is available"""
        try:
            import socket
            import requests
            
            # Check if spark-master hostname resolves
            try:
                socket.gethostbyname('spark-master')
                logger.info("✅ spark-master hostname resolved")
            except socket.gaierror:
                logger.warning("⚠️ spark-master hostname resolution failed")
                return False
            
            # Check if Spark Master UI is accessible
            try:
                response = requests.get("http://spark-master:8080", timeout=5)
                if response.status_code == 200:
                    logger.info("✅ Spark Master UI is accessible")
                    return True
                else:
                    logger.warning(f"⚠️ Spark Master UI returned status: {response.status_code}")
                    return False
            except requests.exceptions.RequestException as e:
                logger.warning(f"⚠️ Spark Master UI not accessible: {e}")
                return False
                
        except Exception as e:
            logger.warning(f"⚠️ Error checking Spark cluster availability: {e}")
            return False
    
    def process_kafka_message(self, message: str) -> Dict[str, Any]:
        """
        Process Kafka message with Spark, database ingestion, and notifications
        Falls back to pure Python processing if Spark is unavailable
        
        Args:
            message: JSON string from Kafka
            
        Returns:
            Dict with processing results
        """
        try:
            total_start = time.monotonic()
            logger.info("⏱️ E2E: Start processing Kafka message")
            # Parse JSON message
            data = json.loads(message) if message else {}
            
            if not data:
                return {"success": False, "error": "Empty message"}
            
            if not self.spark:
                logger.error("❌ Spark session not initialized")
                return {"success": False, "error": "Spark session not initialized"}
            
            # Convert to DataFrame
            step1_start = time.monotonic()
            logger.info("📊 Step 1: Creating Spark DataFrame...")
            df = self._create_dataframe(data)
            logger.info("✅ DataFrame created (lazy evaluation)")
            try:
                partitions = df.rdd.getNumPartitions()
                logger.info(f"ℹ️ DF partitions: {partitions}")
            except Exception:
                pass
            logger.info(f"⏱️ Step 1 duration: {int((time.monotonic() - step1_start) * 1000)} ms")
            
            # Process with Spark
            step2_start = time.monotonic()
            logger.info("⚙️ Step 2: Processing with Spark transformations...")
            processed_df = self._process_with_spark(df)
            logger.info("✅ Spark processing completed")
            try:
                partitions_proc = processed_df.rdd.getNumPartitions()
                logger.info(f"ℹ️ Processed DF partitions: {partitions_proc}")
            except Exception:
                pass
            logger.info(f"⏱️ Step 2 duration: {int((time.monotonic() - step2_start) * 1000)} ms")
            
            # Convert back to dict/list
            step3_start = time.monotonic()
            logger.info("📤 Step 3: Converting DataFrame to dict...")
            result_data = self._dataframe_to_dict(processed_df)
            logger.info(f"✅ Converted {len(result_data)} records")
            logger.info(f"⏱️ Step 3 duration: {int((time.monotonic() - step3_start) * 1000)} ms")
            
            # Database ingestion (can be disabled via env)
            import os as _os_proc
            if (_os_proc.environ.get('DISABLE_DB_INGESTION') or 'false').lower() == 'true':
                logger.info("⏭️ Skipping database ingestion (DISABLE_DB_INGESTION=true)")
                db_success = False
            else:
                db_start = time.monotonic()
                db_success = self._ingest_to_database(result_data)
                logger.info(f"⏱️ DB ingestion duration: {int((time.monotonic() - db_start) * 1000)} ms")
            
            # Send notifications
            notif_start = time.monotonic()
            notification_success = self._send_notifications(result_data)
            logger.info(f"⏱️ Notifications duration: {int((time.monotonic() - notif_start) * 1000)} ms")
            
            logger.info(f"✅ Spark processed {len(result_data)} records successfully")
            logger.info(f"📊 Database ingestion: {'✅' if db_success else '❌'}")
            logger.info(f"📱 Notifications: {'✅' if notification_success else '❌'}")
            logger.info(f"⏱️ E2E: Total duration: {int((time.monotonic() - total_start) * 1000)} ms")
            
            return {
                "success": True,
                "data": result_data,
                "processed_count": len(result_data),
                "processing_time": datetime.now(timezone.utc).isoformat(),
                "database_ingested": db_success,
                "notifications_sent": notification_success
            }
            
        except Exception as e:
            err_text = str(e)
            logger.error(f"❌ Spark processing error: {err_text}")
            # Auto-fallback: if cluster-side failure like Master removed application or AppClient missing
            if any(x in err_text for x in [
                "Master removed our application",
                "Could not find AppClient",
                "AppClient",
                "RpcEndpointRef",
                "/usr/local/bin/python",
                "Cannot run program \"/usr/local/bin/python\"",
                "PythonWorkerFactory",
                "VALIDATION_TIMEOUT",
            ]):
                try:
                    logger.warning("♻️ Falling back to Spark local mode due to cluster failure...")
                    self._initialize_spark(force_local=True)
                    if not self.spark:
                        raise RuntimeError("Spark local re-initialization failed")
                    # Retry once in local mode
                    total_start = time.monotonic()
                    logger.info("⏱️ E2E (retry local): Start")
                    df = self._create_dataframe(data)
                    processed_df = self._process_with_spark(df)
                    result_data = self._dataframe_to_dict(processed_df)
                    logger.info(f"⏱️ E2E (retry local): {int((time.monotonic() - total_start) * 1000)} ms")
                    db_success = self._ingest_to_database(result_data)
                    notification_success = self._send_notifications(result_data)
                    return {
                        "success": True,
                        "data": result_data,
                        "processed_count": len(result_data),
                        "processing_time": datetime.now(timezone.utc).isoformat(),
                        "database_ingested": db_success,
                        "notifications_sent": notification_success,
                        "fallback_local": True,
                    }
                except Exception as e2:
                    logger.error(f"❌ Local fallback failed: {e2}")
                    return {"success": False, "error": str(e2)}
            return {"success": False, "error": err_text}
    
    def _create_dataframe(self, data: Dict[str, Any]) -> 'DataFrame':
        """Create Spark DataFrame from data"""
        try:
            logger.info("🔧 Creating DataFrame from data...")
            
            # Handle both single object and array
            if isinstance(data, list):
                records = data
                logger.info(f"📊 Input: {len(records)} records (list)")
            else:
                records = [data]
                logger.info("📊 Input: 1 record (dict)")
            
            # Create DataFrame
            logger.info("⚙️ Calling spark.createDataFrame()...")
            df = self.spark.createDataFrame(records)
            logger.info("✅ DataFrame created successfully")
            
            # Add processing timestamp
            from pyspark.sql.functions import current_timestamp
            df = df.withColumn("processing_time", current_timestamp())
            logger.info("✅ Added processing_time column")
            
            return df
            
        except Exception as e:
            logger.error(f"❌ Error creating DataFrame: {e}")
            import traceback
            logger.error(f"❌ Traceback: {traceback.format_exc()}")
            raise
    
    def _process_with_spark(self, df: 'DataFrame') -> 'DataFrame':
        """Process DataFrame with Spark transformations"""
        try:
            step_start = time.monotonic()
            logger.info("⏱️ Spark step 1/4: Clean & validate data...")
            # 1. Data cleaning and validation
            df_cleaned = self._clean_data(df)
            logger.info(f"⏱️ Spark step 1/4 done in {int((time.monotonic() - step_start) * 1000)} ms")
            try:
                p_clean = df_cleaned.rdd.getNumPartitions()
                logger.info(f"ℹ️ Cleaned DF partitions: {p_clean}")
            except Exception:
                pass
            
            # 2. Feature engineering
            step_start = time.monotonic()
            logger.info("⏱️ Spark step 2/4: Feature engineering...")
            df_features = self._add_features(df_cleaned)
            logger.info(f"⏱️ Spark step 2/4 done in {int((time.monotonic() - step_start) * 1000)} ms")
            
            # 3. Calculate WQI
            step_start = time.monotonic()
            logger.info("⏱️ Spark step 3/4: Calculate WQI...")
            df_wqi = self._calculate_wqi_spark(df_features)
            logger.info(f"⏱️ Spark step 3/4 done in {int((time.monotonic() - step_start) * 1000)} ms")
            
            # 4. Add quality indicators
            step_start = time.monotonic()
            logger.info("⏱️ Spark step 4/4: Add quality indicators...")
            df_final = self._add_quality_indicators(df_wqi)
            logger.info(f"⏱️ Spark step 4/4 done in {int((time.monotonic() - step_start) * 1000)} ms")
            
            return df_final
            
        except Exception as e:
            logger.error(f"❌ Error in Spark processing: {e}")
            raise
    
    def _clean_data(self, df: 'DataFrame') -> 'DataFrame':
        """Clean and validate data - reject invalid data instead of replacing"""
        try:
            from pyspark.sql import functions as F
            import os as _os
            # Build validity condition once
            valid_cond = (
                col('ph').isNotNull() &
                col('temperature').isNotNull() &
                col('do').isNotNull() &
                col('station_id').isNotNull() &
                col('ph').between(0, 14) &
                col('temperature').between(-10, 50) &
                col('do').between(0, 20) &
                (col('station_id') >= 0)
            )
            # Choose validation mode: fast (default) or full via env
            validation_mode = (_os.environ.get('SPARK_VALIDATION_MODE') or 'fast').lower()
            if validation_mode not in ("fast", "full"):
                validation_mode = "fast"
            logger.info(f"🧪 Validation mode: {validation_mode}")

            # Apply filter once and cache to avoid recomputation downstream
            df_cleaned = df.filter(valid_cond).persist()
            try:
                p_after = df_cleaned.rdd.getNumPartitions()
                logger.info(f"ℹ️ Cleaned DF partitions (post-filter, cached): {p_after}")
            except Exception:
                pass

            if validation_mode == "full":
                # Single-pass aggregation for both counts
                agg_start = time.monotonic()
                with self._heartbeat("Spark count(validation)"):
                    stats = df.select(
                        F.count(F.lit(1)).alias("original_count"),
                        F.sum(valid_cond.cast("int")).alias("cleaned_count")
                    ).first()
                agg_ms = int((time.monotonic() - agg_start) * 1000)
                original_count = int(stats[0] or 0)
                cleaned_count = int(stats[1] or 0)
                logger.info(f"⏱️ Action: single-pass counts took {agg_ms} ms (original={original_count}, cleaned={cleaned_count})")
                removed_count = original_count - cleaned_count
                if removed_count > 0:
                    logger.warning(f"⚠️ Removed {removed_count} invalid records (null values or out of range)")
                    logger.info(f"✅ Valid records remaining: {cleaned_count}")
                else:
                    logger.info(f"✅ All {cleaned_count} records are valid")
                if cleaned_count == 0:
                    raise ValueError("❌ No valid data records found after cleaning")
                if removed_count > 0:
                    self._send_data_quality_alert(removed_count, original_count)
            else:
                # Fast validation: avoid full counts; just ensure non-empty
                fast_start = time.monotonic()
                timeout_sec = int((_os.environ.get('SPARK_VALIDATION_TIMEOUT_SEC') or '20'))
                logger.info(f"⏳ Fast validation (limit(1).count()) with timeout {timeout_sec}s")
                with self._heartbeat("Spark fast-check(limit(1).count())", interval_seconds=5):
                    exists = df_cleaned.limit(1).count()
                elapsed_ms = int((time.monotonic() - fast_start) * 1000)
                logger.info(f"⏱️ Fast validation check took {elapsed_ms} ms")
                if elapsed_ms > timeout_sec * 1000:
                    raise RuntimeError("VALIDATION_TIMEOUT: Spark fast validation exceeded timeout")
                if exists == 0:
                    raise ValueError("❌ No valid data records found after cleaning (fast mode)")
                logger.info("✅ Fast validation passed (counts skipped)")

            return df_cleaned
            
        except Exception as e:
            logger.error(f"❌ Error cleaning data: {e}")
            raise
    
    def _send_data_quality_alert(self, removed_count: int, original_count: int):
        """Send alert when data quality issues are detected"""
        try:
            loss_percentage = (removed_count / original_count) * 100
            
            alert_data = {
                "alert_type": "DATA_QUALITY_ISSUE",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "message": f"Data quality issue detected: {removed_count}/{original_count} records ({loss_percentage:.1f}%) were invalid",
                "details": {
                    "removed_records": removed_count,
                    "total_records": original_count,
                    "loss_percentage": loss_percentage,
                    "reason": "Null values or out-of-range measurements detected"
                }
            }
            
            # Log the alert
            logger.warning(f"🚨 DATA QUALITY ALERT: {alert_data['message']}")
            
            # Here you can add notification logic (email, Slack, etc.)
            # For now, just log the alert
            
        except Exception as e:
            logger.error(f"❌ Error sending data quality alert: {e}")
    
    def _add_features(self, df: 'DataFrame') -> 'DataFrame':
        """Add engineered features"""
        try:
            # Add temporal features
            from pyspark.sql.functions import hour, dayofweek, month, year
            
            df_features = df.withColumn('hour', hour('measurement_time')) \
                           .withColumn('day_of_week', dayofweek('measurement_time')) \
                           .withColumn('month', month('measurement_time')) \
                           .withColumn('year', year('measurement_time'))
            
            # Add interaction features
            df_features = df_features.withColumn(
                'ph_temp_interaction', 
                col('ph') * col('temperature')
            )
            
            df_features = df_features.withColumn(
                'ph_do_interaction',
                col('ph') * col('do')
            )
            
            df_features = df_features.withColumn(
                'temp_do_interaction',
                col('temperature') * col('do')
            )
            
            return df_features
            
        except Exception as e:
            logger.error(f"❌ Error adding features: {e}")
            raise
    
    def _calculate_wqi_spark(self, df: 'DataFrame') -> 'DataFrame':
        """Calculate WQI using Spark"""
        try:
            from pyspark.sql.functions import when, round
            
            # Calculate sub-indices
            df_wqi = df.withColumn(
                'ph_score',
                when((col('ph') >= 6.0) & (col('ph') <= 10.0), 
                     (col('ph') - 6.0) * 25)
                .otherwise(0)
            )
            
            from pyspark.sql.functions import abs as spark_abs
            
            df_wqi = df_wqi.withColumn(
                'temp_score',
                when((col('temperature') >= 15) & (col('temperature') <= 30),
                     100 - spark_abs(col('temperature') - 22) * 2)
                .otherwise(0)
            )
            
            df_wqi = df_wqi.withColumn(
                'do_score',
                when(col('do') >= 0, col('do') * 10)
                .otherwise(0)
            )
            
            # Calculate final WQI
            df_wqi = df_wqi.withColumn(
                'wqi',
                round(
                    col('ph_score') * 0.3 + 
                    col('temp_score') * 0.3 + 
                    col('do_score') * 0.4, 2
                )
            )
            
            return df_wqi
            
        except Exception as e:
            logger.error(f"❌ Error calculating WQI: {e}")
            raise
    
    def _add_quality_indicators(self, df: 'DataFrame') -> 'DataFrame':
        """Add quality indicators and alerts"""
        try:
            from pyspark.sql.functions import when, lit
            
            # Add quality status
            df_indicators = df.withColumn(
                'quality_status',
                when(col('wqi') >= 90, 'excellent')
                .when(col('wqi') >= 70, 'good')
                .when(col('wqi') >= 50, 'moderate')
                .when(col('wqi') >= 25, 'poor')
                .otherwise('very_poor')
            )
            
            # Add alert level
            df_indicators = df_indicators.withColumn(
                'alert_level',
                when(col('wqi') < 50, 'critical')
                .when(col('wqi') < 60, 'warning')
                .when(col('wqi') > 80, 'excellent')
                .otherwise('normal')
            )
            
            # Add processing metadata
            df_indicators = df_indicators.withColumn(
                'spark_processed', lit(True)
            )
            
            return df_indicators
            
        except Exception as e:
            logger.error(f"❌ Error adding quality indicators: {e}")
            raise
    
    def _dataframe_to_dict(self, df: 'DataFrame') -> list:
        """Convert DataFrame to list of dictionaries"""
        try:
            # Collect data
            collect_start = time.monotonic()
            logger.info("⏱️ Action: collecting DataFrame to driver...")
            with self._heartbeat("Spark collect(results)"):
                rows = df.collect()
            logger.info(f"⏱️ Action: collect() took {int((time.monotonic() - collect_start) * 1000)} ms; rows={len(rows)}")
            
            # Convert to list of dicts
            convert_start = time.monotonic()
            result = []
            for row in rows:
                row_dict = row.asDict()
                # Convert datetime objects to strings
                for key, value in row_dict.items():
                    if isinstance(value, datetime):
                        row_dict[key] = value.isoformat()
                result.append(row_dict)
            logger.info(f"⏱️ Action: rows->dict conversion took {int((time.monotonic() - convert_start) * 1000)} ms")
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Error converting DataFrame to dict: {e}")
            raise
    
    def _ingest_to_database(self, processed_data: list) -> bool:
        """
        Ingest processed data to database
        
        Args:
            processed_data: List of processed records
            
        Returns:
            bool: Success status
        """
        try:
            logger.info(f"📊 Ingesting {len(processed_data)} records to database...")
            
            success_count = 0
            for record in processed_data:
                try:
                    # Prepare raw data record
                    raw_record = {
                        "station_id": int(record.get("station_id", 0)),
                        "measurement_time": self._parse_measurement_time(record.get("measurement_time")),
                        "ph": float(record.get("ph", 7.0)),
                        "temperature": float(record.get("temperature", 25.0)),
                        "do": float(record.get("do", 8.0)),
                    }
                    # Prefer inserting processed data with WQI when available
                    wqi_value = record.get("wqi")
                    if wqi_value is not None:
                        processed_record = {
                            "station_id": raw_record["station_id"],
                            "measurement_time": raw_record["measurement_time"],
                            "ph": raw_record["ph"],
                            "temperature": raw_record["temperature"],
                            "do": raw_record["do"],
                            "wqi": float(wqi_value),
                        }
                        if db_manager.insert_processed_data(processed_record):
                            success_count += 1
                            logger.debug(f"✅ Inserted processed data with WQI for station {processed_record['station_id']}")
                        else:
                            logger.warning(f"⚠️ Failed to insert processed data, falling back to raw for station {raw_record['station_id']}")
                            if db_manager.insert_raw_data(raw_record):
                                success_count += 1
                                logger.debug(f"✅ Inserted raw data for station {raw_record['station_id']}")
                            else:
                                logger.warning(f"⚠️ Failed to insert raw data for station {raw_record['station_id']}")
                    else:
                        # No WQI available, compute from inputs and insert processed
                        ph_v = raw_record["ph"]
                        temp_v = raw_record["temperature"]
                        do_v = raw_record["do"]
                        ph_score = (ph_v - 6.0) * 25 if 6.0 <= ph_v <= 10.0 else 0.0
                        temp_score = 100 - abs(temp_v - 22.0) * 2 if 15.0 <= temp_v <= 30.0 else 0.0
                        do_score = do_v * 10 if do_v >= 0 else 0.0
                        wqi_comp = round(ph_score * 0.3 + temp_score * 0.3 + do_score * 0.4, 2)
                        processed_record = {
                            "station_id": raw_record["station_id"],
                            "measurement_time": raw_record["measurement_time"],
                            "ph": ph_v,
                            "temperature": temp_v,
                            "do": do_v,
                            "wqi": float(wqi_comp),
                        }
                        if db_manager.insert_processed_data(processed_record):
                            success_count += 1
                            logger.debug(f"✅ Computed and inserted WQI for station {processed_record['station_id']}")
                        else:
                            logger.warning(f"⚠️ Failed to insert processed data, falling back to raw for station {raw_record['station_id']}")
                            if db_manager.insert_raw_data(raw_record):
                                success_count += 1
                                logger.debug(f"✅ Inserted raw data for station {raw_record['station_id']}")
                            else:
                                logger.warning(f"⚠️ Failed to insert raw data for station {raw_record['station_id']}")
                        
                except Exception as e:
                    logger.error(f"❌ Error inserting record: {e}")
                    continue
            
            success_rate = success_count / len(processed_data) if processed_data else 0
            logger.info(f"📊 Database ingestion: {success_count}/{len(processed_data)} records ({success_rate:.1%})")
            
            return success_count > 0
            
        except Exception as e:
            logger.error(f"❌ Database ingestion error: {e}")
            return False
    
    def _parse_measurement_time(self, mt) -> datetime:
        """Parse measurement time from various formats"""
        try:
            if isinstance(mt, str):
                # Handle ISO strings including with timezone or trailing Z
                mt_str = mt[:-1] if mt.endswith("Z") else mt
                return datetime.fromisoformat(mt_str)
            elif isinstance(mt, (int, float)):
                return datetime.fromtimestamp(mt, tz=timezone.utc)
            else:
                return datetime.now(timezone.utc)
        except Exception:
            return datetime.now(timezone.utc)
    
    def _send_notifications(self, processed_data: list) -> bool:
        """
        Send notifications for processed data
        
        Args:
            processed_data: List of processed records
            
        Returns:
            bool: Success status
        """
        try:
            logger.info(f"📱 Sending notifications for {len(processed_data)} records...")
            
            success_count = 0
            for record in processed_data:
                try:
                    station_id = int(record.get("station_id", 0))
                    wqi_value = float(record.get("wqi", 50.0))
                    alert_level = record.get("alert_level", "normal")
                    
                    # Send notifications for all levels (with different priorities)
                    notification_sent = self._send_single_notification(
                        station_id, wqi_value, alert_level
                    )
                    if notification_sent:
                        success_count += 1
                        logger.debug(f"✅ Sent notification for station {station_id} (WQI: {wqi_value}, Level: {alert_level})")
                    else:
                        logger.warning(f"⚠️ Failed to send notification for station {station_id}")
                    
                except Exception as e:
                    logger.error(f"❌ Error sending notification: {e}")
                    continue
            
            logger.info(f"📱 Notifications: {success_count} sent successfully")
            return success_count > 0
            
        except Exception as e:
            logger.error(f"❌ Notification error: {e}")
            return False
    
    def _get_notification_content(self, station_id: int, wqi_value: float, alert_level: str) -> tuple:
        """
        Get notification content (for testing)
        
        Args:
            station_id: Station ID
            wqi_value: WQI value
            alert_level: Alert level
            
        Returns:
            tuple: (title, message, status)
        """
        # Determine notification content based on alert level
        if alert_level == "critical":
            title = f"🚨 Cảnh báo chất lượng nước NGHIÊM TRỌNG - Trạm {station_id}"
            message = f"WQI hiện tại là {wqi_value:.1f}. Cần xử lý ngay!"
            status = "critical"
        elif alert_level == "warning":
            title = f"⚠️ Cảnh báo chất lượng nước - Trạm {station_id}"
            message = f"WQI hiện tại là {wqi_value:.1f}. Vui lòng theo dõi sát."
            status = "warning"
        elif alert_level == "excellent":
            title = f"✅ Chất lượng nước RẤT TỐT - Trạm {station_id}"
            message = f"WQI hiện tại là {wqi_value:.1f}. Chất lượng nước rất tốt."
            status = "excellent"
        elif alert_level == "normal":
            title = f"📊 Cập nhật chất lượng nước - Trạm {station_id}"
            message = f"WQI hiện tại là {wqi_value:.1f}. Trạng thái bình thường."
            status = "info"
        else:
            title = f"📊 Báo cáo chất lượng nước - Trạm {station_id}"
            message = f"WQI hiện tại là {wqi_value:.1f}. Mức độ: {alert_level}"
            status = "info"
        # Map to API-accepted enum: 'good', 'danger', 'info', 'warning'
        status_map = {
            "critical": "danger",
            "warning": "warning",
            "excellent": "good",
            "info": "info",
        }
        api_status = status_map.get(status, "info")
        
        return title, message, api_status

    def _send_single_notification(self, station_id: int, wqi_value: float, alert_level: str) -> bool:
        """
        Send single notification
        
        Args:
            station_id: Station ID
            wqi_value: WQI value
            alert_level: Alert level
            
        Returns:
            bool: Success status
        """
        try:
            # Get notification content
            title, message, status = self._get_notification_content(station_id, wqi_value, alert_level)
            
            # Send notification via API
            return self._push_notification(
                account_id=3,
                title=title,
                message=message,
                status=status
            )
            
        except Exception as e:
            logger.error(f"❌ Error in single notification: {e}")
            return False
    
    def _push_notification(self, account_id: int, title: str, message: str, status: str) -> bool:
        """
        Send push notification to API
        
        Args:
            account_id: Account/Station ID
            title: Notification title
            message: Notification message
            status: Notification status
            
        Returns:
            bool: Success status
        """
        try:
            url = "https://datamanagerment.anhkiet.xyz/notifications/send-notification"
            payload = {
                "account_id": str(account_id),
                "title": title,
                "message": message,
                "status": status
            }
            
            response = requests.post(url, json=payload, timeout=10)
            if response.status_code == 200:
                logger.debug(f"✅ Push notification sent successfully for station {account_id}")
                return True
            else:
                logger.error(f"❌ Push notification failed: {response.text}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Error sending push notification: {e}")
            return False

    def close(self):
        """Close Spark session"""
        if self.spark:
            self.spark.stop()
            logger.info("✅ Spark session closed")

# Global consumer instance
_consumer = None

def get_spark_consumer():
    """Get or create Spark consumer instance"""
    global _consumer
    
    # Reuse existing consumer if Spark session is still active
    if _consumer is not None and _consumer.spark is not None:
        try:
            # Check if Spark session is still active
            _ = _consumer.spark.sparkContext.master
            logger.info("♻️ Reusing existing Spark consumer instance")
            return _consumer
        except Exception as e:
            logger.warning(f"⚠️ Existing consumer is stale: {e}")
            _consumer = None
    
    # Create new consumer
    logger.info("🔄 Creating new Spark consumer instance...")
    _consumer = SparkKafkaConsumer()
    return _consumer

def process_kafka_message_with_spark(message: str) -> Dict[str, Any]:
    """
    Process Kafka message with Spark (main function)
    
    Args:
        message: JSON string from Kafka
        
    Returns:
        Dict with processing results
    """
    try:
        consumer = get_spark_consumer()
        result = consumer.process_kafka_message(message)
        
        logger.info(f"📊 Spark processing result: {result.get('success', False)}")
        return result
        
    except Exception as e:
        logger.error(f"❌ Error in process_kafka_message_with_spark: {e}")
        return {"success": False, "error": str(e)}

def run_spark_kafka_consumer():
    """Run Spark Kafka consumer (for background processing)"""
    try:
        logger.info("🚀 Starting Spark Kafka consumer...")
        
        consumer = get_spark_consumer()
        if not consumer.spark:
            logger.error("❌ Failed to initialize Spark session")
            return
        
        # This would be used for continuous Kafka consumption
        # For now, just log that consumer is ready
        logger.info("✅ Spark Kafka consumer is ready for processing")
        
    except Exception as e:
        logger.error(f"❌ Error starting Spark consumer: {e}")

if __name__ == "__main__":
    # Test the consumer
    test_message = '{"station_id": 1, "ph": 7.2, "temperature": 25.5, "do": 8.1, "measurement_time": "2024-01-15T10:30:00Z"}'
    result = process_kafka_message_with_spark(test_message)
    print(f"Test result: {result}")
