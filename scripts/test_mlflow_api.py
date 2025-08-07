#!/usr/bin/env python3
"""
Test script for MLflow API functionality
Tests the new model registry API without deprecated stages
"""

import mlflow
from mlflow.tracking import MlflowClient
import logging
import requests

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_mlflow_connection():
    """Test MLflow server connectivity"""
    try:
        response = requests.get("http://77.37.44.237:5003/health", timeout=10)
        if response.status_code == 200:
            logger.info("✅ MLflow server is accessible")
            return True
        else:
            logger.error(f"❌ MLflow server returned status {response.status_code}")
            return False
    except Exception as e:
        logger.error(f"❌ Cannot connect to MLflow server: {e}")
        return False

def test_model_registry_api():
    """Test the new model registry API"""
    try:
        mlflow.set_tracking_uri("http://77.37.44.237:5003")
        client = MlflowClient()
        
        # Test models to check
        test_models = ["water_quality_best_model", "water_quality_scaler"]
        
        for model_name in test_models:
            logger.info(f"🔍 Testing model: {model_name}")
            
            try:
                # Get all versions of the model
                versions = client.search_model_versions(f"name='{model_name}'")
                if versions:
                    logger.info(f"✅ Found {len(versions)} versions for {model_name}")
                    
                    # Get the latest version
                    latest_version = max(versions, key=lambda v: v.version)
                    logger.info(f"📋 Latest version: {latest_version.version}")
                    
                    # Check if model has Production alias
                    try:
                        production_version = client.get_model_version_by_alias(model_name, "Production")
                        logger.info(f"✅ {model_name} has Production alias: v{production_version.version}")
                    except Exception as e:
                        logger.info(f"ℹ️ {model_name} has no Production alias: {e}")
                        
                else:
                    logger.warning(f"⚠️ No versions found for {model_name}")
                    
            except Exception as e:
                logger.error(f"❌ Error testing {model_name}: {e}")
                
    except Exception as e:
        logger.error(f"❌ Error in test_model_registry_api: {e}")

def test_alias_operations():
    """Test alias operations"""
    try:
        mlflow.set_tracking_uri("http://77.37.44.237:5003")
        client = MlflowClient()
        
        model_name = "water_quality_best_model"
        
        # Get latest version
        versions = client.search_model_versions(f"name='{model_name}'")
        if not versions:
            logger.warning(f"⚠️ No versions found for {model_name}")
            return
            
        latest_version = max(versions, key=lambda v: v.version)
        version = latest_version.version
        
        logger.info(f"📋 Testing alias operations for {model_name} v{version}")
        
        # Test setting an alias
        try:
            client.set_registered_model_alias(
                name=model_name,
                alias="TestAlias",
                version=version
            )
            logger.info(f"✅ Successfully set TestAlias for {model_name} v{version}")
            
            # Test getting model by alias
            alias_version = client.get_model_version_by_alias(model_name, "TestAlias")
            logger.info(f"✅ Successfully retrieved model by alias: v{alias_version.version}")
            
            # Test deleting alias
            client.delete_registered_model_alias(model_name, "TestAlias")
            logger.info(f"✅ Successfully deleted TestAlias")
            
        except Exception as e:
            logger.error(f"❌ Error in alias operations: {e}")
            
    except Exception as e:
        logger.error(f"❌ Error in test_alias_operations: {e}")

if __name__ == "__main__":
    logger.info("🧪 Starting MLflow API tests...")
    
    # Test 1: Connection
    if test_mlflow_connection():
        # Test 2: Model Registry API
        test_model_registry_api()
        
        # Test 3: Alias Operations
        test_alias_operations()
    else:
        logger.error("❌ Cannot proceed with tests - MLflow server not accessible")
    
    logger.info("🏁 MLflow API tests completed") 