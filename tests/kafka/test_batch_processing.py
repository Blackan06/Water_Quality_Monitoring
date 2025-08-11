import unittest
import json
from datetime import datetime
from unittest.mock import Mock, patch, MagicMock
import sys
import os

# Add paths for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
from include.iot_streaming.kafka_consumer import kafka_consumer_task
from include.iot_streaming.database_manager import DatabaseManager

class TestBatchProcessing(unittest.TestCase):
    
    def setUp(self):
        """Set up test fixtures"""
        self.db_manager = DatabaseManager()
        
    @patch('include.iot_streaming.kafka_consumer.KafkaConsumer')
    @patch('include.iot_streaming.kafka_consumer.db_manager')
    def test_single_station_message(self, mock_db_manager, mock_kafka_consumer):
        """Test processing single station message"""
        # Mock Kafka message
        mock_message = Mock()
        mock_message.value = {
            'station_id': 1,
            'measurement_time': '2024-01-01T10:00:00',
            'ph': 7.2,
            'temperature': 25.5,
            'do': 8.0
        }
        
        # Mock Kafka consumer
        mock_consumer = Mock()
        mock_consumer.__iter__.return_value = [mock_message]
        mock_kafka_consumer.return_value = mock_consumer
        
        # Mock database manager
        mock_db_manager.insert_raw_data.return_value = True
        
        # Test kafka consumer task
        result = kafka_consumer_task()
        
        # Assertions
        self.assertIn("Batch processed", result)
        mock_db_manager.insert_raw_data.assert_called_once()
        
        # Check the data passed to database
        call_args = mock_db_manager.insert_raw_data.call_args[0][0]
        self.assertEqual(call_args['station_id'], 1)
        self.assertEqual(call_args['ph'], 7.2)
        self.assertEqual(call_args['temperature'], 25.5)
        self.assertEqual(call_args['do'], 8.0)
    
    @patch('include.iot_streaming.kafka_consumer.KafkaConsumer')
    @patch('include.iot_streaming.kafka_consumer.db_manager')
    def test_multiple_stations_message(self, mock_db_manager, mock_kafka_consumer):
        """Test processing message with multiple stations using 'messages' format"""
        # Mock Kafka message with multiple stations
        mock_message = Mock()
        mock_message.value = {
            'messages': [
                {
                    'station_id': 1,
                    'measurement_time': '2024-01-01T10:00:00',
                    'ph': 7.2,
                    'temperature': 25.5,
                    'do': 8.0
                },
                {
                    'station_id': 2,
                    'measurement_time': '2024-01-01T10:01:00',
                    'ph': 6.8,
                    'temperature': 26.0,
                    'do': 7.5
                }
            ]
        }
        
        # Mock Kafka consumer
        mock_consumer = Mock()
        mock_consumer.__iter__.return_value = [mock_message]
        mock_kafka_consumer.return_value = mock_consumer
        
        # Mock database manager
        mock_db_manager.insert_raw_data.return_value = True
        
        # Test kafka consumer task
        result = kafka_consumer_task()
        
        # Assertions
        self.assertIn("Batch processed", result)
        # Should be called twice (once for each station)
        self.assertEqual(mock_db_manager.insert_raw_data.call_count, 2)
        
        # Check calls for both stations
        calls = mock_db_manager.insert_raw_data.call_args_list
        self.assertEqual(calls[0][0][0]['station_id'], 1)
        self.assertEqual(calls[1][0][0]['station_id'], 2)
    
    @patch('include.iot_streaming.kafka_consumer.KafkaConsumer')
    @patch('include.iot_streaming.kafka_consumer.db_manager')
    def test_legacy_stations_message(self, mock_db_manager, mock_kafka_consumer):
        """Test processing message with multiple stations using legacy 'stations' format"""
        # Mock Kafka message with multiple stations (legacy format)
        mock_message = Mock()
        mock_message.value = {
            'stations': [
                {
                    'station_id': 1,
                    'measurement_time': '2024-01-01T10:00:00',
                    'ph': 7.2,
                    'temperature': 25.5,
                    'do': 8.0
                },
                {
                    'station_id': 2,
                    'measurement_time': '2024-01-01T10:01:00',
                    'ph': 6.8,
                    'temperature': 26.0,
                    'do': 7.5
                }
            ]
        }
        
        # Mock Kafka consumer
        mock_consumer = Mock()
        mock_consumer.__iter__.return_value = [mock_message]
        mock_kafka_consumer.return_value = mock_consumer
        
        # Mock database manager
        mock_db_manager.insert_raw_data.return_value = True
        
        # Test kafka consumer task
        result = kafka_consumer_task()
        
        # Assertions
        self.assertIn("Batch processed", result)
        # Should be called twice (once for each station)
        self.assertEqual(mock_db_manager.insert_raw_data.call_count, 2)
        
        # Check calls for both stations
        calls = mock_db_manager.insert_raw_data.call_args_list
        self.assertEqual(calls[0][0][0]['station_id'], 1)
        self.assertEqual(calls[1][0][0]['station_id'], 2)
    
    @patch('include.iot_streaming.kafka_consumer.KafkaConsumer')
    @patch('include.iot_streaming.kafka_consumer.db_manager')
    def test_multiple_messages_batch(self, mock_db_manager, mock_kafka_consumer):
        """Test processing multiple messages in a batch"""
        # Mock multiple Kafka messages
        mock_message1 = Mock()
        mock_message1.value = {
            'station_id': 1,
            'measurement_time': '2024-01-01T10:00:00',
            'ph': 7.2,
            'temperature': 25.5,
            'do': 8.0
        }
        
        mock_message2 = Mock()
        mock_message2.value = {
            'messages': [
                {
                    'station_id': 2,
                    'measurement_time': '2024-01-01T10:01:00',
                    'ph': 6.8,
                    'temperature': 26.0,
                    'do': 7.5
                },
                {
                    'station_id': 3,
                    'measurement_time': '2024-01-01T10:02:00',
                    'ph': 7.5,
                    'temperature': 24.5,
                    'do': 8.2
                }
            ]
        }
        
        # Mock Kafka consumer
        mock_consumer = Mock()
        mock_consumer.__iter__.return_value = [mock_message1, mock_message2]
        mock_kafka_consumer.return_value = mock_consumer
        
        # Mock database manager
        mock_db_manager.insert_raw_data.return_value = True
        
        # Test kafka consumer task
        result = kafka_consumer_task()
        
        # Assertions
        self.assertIn("Batch processed", result)
        # Should be called 3 times (1 + 2 stations)
        self.assertEqual(mock_db_manager.insert_raw_data.call_count, 3)
        
        # Check all station IDs were processed
        calls = mock_db_manager.insert_raw_data.call_args_list
        station_ids = [call[0][0]['station_id'] for call in calls]
        self.assertEqual(set(station_ids), {1, 2, 3})
    
    @patch('include.iot_streaming.kafka_consumer.KafkaConsumer')
    @patch('include.iot_streaming.kafka_consumer.db_manager')
    def test_invalid_message_format(self, mock_db_manager, mock_kafka_consumer):
        """Test handling invalid message format"""
        # Mock invalid Kafka message
        mock_message = Mock()
        mock_message.value = {
            'invalid_field': 'invalid_value'
        }
        
        # Mock Kafka consumer
        mock_consumer = Mock()
        mock_consumer.__iter__.return_value = [mock_message]
        mock_kafka_consumer.return_value = mock_consumer
        
        # Mock database manager
        mock_db_manager.insert_raw_data.return_value = True
        
        # Test kafka consumer task
        result = kafka_consumer_task()
        
        # Assertions
        self.assertIn("Batch processed", result)
        # Should not call database insert for invalid message
        mock_db_manager.insert_raw_data.assert_not_called()
    
    @patch('include.iot_streaming.kafka_consumer.KafkaConsumer')
    @patch('include.iot_streaming.kafka_consumer.db_manager')
    def test_missing_required_fields(self, mock_db_manager, mock_kafka_consumer):
        """Test handling message with missing required fields"""
        # Mock Kafka message with missing fields
        mock_message = Mock()
        mock_message.value = {
            'station_id': 1,
            'measurement_time': '2024-01-01T10:00:00',
            'ph': 7.2,
            # Missing temperature and do
        }
        
        # Mock Kafka consumer
        mock_consumer = Mock()
        mock_consumer.__iter__.return_value = [mock_message]
        mock_kafka_consumer.return_value = mock_consumer
        
        # Mock database manager
        mock_db_manager.insert_raw_data.return_value = True
        
        # Test kafka consumer task
        result = kafka_consumer_task()
        
        # Assertions
        self.assertIn("Batch processed", result)
        # Should not call database insert for invalid data
        mock_db_manager.insert_raw_data.assert_not_called()
    
    @patch('include.iot_streaming.kafka_consumer.KafkaConsumer')
    @patch('include.iot_streaming.kafka_consumer.db_manager')
    def test_database_insert_failure(self, mock_db_manager, mock_kafka_consumer):
        """Test handling database insert failure"""
        # Mock Kafka message
        mock_message = Mock()
        mock_message.value = {
            'station_id': 1,
            'measurement_time': '2024-01-01T10:00:00',
            'ph': 7.2,
            'temperature': 25.5,
            'do': 8.0
        }
        
        # Mock Kafka consumer
        mock_consumer = Mock()
        mock_consumer.__iter__.return_value = [mock_message]
        mock_kafka_consumer.return_value = mock_consumer
        
        # Mock database manager to fail
        mock_db_manager.insert_raw_data.return_value = False
        
        # Test kafka consumer task
        result = kafka_consumer_task()
        
        # Assertions
        self.assertIn("Batch processed", result)
        # Should still call database insert but it fails
        mock_db_manager.insert_raw_data.assert_called_once()

if __name__ == '__main__':
    unittest.main() 