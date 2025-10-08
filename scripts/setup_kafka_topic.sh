#!/bin/bash

# Script to check and create Kafka topic for water quality data
# This script will be run in the Kafka container

set -e

echo "🔍 Checking Kafka topic setup..."

# Wait for Kafka to be ready
echo "⏳ Waiting for Kafka to be ready..."
until kafka-topics.sh --bootstrap-server kafka:9092 --list > /dev/null 2>&1; do
    echo "⏳ Kafka not ready yet, waiting..."
    sleep 5
done

echo "✅ Kafka is ready!"

# Check if topic exists
TOPIC_NAME="water-quality-data"
echo "🔍 Checking if topic '$TOPIC_NAME' exists..."

if kafka-topics.sh --bootstrap-server kafka:9092 --list | grep -q "^$TOPIC_NAME$"; then
    echo "✅ Topic '$TOPIC_NAME' already exists!"
    
    # Show topic details
    echo "📊 Topic details:"
    kafka-topics.sh --bootstrap-server kafka:9092 --describe --topic "$TOPIC_NAME"
else
    echo "❌ Topic '$TOPIC_NAME' does not exist. Creating..."
    
    # Create the topic
    kafka-topics.sh \
        --bootstrap-server kafka:9092 \
        --create \
        --topic "$TOPIC_NAME" \
        --partitions 3 \
        --replication-factor 1 \
        --config cleanup.policy=delete \
        --config retention.ms=604800000 \
        --config segment.ms=86400000
    
    if [ $? -eq 0 ]; then
        echo "✅ Topic '$TOPIC_NAME' created successfully!"
        
        # Show topic details
        echo "📊 Topic details:"
        kafka-topics.sh --bootstrap-server kafka:9092 --describe --topic "$TOPIC_NAME"
    else
        echo "❌ Failed to create topic '$TOPIC_NAME'"
        exit 1
    fi
fi

# List all topics
echo "📋 All available topics:"
kafka-topics.sh --bootstrap-server kafka:9092 --list

echo "🎉 Kafka topic setup completed!"
