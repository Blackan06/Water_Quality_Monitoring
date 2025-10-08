#!/bin/bash

# Kafka startup script with automatic topic creation
set -e

echo "🚀 Starting Kafka with automatic topic setup..."

# Start Kafka in background using the default command
echo "⏳ Starting Kafka server..."
/opt/kafka/bin/kafka-server-start.sh /opt/kafka/config/kraft/server.properties &
KAFKA_PID=$!

# Wait for Kafka to be ready
echo "⏳ Waiting for Kafka to be ready..."
sleep 45

# Check if Kafka is ready
echo "🔍 Checking Kafka readiness..."
for i in {1..30}; do
    if kafka-topics.sh --bootstrap-server localhost:9092 --list > /dev/null 2>&1; then
        echo "✅ Kafka is ready!"
        break
    else
        echo "⏳ Kafka not ready yet, waiting... (attempt $i/30)"
        sleep 5
    fi
    
    if [ $i -eq 30 ]; then
        echo "❌ Kafka failed to start after 150 seconds"
        kill $KAFKA_PID 2>/dev/null || true
        exit 1
    fi
done

# Run topic setup script
echo "🔧 Running topic setup..."
if [ -f "/setup_kafka_topic.sh" ]; then
    chmod +x /setup_kafka_topic.sh
    /setup_kafka_topic.sh
    echo "✅ Topic setup completed!"
else
    echo "⚠️ Topic setup script not found, skipping..."
fi

# Keep Kafka running
echo "🎉 Kafka is running with topics configured!"
wait $KAFKA_PID
