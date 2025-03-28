from kafka import KafkaConsumer
from json import loads

KAFKA_CONSUMER_GROUP_NAME_CONS = "wqi_consumer_group"
KAFKA_TOPIC_NAME_CONS = "water-quality-data"
KAFKA_BOOTSTRAP_SERVERS_CONS = 'kafka:9092'

def kafka_consumer_task(**kwargs):
    print("Kafka Consumer Application Started ... ")
    try:
        consumer = KafkaConsumer(
            KAFKA_TOPIC_NAME_CONS,
            bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS_CONS,
            auto_offset_reset='latest',
            enable_auto_commit=True,
            group_id=KAFKA_CONSUMER_GROUP_NAME_CONS,
            value_deserializer=lambda x: loads(x.decode('utf-8'))
        )
        
        # Pull the data that was pushed from the producer task
        data = kwargs['ti'].xcom_pull(task_ids='kafka_producer_task', key='kafka_data')
        print(f"Consuming data passed from producer: {data}")
        
        # Consume messages continuously and process
        for message in consumer:
            print("Key: ", message.key)
            message_value = message.value
            print("Message received: ", message_value)

    except Exception as ex:
        print("Failed to read kafka message.")
        print(ex)

    print("Kafka Consumer Application Completed.")
