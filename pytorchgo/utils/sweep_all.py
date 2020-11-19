from kafka import KafkaConsumer
import msgpack

consumer = KafkaConsumer('fizzbuzz',bootstrap_servers='127.0.0.1:1234', api_version=(0,10))
for msg in consumer:
     print(msg.value)
     assert isinstance(msg.value, dict)