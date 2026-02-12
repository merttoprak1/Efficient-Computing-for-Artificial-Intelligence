import time
import json
import paho.mqtt.client as mqtt
import adafruit_dht
import board
import uuid

BROKER_ADDRESS = "broker.emqx.io"
PORT = 1883
TOPIC = "s336966_s338142_s343124"  
              
DHT_PIN = board.D4                 

# Initialize the sensor 
dht_device = adafruit_dht.DHT11(DHT_PIN)

# Device mac address retrieval
def get_mac_address():
    mac_num = uuid.getnode()
    mac = ':'.join(['{:02x}'.format((mac_num >> elements) & 0xff) 
                    for elements in range(0, 2*6, 2)][::-1])
    return mac

 # reading temperature and humidity
def read_sensor():
    try:
        temperature = dht_device.temperature
        humidity = dht_device.humidity
        return humidity, temperature
    except RuntimeError as error:
        return None, None
    except Exception as error:
        dht_device.exit()
        raise error

def publish_data():
    client = mqtt.Client()
    client.connect(BROKER_ADDRESS, PORT)
    
    mac_address = get_mac_address()
    
    print(f"Starting publisher on topic: {TOPIC}...")
    
    try:
        while True:
            # Measure Data
            humidity, temperature = read_sensor()
            
            if humidity is not None and temperature is not None:
                # JSON Message
                timestamp_ms = int(time.time() * 1000)
                
                payload = {
                    "mac_address": mac_address,
                    "timestamp": timestamp_ms,
                    "data": [
                        {"name": "temperature", "value": int(temperature)},
                        {"name": "humidity", "value": int(humidity)}
                    ]
                }
                
                payload_json = json.dumps(payload)
                
                # Publish the message
                client.publish(TOPIC, payload_json)
                print(f"Published: {payload_json}")
            else:
                # wait for the next loop
                pass 

            time.sleep(5)
            
    except KeyboardInterrupt:
        print("Stopping publisher...")
        client.disconnect()
        dht_device.exit()

if __name__ == "__main__":
    publish_data()