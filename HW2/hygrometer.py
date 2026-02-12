import argparse
import time
import threading
import queue
import sys

import board
import adafruit_dht
import redis
import sounddevice as sd

import torch
import torchaudio
import numpy as np
import onnxruntime as ort       
from scipy.special import softmax 


# Audio Recording Parameters
CHANNELS = 1
BIT_DEPTH = 'int16'
MIC_SAMPLE_RATE = 48000
RECORDING_DURATION = 1  # Seconds
MODEL_SAMPLE_RATE = 16000 # The KWS model expects 16kHz audio

# Model Paths
PATH_FRONTEND = "Group1_frontend.onnx"
PATH_BACKEND = "Group1_model.onnx"

# Class Labels: 0 = "stop", 1 = "up"
COMMAND_LABELS = ["stop", "up"]

# Redis Configuration
DEVICE_ID = "e45f01e89915"
KEY_TEMPERATURE = f"{DEVICE_ID}:temperature"
KEY_HUMIDITY = f"{DEVICE_ID}:humidity"

# Global State Variables:
monitoring_enabled = False  # Data collection starts in the disabled state
system_active = True
microphone_buffer = queue.Queue()


def find_usb_microphone():
    # Available audio devices are scanned to locate a USB microphone.
    # The index is returned if found, otherwise None is returned.
    available_devices = sd.query_devices()
    for index, device in enumerate(available_devices):
        # A device is considered valid if 'USB' is in the name and it has input channels
        if 'USB' in device['name'] and device['max_input_channels'] > 0:
            print(f"[Audio Setup] USB Microphone detected: {device['name']} at Index {index}")
            return index
            
    print("[Audio Setup] Warning: No USB microphone found. The default system device will be used.")
    return None


def audio_capture_callback(incoming_data, frames, time_info, status):
    # Callback function triggered by the audio stream.
    # Incoming 48kHz audio data is copied and placed into the processing queue.
    if status:
        print(f"[Audio Stream] Status Warning: {status}")
    
    # A copy of the raw audio buffer is added to the thread-safe queue
    microphone_buffer.put(incoming_data.copy())


def process_voice_commands(frontend_session, backend_session, audio_resampler):
    # Audio is continuously retrieved from the buffer and processed through the ONNX models.
    # The system state is toggled based on recognized commands.
    
    global monitoring_enabled, system_active
    
    # A high confidence threshold is enforced to prevent false positives
    CONFIDENCE_THRESHOLD = 0.999
    
    print(f"[Voice Control] Listening for commands: {COMMAND_LABELS} (Required Confidence > {CONFIDENCE_THRESHOLD:.1%})...")

    while system_active:
        try:
            # The loop waits up to 1 second for new audio data to arrive
            raw_audio = microphone_buffer.get(timeout=1.0)
        except queue.Empty:
            continue

        # Pre-processing Pipeline 
        
        # 1. The raw integer array is converted into a PyTorch floating-point tensor
        audio_tensor = torch.from_numpy(raw_audio).to(torch.float32)

        # 2. Dimensions are transposed: (samples, channels) -> (channels, samples)
        audio_tensor = audio_tensor.transpose(0, 1)

        # 3. Normalization is applied to scale values between -1 and 1
        audio_tensor = audio_tensor / 32768.0 
        
        # 4. The audio is downsampled from 48kHz to 16kHz
        resampled_audio = audio_resampler(audio_tensor)

        # 5. A batch dimension is added to create the shape (1, 1, 16000) for the model
        model_input = resampled_audio.numpy()[None, :]

        # Inference Pipeline:
        try:
            # Feature Extraction (Frontend Model)
            input_name_fe = frontend_session.get_inputs()[0].name
            features = frontend_session.run(None, {input_name_fe: model_input})[0]
            
            # Classification (Backend Model)
            input_name_be = backend_session.get_inputs()[0].name
            logits = backend_session.run(None, {input_name_be: features})[0]

            # Probabilities are calculated using Softmax
            probabilities = softmax(logits[0])
            
            # The highest probability and its corresponding index are identified
            highest_prob = np.max(probabilities)     
            predicted_index = np.argmax(probabilities)  

            # The index is mapped to a human-readable label
            if predicted_index < len(COMMAND_LABELS):
                detected_command = COMMAND_LABELS[predicted_index]
            else:
                detected_command = "unknown"

            # Control Logic: 
            if highest_prob > CONFIDENCE_THRESHOLD:
                
                # Command: "up" -> Start recording environmental data
                if detected_command == "up":
                    if not monitoring_enabled:
                        monitoring_enabled = True
                        print(f" [Voice Control] Command 'UP' recognized ({highest_prob:.5f}).")
                        print(" [System] Data collection has been ENABLED.")

                # Command: "stop" -> Stop recording environmental data
                elif detected_command == "stop":
                    if monitoring_enabled:
                        monitoring_enabled = False
                        print(f" [Voice Control] Command 'STOP' recognized ({highest_prob:.5f}).")
                        print(" [System] Data collection has been DISABLED.")

            # If confidence is low, the state remains unchanged

        except Exception as error:
            print(f"[Voice Control] Error during inference: {error}")


def monitor_environment(redis_connection, dht_device):
    # Background thread that reads sensor data and uploads it to Redis
    # only when 'monitoring_enabled' is set to True.
    
    print("[Sensor Monitor] Background thread initialized.")
    
    while system_active:
        if monitoring_enabled:
            try:
                # The temperature and humidity are read from the sensor
                current_temp = dht_device.temperature
                current_humid = dht_device.humidity

                if current_temp is not None and current_humid is not None:
                    # Current time is converted to milliseconds
                    timestamp = int(time.time() * 1000)

                    # Data points are added to the Redis TimeSeries
                    redis_connection.ts().add(KEY_TEMPERATURE, timestamp, current_temp)
                    redis_connection.ts().add(KEY_HUMIDITY, timestamp, current_humid)
                    
                    print(f"[Sensor Monitor] Data Uploaded: Temp={current_temp}°C, Humidity={current_humid}%")
                else:
                    print("[Sensor Monitor] Reading failed. Retrying...")

            except RuntimeError as error:
                # Common DHT11 timing errors are caught here
                print(f"[Sensor Monitor] Sensor Read Error: {error}")
            except redis.RedisError as error:
                print(f"[Sensor Monitor] Database Error: {error}")
            
            # A 5-second delay is applied between readings
            time.sleep(5)
        else:
            # Check the state again in 1 second
            time.sleep(1)


def main():
    global system_active

    # Command line arguments are defined and parsed
    parser = argparse.ArgumentParser(description="Voice-Controlled Smart Hygrometer System")
    parser.add_argument('--host', type=str, required=True, help="Redis Cloud host address")
    parser.add_argument('--port', type=int, required=True, help="Redis Cloud port number")
    parser.add_argument('--user', type=str, required=True, help="Redis Cloud username")
    parser.add_argument('--password', type=str, required=True, help="Redis Cloud password")
    args = parser.parse_args()

    print("Establishing connection to Redis...")
    try:
        redis_client = redis.Redis(
            host=args.host,
            port=args.port,
            username=args.user,
            password=args.password,
            decode_responses=True
        )
        # TimeSeries keys are created if they do not typically exist
        try:
            redis_client.ts().create(KEY_TEMPERATURE)
            redis_client.ts().create(KEY_HUMIDITY)
        except redis.ResponseError:
            # Keys likely already exist, so this error is ignored
            pass 

    except redis.RedisError as error:
        print(f"FATAL: Unable to connect to Redis: {error}")
        return

    print("Initializing DHT-11 sensor on GPIO Pin 4...")
    try:
        sensor_device = adafruit_dht.DHT11(board.D4)
    except Exception as error:
        print(f"FATAL: Sensor initialization failed: {error}")
        return

    print(f"Loading ONNX models from disk ({PATH_FRONTEND}, {PATH_BACKEND})...")
    try:
        # Inference sessions are created for both parts of the model
        session_frontend = ort.InferenceSession(PATH_FRONTEND)
        session_backend = ort.InferenceSession(PATH_BACKEND)
        
        # A resampler is configured to convert mic input to model requirements
        audio_resampler = torchaudio.transforms.Resample(
            orig_freq=MIC_SAMPLE_RATE, 
            new_freq=MODEL_SAMPLE_RATE
        )
        print("Models loaded successfully.")
    except Exception as error:
        print(f"FATAL: Failed to load ONNX models: {error}")
        return

    # Background threads are instantiated
    env_thread = threading.Thread(target=monitor_environment, args=(redis_client, sensor_device))
    voice_thread = threading.Thread(target=process_voice_commands, args=(session_frontend, session_backend, audio_resampler))
    
    # Threads are started
    env_thread.start()
    voice_thread.start()

    # Audio input configuration
    # Block size represents one second of audio data
    buffer_block_size = int(MIC_SAMPLE_RATE * RECORDING_DURATION)
    
    microphone_index = find_usb_microphone()
    if microphone_index is None:
        print("WARNING: Falling back to default system audio device.")
    
    audio_stream = sd.InputStream(
        device=microphone_index,
        samplerate=MIC_SAMPLE_RATE,
        channels=CHANNELS,
        dtype=BIT_DEPTH,
        blocksize=buffer_block_size,
        callback=audio_capture_callback
    )
    
    print(f"\nSystem is running. Initial State: DISABLED.")
    print(f"Say 'up' to enable collection, 'stop' to disable.")
    print("Press Ctrl+C to exit safely.\n")
    
    # The main thread keeps the audio stream active until interrupted
    with audio_stream:
        try:
            while system_active:
                time.sleep(0.5)
        except KeyboardInterrupt:
            print("\nCtrl+C detected. Initiating shutdown sequence...")
            system_active = False
            
    # Cleanup operations
    print("Waiting for background threads to terminate...")
    env_thread.join()
    voice_thread.join()
    sensor_device.exit()
    redis_client.close()
    print("Shutdown complete.")

if __name__ == "__main__":
    main()



"""


python3 /home/ecai/workdir/hygrometer_2.py \
    --host redis-11660.c293.eu-central-1-1.ec2.cloud.redislabs.com \
    --port 11660 \
    --user default \
    --password "JdndGHAyZcOlpSJkvWL4wxAPWoaOSovb"


"""




