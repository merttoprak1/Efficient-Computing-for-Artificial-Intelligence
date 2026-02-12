import argparse  
import time      
import threading 
import queue     
import string    
import board                 # Raspberry Pi GPIO pins
import adafruit_dht          # DHT temperature/humidity sensor
import redis                
import sounddevice as sd     # For capturing audio from the microphone
import torch                 
import torchaudio            # (resampling)
import numpy as np           # (sounddevice gives NumPy arrays)
from transformers import WhisperProcessor, WhisperForConditionalGeneration 

# Start with data collection off.
# The VUI thread controls (sets to True/False)
is_collecting = False

# False when it catches a ctrl+C.
keep_running = True

# Producer: audio_callback 
# Consumer: vui_loop 
audio_queue = queue.Queue()

# Setup our audio recording settings 
AUDIO_CHANNELS = 1                               # Mono audio
AUDIO_BIT_DEPTH = 'int16'                        # 16-bit audio
AUDIO_SAMPLING_RATE = 48000                      # 48kHz 
AUDIO_WINDOW_SEC = 1                             # 1-second chunks of audio at a time
WHISPER_SAMPLING_RATE = 16000                    # Whisper 16kHz audio

# Redis TimeSeries 
MAC_ADDRESS = "e45f01e89915"

TEMP_KEY = f"{MAC_ADDRESS}:temperature"
HUMID_KEY = f"{MAC_ADDRESS}:humidity"


# Audio Callback

def audio_callback(indata, frames, time, status):
    if status:
        # Print any audio-related errors 
        print(f"[Audio] Status: {status}")
    
    # We must put a copy of the data. 'indata' is a buffer that sounddevice will reuse. If we just put 'indata' in the queue, it might be
    # overwritten with new audio before the VUI thread can process it.
    audio_queue.put(indata.copy())

#  VUI Processing Loop 

def vui_loop(processor, model, resampler):
    
    # Runs in a continuous loop, waiting for audio to appear in the 
    #"audio_queue", processing it through Whisper, and updating the global
    #'is_collecting' flag if a command is heard
    
    # Need to declare 'global' because we are modifying these variables, not just reading them.
    global is_collecting, keep_running
    
    print("[VUI] VUI thread started. Listening for 'up' or 'stop'...")

    while keep_running:
        try:
            # Wait for the next 1-second audio chunk to arrive from the callback.If it blocks, it can't check 'keep_running', so it would
            # never shut down.
            audio_int16 = audio_queue.get(timeout=1.0)
        
        except queue.Empty:
            #  No audio has arrived in the last second
            continue 

        
        # Convert from NumPy array (int16) to a PyTorch tensor (float32).
        # ML models always work with floating-point numbers.
        audio_tensor = torch.from_numpy(audio_int16).to(torch.float32)

        # Change layout from (samples, channels) to (channels, samples).
        # Sounddevice gives (48000, 1), but PyTorch audio needs (1, 48000).
        audio_tensor = audio_tensor.transpose(0, 1)

        # Normalize the audio. 'int16' audio goes from -32768 to +32767.
        # Nrmalize it to be in the range -1.0 to 1.0.
        audio_tensor = audio_tensor / 32768.0

        # Downsample from 48kHz to the 16kHz that Whisper requires.
        audio_16k = resampler(audio_tensor)

        # Get rid of the channel dimension.
        # We now have (1, 16000).
        # Removes the dimension at index 0.
        audio_16k = audio_16k.squeeze(0)

        # Feed the clean audio tensor into the Whisper processor.
        input_features = processor(
            audio_16k, 
            sampling_rate=WHISPER_SAMPLING_RATE, 
            return_tensors="pt"  # Return PyTorch tensors
        ).input_features
        
        # Run the actual voice-to-text model (inference).
        # Generates a list of "token IDs" representing the predicted text.
        predicted_ids = model.generate(
            input_features,
            forced_decoder_ids=None, # Not forcing any specific output
            suppress_tokens=[]       # Not suppressing any words
        )

        # Decode the model's output (token IDs) into a human-readable text string.
        transcription = processor.batch_decode(
            predicted_ids, 
            skip_special_tokens=True # Don't include tokens like "startoftext"
        )[0] # Get the first 

        # Clean up the text, "Stop." or "  stop  "
        command = transcription.translate(
            str.maketrans('', '', string.punctuation)
        ).replace(" ", "").lower()

    
        if command == "up":
            if not is_collecting:
                is_collecting = True
                print("\n[VUI] 'up' detected. === DATA COLLECTION ENABLED ===\n")
        
        elif command == "stop":
            # Only flip the switch if we're currently running.
            if is_collecting:
                is_collecting = False
                print("\n[VUI] 'stop' detected. === DATA COLLECTION DISABLED ===\n")
        
# Sensor & Data Collection
def sensor_loop(redis_client, dht_sensor):

    # If it's True, it reads the sensor every 5 seconds and uploads to Redis
    # If it's False, it does nothing and sleeps
    
    print("[Sensor] Sensor thread started.")
    
    while keep_running:
        if is_collecting:
            # Data Collection and Upload
            try:
                # Measure temperature and humidity
                temperature = dht_sensor.temperature
                humidity = dht_sensor.humidity

                # Check if the read was successful
                if temperature is not None and humidity is not None:
                    
                    # Get current timestamp in milliseconds
                    timestamp_ms = int(time.time() * 1000)

                    # Send data to Redis TimeSeries :
                    redis_client.ts().add(TEMP_KEY, timestamp_ms, temperature)
                    redis_client.ts().add(HUMID_KEY, timestamp_ms, humidity)
                    
                    print(f"[Sensor] Logged to Redis TS: T={temperature}°C, H={humidity}%")
                
                else:
                    print("[Sensor] Failed to read from DHT-11. (This is common, will retry)")

            except RuntimeError as e:
                # DHT sensors  runtime errors
                print(f"[Sensor] Read error: {e}")
            
            except redis.RedisError as e:
                # Lost connection
                print(f"[Sensor] Redis error: {e}")
            
            # Wait 5 seconds before the next reading
            time.sleep(5)
        
        else:
            # Data collection is disabled.
            time.sleep(1)


def main():
    global is_collecting, keep_running
    
    # Get the Redis credentials
    parser = argparse.ArgumentParser(description="Smart Hygrometer System")
    parser.add_argument('--host', type=str, required=True, help="Redis Cloud host")
    parser.add_argument('--port', type=int, required=True, help="Redis Cloud port")
    parser.add_argument('--user', type=str, required=True, help="Redis Cloud username")
    parser.add_argument('--password', type=str, required=True, help="Redis Cloud password")
    args = parser.parse_args()

    #Connect to Redis
    print("\nConnecting to Redis \n")
    try:
        r = redis.Redis(
            host=args.host,
            port=args.port,
            username=args.user,
            password=args.password,
            decode_responses=True # Get strings back from Redis
        )
        # Test the connection
        r.ping()
        print("Redis connection successful.")

        # Ensure TimeSeries keys exist
        try:
            # Create the temperature key
            r.ts().create(TEMP_KEY, duplicate_policy='last')
            # Create the humidity key
            r.ts().create(HUMID_KEY, duplicate_policy='last')
            print("Redis TimeSeries keys ensured.")
        except redis.ResponseError:
            print("Redis TimeSeries keys already exist.")

    except redis.RedisError as e:
        print(f"ERROR: Could not connect to Redis: {e}")
        return 

    # Initialize DHT-11 Sensor
    print("Initializing DHT-11 sensor (GPIO 4)...")
    try:
        # Tell the library the sensor is connected to GPIO Pin 4
        dht_sensor = adafruit_dht.DHT11(board.D4)
        # Give it a quick test read to make sure it's working
        dht_sensor.temperature 
        print("DHT-11 sensor found.")
    except Exception as e:
        print(f"ERROR: Could not initialize DHT-11: {e}")
        return 

    # Load the Whisper model
    print("Loading Whisper 'tiny.en' model... (this might take a moment)")
    try:
        # to recognize commands ("up", "stop").
        model_name = 'openai/whisper-tiny.en'
        processor = WhisperProcessor.from_pretrained(model_name)
        model = WhisperForConditionalGeneration.from_pretrained(model_name)
        
        # Initialize the resampler object to convert 48kHz -> 16kHz
        resampler = torchaudio.transforms.Resample(
            orig_freq=AUDIO_SAMPLING_RATE, 
            new_freq=WHISPER_SAMPLING_RATE
        )
        print("Whisper model loaded.")
    except Exception as e:
        print(f"ERROR: Could not load Whisper model: {e}")
        return
    
    print("Starting worker threads...")
    # Create the sensor thread
    sensor_thread = threading.Thread(
        target=sensor_loop, 
        args=(r, dht_sensor)
    )
    # Create the VUI thread
    vui_thread = threading.Thread(
        target=vui_loop, 
        args=(processor, model, resampler)
    )
    
    # Start both threads:
    sensor_thread.start()
    vui_thread.start()

    # Start Audio Stream::
    
    # Calculate how many audio frames are in the 1 second window
    block_size = int(AUDIO_SAMPLING_RATE * AUDIO_WINDOW_SEC) # 48000 * 1 = 48000
    
    # Stream object:
    stream = sd.InputStream(
        samplerate=AUDIO_SAMPLING_RATE, # 48kHz
        channels=AUDIO_CHANNELS,        # 1 (mono)
        dtype=AUDIO_BIT_DEPTH,          # int16
        blocksize=block_size,           # 1-second chunks
        callback=audio_callback         # This is the function to call
    )
    
    print(f"\nSystem running. Initial state: DISABLED.")
    print("Say 'up' to enable, 'stop' to disable.")
    print("Press Ctrl+C to quit.\n")
    
    with stream:
        try:
            while keep_running:
                time.sleep(1)
        except KeyboardInterrupt:
            # User pressed ctrl+c
            print("\nCtrl+C detected. Shutting down...")
            keep_running = False
            
    print("Waiting for threads to close...")
    sensor_thread.join()
    vui_thread.join()
    
    dht_sensor.exit()
    r.close()
    print("Shutdown complete.")


if __name__ == "__main__":
    main()


# Command line to run the code
    """
    python3 /home/ecai/workdir/hygrometer.py \
    --host redis-18331.c55.eu-central-1-1.ec2.cloud.redislabs.com \
    --port 18331 \
    --user ... \
    --password "..."
    """


