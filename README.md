# Efficient Computing for Artificial Intelligence

Homework assignments for the **Efficient Computing for Artificial Intelligence** course. Each homework builds an end-to-end IoT system on a **Raspberry Pi**, progressively introducing edge AI, model optimization, and cloud communication patterns.

---

## 📁 Repository Structure

```
├── HW1/                  # Voice-Controlled Smart Hygrometer (Whisper)
├── HW1-Assignment.pdf
├── HW2/                  # Optimized KWS Hygrometer (ONNX on-device)
├── HW2-Assignment.pdf
├── HW3/                  # MQTT Pub/Sub & REST API Data Pipeline
├── HW3-Assignment.pdf
└── README.md
```

---

## HW1 — Voice-Controlled Smart Hygrometer

A voice-activated environmental monitoring system running on a Raspberry Pi. A **DHT-11** sensor measures temperature and humidity, while an **OpenAI Whisper** speech-to-text model listens for voice commands to start and stop data collection.

### Key Components

| File | Description |
|------|-------------|
| `hygrometer.py` | Main application — multi-threaded system with audio capture, Whisper inference, sensor reading, and Redis upload |
| `msc_dataset.py` | Custom PyTorch `Dataset` class for loading the Mini Speech Commands dataset |
| `report.pdf` | Detailed assignment report |

### Architecture

- **Voice UI Thread** — Captures 1-second audio chunks at 48 kHz via USB microphone, resamples to 16 kHz, and runs Whisper `tiny.en` inference to detect `"up"` (start) and `"stop"` (halt) commands.
- **Sensor Thread** — Reads DHT-11 temperature/humidity every 5 seconds (when enabled) and pushes timestamped data to **Redis TimeSeries** on Redis Cloud.
- **Technologies:** Python, PyTorch, Hugging Face Transformers, torchaudio, sounddevice, Redis TimeSeries, Adafruit DHT

---

## HW2 — Optimized Keyword Spotting Hygrometer

An evolution of HW1 that replaces the heavy Whisper model with a **lightweight, quantized ONNX model** for on-device keyword spotting (KWS), dramatically reducing latency and resource usage.

### Key Components

| File | Description |
|------|-------------|
| `training.ipynb` | End-to-end training pipeline — DSCNN model definition, MelSpectrogram feature extraction, training loop, ONNX export, and static quantization |
| `hygrometer.py` | Optimized application using ONNX Runtime for inference instead of Whisper |
| `Group1_frontend.onnx` | Quantized feature extraction model (MelSpectrogram) |
| `Group1_model.onnx` | Quantized DSCNN classification model |
| `report.pdf` | Detailed assignment report |

### Architecture

- **Model:** Depthwise Separable CNN (DSCNN) trained on Mini Speech Commands to classify `"up"` and `"stop"` keywords.
- **Optimization Pipeline:** PyTorch → ONNX export → Static quantization (INT8) to achieve <100 KB model size with sub-5ms inference latency on Raspberry Pi.
- **Inference:** Two-stage ONNX Runtime pipeline — frontend (feature extraction) → backend (classification) with a 99.9% confidence threshold to prevent false positives.
- **Technologies:** Python, PyTorch, torchaudio, ONNX Runtime, SciPy, Redis TimeSeries, Adafruit DHT

---

## HW3 — MQTT Pub/Sub & REST API Data Pipeline

A distributed IoT data pipeline using **MQTT** for real-time sensor data streaming and a **REST API** for historical data retrieval and visualization.

### Key Components

| File | Description |
|------|-------------|
| `publisher.py` | Raspberry Pi script — reads DHT-11 sensor data and publishes JSON payloads to an MQTT broker every 5 seconds |
| `subscriber.ipynb` | MQTT subscriber — listens for sensor messages and stores them in Redis TimeSeries |
| `rest_server.ipynb` | CherryPy REST API server — exposes endpoints for health checks (`/status`) and historical data retrieval (`/data/{mac_address}?count=N`) |
| `rest_client.ipynb` | API consumer — fetches historical data via REST, displays it in a DataFrame, and generates temperature/humidity plots |
| `report.pdf` | Detailed assignment report |

### Architecture

```
┌──────────────┐    MQTT     ┌──────────────┐            ┌───────────────┐
│  Raspberry   │ ──────────► │  Subscriber  │ ─────────► │  Redis Cloud  │
│  Pi + DHT-11 │  (broker.   │  (Deepnote)  │  TimeSeries│  (TimeSeries) │
│  publisher   │   emqx.io)  └──────────────┘            └───────┬───────┘
└──────────────┘                                                 │
                                                                 │
                              ┌──────────────┐                   │
                              │  REST Server │ ◄─────────────────┘
                              │  (CherryPy)  │
                              └──────┬───────┘
                                     │ HTTP
                              ┌──────▼───────┐
                              │  REST Client │
                              │  (Pandas +   │
                              │  Matplotlib) │
                              └──────────────┘
```

- **Technologies:** Python, Paho MQTT, CherryPy, Redis TimeSeries, Pandas, Matplotlib, Adafruit DHT

---

## 🛠️ Hardware Requirements

- Raspberry Pi (tested on RPi 4)
- DHT-11 temperature & humidity sensor (GPIO Pin 4)
- USB microphone (HW1 & HW2)

## ☁️ Cloud Services

- **Redis Cloud** — TimeSeries database for storing sensor readings
- **EMQX Public Broker** — MQTT message broker (`broker.emqx.io:1883`) for HW3
- **Deepnote** — Cloud notebook environment used for subscriber, server, and client notebooks
