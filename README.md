# OpenCADS: Chip Anomaly Detection System

<div align="center">

[![GitHub stars](https://img.shields.io/github/stars/jingfrank/OpenCADS?style=social)](https://github.com/jingfrank/OpenCADS)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/Python-3.9%20%7C%203.10%20%7C%203.11%20%7C%203.13-blue)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100%2B-009688.svg?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C.svg?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![React](https://img.shields.io/badge/React-18-61DAFB.svg?logo=react&logoColor=black)](https://react.dev/)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED.svg?logo=docker&logoColor=white)](https://www.docker.com/)

[**简体中文**](./README_CN.md) | [**English**](./README.md) | [**Architecture Design**](./docs/DESIGN.md) | [**API Docs**](./docs/API_DOCS.md) | [**Benchmark**](./docs/BENCHMARK.md)

</div>

---

## 📌 Overview

**OpenCADS (Chip Anomaly Detection System)** is a full-stack, end-to-end industrial visual anomaly detection platform powered by the **PatchCore** unsupervised algorithm. 

In industrial semiconductor and PCB manufacturing, defective samples (negative cases) are extremely rare and costly to collect. OpenCADS resolves this pain point by **training exclusively on defect-free (normal) images**. During inspection, it accurately predicts anomaly scores, makes threshold-based judgments, and generates fine-grained pixel-level anomaly heatmaps and binary masks.

---

## ✨ Key Features

- **Unsupervised Anomaly Detection (PatchCore)**
  - Utilizes a pretrained WideResNet-50 backbone extracting intermediate representations from layers 2 & 3.
  - Generates rich multiscale local neighborhood patch features (1024-dim).
- **Extreme Memory & Inference Acceleration**
  - **Coreset Subsampling**: Preserves distribution coverage while reducing memory bank size by ~90%.
  - **FAISS Nearest Neighbor Index**: GPU/CPU accelerated K-NN scoring for sub-second defect query.
- **Dynamic Adaptive Thresholding**
  - Calibrates detection thresholds at training time using statistical distributions: $\text{Threshold} = \mu_{\text{normal}} + 3\sigma_{\text{normal}}$.
- **Full-Stack Industrial UI & Inspection Console**
  - **Datasets Management**: Upload zipped images, browse samples, and manage classification tags.
  - **Async Training Pipeline**: Non-blocking training engine with stage tracking and real-time progress feedback.
  - **Model Hot-Swapping**: Instant zero-downtime model switching and metadata inspection.
  - **Real-Time Visualizer**: Dual-view visualization comparing raw defect images, binary masks, and Jet colormap overlay heatmaps.
- **Production-Ready & Cloud-Native Deployment**
  - **Out-of-the-Box**: Single-command startup with embedded SQLite mode via Docker Compose.
  - **Enterprise Scalability**: Pluggable PostgreSQL + Redis + Celery distributed worker support for batch workloads.

---

## 📊 Performance Benchmark

Benchmarked on single-image end-to-end inference (256×256, WideResNet-50 backbone, 10% Coreset):

| Computing Platform | Model Inference Time (ms) | End-to-End Processing Time (ms) | Inference Speedup vs CPU |
| :--- | :---: | :---: | :---: |
| **Edge CPU (x86)** | 801.53 ms | 857.28 ms | **1.0x (Baseline)** |
| **NVIDIA Tesla T4** | 151.70 ms | 241.07 ms | **5.3x** |
| **NVIDIA Tesla V100** | **29.71 ms** | **157.38 ms** | **27.0x** |

> *Note: End-to-End processing includes image preprocessing, network forward pass, FAISS distance query, Gaussian blur smoothing, and base64 heatmap synthesis.*

---

## 🏗️ System Architecture

```
┌────────────────────────────────────────────────────────────────────────┐
│                        Industrial Web Console (React 18)               │
│         [Dataset Manager]   [Training Dashboard]   [Inference Studio]  │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │ HTTP / REST API (Port 3000 -> 8000)
┌───────────────────────────────────▼────────────────────────────────────┐
│                       FastAPI Application Gateway                      │
│      ├── /api/datasets        ├── /api/models                          │
│      ├── /api/train           └── /api/predict (Inference Engine)      │
└───────────────┬───────────────────────────────────┬────────────────────┘
                │                                   │
┌───────────────▼────────────────┐ ┌────────────────▼────────────────────┐
│      SQLModel Persistence      │ │      PatchCore Algorithm Engine     │
│  SQLite (Dev) / Postgres (Prod)│ │  - WideResNet-50 Feature Extractor  │
│  - Training Job States         │ │  - Patch Extractor & Aggregator     │
│  - Inspection History & Labels │ │  - Coreset Subsampler               │
│                                │ │  - FAISS Vector Memory Bank Index   │
└────────────────────────────────┘ └─────────────────────────────────────┘
```

---

## 🚀 Quick Start

### 1. Run with Docker Compose (Recommended)

Make sure you have [Docker](https://www.docker.com/) and [Docker Compose](https://docs.docker.com/compose/) installed.

```bash
# Clone the repository
git clone https://github.com/jingfrank/OpenCADS.git
cd OpenCADS

# Start all services with one command
docker-compose up -d --build
```

Access the web console:
- **Frontend Inspection Web UI**: [http://localhost:3000](http://localhost:3000)
- **Backend Swagger API Docs**: [http://localhost:8000/docs](http://localhost:8000/docs)

To stop services:
```bash
docker-compose down
```

---

### 2. Local Manual Setup

#### Backend Setup
```bash
cd backend
python -m venv venv

# On Linux/macOS:
source venv/bin/activate
# On Windows:
venv\Scripts\activate

pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

#### Frontend Setup
```bash
cd frontend
npm install
npm run dev
# Frontend dev server will be running on http://localhost:3000
```

---

## 🧪 Testing

Run backend automated unit tests and API integration suite:

```bash
cd backend
pytest tests/ -v
```

---

## 📂 Project Structure

```
OpenCADS/
├── backend/
│   ├── app/                      # DDD Architecture
│   │   ├── core/                 # DB, Celery & Config
│   │   └── domain/               # Datasets, Models, Training, Detection
│   ├── engine/                   # PatchCore Algorithm Core
│   │   ├── feature_extractor.py  # WideResNet Backbone
│   │   ├── sampler.py            # Coreset Subsampling
│   │   ├── memorybank.py         # FAISS Index Wrapper
│   │   ├── trainer.py            # Offline Training & Thresholding
│   │   └── inference.py          # Online Inference & Heatmap Rendering
│   ├── datasets/                 # Training and inspection datasets
│   ├── tests/                    # Pytest test suite
│   ├── Dockerfile
│   └── requirements.txt
├── frontend/
│   ├── src/
│   │   ├── components/           # UI Components (Layout, Dashboard, etc.)
│   │   └── pages/                # Datasets, Training, Models, Detection
│   ├── package.json
│   └── Dockerfile
├── docs/                         # Specifications & Benchmarks
│   ├── DESIGN.md                 # Software Architecture & Flow
│   ├── USER_GUIDE.md             # Operations Manual
│   ├── API_DOCS.md               # RESTful API Specifications
│   └── BENCHMARK.md              # Inference Hardware Benchmark Report
├── docker-compose.yml            # Default lightweight compose
├── docker-compose.prod.yml       # Production distributed compose (Postgres+Redis+Celery)
├── LICENSE                       # MIT License
└── README.md
```

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).
