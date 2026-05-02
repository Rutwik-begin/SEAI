# AI-Powered Plant Disease Detection System

## Overview

This project is a containerized AI-powered plant disease detection platform that combines computer vision, GPU inference serving, and Large Language Model (LLM) augmentation.

The system detects plant diseases from uploaded leaf images using an EfficientNet-B0 deep learning model deployed through NVIDIA Triton Inference Server with ONNX Runtime acceleration. After prediction, the system uses Gemini 2.5 Flash to generate farmer-friendly disease explanations, treatment recommendations, and prevention guidance.

---

# Features

* Plant disease classification using EfficientNet-B0
* 38 plant disease categories
* GPU-accelerated inference using NVIDIA Triton Server
* ONNX Runtime deployment optimization
* Dockerized deployment architecture
* Gemini 2.5 Flash integration for AI-generated explanations
* Flask-based web interface
* Dynamic disease guidance generation
* Fault-tolerant backend handling
* Containerized multi-service orchestration using Docker Compose

---

# Dataset

Dataset Used:

* PlantVillage Dataset

Dataset Characteristics:

* 38 disease classes
* Multiple crop categories
* Healthy and diseased plant samples
* RGB leaf images

---

# Model Information

## Classification Model

| Component         | Value           |
| ----------------- | --------------- |
| Architecture      | EfficientNet-B0 |
| Framework         | PyTorch         |
| Input Size        | 224 × 224       |
| Output Classes    | 38              |
| Deployment Format | ONNX            |

---

# Model Performance

## Validation Metrics

| Metric                   | Score  |
| ------------------------ | ------ |
| Best Validation Accuracy | 95.29% |

## Test Metrics

| Metric        | Score  |
| ------------- | ------ |
| Test Accuracy | 94.71% |
| Precision     | 94.73% |
| Recall        | 94.71% |
| F1-Score      | 94.65% |

---

# System Architecture

The system consists of the following major components:

1. Flask Web Application
2. Triton Inference Server
3. ONNX Runtime Backend
4. EfficientNet-B0 ONNX Model
5. NVIDIA GPU (CUDA)
6. Gemini 2.5 Flash API
7. Docker Compose Orchestration

---

# Inference Workflow

1. User uploads plant leaf image through web interface.
2. Flask backend preprocesses image.
3. Flask sends inference request to Triton Server.
4. Triton performs GPU inference using ONNX Runtime.
5. Predicted disease label is returned to Flask.
6. Flask sends disease label to Gemini 2.5 Flash.
7. Gemini generates:

   * Disease overview
   * Causes
   * Symptoms
   * Treatment recommendations
   * Prevention guidance
8. Final result is displayed in browser.

---

# Project Structure

```text
Plant_AI/
│
├── Flask/
│   ├── app.py
│   ├── llm_client.py
│   ├── triton_client.py
│   ├── requirements.txt
│   ├── templates/
│   ├── static/
│   └── .env
│
├── deployment/
│   └── model_repository/
│       └── plant_disease_classifier/
│           ├── config.pbtxt
│           └── 1/
│               ├── model.onnx
│               └── model.onnx.data
│
├── docker/
│   ├── docker-compose.yml
│   ├── Flask.Dockerfile
│   └── .dockerignore
│
└── artifacts/
```

---

# Technologies Used

## Machine Learning

* PyTorch
* torchvision
* ONNX
* EfficientNet-B0

## Deployment

* NVIDIA Triton Inference Server
* ONNX Runtime
* CUDA
* Docker
* Docker Compose

## Backend

* Flask
* NumPy
* Pillow
* Triton HTTP Client

## Generative AI

* Gemini 2.5 Flash
* Google GenAI SDK

---

# Docker Deployment

## Prerequisites

* Docker Desktop
* NVIDIA GPU
* NVIDIA Container Toolkit
* CUDA-compatible drivers

---

# Run Application

Navigate to:

```powershell
cd docker
```

Start services:

```powershell
docker compose up --build
```

---

# Access Application

Open browser:

```text
http://localhost:5000
```

---

# Triton Endpoints

| Service        | Port |
| -------------- | ---- |
| Triton HTTP    | 8000 |
| Triton gRPC    | 8001 |
| Triton Metrics | 8002 |
| Flask App      | 5000 |

---

# Gemini API Setup

Create:

```text
Flask/.env
```

Add:

```env
GEMINI_API_KEY=your_api_key_here
```

---

# Example Output

The system predicts plant disease classes and generates AI-powered agricultural guidance including:

* Disease overview
* Causes
* Symptoms
* Treatment recommendations
* Prevention techniques

---

# Screenshots

## Web Interface

(Add screenshot here)

## Prediction Output

(Add screenshot here)

## Docker Deployment

(Add screenshot here)

## Triton Logs

(Add screenshot here)

## System Architecture Diagram

(Add diagram here)

---

# Future Improvements

* Mobile application integration
* Real-time camera detection
* Multi-language support
* Database logging
* User authentication
* Cloud deployment
* Disease severity estimation
* Recommendation personalization

---

# Conclusion

This project demonstrates a complete end-to-end AI deployment pipeline integrating:

* Deep Learning
* GPU inference serving
* ONNX optimization
* Containerized deployment
* Generative AI augmentation
* Modern MLOps architecture

The platform provides scalable and efficient plant disease diagnosis with AI-generated agricultural guidance.
