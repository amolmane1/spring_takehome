# Wealth Potential Estimator

This project provides a FastAPI-based REST service that estimates a person’s net worth based on facial similarity to a reference set of individuals.

## Architecture Decisions

Dockerized Build & Bake-in Data:

I've used docker for reliable reproducibility.
I have pre generated the reference_data.json (it takes a long time to run) so that you don't have to.

Folder Layout:

spring_takehome/ ← project root
├── app/ ← application code (FastAPI + data generator)
│ ├── main.py ← FastAPI endpoints and model loading
│ └── generate_data.py ← LFW fetch + embedding + net-worth sampling
├── data/ ← baked reference_data.json lives here
├── Dockerfile
├── requirements.txt
└── README.md

FastAPI:

- Chosen for simplicity and performance in serving a lightweight REST API.

Face Processing & Embedding:

- MTCNN (from facenet-pytorch) for face detection and cropping to a standard 160×160 RGB tensor.
- InceptionResnetV1 pretrained on VGGFace2 to generate 512-dimensional, L2-normalized embeddings.

Similarity & Estimation Logic:

- Cosine similarity between input embedding and all reference embeddings.
- Top-3 most similar individuals retrieved; their net worths are averaged using the similarity scores as weights.

Caching:

- @lru_cache on settings loader, and reference-data loader to avoid repeated I/O and model re-instantiation per request.

## How to Run / Deploy

Prerequisites

Docker installed (or Python 3.8+ if running locally).

1. Build & Generate Data in Docker

Build the image:

docker build -t wealth-estimator .

2. Run the Service

docker run -d --name wealth-api \
 -p 80:80 \
 -p 8888:8888 \
 wealth-estimator

FastAPI is available at http://<host>:80/predict

JupyterLab (no token) is available at http://<host>:8888

3. Test the Endpoint

curl -X POST "http://localhost/predict" \
 -F "file=@/path/to/your_image.jpg" \
 -H "Accept: application/json"

## Assumptions Made

- Dataset: Uses sklearn’s LFW (Labeled Faces in the Wild) dataset as a stand-in for real customer data.
- Net Worth Sampling: Simulated via a normal distribution (mean=500,000, sd=50,000). A high net worth individual is someone whose net worth is in this vicinity.
