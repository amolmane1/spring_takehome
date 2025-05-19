# Wealth Potential Estimator

This project provides a FastAPI-based REST service that estimates a person’s net worth based on facial similarity to a reference set of individuals.

## Architecture Decisions

#### Project Layout:

```text
spring_takehome/ ← project root
├── app/ ← application code
│   ├── generate_data.py ← script to build reference_data.json
│   ├── main.py ← FastAPI entrypoint
│   ├── schemas.py ← Pydantic request/response models
│   ├── services.py ← face-processing & prediction logic
│   ├── settings.py ← config via Pydantic BaseSettings
│   └── __init__.py
├── data/ ← baked-in reference data (`reference_data.json`) + sample images
├── .gitignore
├── Dockerfile
├── README.md
└── requirements.txt
```

#### Dockerized Build & Bake-in Data:

- I've used docker for reliable reproducibility.
- I have pre generated the reference_data.json (it takes a long time to run) so that you don't have to. The code to generate reference_data.json is in app/generate_data.py.

#### FastAPI:

- Chosen for simplicity and performance in serving a lightweight REST API.

#### Pydantic

- All inputs and outputs are validated via Pydantic models in `schemas.py`.

#### Error handling

- All inputs and outputs are validated via Pydantic models in `schemas.py`.

#### Face Processing & Embedding:

- MTCNN (from facenet-pytorch) for face detection and cropping to a standard 160×160 RGB tensor.
- InceptionResnetV1 pretrained on VGGFace2 to generate 512-dimensional, L2-normalized embeddings.

#### Similarity & Estimation Logic:

- Cosine similarity between input embedding and all reference embeddings.
- Top-3 most similar individuals retrieved; their net worths are averaged using the similarity scores as weights. This average is used as the estimated net worth of the person in the input picture.

#### Caching:

- `@lru_cache()` on settings loader, model loader, and reference loader to avoid repeated disk I/O and model instantiation on each request.

## How to Run / Deploy

#### Prerequisites

Docker installed (or Python 3.8+ if running locally).

#### 0. Clone repo

```
git clone https://github.com/amolmane1/spring_takehome.git
cd spring_takehome
```

#### 1. Build & Generate Data in Docker

At the root of the repo, build the image:

```
docker build -t wealth-estimator .
```

#### 2. Run the Service

```
docker run -d \
  --name wealth-dev \
  -p 80:80 \
  wealth-estimator:latest
```

3. Test the Endpoint

Test using FastAPI UI:
Go to:

```
http://localhost:80/docs#/default/predict_predict_post
```

Click on `Try it out` and upload any image (you can use the sample jpgs in the /data folder!).

Test using curl:

```
curl -X POST "http://localhost/predict" \
 -F "file=@/path/to/your_image.jpg" \
 -H "Accept: application/json"
```

## Assumptions Made

- Dataset: Uses sklearn’s LFW (Labeled Faces in the Wild) dataset as a stand-in for real customer data.
- Net Worth Sampling: Simulated via a normal distribution (mean=500,000, sd=50,000). A high net worth individual is someone whose net worth is in this vicinity.
