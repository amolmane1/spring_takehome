# Wealth Potential Estimator

This project provides a FastAPI-based REST service that estimates a person’s net worth based on facial similarity to a reference set of individuals.

## Architecture Decisions

#### Project Layout:

SPRING_TAKEHOME/
├── app/

│ ├── generate_data.py

│ └── main.py

├── data/

│ ├── amol.jpg

│ ├── reference_data.json

│ └── room.jpg

├── Dockerfile

├── README.md

└── requirements.txt

#### Dockerized Build & Bake-in Data:

I've used docker for reliable reproducibility.
I have pre generated the reference_data.json (it takes a long time to run) so that you don't have to. The code to generate reference_data.json is in app/generate_data.py.

#### FastAPI:

- Chosen for simplicity and performance in serving a lightweight REST API.

#### Pydantic

I used Pydantic models to validate the inputs to the endpoint

#### Error handling

I return appropriate error messages to handle commonly anticipated issues.

#### Face Processing & Embedding:

- MTCNN (from facenet-pytorch) for face detection and cropping to a standard 160×160 RGB tensor.
- InceptionResnetV1 pretrained on VGGFace2 to generate 512-dimensional, L2-normalized embeddings.

#### Similarity & Estimation Logic:

- Cosine similarity between input embedding and all reference embeddings.
- Top-3 most similar individuals retrieved; their net worths are averaged using the similarity scores as weights.

#### Caching:

- @lru_cache on settings loader, and reference-data loader to avoid repeated I/O and model re-instantiation per request.

## How to Run / Deploy

#### Prerequisites

Docker installed (or Python 3.8+ if running locally).

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

I have my own selfie (amol.jpg) and a picture of a room (room.jpg) in the data folder. You can use that to test the endpoint.

Test using FastAPI UI:
Go to:

```
http://localhost/docs#/default/predict_predict_post
```

Test using curl:

```
curl -X POST "http://localhost/predict" \
 -F "file=@/path/to/your_image.jpg" \
 -H "Accept: application/json"
```

## Assumptions Made

- Dataset: Uses sklearn’s LFW (Labeled Faces in the Wild) dataset as a stand-in for real customer data.
- Net Worth Sampling: Simulated via a normal distribution (mean=500,000, sd=50,000). A high net worth individual is someone whose net worth is in this vicinity.
