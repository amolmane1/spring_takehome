import io
import json
import logging
from functools import lru_cache
from typing import List

import numpy as np
import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel
from pydantic_settings import BaseSettings
from PIL import Image, UnidentifiedImageError
from sklearn.metrics.pairwise import cosine_similarity
from facenet_pytorch import MTCNN, InceptionResnetV1

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("wealth-estimator")

# Settings class for configuration
class Settings(BaseSettings):
    reference_data_path: str = "data/reference_data.json"
    image_size: int = 160
    mtcnn_margin: int = 0

@lru_cache()
def get_settings() -> Settings:
    """Load settings from environment or .env file."""
    return Settings()

@lru_cache()
def load_reference_data(path: str):
    """Load embeddings, names, and net worths from JSON file."""
    with open(path, "r") as f:
        data = json.load(f)
    embeddings = np.array(data["embedding"])
    names = data["name"]
    net_worths = np.array(data["net worth"])
    return embeddings, names, net_worths

def init_models(settings: Settings):
    """Initialize MTCNN and InceptionResnetV1 models."""
    # model to identify if there is a face in the picture, and crop the picture to a standardized size and frame (just the face)
    mtcnn = MTCNN(image_size=settings.image_size, margin=settings.mtcnn_margin)
    # model to embed the framed face
    resnet = InceptionResnetV1(pretrained="vggface2").eval()
    return mtcnn, resnet

# Pydantic models for request/response validation
class SimilarIndividual(BaseModel):
    name: str
    score: float

class PredictionResponse(BaseModel):
    estimated_net_worth: float
    similar_individuals: List[SimilarIndividual]

# Application factory
def create_app() -> FastAPI:
    settings = get_settings()
    embeddings, names, net_worths = load_reference_data(settings.reference_data_path)
    mtcnn, resnet = init_models(settings)

    app = FastAPI(title="Wealth Potential Estimator")

    @app.post("/predict", response_model=PredictionResponse)
    async def predict(file: UploadFile = File(...)) -> PredictionResponse:
        # Validate file type
        if not file.content_type.startswith("image/"):
            logger.error("Invalid file type: %s", file.content_type)
            raise HTTPException(status_code=415, detail="Unsupported file type")

        # Read and parse image
        try:
            data = await file.read()
            img = Image.open(io.BytesIO(data)).convert("RGB")
        except UnidentifiedImageError:
            logger.exception("Failed to parse image")
            raise HTTPException(status_code=400, detail="Invalid image file")

        # Detect face
        face_tensor = mtcnn(img)
        if face_tensor is None:
            logger.warning("No face detected in image")
            raise HTTPException(status_code=400, detail="No face detected")

        # Compute embedding
        batch = face_tensor.unsqueeze(0)
        with torch.no_grad():
            emb = resnet(batch)
            emb = emb / emb.norm(dim=1, keepdim=True)
        emb_np = emb.cpu().numpy()

        # Similarity search
        scores = cosine_similarity(emb_np, embeddings)[0]
        top_idx = scores.argsort()[::-1][:3]

        similar = [
            SimilarIndividual(name=names[i], score=float(scores[i]))
            for i in top_idx
        ]
        # calculate the estimated net worth of the person by getting the weighed average of the net worths of the 3 most similar individuals
        selected_scores = scores[top_idx]
        selected_net_worths = net_worths[top_idx]
        estimated_worth = float(
            np.dot(selected_scores, selected_net_worths) / selected_scores.sum()
        )

        return PredictionResponse(
            estimated_net_worth=estimated_worth,
            similar_individuals=similar,
        )

    return app

app = create_app()
