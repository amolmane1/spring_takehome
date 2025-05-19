import io, json
import numpy as np
import torch
from PIL import Image, UnidentifiedImageError
from facenet_pytorch import MTCNN, InceptionResnetV1
from sklearn.metrics.pairwise import cosine_similarity
from functools import lru_cache

from .settings import settings

# one-time init
@lru_cache()
def _init_models():
    mtcnn = MTCNN(image_size=settings.image_size, margin=settings.margin)
    resnet = InceptionResnetV1(pretrained="vggface2").eval()
    return mtcnn, resnet

# one-time init
@lru_cache()
def _init_reference():
    with open(settings.reference_data_path, "r") as f:
        d = json.load(f)
    return np.array(d["embedding"]), d["name"], np.array(d["net worth"])

_MTCCN, _RESNET            = _init_models()
_EMBEDDINGS, _NAMES, _NET_WORTHS = _init_reference()

def predict_from_bytes(data: bytes):
    try:
        img = Image.open(io.BytesIO(data)).convert("RGB")
    except UnidentifiedImageError:
        raise ValueError("invalid image file")

    # detect face and crop to a standard 160×160 RGB tensor.
    face = _MTCCN(img)
    if face is None:
        raise ValueError("no face detected")

    # generate embeddings using face
    with torch.no_grad():
        emb = _RESNET(face.unsqueeze(0))
        emb = emb / emb.norm(dim=1, keepdim=True)

    # Similarity search
    emb_np = emb.cpu().numpy()
    scores = cosine_similarity(emb_np, _EMBEDDINGS)[0]
    idx    = scores.argsort()[::-1][:3]

    similar_people = [{"name": _NAMES[i], "score": float(scores[i])} for i in idx]
    weights = scores[idx]
    # calculate the estimated net worth of the person by getting the weighed average of the net worths of the 3 most similar individuals
    net_worth_estimate = float((weights @ _NET_WORTHS[idx]) / weights.sum())

    return net_worth_estimate, similar_people