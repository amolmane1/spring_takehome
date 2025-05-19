from pydantic import BaseModel
from typing import List

class SimilarIndividual(BaseModel):
    name: str
    score: float

class PredictionResponse(BaseModel):
    estimated_net_worth: float
    similar_individuals: List[SimilarIndividual]