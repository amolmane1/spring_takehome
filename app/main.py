from fastapi import FastAPI, File, UploadFile, HTTPException
from .services import predict_from_bytes
from .schemas import PredictionResponse

app = FastAPI(title="Wealth Estimator")

@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(415, "Unsupported file type")

    data = await file.read()
    try:
        net_worth_estimate, similar_people = predict_from_bytes(data)
    except ValueError as e:
        raise HTTPException(400, str(e))

    return {"estimated_net_worth": net_worth_estimate, "similar_individuals": similar_people}