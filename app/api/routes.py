from fastapi import APIRouter, File, UploadFile
from app.models.detector import load_model
from app.models.inference import run_inference

router = APIRouter()

model = load_model()

@router.post("/predict")
async def predict(file: UploadFile = File(...)):
    image_bytes = await file.read()
    detections = run_inference(model, image_bytes)
    return {"detections": detections}
