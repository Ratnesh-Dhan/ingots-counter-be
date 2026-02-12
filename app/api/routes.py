from fastapi import APIRouter, File, UploadFile
from fastapi.responses import Response
from app.services.detector import load_model
from app.services.inference import run_inference
from app.services.test import test_image

router = APIRouter()

model = load_model()

@router.get("/")
async def root():
    return {"message": "Hello World"}

@router.post("/predict")
async def predict(file: UploadFile = File(...)):
    image_bytes = await file.read()
    annotated_bytes, count = run_inference(model, image_bytes)
    print(count)
    return Response(
        content=annotated_bytes,
        media_type="image/jpeg",
        headers={"X-Count": str(count)}
    )

@router.post("/test")
async def test(file: UploadFile = File(...)):
    image_bytes = await file.read()
    test_image(image_bytes)
    return {"message": "Image tested successfully"}