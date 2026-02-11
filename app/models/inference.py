import io
from PIL import Image
import torchvision.transforms as T
from app.core.config import DEVICE, SCORE_THRESHOLD

transform = T.Compose([
    T.ToTensor()
])

def run_inference(model, image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image_tensor = transform(image).to(DEVICE)

    with torch.no_grad():
        outputs = model([image_tensor])

    boxes = outputs[0]["boxes"].cpu().tolist()
    scores = outputs[0]["scores"].cpu().tolist()
    labels = outputs[0]["labels"].cpu().tolist()

    results = []
    for box, score, label in zip(boxes, scores, labels):
        if score > SCORE_THRESHOLD:
            results.append({
                "box": box,
                "score": score,
                "label": int(label)
            })

    return results
