import io
import torch
from PIL import Image
import torchvision.transforms as T
from app.core.config import DEVICE, SCORE_THRESHOLD
import cv2
import numpy as np

from app.services.supports import nms
transform = T.Compose([
    T.ToTensor()
])



def run_inference(model, image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image_tensor = transform(image).to(DEVICE)

    with torch.no_grad():
        outputs = model([image_tensor])

    # boxes = outputs[0]["boxes"].cpu().tolist()
    boxes = outputs[0]["boxes"].cpu().numpy()
    scores = outputs[0]["scores"].cpu().numpy()
    labels = outputs[0]["labels"].cpu().numpy()

    ingots = []
    for box, score, label in zip(boxes, scores, labels):
        if score < SCORE_THRESHOLD:
            continue

        x1, y1, x2, y2 = map(int, box)
        color = (0, 255, 0) if label == 1 else (255, 0, 0)
        if label == 1:
            ingots.append({'coordinates': [x1, y1, x2, y2],
                                    'conf': score})
        else:
            continue

    filtered = []
    filtered = nms(filtered, iou_thresh=0.5)
    image = np.array(image)
    for ingot in ingots:
    # for ingot in filtered:
        x1, y1, x2, y2 = ingot['coordinates']
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            image,
            # f"ingot {ingot['conf']:.2f}",
            f"{ingot['conf']:.2f}",
            (x2-30, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.3, (0, 255, 0), 1)

    count = len(ingots)

    # Draw total count top-left
    cv2.putText(
        image,
        f"Total Ingots: {count}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 0, 255),
        2
    )

    # Convert back to bytes
    _, buffer = cv2.imencode(".jpg", image)
    annotated_bytes = buffer.tobytes()

    return annotated_bytes, count