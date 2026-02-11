import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from app.core.config import DEVICE, MODEL_PATH, NUM_CLASSES

def load_model():
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=None)

    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, NUM_CLASSES)

    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()

    return model
