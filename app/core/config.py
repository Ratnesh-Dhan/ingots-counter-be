import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "app/model/fasterrcnn_best.pth"
NUM_CLASSES = 4
SCORE_THRESHOLD = 0.55
