import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "model.pth"
NUM_CLASSES = 4
SCORE_THRESHOLD = 0.5
