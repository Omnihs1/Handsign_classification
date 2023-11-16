from torchmetrics.classification import MulticlassAccuracy
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
Accuracy = MulticlassAccuracy(num_classes = 263).to(device)
