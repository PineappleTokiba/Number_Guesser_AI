"""One-time converter from the training checkpoints to Node-friendly ONNX files."""

from pathlib import Path

import torch
from torch import nn

root = Path(__file__).resolve().parent / "models"
example = torch.zeros(1, 1, 28, 28)

models = {
    "mlp": nn.Sequential(
        nn.Flatten(), nn.Linear(28 * 28, 128), nn.ReLU(), nn.Linear(128, 10)
    ),
    "cnn": nn.Sequential(
        nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
        nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
        nn.Flatten(), nn.Linear(64 * 7 * 7, 128), nn.ReLU(), nn.Dropout(0.2), nn.Linear(128, 10),
    ),
}

for name, model in models.items():
    checkpoint = root / f"mnist_{name}_model.pth"
    model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
    model.eval()

for name, model in models.items():
    destination = root / f"mnist_{name}_model.onnx"
    torch.onnx.export(
        model,
        (example,),
        destination,
        input_names=["image"],
        output_names=["logits"],
        opset_version=18,
        dynamo=False,
    )
    print(f"Exported {destination.name}")
