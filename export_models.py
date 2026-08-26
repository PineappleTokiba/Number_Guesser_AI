"""One-time converter from the training checkpoints to Node-friendly ONNX files."""

from pathlib import Path

import torch
from torch import nn

root = Path(__file__).resolve().parent / "models"
web_models = Path(__file__).resolve().parent / "app" / "static" / "models"
example = torch.zeros(1, 1, 28, 28)


class MLP(nn.Module):
    def __init__(self, output_size=10):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28 * 28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, output_size),
        )

    def forward(self, x):
        x = self.flatten(x)
        return self.linear_relu_stack(x)


models = {
    "mnist_mlp_model": (MLP(), root / "mnist_mlp_model.pth"),
    "mnist_cnn_model": (nn.Sequential(
        nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
        nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
        nn.Flatten(), nn.Linear(64 * 7 * 7, 128), nn.ReLU(), nn.Dropout(0.2), nn.Linear(128, 10),
    ), root / "mnist_cnn_model.pth"),
    "emnist_bymerge_model": (MLP(output_size=47), root / "MLP_EMNIST.pth"),
}

for name, (model, checkpoint) in models.items():
    if not checkpoint.exists():
        print(f"Skipped {name}: {checkpoint.name} was not found")
        continue

    model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
    model.eval()
    destination = web_models / f"{name}.onnx"
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
