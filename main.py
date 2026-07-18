from torchvision import DataLoader
from torchvision import datasets, transforms

train_dataset = datasets.MNIST(
    root="data",
    train=True,
    download=True
)

test_dataset = datasets.MNIST(
    root="data",
    train=False,
    download=True
)

train_loader = DataLoader(
    train_dataset,
    batch_size=64,
    shuffle=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=64,
    shuffle=False
)