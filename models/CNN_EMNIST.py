import torch
import torch.nn as nn
from pathlib import Path
from torch.utils.data import DataLoader, Subset, random_split
from torchvision import datasets, transforms

def train_transform():
    return transforms.Compose([
        transforms.RandomAffine(
            degrees=10,
            translate=(0.1, 0.1),
            scale=(0.9, 1.1),
        ),
        transforms.ToTensor(),
    ])

class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()

        self.model = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),

            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.15),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),

            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.25),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),

            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(),

            nn.Flatten(),
            nn.Linear(128 * 7 * 7, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, 47),
        )

    def forward(self, x):
        return self.model(x)

def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

def evaluate(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    return 100 * correct, test_loss

transform = transforms.ToTensor()

train_data = datasets.EMNIST(
    root="data",
    split="bymerge",
    train=True,
    download=True,
    transform=train_transform(),
)

# Use the same samples without random augmentation when measuring validation accuracy.
validation_source = datasets.EMNIST(
    root="data",
    split="bymerge",
    train=True,
    download=True,
    transform=transform,
)

# Reproducibly reserve 10% of the training set for validation.
validation_size = len(train_data) // 10
training_size = len(train_data) - validation_size
split_generator = torch.Generator().manual_seed(42)
train_split, validation_split = random_split(
    range(len(train_data)),
    [training_size, validation_size],
    generator=split_generator,
)
train_data = Subset(train_data, train_split.indices)
validation_data = Subset(validation_source, validation_split.indices)

test_data = datasets.EMNIST(
    root="data",
    split="bymerge",
    train=False,
    download=True,
    transform=transform,
)

train_loader = DataLoader(
    train_data,
    batch_size=64,
    shuffle=True
)

test_loader = DataLoader(
    test_data,
    batch_size=64,
    shuffle=False
)

validation_loader = DataLoader(
    validation_data,
    batch_size=64,
    shuffle=False,
)

device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
print(f"Using {device} device")

model = NeuralNetwork().to(device)

loss_fn = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001,
    weight_decay=1e-4,
)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="max",
    factor=0.5,
    patience=2,
    min_lr=1e-5,
)

epochs = 20
checkpoint_path = Path(__file__).resolve().parent / "CNN_EMNIST.pth"
best_validation_accuracy = 0.0

for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train(train_loader, model, loss_fn, optimizer)
    validation_accuracy, validation_loss = evaluate(validation_loader, model, loss_fn)

    if validation_accuracy > best_validation_accuracy:
        best_validation_accuracy = validation_accuracy
        torch.save(model.state_dict(), checkpoint_path)
        save_message = " (saved best model)"
    else:
        save_message = ""

    scheduler.step(validation_accuracy)
    current_lr = optimizer.param_groups[0]["lr"]

    print(
        f"Validation accuracy: {validation_accuracy:.2f}%, "
        f"Avg loss: {validation_loss:.6f}, "
        f"Learning rate: {current_lr:.6f}{save_message}\n"
    )

# Evaluate the official test set once using the best validation checkpoint.
model.load_state_dict(
    torch.load(checkpoint_path, map_location=device, weights_only=True)
)
test_accuracy, test_loss = evaluate(test_loader, model, loss_fn)

print(f"Test accuracy: {test_accuracy:.2f}%, Avg loss: {test_loss:.6f}")
print(f"Best validation accuracy: {best_validation_accuracy:.2f}%")
print(f"Best model saved to: {checkpoint_path}")
