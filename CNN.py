import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# Original transform (kept for comparison)
# transform = transforms.Compose([
#     transforms.ToTensor(),
#     transforms.Normalize((0.1307,), (0.3081,))
# ])

# Augment only the training images to imitate different handwriting styles.
train_transform = transforms.Compose([
    transforms.RandomAffine(
        degrees=10,
        translate=(0.1, 0.1),
        scale=(0.9, 1.1)
    ),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# Test images should remain unchanged so accuracy is measured fairly.
test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset = datasets.MNIST(
    root="data",
    train=True,
    download=True,
    # transform=transform  # Original transform
    transform=train_transform
)

test_dataset = datasets.MNIST(
    root="data",
    train=False,
    download=True,
    # transform=transform  # Original transform
    transform=test_transform
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

# Original CNN (kept for comparison)
# model = nn.Sequential(
#     nn.Conv2d(1, 32, kernel_size=3, padding=1),
#     nn.ReLU(),
#     nn.MaxPool2d(2),
#
#     nn.Conv2d(32, 64, kernel_size=3, padding=1),
#     nn.ReLU(),
#     nn.MaxPool2d(2),
#
#     nn.Flatten(),
#     nn.Linear(64 * 7 * 7, 128),
#     nn.ReLU(),
#     nn.Dropout(0.2),
#     nn.Linear(128, 10)
# )

# CNN with batch normalization
model = nn.Sequential(
    nn.Conv2d(1, 32, kernel_size=3, padding=1),
    nn.BatchNorm2d(32),
    nn.ReLU(),
    nn.MaxPool2d(2),

    nn.Conv2d(32, 64, kernel_size=3, padding=1),
    nn.BatchNorm2d(64),
    nn.ReLU(),
    nn.MaxPool2d(2),

    nn.Flatten(),
    nn.Linear(64 * 7 * 7, 128),
    nn.ReLU(),
    nn.Dropout(0.2),
    nn.Linear(128, 10)
)

loss_function = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001
)

epochs = 10

for epoch in range(epochs):
    model.train()
    total_loss = 0

    for images, labels in train_loader:
        optimizer.zero_grad()

        predictions = model(images)
        loss = loss_function(predictions, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    average_loss = total_loss / len(train_loader)

    print(
        f"Epoch {epoch + 1}/{epochs}, "
        f"Loss: {average_loss:.4f}"
    )

model.eval()

correct = 0
total = 0

with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images)
        predictions = outputs.argmax(dim=1)

        total += labels.size(0)
        correct += (predictions == labels).sum().item()

accuracy = 100 * correct / total

print(f"Test accuracy: {accuracy:.2f}%")

torch.save(model.state_dict(), "mnist_cnn_model.pth")
