import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision import datasets
import numpy as np
from matplotlib import pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Transform definition
transform = transforms.Compose(
    [
        transforms.RandomRotation(degrees=15),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ]
)

# Use PyTorch's built-in MNIST dataset
train_dataset = datasets.MNIST(
    root="data",
    train=True,
    download=True,
    transform=transform,
)

test_dataset = datasets.MNIST(
    root="data",
    train=False,
    download=True,
    transform=transform,
)

print(f"Train Size: {len(train_dataset)}, Test Size: {len(test_dataset)}")

# Create data loaders
batch_size = 64
train_loader = DataLoader(
    dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=0
)
test_loader = DataLoader(
    dataset=test_dataset, batch_size=batch_size, shuffle=False, num_workers=0
)

# Test the data loader
for example_data, example_labels in train_loader:
    example_image = example_data[0]
    print("Input Size:", example_data.size())

    # Visualize the first image
    example_image_numpy = (
        example_image.squeeze().numpy()
    )  # Remove channel dimension for grayscale
    plt.imshow(example_image_numpy, cmap="gray")
    plt.title(f"Label: {example_labels[0]}")
    plt.show()
    break


# Your SimpleCNN class remains the same
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        # Convolutional layers
        self.conv1 = nn.Conv2d(
            in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1
        )
        self.conv2 = nn.Conv2d(
            in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1
        )
        self.conv3 = nn.Conv2d(
            in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1
        )

        # Activation and pooling layers
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Automatically calculate the input size for fully connected layer
        self.fc_input_size = self._get_conv_output_size()

        # Fully connected layers
        self.fc1 = nn.Linear(self.fc_input_size, 128)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, 20)
        self.fc3 = nn.Linear(20, 10)

    def _get_conv_output_size(self):
        """Automatically calculate the output size of convolutional layers"""
        with torch.no_grad():
            # Test with MNIST image size (1, 28, 28)
            x = torch.zeros(1, 1, 28, 28)

            # Forward pass through conv layers
            x = self.pool(self.relu(self.conv1(x)))  # 28x28 -> 14x14
            x = self.pool(self.relu(self.conv2(x)))  # 14x14 -> 7x7
            x = self.pool(self.relu(self.conv3(x)))  # 7x7 -> 3x3

            return x.view(1, -1).size(1)

    def forward(self, x):
        # Convolutional layers with ReLU and pooling
        x = self.pool(self.relu(self.conv1(x)))  # (batch, 32, 14, 14)
        x = self.pool(self.relu(self.conv2(x)))  # (batch, 64, 7, 7)
        x = self.pool(self.relu(self.conv3(x)))  # (batch, 128, 3, 3)

        # Flatten the tensor for fully connected layers
        x = x.view(x.size(0), -1)

        # Fully connected layers
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.fc3(x)  # Final output layer (no activation)

        return x


# Model training
model = SimpleCNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 10  # Reduced for testing
running_loss = 0.0

for epoch in range(num_epochs):
    for i, data in enumerate(train_loader, 0):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()

        outputs = model(inputs)  # No need for .float() since data is already float
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if i % 100 == 99:
            print(f"[{epoch + 1}, {i + 1}] loss: {running_loss / 100:.3f}")
            running_loss = 0.0

print("Finished Training")

model.eval()
predictions = []

with torch.no_grad():
    for data in test_loader:
        data = data.to(device)

        outputs = model(data)
        _, predicted = torch.max(outputs.data, 1)
        predictions.extend(predicted.cpu().tolist())

submission = pd.DataFrame({
    "ImageId": range(1, len(predictions) + 1),
    "Label": predictions
})

submission.to_csv('submission.csv', index=False)
