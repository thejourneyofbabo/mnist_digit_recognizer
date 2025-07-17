# Import necessary libraries
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.data import DataLoader, random_split
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
import wandb

# Configuration
config = {
    "learning_rate": 0.001,
    "epochs": 40,
    "batch_size": 64,
    "val_split": 0.2,
    "architecture": "SimpleCNN"
}

# Initialize W&B
wandb.init(
    project="mnist-cnn", 
    name="mnist-cnn-core",
    group="1x1-fc", 
    tags=["mnist-cnn", "core"],
    config=config)

# Check device availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
wandb.config.update({"device": str(device)})

# Data preprocessing
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Load datasets
full_train_dataset = datasets.MNIST(root="data", train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root="data", train=False, download=True, transform=transform)

# Split training data
val_size = int(len(full_train_dataset) * config["val_split"])
train_size = len(full_train_dataset) - val_size
train_dataset, val_dataset = random_split(full_train_dataset, [train_size, val_size])

print(f'Train Size: {len(train_dataset)}, Val Size: {len(val_dataset)}, Test Size: {len(test_dataset)}')

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=config["batch_size"], shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=config["batch_size"], shuffle=False)

# Define CNN model
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        
        # Pooling and activation
        self.pool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU()
        
        # Fully connected layers
        self.fc1 = nn.Linear(64 * 3 * 3, 128)
        self.bn_fc1 = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, 10)
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, x):
        # Conv layers with batch norm and pooling
        x = self.pool(self.relu(self.bn1(self.conv1(x))))  # 28x28 -> 14x14
        x = self.pool(self.relu(self.bn2(self.conv2(x))))  # 14x14 -> 7x7
        x = self.pool(self.relu(self.bn3(self.conv3(x))))  # 7x7 -> 3x3
        
        # Flatten and FC layers
        x = x.view(-1, 64 * 3 * 3)
        x = self.relu(self.bn_fc1(self.fc1(x)))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x
    

class SimpleCNN_1x1(nn.Module):
    def __init__(self):
        super(SimpleCNN_1x1, self).__init__()
        self.conv1 = nn.Conv2d(1, 24, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(24)
        self.conv2 = nn.Conv2d(24, 48, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(48)
        self.conv3 = nn.Conv2d(48, 72, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(72)

        self.pool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU()

        self.conv1x1_1 = nn.Conv2d(72, 128, kernel_size=1)
        self.bn_fc1 = nn.BatchNorm2d(128)
        self.conv1x1_2 = nn.Conv2d(128, 64, kernel_size=1)
        self.bn_fc2 = nn.BatchNorm2d(64)
        self.conv1x1_3 = nn.Conv2d(64, 10, kernel_size=1)

        self.dropout = nn.Dropout2d(0.4)
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        x = self.pool(self.relu(self.bn1(self.conv1(x))))
        x = self.pool(self.relu(self.bn2(self.conv2(x))))
        x = self.pool(self.relu(self.bn3(self.conv3(x))))
        x = self.dropout(self.relu(self.bn_fc1(self.conv1x1_1(x))))
        x = self.dropout(self.relu(self.bn_fc2(self.conv1x1_2(x))))
        x = self.conv1x1_3(x)
        x = self.global_avg_pool(x)
        return x.view(x.size(0), -1)

# Model setup
model = SimpleCNN_1x1().to(device)
# Initialize model, loss, and optimizer
# model = SimpleCNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"])

print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

# Training function
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=5):
    wandb.watch(model, criterion, log="all", log_freq=10)
    
    train_losses = []
    val_losses = []
    val_accuracies = []
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        epoch_loss = 0.0
        
        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
            # Log every batch
            wandb.log({
                "batch_train_loss": loss.item(),
                "global_step": epoch * len(train_loader) + i
            })
            
            # Print progress every 100 batches
            if i % 100 == 99:
                print(f'  [Epoch {epoch + 1}, Batch {i + 1:4d}/{len(train_loader)}] Loss: {loss.item():.4f}')
        
        avg_train_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        avg_val_loss = val_loss / len(val_loader)
        val_accuracy = 100 * correct / total
        val_losses.append(avg_val_loss)
        val_accuracies.append(val_accuracy)
        
        # Log epoch metrics to W&B with global_step
        current_step = (epoch + 1) * len(train_loader)
        wandb.log({
            "epoch": epoch + 1,
            "train_loss": avg_train_loss,
            "val_loss": avg_val_loss,
            "val_accuracy": val_accuracy,
            "global_step": current_step
        })
        
        print(f'Epoch {epoch + 1}/{num_epochs} - Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Val Acc: {val_accuracy:.2f}%')
    
    return train_losses, val_losses, val_accuracies

# Evaluation function
def evaluate_model(model, test_loader):
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    return np.array(all_preds), np.array(all_labels)

# Train the model
print("\nStarting training...")
train_losses, val_losses, val_accuracies = train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=config["epochs"])

# Evaluate the model
print("\nEvaluating model...")
predictions, true_labels = evaluate_model(model, test_loader)

# Calculate metrics
accuracy = accuracy_score(true_labels, predictions)
f1_macro = f1_score(true_labels, predictions, average='macro')
f1_weighted = f1_score(true_labels, predictions, average='weighted')
conf_matrix = confusion_matrix(true_labels, predictions)

print(f"\nTest Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
print(f"F1 Score (Macro): {f1_macro:.4f}")
print(f"F1 Score (Weighted): {f1_weighted:.4f}")

# Log test metrics to W&B
wandb.log({
    "test_accuracy": accuracy * 100,
    "test_f1_macro": f1_macro,
    "test_f1_weighted": f1_weighted
})

# Log confusion matrix to W&B
wandb.log({"confusion_matrix": wandb.plot.confusion_matrix(
    probs=None,
    y_true=true_labels,
    preds=predictions,
    class_names=[str(i) for i in range(10)])
})

# Classification report
print("\nClassification Report:")
print(classification_report(true_labels, predictions))

# Save model
torch.save(model.state_dict(), 'mnist_cnn_model.pth')
print("\nModel saved as 'mnist_cnn_model.pth'")

# Save model to W&B
wandb.save('mnist_cnn_model.pth')

# Log final metrics summary
wandb.summary["final_test_accuracy"] = accuracy * 100
wandb.summary["final_test_f1_macro"] = f1_macro
wandb.summary["final_test_f1_weighted"] = f1_weighted
wandb.summary["total_parameters"] = sum(p.numel() for p in model.parameters())
wandb.summary["final_train_loss"] = train_losses[-1]
wandb.summary["final_val_loss"] = val_losses[-1]

# Finish W&B run
wandb.finish()