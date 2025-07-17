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
    "architecture": "SimpleCNN_1x1",
    "early_stopping_enabled": True,
    "early_stopping_patience": 7,
    "early_stopping_min_delta": 0.001
}

# Initialize W&B
wandb.init(
    project="mnist-cnn", 
    name="mnist-cnn-optimized",
    group="1x1-fc", 
    tags=["mnist-cnn", "optimized"],
    config=config)

# Check device availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
wandb.config.update({"device": str(device)})

# Data preprocessing and loading
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Load and split datasets
full_train_dataset = datasets.MNIST(root="data", train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root="data", train=False, download=True, transform=transform)

val_size = int(len(full_train_dataset) * config["val_split"])
train_size = len(full_train_dataset) - val_size
train_dataset, val_dataset = random_split(full_train_dataset, [train_size, val_size])

print(f'Dataset sizes - Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}')

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
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"])

print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

# Early Stopping class
class EarlyStopping:
    def __init__(self, patience=7, min_delta=0.001, restore_best_weights=True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_score = None
        self.counter = 0
        self.best_weights = None
        
    def __call__(self, val_score, model):
        if self.best_score is None:
            self.best_score = val_score
            self.save_checkpoint(model)
        elif val_score < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                if self.restore_best_weights:
                    model.load_state_dict(self.best_weights)
                return True
        else:
            self.best_score = val_score
            self.save_checkpoint(model)
            self.counter = 0
        return False
    
    def save_checkpoint(self, model):
        self.best_weights = model.state_dict().copy()

# Training function
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=5):
    wandb.watch(model, criterion, log="gradients", log_freq=100)
    
    # Initialize early stopping conditionally
    early_stopping = None
    if config["early_stopping_enabled"]:
        early_stopping = EarlyStopping(
            patience=config["early_stopping_patience"], 
            min_delta=config["early_stopping_min_delta"]
        )
        print(f"Early stopping enabled (patience={config['early_stopping_patience']})")
    else:
        print("Early stopping disabled")
    
    train_losses, val_losses, val_accuracies = [], [], []
    
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
            
            # Log every batch to W&B
            wandb.log({
                "batch_train_loss": loss.item(),
                "global_step": epoch * len(train_loader) + i
            })
            
            # Print progress every 200 batches
            if i % 200 == 199:
                print(f'  [Epoch {epoch + 1}, Batch {i + 1:4d}/{len(train_loader)}] Loss: {loss.item():.4f}')
        
        avg_train_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # Validation phase
        val_metrics = evaluate_epoch(model, val_loader, criterion, device)
        val_losses.append(val_metrics['loss'])
        val_accuracies.append(val_metrics['accuracy'])
        
        # Log epoch metrics to W&B
        wandb.log({
            "epoch": epoch + 1,
            "train_loss": avg_train_loss,
            "val_loss": val_metrics['loss'],
            "val_accuracy": val_metrics['accuracy'],
            "val_f1_macro": val_metrics['f1_macro'],
            "val_f1_weighted": val_metrics['f1_weighted'],
            "global_step": (epoch + 1) * len(train_loader)
        })
        
        print(f'Epoch {epoch + 1}/{num_epochs} - Train Loss: {avg_train_loss:.4f}, '
              f'Val Loss: {val_metrics["loss"]:.4f}, Val Acc: {val_metrics["accuracy"]:.2f}%, '
              f'Val F1: {val_metrics["f1_macro"]:.4f}')
        
        # Early stopping check
        if early_stopping and early_stopping(val_metrics['accuracy'], model):
            print(f'Early stopping triggered at epoch {epoch + 1}')
            wandb.log({
                "early_stopping_epoch": epoch + 1, 
                "early_stopping_triggered": True,
                "global_step": (epoch + 1) * len(train_loader)
            })
            break
    
    # Log final training info
    wandb.log({
        "total_epochs_trained": len(train_losses),
        "early_stopping_enabled": config["early_stopping_enabled"],
        **({"best_val_accuracy": early_stopping.best_score} if early_stopping else {})
    })
    
    return train_losses, val_losses, val_accuracies

# Helper function for validation evaluation
def evaluate_epoch(model, data_loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []
    
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    accuracy = 100 * correct / total
    f1_macro = f1_score(all_labels, all_preds, average='macro')
    f1_weighted = f1_score(all_labels, all_preds, average='weighted')
    
    return {
        'loss': total_loss / len(data_loader),
        'accuracy': accuracy,
        'f1_macro': f1_macro,
        'f1_weighted': f1_weighted,
        'predictions': all_preds,
        'labels': all_labels
    }

# Evaluation function
def evaluate_model(model, test_loader):
    return evaluate_epoch(model, test_loader, nn.CrossEntropyLoss(), device)

# Train the model
print("\nStarting training...")
train_losses, val_losses, val_accuracies = train_model(
    model, train_loader, val_loader, criterion, optimizer, num_epochs=config["epochs"]
)

# Evaluate the model
print("\nEvaluating model...")
test_results = evaluate_model(model, test_loader)

print(f"\nTest Results:")
print(f"Accuracy: {test_results['accuracy']:.2f}%")
print(f"F1 Score (Macro): {test_results['f1_macro']:.4f}")
print(f"F1 Score (Weighted): {test_results['f1_weighted']:.4f}")

# Log test metrics and confusion matrix
wandb.log({
    "test_accuracy": test_results['accuracy'],
    "test_f1_macro": test_results['f1_macro'],
    "test_f1_weighted": test_results['f1_weighted'],
    "confusion_matrix": wandb.plot.confusion_matrix(
        probs=None,
        y_true=test_results['labels'],
        preds=test_results['predictions'],
        class_names=[str(i) for i in range(10)]
    )
})

# Print classification report
print("\nClassification Report:")
print(classification_report(test_results['labels'], test_results['predictions']))

# Save model
model_path = 'mnist_cnn_model.pth'
torch.save(model.state_dict(), model_path)
print(f"\nModel saved as '{model_path}'")
wandb.save(model_path)

# Log final metrics summary
wandb.summary.update({
    "final_test_accuracy": test_results['accuracy'],
    "final_test_f1_macro": test_results['f1_macro'],
    "final_test_f1_weighted": test_results['f1_weighted'],
    "total_parameters": sum(p.numel() for p in model.parameters()),
    "epochs_trained": len(train_losses),
    "early_stopping_enabled": config["early_stopping_enabled"],
    **({"early_stopping_patience": config["early_stopping_patience"]} if config["early_stopping_enabled"] else {})
})

wandb.finish()
print("Training completed!")