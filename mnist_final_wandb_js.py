import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import classification_report, f1_score, precision_score, recall_score, confusion_matrix
import wandb
import numpy as np

# Configuration
config = {
    "learning_rate": 0.001,
    "epochs": 40,
    "batch_size": 64,
    "val_split": 0.15, # Validation split ratio
    # Early stopping parameters
    "early_stopping_patience": 7, # Early stopping patience
    # Learning rate scheduler parameters
    "gamma": 0.1, # Learning rate decay factor
    "step_size": 10, # Step size for learning rate scheduler
}

# Initialize W&B
wandb.init(
    project="mnist-final-wandb-js",
    name="mnist-final-v0(base-setup)",
    config=config)

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Data preprocessing
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Load datasets
full_train_dataset = datasets.MNIST(root="data", train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root="data", train=False, download=True, transform=transform)

# Split train/validation
val_size = int(len(full_train_dataset) * config["val_split"])
train_size = len(full_train_dataset) - val_size
train_dataset, val_dataset = random_split(full_train_dataset, [train_size, val_size])

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=config["batch_size"], shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=config["batch_size"], shuffle=False)

# CNN Model
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 24, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(24)
        self.conv2 = nn.Conv2d(24, 48, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(48)
        self.conv3 = nn.Conv2d(48, 72, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(72)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv1x1_1 = nn.Conv2d(72, 128, kernel_size=1)
        self.bn4 = nn.BatchNorm2d(128)
        self.conv1x1_2 = nn.Conv2d(128, 64, kernel_size=1)
        self.bn5 = nn.BatchNorm2d(64)
        self.conv1x1_3 = nn.Conv2d(64, 10, kernel_size=1)
        self.dropout = nn.Dropout2d(0.4)
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.pool(self.relu(self.bn1(self.conv1(x))))
        x = self.pool(self.relu(self.bn2(self.conv2(x))))
        x = self.pool(self.relu(self.bn3(self.conv3(x))))
        x = self.dropout(self.relu(self.bn4(self.conv1x1_1(x))))
        x = self.dropout(self.relu(self.bn5(self.conv1x1_2(x))))
        x = self.conv1x1_3(x)
        x = self.global_avg_pool(x)
        return x.view(x.size(0), -1)

# Model setup
model = SimpleCNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"])
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=config["step_size"], gamma=config["gamma"])

print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

# Early Stopping
class EarlyStopping:
    def __init__(self, patience=5):
        self.patience = patience
        self.best_score = None
        self.counter = 0
        self.best_weights = None
        
    def __call__(self, val_score, model):
        if self.best_score is None:
            self.best_score = val_score
            self.save_checkpoint(model)
        elif val_score <= self.best_score:
            self.counter += 1
            if self.counter >= self.patience:
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
def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs=20):
    early_stopping = EarlyStopping(patience=config["early_stopping_patience"])
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        epoch_loss = 0.0
        
        for batch_idx, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            
            # Print batch loss every 100 batches
            if batch_idx % 100 == 0:
                print(f'  Epoch {epoch + 1}/{num_epochs}, Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}')
        
        avg_train_loss = epoch_loss / len(train_loader)
        
        # Validation
        val_metrics = evaluate_model(model, val_loader)
        
        # Learning rate scheduler step
        scheduler.step()
        
        # Log to W&B
        wandb.log({
            "epoch": epoch + 1,
            "train_loss": avg_train_loss,
            "val_loss": val_metrics['loss'],
            "val_accuracy": val_metrics['accuracy'],
            "val_f1_macro": val_metrics['f1_macro'],
            "learning_rate": optimizer.param_groups[0]['lr']
        })
        
        print(f'Epoch {epoch + 1}/{num_epochs} - Train Loss: {avg_train_loss:.4f}, '
              f'Val Loss: {val_metrics["loss"]:.4f}, Val Acc: {val_metrics["accuracy"]:.2f}%, '
              f'Val F1: {val_metrics["f1_macro"]:.4f}, LR: {optimizer.param_groups[0]["lr"]:.6f}')
        
        # Early stopping
        if early_stopping(val_metrics['accuracy'], model):
            print(f'Early stopping at epoch {epoch + 1}')
            break

# Evaluation function
def evaluate_model(model, data_loader):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []
    
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = nn.CrossEntropyLoss()(outputs, labels)
            total_loss += loss.item()
            
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Convert to numpy arrays
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    # Calculate metrics
    accuracy = 100 * correct / total
    precision_macro = precision_score(all_labels, all_preds, average='macro', zero_division=0)
    recall_macro = recall_score(all_labels, all_preds, average='macro', zero_division=0)
    f1_macro = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    
    # Per-class metrics
    precision_per_class = precision_score(all_labels, all_preds, average=None, zero_division=0)
    recall_per_class = recall_score(all_labels, all_preds, average=None, zero_division=0)
    f1_per_class = f1_score(all_labels, all_preds, average=None, zero_division=0)
    
    # Confusion matrix
    conf_matrix = confusion_matrix(all_labels, all_preds)
    
    return {
        'loss': total_loss / len(data_loader),
        'accuracy': accuracy,
        'precision_macro': precision_macro,
        'recall_macro': recall_macro,
        'f1_macro': f1_macro,
        'precision_per_class': precision_per_class,
        'recall_per_class': recall_per_class,
        'f1_per_class': f1_per_class,
        'confusion_matrix': conf_matrix,
        'predictions': all_preds,
        'labels': all_labels
    }

# Metrics display
def print_metrics(metrics):
    print(f"\nEvaluation Results:")
    print(f"Accuracy: {metrics['accuracy']:.2f}%")
    print(f"Precision: {metrics['precision_macro']:.4f}")
    print(f"Recall: {metrics['recall_macro']:.4f}")
    print(f"F1 Score: {metrics['f1_macro']:.4f}")
    
    print(f"\nPer-class metrics:")
    for i in range(len(metrics['precision_per_class'])):
        print(f"Class {i}: P={metrics['precision_per_class'][i]:.3f}, "
              f"R={metrics['recall_per_class'][i]:.3f}, "
              f"F1={metrics['f1_per_class'][i]:.3f}")

# W&B logging (removed individual test metrics - only confusion matrix will be logged)
def log_test_metrics(metrics):
    # Test metrics are only printed, not logged to W&B
    # Only confusion matrix will be logged as it provides meaningful visualization
    pass

# Train the model
print("Starting training...")
train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs=config["epochs"])

# Evaluate the model
print("Evaluating model...")
test_results = evaluate_model(model, test_loader)

# Display and log results
print_metrics(test_results)
log_test_metrics(test_results)

# Log confusion matrix
wandb.log({
    "confusion_matrix": wandb.plot.confusion_matrix(
        probs=None,
        y_true=test_results['labels'],
        preds=test_results['predictions'],
        class_names=[str(i) for i in range(10)]
    )
})

# Classification report
print("\nClassification Report:")
print(classification_report(test_results['labels'], test_results['predictions'], 
                          target_names=[f"Digit {i}" for i in range(10)]))

# Save model
torch.save(model.state_dict(), 'mnist_cnn_model.pth')
print("Model saved as 'mnist_cnn_model.pth'")

wandb.finish()
print("Training completed!")