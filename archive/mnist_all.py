# %% [markdown]
# # MNIST Project with Real-time Training Visualization
# > **By Jisang Yun**

# %%
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision 
import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('TkAgg')  # Interactive backend
import matplotlib.pyplot as plt
plt.ion()  # Interactive mode on
from IPython.display import clear_output
# Optional: uncomment if sklearn and seaborn are available
# from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
# import seaborn as sns

# Enable interactive plotting
plt.ion()  # Interactive mode on

# %%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# %%
transform = transforms.Compose([
    transforms.RandomRotation(degrees=15),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_dataset = datasets.MNIST(root="data", train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root="data", train=False, download=True, transform=transform)

print(f'Train Size: {len(train_dataset)}, Test Size: {len(test_dataset)}')

# %%
batch_size = 64
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

# %%
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.shortcut = nn.Sequential()

    def forward(self, x):
        residual = self.shortcut(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual
        out = self.relu(out)
        return out

# %%
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        
        self.res_block1 = ResidualBlock(1, 32)
        self.res_block2 = ResidualBlock(32, 64)
        self.res_block3 = ResidualBlock(64, 128)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.relu = nn.ReLU()
        
        self.fc_input_size = self._get_conv_output_size()
        self.fc1 = nn.Linear(self.fc_input_size, 128)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, 20)
        self.fc3 = nn.Linear(20, 10)

    def _get_conv_output_size(self):
        with torch.no_grad():
            x = torch.zeros(1, 1, 28, 28)
            x = self.pool(self.res_block1(x))
            x = self.pool(self.res_block2(x))
            x = self.pool(self.res_block3(x))
            return x.view(1, -1).size(1)

    def forward(self, x):
        x = self.pool(self.res_block1(x))
        x = self.pool(self.res_block2(x))
        x = self.pool(self.res_block3(x))
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# %%
model = SimpleCNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# %%
def plot_training_progress(train_losses, epoch_losses, current_epoch, total_epochs):
    # For terminal/script execution, use separate windows
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    if train_losses:
        ax1.clear()
        ax1.plot(train_losses, alpha=0.7, linewidth=0.8, color='blue')
        ax1.set_title(f'Training Loss per Batch (Epoch {current_epoch}/{total_epochs})')
        ax1.set_xlabel('Batch Number')
        ax1.set_ylabel('Loss')
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(0, max(1.0, max(train_losses) if train_losses else 1.0))
    
    if epoch_losses:
        ax2.clear()
        epochs_range = range(1, len(epoch_losses) + 1)
        ax2.plot(epochs_range, epoch_losses, 'r-', linewidth=2, marker='o', markersize=4)
        ax2.set_title('Average Training Loss per Epoch')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Average Loss')
        ax2.grid(True, alpha=0.3)
        ax2.set_xlim(0, total_epochs)
        if epoch_losses:
            ax2.set_ylim(0, max(epoch_losses) * 1.1)
    
    plt.tight_layout()
    plt.draw()
    plt.pause(0.001)  # Small pause to update display
    
    if train_losses:
        print(f"Current Batch Loss: {train_losses[-1]:.4f}")
    if epoch_losses:
        print(f"Last Epoch Average Loss: {epoch_losses[-1]:.4f}")
        print(f"Best Epoch Loss: {min(epoch_losses):.4f}")

# Global figure for persistent plotting
training_fig = None

# %%
num_epochs = 50
train_losses = []
epoch_losses = []
running_loss = 0.0
plot_interval = 50

print("Starting Training...")

for epoch in range(num_epochs):
    epoch_loss = 0.0
    batch_count = 0
    
    for i, data in enumerate(train_loader, 0):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs.float())
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        current_loss = loss.item()
        train_losses.append(current_loss)
        epoch_loss += current_loss
        batch_count += 1
        running_loss += current_loss
        
        if i % 100 == 99:
            avg_loss = running_loss / 100
            print(f"[{epoch + 1}, {i + 1:5d}] loss: {avg_loss:.3f}")
            running_loss = 0.0
        
        if i % plot_interval == 0:
            plot_training_progress(train_losses, epoch_losses, epoch + 1, num_epochs)
    
    avg_epoch_loss = epoch_loss / batch_count if batch_count > 0 else 0
    epoch_losses.append(avg_epoch_loss)
    plot_training_progress(train_losses, epoch_losses, epoch + 1, num_epochs)
    
    print(f"Epoch {epoch + 1} completed. Average Loss: {avg_epoch_loss:.4f}")

print('Training Finished!')

# %%
plt.figure(figsize=(15, 5))

plt.subplot(1, 2, 1)
plt.plot(train_losses, alpha=0.7, linewidth=0.8, color='blue')
plt.title('Final Training Loss per Batch')
plt.xlabel('Batch Number')
plt.ylabel('Loss')
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.plot(range(1, num_epochs + 1), epoch_losses, 'r-', linewidth=2, marker='o', markersize=3)
plt.title('Final Average Training Loss per Epoch')
plt.xlabel('Epoch')
plt.ylabel('Average Loss')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print(f"Initial Loss: {train_losses[0]:.4f}")
print(f"Final Loss: {train_losses[-1]:.4f}")
print(f"Best Epoch Loss: {min(epoch_losses):.4f}")
print(f"Loss Reduction: {train_losses[0] - train_losses[-1]:.4f}")

# %%
def evaluate_model(model, test_loader, device):
    model.eval()
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for data, targets in test_loader:
            data, targets = data.to(device), targets.to(device)
            outputs = model(data)
            _, predicted = torch.max(outputs, 1)
            all_predictions.extend(predicted.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
    
    return np.array(all_predictions), np.array(all_targets)

def calculate_metrics_manual(y_true, y_pred, num_classes=10):
    """Calculate metrics manually without sklearn"""
    correct = np.sum(y_true == y_pred)
    total = len(y_true)
    accuracy = correct / total
    
    # Per-class metrics
    precision_per_class = []
    recall_per_class = []
    f1_per_class = []
    
    for class_id in range(num_classes):
        # True positives, false positives, false negatives
        tp = np.sum((y_true == class_id) & (y_pred == class_id))
        fp = np.sum((y_true != class_id) & (y_pred == class_id))
        fn = np.sum((y_true == class_id) & (y_pred != class_id))
        
        # Precision and recall
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        precision_per_class.append(precision)
        recall_per_class.append(recall)
        f1_per_class.append(f1)
    
    # Macro averages
    precision_macro = np.mean(precision_per_class)
    recall_macro = np.mean(recall_per_class)
    f1_macro = np.mean(f1_per_class)
    
    # Weighted averages
    class_counts = [np.sum(y_true == i) for i in range(num_classes)]
    weights = np.array(class_counts) / total
    precision_weighted = np.sum(np.array(precision_per_class) * weights)
    recall_weighted = np.sum(np.array(recall_per_class) * weights)
    f1_weighted = np.sum(np.array(f1_per_class) * weights)
    
    return {
        'accuracy': accuracy,
        'precision_macro': precision_macro,
        'recall_macro': recall_macro,
        'f1_macro': f1_macro,
        'precision_weighted': precision_weighted,
        'recall_weighted': recall_weighted,
        'f1_weighted': f1_weighted,
        'precision_per_class': precision_per_class,
        'recall_per_class': recall_per_class,
        'f1_per_class': f1_per_class
    }

def plot_confusion_matrix_manual(y_true, y_pred, class_names=None):
    """Plot confusion matrix without seaborn"""
    num_classes = len(set(y_true)) if class_names is None else len(class_names)
    cm = np.zeros((num_classes, num_classes), dtype=int)
    
    for true_label, pred_label in zip(y_true, y_pred):
        cm[true_label, pred_label] += 1
    
    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation='nearest', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.colorbar()
    
    classes = class_names or [str(i) for i in range(num_classes)]
    tick_marks = np.arange(num_classes)
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)
    
    # Add text annotations
    thresh = cm.max() / 2.
    for i in range(num_classes):
        for j in range(num_classes):
            plt.text(j, i, format(cm[i, j], 'd'),
                    horizontalalignment="center",
                    color="white" if cm[i, j] > thresh else "black")
    
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.show()
    return cm

def print_classification_report_manual(y_true, y_pred, class_names=None):
    """Print classification report manually"""
    metrics = calculate_metrics_manual(y_true, y_pred)
    classes = class_names or [str(i) for i in range(len(metrics['precision_per_class']))]
    
    print(f"{'Class':<10} {'Precision':<10} {'Recall':<10} {'F1-Score':<10} {'Support':<10}")
    print("-" * 50)
    
    for i, class_name in enumerate(classes):
        support = np.sum(y_true == i)
        print(f"{class_name:<10} {metrics['precision_per_class'][i]:<10.3f} "
              f"{metrics['recall_per_class'][i]:<10.3f} {metrics['f1_per_class'][i]:<10.3f} {support:<10}")
    
    print("-" * 50)
    print(f"{'macro avg':<10} {metrics['precision_macro']:<10.3f} "
          f"{metrics['recall_macro']:<10.3f} {metrics['f1_macro']:<10.3f} {len(y_true):<10}")
    print(f"{'weighted avg':<10} {metrics['precision_weighted']:<10.3f} "
          f"{metrics['recall_weighted']:<10.3f} {metrics['f1_weighted']:<10.3f} {len(y_true):<10}")

# Check if sklearn is available
try:
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
    import seaborn as sns
    SKLEARN_AVAILABLE = True
    print("Using sklearn for metrics calculation")
except ImportError:
    SKLEARN_AVAILABLE = False
    print("sklearn not available, using manual implementation")

def calculate_metrics(y_true, y_pred):
    if SKLEARN_AVAILABLE:
        accuracy = accuracy_score(y_true, y_pred)
        precision_macro = precision_score(y_true, y_pred, average='macro')
        recall_macro = recall_score(y_true, y_pred, average='macro')
        f1_macro = f1_score(y_true, y_pred, average='macro')
        precision_weighted = precision_score(y_true, y_pred, average='weighted')
        recall_weighted = recall_score(y_true, y_pred, average='weighted')
        f1_weighted = f1_score(y_true, y_pred, average='weighted')
        
        return {
            'accuracy': accuracy,
            'precision_macro': precision_macro,
            'recall_macro': recall_macro,
            'f1_macro': f1_macro,
            'precision_weighted': precision_weighted,
            'recall_weighted': recall_weighted,
            'f1_weighted': f1_weighted
        }
    else:
        return calculate_metrics_manual(y_true, y_pred)

def plot_confusion_matrix(y_true, y_pred, class_names=None):
    if SKLEARN_AVAILABLE:
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=class_names or range(len(cm)),
                    yticklabels=class_names or range(len(cm)))
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.show()
        return cm
    else:
        return plot_confusion_matrix_manual(y_true, y_pred, class_names)

def plot_per_class_metrics(y_true, y_pred, class_names=None):
    if SKLEARN_AVAILABLE:
        precision_per_class = precision_score(y_true, y_pred, average=None)
        recall_per_class = recall_score(y_true, y_pred, average=None)
        f1_per_class = f1_score(y_true, y_pred, average=None)
    else:
        metrics = calculate_metrics_manual(y_true, y_pred)
        precision_per_class = metrics['precision_per_class']
        recall_per_class = metrics['recall_per_class']
        f1_per_class = metrics['f1_per_class']
    
    classes = class_names or [f'Class {i}' for i in range(len(precision_per_class))]
    x = np.arange(len(classes))
    width = 0.25
    
    plt.figure(figsize=(12, 6))
    plt.bar(x - width, precision_per_class, width, label='Precision', alpha=0.8)
    plt.bar(x, recall_per_class, width, label='Recall', alpha=0.8)
    plt.bar(x + width, f1_per_class, width, label='F1-Score', alpha=0.8)
    
    plt.xlabel('Classes')
    plt.ylabel('Score')
    plt.title('Per-Class Performance Metrics')
    plt.xticks(x, classes)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 1.0)
    plt.tight_layout()
    plt.show()
    
    return precision_per_class, recall_per_class, f1_per_class

# %%
print("Model Evaluation")
predictions, true_labels = evaluate_model(model, test_loader, device)
metrics = calculate_metrics(true_labels, predictions)

print("Overall Performance Metrics:")
print(f"Accuracy: {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
print(f"Precision (Macro): {metrics['precision_macro']:.4f}")
print(f"Precision (Weighted): {metrics['precision_weighted']:.4f}")
print(f"Recall (Macro): {metrics['recall_macro']:.4f}")
print(f"Recall (Weighted): {metrics['recall_weighted']:.4f}")
print(f"F1-Score (Macro): {metrics['f1_macro']:.4f}")
print(f"F1-Score (Weighted): {metrics['f1_weighted']:.4f}")

mnist_classes = [str(i) for i in range(10)]
print(f"\nTotal Test Samples: {len(predictions)}")
print(f"Correctly Classified: {np.sum(predictions == true_labels)}")
print(f"Misclassified: {np.sum(predictions != true_labels)}")

# %%
print("Detailed Classification Report:")
if SKLEARN_AVAILABLE:
    report = classification_report(true_labels, predictions, target_names=mnist_classes)
    print(report)
else:
    print_classification_report_manual(true_labels, predictions, mnist_classes)

# %%
print("Confusion Matrix:")
confusion_mat = plot_confusion_matrix(true_labels, predictions, mnist_classes)

# %%
print("Per-Class Performance:")
precision_per_class, recall_per_class, f1_per_class = plot_per_class_metrics(
    true_labels, predictions, mnist_classes
)

# %%
def show_misclassified_examples(model, test_loader, device, num_examples=8):
    model.eval()
    misclassified = []
    
    with torch.no_grad():
        for data, targets in test_loader:
            data, targets = data.to(device), targets.to(device)
            outputs = model(data)
            _, predicted = torch.max(outputs, 1)
            
            mask = predicted != targets
            if mask.any():
                misclassified_data = data[mask]
                misclassified_targets = targets[mask]
                misclassified_preds = predicted[mask]
                
                for i in range(min(len(misclassified_data), num_examples - len(misclassified))):
                    misclassified.append({
                        'image': misclassified_data[i].cpu(),
                        'true_label': misclassified_targets[i].cpu().item(),
                        'predicted_label': misclassified_preds[i].cpu().item()
                    })
                
                if len(misclassified) >= num_examples:
                    break
    
    if misclassified:
        fig, axes = plt.subplots(2, 4, figsize=(12, 6))
        fig.suptitle('Misclassified Examples', fontsize=16)
        
        for i, example in enumerate(misclassified[:8]):
            row, col = i // 4, i % 4
            img = example['image'].squeeze().numpy()
            
            axes[row, col].imshow(img, cmap='gray')
            axes[row, col].set_title(f'True: {example["true_label"]}, Pred: {example["predicted_label"]}')
            axes[row, col].axis('off')
        
        plt.tight_layout()
        plt.show()

print("Error Analysis - Misclassified Examples:")
show_misclassified_examples(model, test_loader, device)

# %%
print("Generating Submission File...")
submission_predictions = []

model.eval()
with torch.no_grad():
    for data in test_loader:
        data = data.to(device)
        outputs = model(data)
        _, predicted = torch.max(outputs.data, 1)
        submission_predictions.extend(predicted.cpu().tolist())

submission = pd.DataFrame({
    "ImageId": range(1, len(submission_predictions) + 1),
    "Label": submission_predictions
})

submission.to_csv('submission.csv', index=False)
print(f"Submission file saved! Total predictions: {len(submission_predictions)}")

# %%
print("Final Model Performance Summary:")
print(f"Test Accuracy: {metrics['accuracy']*100:.2f}%")
print(f"Macro F1-Score: {metrics['f1_macro']:.4f}")
print(f"Weighted F1-Score: {metrics['f1_weighted']:.4f}")
print(f"Total Parameters: {sum(p.numel() for p in model.parameters()):,}")
print(f"Trainable Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")