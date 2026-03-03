import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import transforms, utils
from pathlib import Path
from PIL import Image
import torch.nn.functional as F
from torchvision.models import resnet18
from torch.optim.lr_scheduler import StepLR
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix, precision_recall_curve
import numpy as np
import os
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

# Create output directory with timestamp
def create_output_dir(base_path, dataset_name):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(f"{base_path}/outputs_{dataset_name}_{timestamp}")
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir

# Focal Loss Implementation
class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

# Custom Dataset with automatic class detection
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, data_path, transform=None):
        self.data_path = data_path
        self.transform = transform
        self.image_paths = list(Path(data_path).glob("*.png"))
        
        # Automatically determine number of classes
        self.classes = sorted(list(set(
            int(img_path.stem.split("_")[-1][0]) for img_path in self.image_paths
        )))
        self.num_classes = len(self.classes)
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        self.idx_to_class = {idx: cls for cls, idx in self.class_to_idx.items()}

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = int(img_path.stem.split("_")[-1][0])
        label_idx = self.class_to_idx[label]

        with open(img_path, 'rb') as f:
            image = Image.open(f).convert('RGB')

        if self.transform:
            image = self.transform(image)
        return image, label_idx

# Function to create sample mosaic
def create_sample_mosaic(dataset, output_path, n_samples=5):
    # Get one sample per class
    samples = {}
    for img, label_idx in dataset:
        cls = dataset.idx_to_class[label_idx]
        if cls not in samples:
            samples[cls] = img
            if len(samples) == dataset.num_classes:
                break
    
    # Create mosaic grid
    fig, axes = plt.subplots(1, len(samples), figsize=(15, 3))
    if len(samples) == 1:  # Handle case with only 1 class
        axes = [axes]
    
    for idx, (cls, img) in enumerate(sorted(samples.items())):
        # Convert tensor to numpy and denormalize if needed
        if isinstance(img, torch.Tensor):
            img = img.numpy().transpose((1, 2, 0))
            img = (img - img.min()) / (img.max() - img.min())
        
        axes[idx].imshow(img)
        axes[idx].set_title(f"Class {cls}")
        axes[idx].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path / "data_samples.png")
    plt.close()

# Function to plot data distribution
def plot_data_distribution(dataset, output_path):
    # Count samples per class
    class_counts = Counter()
    for _, label_idx in dataset:
        cls = dataset.idx_to_class[label_idx]
        class_counts[cls] += 1
    
    # Create bar plot
    classes = sorted(class_counts.keys())
    counts = [class_counts[cls] for cls in classes]
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(range(len(classes)), counts, color='skyblue', edgecolor='navy', alpha=0.7)
    plt.xlabel('Class')
    plt.ylabel('Number of Samples')
    plt.title('Data Distribution by Class')
    plt.xticks(range(len(classes)), [f'Class {cls}' for cls in classes])
    
    # Add value labels on bars
    for bar, count in zip(bars, counts):
        plt.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.1,
                f'{count}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(output_path / "data_distribution.png", dpi=300, bbox_inches='tight')
    plt.close()

# Function to plot confusion matrix
def plot_confusion_matrix(y_true, y_pred, class_names, output_path):
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=[f'Class {name}' for name in class_names],
                yticklabels=[f'Class {name}' for name in class_names])
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(output_path / "confusion_matrix.png", dpi=300, bbox_inches='tight')
    plt.close()

# Function to plot precision-recall curves
def plot_precision_recall_curves(y_true, y_probs, class_names, output_path):
    plt.figure(figsize=(12, 8))
    
    for i, class_name in enumerate(class_names):
        # Create binary labels for current class
        y_true_binary = (np.array(y_true) == i).astype(int)
        y_scores = np.array(y_probs)[:, i]
        
        precision, recall, _ = precision_recall_curve(y_true_binary, y_scores)
        plt.plot(recall, precision, label=f'Class {class_name}')
    
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curves')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path / "precision_recall_curves.png", dpi=300, bbox_inches='tight')
    plt.close()

# Function to plot training metrics
def plot_training_metrics(train_metrics, val_metrics, output_path):
    epochs = range(1, len(train_metrics['loss']) + 1)
    
    # Plot losses
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.plot(epochs, train_metrics['loss'], 'b-', label='Training Loss')
    plt.plot(epochs, val_metrics['loss'], 'r-', label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot accuracies
    plt.subplot(1, 3, 2)
    plt.plot(epochs, train_metrics['accuracy'], 'b-', label='Training Accuracy')
    plt.plot(epochs, val_metrics['accuracy'], 'r-', label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot F1 scores
    plt.subplot(1, 3, 3)
    plt.plot(epochs, train_metrics['f1'], 'b-', label='Training F1')
    plt.plot(epochs, val_metrics['f1'], 'r-', label='Validation F1')
    plt.title('Training and Validation F1 Score')
    plt.xlabel('Epoch')
    plt.ylabel('F1 Score')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path / "training_metrics.png", dpi=300, bbox_inches='tight')
    plt.close()

# Modified ResNet Model
class ResNetClassifier(nn.Module):
    def __init__(self, num_classes, channels=3):
        super(ResNetClassifier, self).__init__()
        self.model = resnet18(pretrained=True)
        
        if channels == 1:
            self.model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, num_classes)

    def forward(self, x):
        return self.model(x)

def train(model, device, train_loader, optimizer, criterion, epoch, output_dir):
    model.train()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        preds = output.argmax(dim=1, keepdim=True)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(target.cpu().numpy())
        
        if batch_idx % 100 == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} '
                  f'({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')
    
    avg_loss = running_loss / len(train_loader)
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='weighted')
    
    # Save training metrics
    with open(output_dir / "training_log.txt", "a") as f:
        f.write(f"Epoch {epoch}: Train Loss: {avg_loss:.4f}, Acc: {accuracy:.4f}, F1: {f1:.4f}\n")
    
    return avg_loss, accuracy, f1

def validate(model, device, val_loader, criterion, epoch, output_dir):
    model.eval()
    val_loss = 0.0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            val_loss += criterion(output, target).item()
            preds = output.argmax(dim=1, keepdim=True)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(target.cpu().numpy())
    
    avg_loss = val_loss / len(val_loader)
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='weighted')
    
    # Save validation metrics
    with open(output_dir / "training_log.txt", "a") as f:
        f.write(f"Epoch {epoch}: Val Loss: {avg_loss:.4f}, Acc: {accuracy:.4f}, F1: {f1:.4f}\n\n")
    
    return avg_loss, accuracy, f1

def test(model, device, test_loader, output_dir):
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            probs = F.softmax(output, dim=1)
            preds = output.argmax(dim=1, keepdim=True)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(target.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    # Flatten the predictions and labels
    all_preds = [pred[0] for pred in all_preds]
    
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='weighted')
    
    # Get unique classes present in the test set and map them back to original class names
    unique_labels = sorted(list(set(all_labels + all_preds)))
    original_classes = test_loader.dataset.dataset.classes
    target_names = [str(original_classes[label]) for label in unique_labels]
    
    report = classification_report(all_labels, all_preds, labels=unique_labels, target_names=target_names)
    
    # Plot confusion matrix
    plot_confusion_matrix(all_labels, all_preds, unique_labels, output_dir)
    
    # Plot precision-recall curves (only if we have probabilities for all classes)
    if len(all_probs) > 0 and len(all_probs[0]) == len(original_classes):
        plot_precision_recall_curves(all_labels, all_probs, original_classes, output_dir)
    
    # Save test results
    with open(output_dir / "test_results.txt", "w") as f:
        f.write(f"Test Accuracy: {accuracy:.4f}\n")
        f.write(f"Test F1 Score: {f1:.4f}\n\n")
        f.write("Classification Report:\n")
        f.write(report)
    
    print("\nTest Results:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print("\nClassification Report:")
    print(report)
    
    return accuracy, f1, report

def main():
    # Hyperparameters
    batch_size = 32
    epochs = 10
    lr = 0.001
    gamma = 0.7
    validation_split = 0.15  # 15% for validation
    test_split = 0.15  # 15% for test
    train_split = 1.0 - validation_split - test_split
    channels_to_keep = 3  # 1 for grayscale, 3 for RGB
    data_path = "thirdmolar_w_classification"  # Your single data directory
    dataset_name = "thirdmolar_w_classification"
    
    # Create output directory
    output_dir = create_output_dir(".", dataset_name)
    print(f"All outputs will be saved to: {output_dir}")
    
    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Data loading and splitting
    if channels_to_keep == 1:
        transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
        ])
    
    # Create full dataset
    full_dataset = CustomDataset(data_path, transform=transform)
    print(f"Found {len(full_dataset)} images in {len(full_dataset.classes)} classes")
    
    # Create sample mosaic before splitting
    create_sample_mosaic(full_dataset, output_dir)
    print("Created sample mosaic image in output directory")
    
    # Plot data distribution
    plot_data_distribution(full_dataset, output_dir)
    print("Created data distribution plot in output directory")
    
    # Split into train, validation and test
    val_size = int(validation_split * len(full_dataset))
    test_size = int(test_split * len(full_dataset))
    train_size = len(full_dataset) - val_size - test_size
    
    train_dataset, val_dataset, test_dataset = random_split(
        full_dataset, [train_size, val_size, test_size]
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    
    # Get number of classes from the dataset
    num_classes = full_dataset.num_classes
    
    # Model, loss, optimizer
    model = ResNetClassifier(num_classes, channels=channels_to_keep).to(device)
    criterion = FocalLoss(gamma=2)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = StepLR(optimizer, step_size=1, gamma=gamma)
    
    # Save hyperparameters
    with open(output_dir / "config.txt", "w") as f:
        f.write(f"Batch size: {batch_size}\n")
        f.write(f"Epochs: {epochs}\n")
        f.write(f"Learning rate: {lr}\n")
        f.write(f"Train split: {train_split:.2f}\n")
        f.write(f"Validation split: {validation_split:.2f}\n")
        f.write(f"Test split: {test_split:.2f}\n")
        f.write(f"Channels: {channels_to_keep}\n")
        f.write(f"Classes: {num_classes}\n")
        f.write(f"Device: {device}\n")
    
    # Training loop
    best_val_loss = float('inf')
    best_model_path = output_dir / "best_model.pth"
    
    # Track metrics for plotting
    train_metrics = {'loss': [], 'accuracy': [], 'f1': []}
    val_metrics = {'loss': [], 'accuracy': [], 'f1': []}
    
    for epoch in range(1, epochs + 1):
        train_loss, train_acc, train_f1 = train(model, device, train_loader, optimizer, criterion, epoch, output_dir)
        val_loss, val_acc, val_f1 = validate(model, device, val_loader, criterion, epoch, output_dir)
        scheduler.step()
        
        # Store metrics
        train_metrics['loss'].append(train_loss)
        train_metrics['accuracy'].append(train_acc)
        train_metrics['f1'].append(train_f1)
        val_metrics['loss'].append(val_loss)
        val_metrics['accuracy'].append(val_acc)
        val_metrics['f1'].append(val_f1)
        
        print(f'\nEpoch {epoch}:')
        print(f'Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | Train F1: {train_f1:.4f}')
        print(f'Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | Val F1: {val_f1:.4f}\n')
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_model_path)
            print("Saved best model!")
    
    # Plot training metrics
    plot_training_metrics(train_metrics, val_metrics, output_dir)
    print("Created training metrics plots in output directory")
    
    # Save final model
    torch.save(model.state_dict(), output_dir / "final_model.pth")
    print(f"\nTraining complete! Best validation loss: {best_val_loss:.4f}")
    
    # Test the best model
    print("\nTesting best model on test set...")
    best_model = ResNetClassifier(num_classes, channels=channels_to_keep).to(device)
    best_model.load_state_dict(torch.load(best_model_path, weights_only=True))
    
    test_acc, test_f1, test_report = test(best_model, device, test_loader, output_dir)
    
    print(f"\nAll outputs saved to: {output_dir}")
    print("\nGenerated files:")
    print("- data_samples.png: Sample images from each class")
    print("- data_distribution.png: Bar plot of class distribution")
    print("- training_metrics.png: Training/validation loss, accuracy, and F1 plots")
    print("- confusion_matrix.png: Confusion matrix for test set")
    print("- precision_recall_curves.png: Precision-recall curves for each class")
    print("- config.txt: Training configuration")
    print("- training_log.txt: Training progress log")
    print("- test_results.txt: Final test results")
    print("- best_model.pth: Best model weights")
    print("- final_model.pth: Final model weights")

if __name__ == '__main__':
    main()