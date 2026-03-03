#!/usr/bin/env python3
"""
Binary Classification using ResNet18
Class 0: Files ending with _0.png
Class 1: Files ending with _1.png, _2.png, _3.png, _4.png, _5.png, _6.png
"""

import os
import glob
import random
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import time

# Set random seeds for reproducibility
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)

class DentalDataset(Dataset):
    """Custom dataset for dental images"""
    
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        label = self.labels[idx]
        
        # Load image
        image = Image.open(image_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

def get_class_from_filename(filename):
    """Extract class from filename based on the last number after splitting by '_'"""
    parts = filename.split('_')
    last_part = parts[-1]  # Get the last part (e.g., "0.png")
    class_num = int(last_part.split('.')[0])  # Extract number before .png
    
    # Binary classification:
    # Class 0: files ending with _0.png -> binary class 0
    # Class 1: files ending with _1.png, _2.png, etc. -> binary class 1
    return 0 if class_num == 0 else 1

def load_data(data_dir):
    """Load and prepare data for binary classification"""
    
    # Get all PNG files
    image_paths = glob.glob(os.path.join(data_dir, "*.png"))
    
    print(f"Found {len(image_paths)} images")
    
    # Extract labels
    labels = []
    valid_paths = []
    
    for path in image_paths:
        filename = os.path.basename(path)
        try:
            label = get_class_from_filename(filename)
            labels.append(label)
            valid_paths.append(path)
        except (ValueError, IndexError) as e:
            print(f"Skipping invalid filename: {filename} - {e}")
    
    print(f"Valid images: {len(valid_paths)}")
    
    # Count classes
    class_counts = {0: labels.count(0), 1: labels.count(1)}
    print(f"Class distribution: {class_counts}")
    
    return valid_paths, labels

def create_data_loaders(image_paths, labels, batch_size=32, val_split=0.2, test_split=0.1):
    """Create train, validation, and test data loaders"""
    
    # First split: separate test set
    train_val_paths, test_paths, train_val_labels, test_labels = train_test_split(
        image_paths, labels, test_size=test_split, random_state=42, stratify=labels
    )
    
    # Second split: separate train and validation
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        train_val_paths, train_val_labels, test_size=val_split/(1-test_split), 
        random_state=42, stratify=train_val_labels
    )
    
    print(f"Train: {len(train_paths)}, Val: {len(val_paths)}, Test: {len(test_paths)}")
    
    # Define transforms
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Create datasets
    train_dataset = DentalDataset(train_paths, train_labels, train_transform)
    val_dataset = DentalDataset(val_paths, val_labels, val_test_transform)
    test_dataset = DentalDataset(test_paths, test_labels, val_test_transform)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    return train_loader, val_loader, test_loader

def create_model(num_classes=2, pretrained=True):
    """Create ResNet18 model for binary classification"""
    
    model = models.resnet18(pretrained=pretrained)
    
    # Modify the final layer for binary classification
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, num_classes)
    
    return model

def train_epoch(model, train_loader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = torch.max(output.data, 1)
        total += target.size(0)
        correct += (predicted == target).sum().item()
        
        if batch_idx % 50 == 0:
            print(f'Batch {batch_idx}, Loss: {loss.item():.4f}')
    
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100. * correct / total
    
    return epoch_loss, epoch_acc

def validate(model, val_loader, criterion, device):
    """Validate the model"""
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            val_loss += criterion(output, target).item()
            
            _, predicted = torch.max(output, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
            
            all_predictions.extend(predicted.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
    
    val_loss /= len(val_loader)
    val_acc = 100. * correct / total
    
    return val_loss, val_acc, all_predictions, all_targets

def plot_training_history(train_losses, train_accuracies, val_losses, val_accuracies):
    """Plot training history"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot losses
    ax1.plot(train_losses, label='Train Loss')
    ax1.plot(val_losses, label='Val Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Plot accuracies
    ax2.plot(train_accuracies, label='Train Accuracy')
    ax2.plot(val_accuracies, label='Val Accuracy')
    ax2.set_title('Training and Validation Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
    print("Training history plot saved as 'training_history.png'")
    plt.close()  # Close the figure to free memory

def main():
    # Configuration
    DATA_DIR = "/home/sanyam/playground/tg-dental-activity/train"
    BATCH_SIZE = 32
    LEARNING_RATE = 0.001
    NUM_EPOCHS = 50
    MODEL_SAVE_PATH = "resnet18_dental_binary_classifier.pth"
    
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Load data
    print("Loading data...")
    image_paths, labels = load_data(DATA_DIR)
    
    # Create data loaders
    print("Creating data loaders...")
    train_loader, val_loader, test_loader = create_data_loaders(
        image_paths, labels, batch_size=BATCH_SIZE
    )
    
    # Create model
    print("Creating model...")
    model = create_model(num_classes=2, pretrained=True)
    model = model.to(device)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.1)
    
    # Training loop
    print("Starting training...")
    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []
    best_val_acc = 0.0
    
    start_time = time.time()
    
    for epoch in range(NUM_EPOCHS):
        print(f'\nEpoch {epoch+1}/{NUM_EPOCHS}')
        print('-' * 20)
        
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # Validate
        val_loss, val_acc, val_preds, val_targets = validate(model, val_loader, criterion, device)
        
        # Step scheduler
        scheduler.step()
        
        # Store metrics
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)
        
        print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
        print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'train_acc': train_acc,
            }, MODEL_SAVE_PATH)
            print(f'New best model saved with validation accuracy: {val_acc:.2f}%')
    
    training_time = time.time() - start_time
    print(f'\nTraining completed in {training_time:.2f} seconds')
    
    # Plot training history
    plot_training_history(train_losses, train_accuracies, val_losses, val_accuracies)
    
    # Load best model for final evaluation
    print("\nLoading best model for final evaluation...")
    checkpoint = torch.load(MODEL_SAVE_PATH)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Final test evaluation
    print("Final evaluation on test set...")
    test_loss, test_acc, test_preds, test_targets = validate(model, test_loader, criterion, device)
    
    print(f'\nFinal Test Results:')
    print(f'Test Loss: {test_loss:.4f}')
    print(f'Test Accuracy: {test_acc:.2f}%')
    
    # Detailed classification report
    print('\nClassification Report:')
    print(classification_report(test_targets, test_preds, target_names=['Class 0', 'Class 1']))
    
    # Confusion matrix
    print('\nConfusion Matrix:')
    cm = confusion_matrix(test_targets, test_preds)
    print(cm)
    
    print(f'\nModel saved as: {MODEL_SAVE_PATH}')
    print(f'Best validation accuracy: {best_val_acc:.2f}%')

if __name__ == "__main__":
    main()
