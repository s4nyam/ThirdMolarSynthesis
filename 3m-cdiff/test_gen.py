import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import resnet18
from PIL import Image
from pathlib import Path
import os
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_recall_fscore_support
from collections import Counter

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Modified ResNet Model (same as in training)
class ResNetClassifier(nn.Module):
    def __init__(self, num_classes, channels=3):
        super(ResNetClassifier, self).__init__()
        self.model = resnet18(pretrained=False)  # No need for pretrained weights here
        
        if channels == 1:
            self.model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, num_classes)

    def forward(self, x):
        return self.model(x)

def load_model(model_path, num_classes=3, channels=3):
    model = ResNetClassifier(num_classes, channels=channels).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model

def predict_image(model, image_path, transform):
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        output = model(image_tensor)
        _, predicted = torch.max(output.data, 1)
    
    return predicted.item()

def extract_true_label_from_filename(filename):
    """Extract the true/generated class label from filename.
    Expected format: {class}_{number}_infer_{timestamp}.png
    """
    try:
        return int(filename.split('_')[0])
    except (ValueError, IndexError):
        return None

def process_test_images(model, test_dir, output_dir, transform):
    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create subdirectory for processed images
    processed_images_dir = output_dir / "processed_images"
    processed_images_dir.mkdir(parents=True, exist_ok=True)
    
    # Get all image paths
    image_paths = list(Path(test_dir).glob("*.png")) + list(Path(test_dir).glob("*.jpeg"))
    
    if not image_paths:
        print(f"No images found in {test_dir}")
        return [], [], []
    
    true_labels = []
    predicted_labels = []
    filenames = []
    
    for img_path in image_paths:
        try:
            # Extract true label from filename
            true_label = extract_true_label_from_filename(img_path.name)
            if true_label is None:
                print(f"Could not extract true label from {img_path.name}, skipping...")
                continue
            
            # Predict label
            predicted_label = predict_image(model, img_path, transform)
            
            # Store results
            true_labels.append(true_label)
            predicted_labels.append(predicted_label)
            filenames.append(img_path.name)
            
            # Create new filename
            stem = img_path.stem
            suffix = img_path.suffix
            new_filename = f"{stem}_predicted_{predicted_label}{suffix}"
            output_path = processed_images_dir / new_filename
            
            # Copy image to new location with new name
            img = Image.open(img_path)
            img.save(output_path)
            
            print(f"Processed: {img_path.name} -> True: {true_label}, Predicted: {predicted_label}")
            
        except Exception as e:
            print(f"Error processing {img_path}: {str(e)}")
    
    return true_labels, predicted_labels, filenames

def create_class_distribution_plot(true_labels, predicted_labels, output_dir, num_classes):
    """Create a bar plot showing distribution of generated vs predicted classes."""
    plt.figure(figsize=(12, 8))
    
    # Count occurrences
    true_counts = Counter(true_labels)
    pred_counts = Counter(predicted_labels)
    
    # Ensure all classes are represented
    classes = list(range(num_classes))
    true_values = [true_counts.get(i, 0) for i in classes]
    pred_values = [pred_counts.get(i, 0) for i in classes]
    
    # Create bar plot
    x = np.arange(len(classes))
    width = 0.35
    
    plt.bar(x - width/2, true_values, width, label='Generated/True Labels', alpha=0.8, color='skyblue')
    plt.bar(x + width/2, pred_values, width, label='Predicted Labels', alpha=0.8, color='lightcoral')
    
    plt.xlabel('Class')
    plt.ylabel('Number of Samples')
    plt.title('Distribution of Generated vs Predicted Classes')
    plt.xticks(x, classes)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for i, (true_val, pred_val) in enumerate(zip(true_values, pred_values)):
        plt.text(i - width/2, true_val + 1, str(true_val), ha='center', va='bottom')
        plt.text(i + width/2, pred_val + 1, str(pred_val), ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'class_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_confusion_matrix_plot(true_labels, predicted_labels, output_dir, num_classes):
    """Create a confusion matrix heatmap."""
    cm = confusion_matrix(true_labels, predicted_labels, labels=list(range(num_classes)))
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=list(range(num_classes)), 
                yticklabels=list(range(num_classes)))
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Class')
    plt.ylabel('True Class')
    plt.tight_layout()
    plt.savefig(output_dir / 'confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    return cm

def create_accuracy_per_class_plot(true_labels, predicted_labels, output_dir, num_classes):
    """Create a bar plot showing accuracy per class."""
    cm = confusion_matrix(true_labels, predicted_labels, labels=list(range(num_classes)))
    
    # Calculate per-class accuracy
    class_accuracies = []
    for i in range(num_classes):
        if cm[i].sum() > 0:  # If there are samples for this class
            accuracy = cm[i, i] / cm[i].sum()
        else:
            accuracy = 0
        class_accuracies.append(accuracy)
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(range(num_classes), class_accuracies, alpha=0.8, color='lightgreen')
    plt.xlabel('Class')
    plt.ylabel('Accuracy')
    plt.title('Per-Class Accuracy')
    plt.ylim(0, 1)
    plt.xticks(range(num_classes))
    plt.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for i, (bar, acc) in enumerate(zip(bars, class_accuracies)):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{acc:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'accuracy_per_class.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_prediction_confidence_plot(model, test_dir, transform, output_dir, true_labels, predicted_labels):
    """Create plots showing prediction confidence distribution."""
    image_paths = list(Path(test_dir).glob("*.png")) + list(Path(test_dir).glob("*.jpg"))
    
    confidences = []
    correct_predictions = []
    
    for img_path, true_label, pred_label in zip(image_paths, true_labels, predicted_labels):
        try:
            image = Image.open(img_path).convert('RGB')
            image_tensor = transform(image).unsqueeze(0).to(device)
            
            with torch.no_grad():
                output = model(image_tensor)
                probabilities = torch.softmax(output, dim=1)
                max_prob = torch.max(probabilities).item()
                
            confidences.append(max_prob)
            correct_predictions.append(true_label == pred_label)
            
        except Exception as e:
            print(f"Error getting confidence for {img_path}: {e}")
            continue
    
    # Confidence distribution plot
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.hist(confidences, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
    plt.xlabel('Prediction Confidence')
    plt.ylabel('Frequency')
    plt.title('Distribution of Prediction Confidence')
    plt.grid(True, alpha=0.3)
    
    # Confidence vs Accuracy plot
    plt.subplot(1, 2, 2)
    correct_conf = [conf for conf, correct in zip(confidences, correct_predictions) if correct]
    incorrect_conf = [conf for conf, correct in zip(confidences, correct_predictions) if not correct]
    
    plt.hist(correct_conf, bins=20, alpha=0.7, label='Correct', color='green', edgecolor='black')
    plt.hist(incorrect_conf, bins=20, alpha=0.7, label='Incorrect', color='red', edgecolor='black')
    plt.xlabel('Prediction Confidence')
    plt.ylabel('Frequency')
    plt.title('Confidence Distribution by Correctness')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'prediction_confidence.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_error_analysis_plot(true_labels, predicted_labels, output_dir, num_classes):
    """Create plots for error analysis."""
    cm = confusion_matrix(true_labels, predicted_labels, labels=list(range(num_classes)))
    
    # Most confused pairs
    confused_pairs = []
    for i in range(num_classes):
        for j in range(num_classes):
            if i != j and cm[i, j] > 0:
                confused_pairs.append((i, j, cm[i, j]))
    
    # Sort by confusion count
    confused_pairs.sort(key=lambda x: x[2], reverse=True)
    
    if confused_pairs:
        # Top 10 most confused pairs
        top_confused = confused_pairs[:min(10, len(confused_pairs))]
        
        plt.figure(figsize=(12, 6))
        pair_labels = [f'{pair[0]}→{pair[1]}' for pair in top_confused]
        counts = [pair[2] for pair in top_confused]
        
        bars = plt.bar(range(len(top_confused)), counts, alpha=0.8, color='salmon')
        plt.xlabel('True → Predicted Class')
        plt.ylabel('Number of Misclassifications')
        plt.title('Top Class Confusion Pairs')
        plt.xticks(range(len(top_confused)), pair_labels, rotation=45)
        
        # Add value labels on bars
        for bar, count in zip(bars, counts):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                    str(count), ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(output_dir / 'error_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()

def create_comprehensive_metrics_plot(true_labels, predicted_labels, output_dir, num_classes):
    """Create a comprehensive metrics visualization."""
    # Calculate metrics
    accuracy = accuracy_score(true_labels, predicted_labels)
    precision, recall, f1, support = precision_recall_fscore_support(true_labels, predicted_labels, 
                                                                     labels=list(range(num_classes)), 
                                                                     average=None, zero_division=0)
    
    # Create subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Precision per class
    axes[0, 0].bar(range(num_classes), precision, alpha=0.8, color='lightblue')
    axes[0, 0].set_title('Precision per Class')
    axes[0, 0].set_xlabel('Class')
    axes[0, 0].set_ylabel('Precision')
    axes[0, 0].set_ylim(0, 1)
    axes[0, 0].grid(True, alpha=0.3)
    
    # Recall per class
    axes[0, 1].bar(range(num_classes), recall, alpha=0.8, color='lightgreen')
    axes[0, 1].set_title('Recall per Class')
    axes[0, 1].set_xlabel('Class')
    axes[0, 1].set_ylabel('Recall')
    axes[0, 1].set_ylim(0, 1)
    axes[0, 1].grid(True, alpha=0.3)
    
    # F1-score per class
    axes[1, 0].bar(range(num_classes), f1, alpha=0.8, color='lightyellow')
    axes[1, 0].set_title('F1-Score per Class')
    axes[1, 0].set_xlabel('Class')
    axes[1, 0].set_ylabel('F1-Score')
    axes[1, 0].set_ylim(0, 1)
    axes[1, 0].grid(True, alpha=0.3)
    
    # Support per class
    axes[1, 1].bar(range(num_classes), support, alpha=0.8, color='lightcoral')
    axes[1, 1].set_title('Support (Number of Samples) per Class')
    axes[1, 1].set_xlabel('Class')
    axes[1, 1].set_ylabel('Number of Samples')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.suptitle(f'Overall Accuracy: {accuracy:.4f}', fontsize=16)
    plt.tight_layout()
    plt.savefig(output_dir / 'comprehensive_metrics.png', dpi=300, bbox_inches='tight')
    plt.close()

def save_detailed_results(true_labels, predicted_labels, filenames, output_dir, num_classes):
    """Save detailed results to CSV and text files."""
    # Create DataFrame with results
    results_df = pd.DataFrame({
        'filename': filenames,
        'true_label': true_labels,
        'predicted_label': predicted_labels,
        'correct': [t == p for t, p in zip(true_labels, predicted_labels)]
    })
    
    # Save to CSV
    results_df.to_csv(output_dir / 'detailed_results.csv', index=False)
    
    # Generate classification report
    report = classification_report(true_labels, predicted_labels, 
                                 labels=list(range(num_classes)), 
                                 target_names=[f'Class_{i}' for i in range(num_classes)],
                                 zero_division=0)
    
    # Save classification report
    with open(output_dir / 'classification_report.txt', 'w') as f:
        f.write("Classification Report\n")
        f.write("=" * 50 + "\n")
        f.write(report)
        f.write("\n\nOverall Accuracy: {:.4f}\n".format(accuracy_score(true_labels, predicted_labels)))
        f.write(f"Total samples: {len(true_labels)}\n")
        
        # Class-wise summary
        f.write("\nClass-wise Summary:\n")
        f.write("-" * 30 + "\n")
        for i in range(num_classes):
            class_true = [j for j in true_labels if j == i]
            class_pred = [j for j in predicted_labels if j == i]
            f.write(f"Class {i}:\n")
            f.write(f"  Generated samples: {len(class_true)}\n")
            f.write(f"  Predicted as this class: {len(class_pred)}\n")
            f.write(f"  Correctly classified: {len([j for j in range(len(true_labels)) if true_labels[j] == i and predicted_labels[j] == i])}\n")
            f.write("\n")

def generate_all_plots(true_labels, predicted_labels, filenames, model, test_dir, transform, output_dir, num_classes):
    """Generate all analysis plots and save results."""
    print("\nGenerating analysis plots...")
    
    # 1. Class distribution plot
    print("Creating class distribution plot...")
    create_class_distribution_plot(true_labels, predicted_labels, output_dir, num_classes)
    
    # 2. Confusion matrix
    print("Creating confusion matrix...")
    create_confusion_matrix_plot(true_labels, predicted_labels, output_dir, num_classes)
    
    # 3. Accuracy per class
    print("Creating accuracy per class plot...")
    create_accuracy_per_class_plot(true_labels, predicted_labels, output_dir, num_classes)
    
    # 4. Prediction confidence analysis
    print("Creating prediction confidence plots...")
    create_prediction_confidence_plot(model, test_dir, transform, output_dir, true_labels, predicted_labels)
    
    # 5. Error analysis
    print("Creating error analysis plot...")
    create_error_analysis_plot(true_labels, predicted_labels, output_dir, num_classes)
    
    # 6. Comprehensive metrics
    print("Creating comprehensive metrics plot...")
    create_comprehensive_metrics_plot(true_labels, predicted_labels, output_dir, num_classes)
    
    # 7. Save detailed results
    print("Saving detailed results...")
    save_detailed_results(true_labels, predicted_labels, filenames, output_dir, num_classes)
    
    print(f"All plots and analysis saved to: {output_dir}")

def main():
    # Configuration
    model_path = "outputs_thirdmolar_20250710_095827/best_model.pth"
    test_dir = "/home/sj/working_dir/third_molar_project_cuda0/1000_images_diff_no_condition"
    
    # Create output directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(f"outputs_test_thirdmolar_20250710_095827_1000images_uncodn_diff"+f"_{timestamp}")
    
    # Transform (should match what was used during training)
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
    ])
    
    # Load model (assuming 7 classes and RGB images based on your training code)
    num_classes = 7  # Update this if your model was trained with different number of classes
    channels = 3      # 1 for grayscale, 3 for RGB
    
    try:
        model = load_model(model_path, num_classes, channels)
        print(f"Model loaded successfully from {model_path}")
        
        # Process test images and get results
        print("\nProcessing test images...")
        true_labels, predicted_labels, filenames = process_test_images(model, test_dir, output_dir, transform)
        
        if not true_labels:
            print("No valid images were processed. Exiting.")
            return
        
        print(f"\nProcessed {len(true_labels)} images successfully")
        print(f"Processed images saved to: {output_dir / 'processed_images'}")
        
        # Generate all analysis plots and save results
        generate_all_plots(true_labels, predicted_labels, filenames, model, test_dir, 
                          transform, output_dir, num_classes)
        
        # Print summary statistics
        print("\n" + "="*50)
        print("SUMMARY STATISTICS")
        print("="*50)
        
        overall_accuracy = accuracy_score(true_labels, predicted_labels)
        print(f"Overall Accuracy: {overall_accuracy:.4f} ({overall_accuracy*100:.2f}%)")
        print(f"Total samples processed: {len(true_labels)}")
        
        # Class distribution
        print("\nClass Distribution (Generated vs Predicted):")
        true_counts = Counter(true_labels)
        pred_counts = Counter(predicted_labels)
        for i in range(num_classes):
            generated = true_counts.get(i, 0)
            predicted = pred_counts.get(i, 0)
            print(f"  Class {i}: Generated={generated}, Predicted={predicted}")
        
        print(f"\nAll analysis plots and results saved to: {output_dir}")
        
    except Exception as e:
        print(f"Error loading model or processing images: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()