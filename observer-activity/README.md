# Dental Binary Classification with ResNet18

This project implements a binary classification model using ResNet18 to classify dental images into two categories:

- **Class 0**: Images ending with `_0.png`
- **Class 1**: Images ending with `_1.png`, `_2.png`, `_3.png`, `_4.png`, `_5.png`, `_6.png`

## Files

- `2.bc.py`: Training script for the binary classification model
- `3.infer.py`: Inference script for making predictions on new images
- `README.md`: This file

## Requirements

Make sure you have the following packages installed:

```bash
pip install torch torchvision pillow scikit-learn matplotlib numpy
```

## Usage

### Training (`2.bc.py`)

The training script will:
1. Load images from the training directory
2. Split them into train/validation/test sets
3. Train a ResNet18 model with transfer learning
4. Save the best model based on validation accuracy
5. Generate training plots and evaluation metrics

```bash
python 2.bc.py
```

**Key features:**
- Data augmentation for training (rotation, flips, color jitter)
- Stratified splitting to maintain class balance
- Learning rate scheduling
- Automatic best model saving
- Comprehensive evaluation with confusion matrix and classification report

**Output:**
- `resnet18_dental_binary_classifier.pth`: Trained model file
- `training_history.png`: Training curves plot
- Console output with training progress and final metrics

### Inference (`3.infer.py`)

The inference script can predict on:
- Single images
- Multiple images in a directory
- Batch processing with CSV output

```bash
# Single image prediction
python 3.infer.py --model resnet18_dental_binary_classifier.pth --image path/to/image.png

# Directory prediction
python 3.infer.py --model resnet18_dental_binary_classifier.pth --directory path/to/images/

# Save results to CSV
python 3.infer.py --model resnet18_dental_binary_classifier.pth --directory path/to/images/ --output results.csv

# Force CPU usage
python 3.infer.py --model resnet18_dental_binary_classifier.pth --image path/to/image.png --device cpu
```

**Output format:**
- Console: Formatted prediction results with confidence scores
- CSV: Structured data with all predictions and probabilities

## Model Architecture

- **Base Model**: ResNet18 (pretrained on ImageNet)
- **Modification**: Final layer replaced with 2-class classifier
- **Input Size**: 224x224 RGB images
- **Output**: Binary classification (Class 0 vs Class 1+)

## Data Processing

1. **Image Loading**: PNG files from the training directory
2. **Label Extraction**: Based on filename pattern `*_*_X.png` where X is the class
3. **Binary Mapping**: 
   - X = 0 → Binary Class 0
   - X ∈ {1,2,3,4,5,6} → Binary Class 1
4. **Preprocessing**: Resize, normalize, augment (training only)

## Training Configuration

- **Batch Size**: 32
- **Learning Rate**: 0.001 (with step decay)
- **Epochs**: 50
- **Optimizer**: Adam
- **Loss Function**: CrossEntropyLoss
- **Data Split**: 70% train, 20% validation, 10% test

## Example Output

### Training
```
Found 4747 images
Valid images: 4747
Class distribution: {0: 2853, 1: 1894}
Train: 3322, Val: 949, Test: 476

Epoch 1/50
--------------------
Batch 0, Loss: 0.6891
...
Train Loss: 0.3456, Train Acc: 85.67%
Val Loss: 0.2890, Val Acc: 88.45%
New best model saved with validation accuracy: 88.45%
```

### Inference
```
Using device: cuda
Model loaded from: resnet18_dental_binary_classifier.pth
Model validation accuracy: 88.45%

================================================================================
PREDICTION RESULTS
================================================================================

1. 00000734_1_0.png
   Predicted Class: Class 0
   Confidence: 0.9234 (92.34%)
   Probabilities:
     Class 0: 0.9234 (92.34%)
     Class 1+: 0.0766 (7.66%)

2. 00000742_2A_1.png
   Predicted Class: Class 1+
   Confidence: 0.8567 (85.67%)
   Probabilities:
     Class 0: 0.1433 (14.33%)
     Class 1+: 0.8567 (85.67%)
```

## Notes

- The model uses transfer learning with ImageNet pretrained weights
- GPU acceleration is automatically detected and used if available
- The training script includes comprehensive evaluation metrics
- The inference script handles errors gracefully and provides detailed output
- All random seeds are set for reproducibility