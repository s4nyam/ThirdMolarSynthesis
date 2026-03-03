#!/usr/bin/env python3
"""
Inference script for dental binary classification model
Loads trained ResNet18 model and performs inference on new images
"""

import os
import argparse
import shutil
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
import numpy as np

class DentalClassifier:
    """Dental binary classification inference class"""
    
    def __init__(self, model_path, device=None):
        """
        Initialize the classifier
        
        Args:
            model_path (str): Path to the trained model file
            device (str): Device to run inference on ('cpu', 'cuda', or None for auto)
        """
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Load model
        self.model = self._load_model(model_path)
        
        # Define transform
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Class names
        self.class_names = ['Class 0', 'Class 1+']
    
    def _load_model(self, model_path):
        """Load the trained model"""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        # Create model architecture
        model = models.resnet18(pretrained=False)
        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features, 2)  # Binary classification
        
        # Load trained weights
        checkpoint = torch.load(model_path, map_location=self.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # Set to evaluation mode
        model.eval()
        model = model.to(self.device)
        
        print(f"Model loaded from: {model_path}")
        print(f"Model validation accuracy: {checkpoint.get('val_acc', 'N/A'):.2f}%")
        
        return model
    
    def predict_single(self, image_path):
        """
        Predict class for a single image
        
        Args:
            image_path (str): Path to the image file
            
        Returns:
            dict: Dictionary containing prediction results
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")
        
        # Load and preprocess image
        image = Image.open(image_path).convert('RGB')
        input_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        # Perform inference
        with torch.no_grad():
            outputs = self.model(input_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            predicted_class = torch.argmax(outputs, dim=1).item()
            confidence = probabilities[0][predicted_class].item()
        
        # Prepare results
        results = {
            'image_path': image_path,
            'predicted_class': predicted_class,
            'predicted_class_name': self.class_names[predicted_class],
            'confidence': confidence,
            'probabilities': {
                'Class 0': probabilities[0][0].item(),
                'Class 1+': probabilities[0][1].item()
            }
        }
        
        return results
    
    def predict_batch(self, image_paths):
        """
        Predict classes for multiple images
        
        Args:
            image_paths (list): List of image file paths
            
        Returns:
            list: List of prediction results for each image
        """
        results = []
        
        for image_path in image_paths:
            try:
                result = self.predict_single(image_path)
                results.append(result)
            except Exception as e:
                print(f"Error processing {image_path}: {str(e)}")
                results.append({
                    'image_path': image_path,
                    'error': str(e)
                })
        
        return results
    
    def predict_directory(self, directory_path, extensions=('.png', '.jpg', '.jpeg')):
        """
        Predict classes for all images in a directory
        
        Args:
            directory_path (str): Path to the directory containing images
            extensions (tuple): Tuple of valid image extensions
            
        Returns:
            list: List of prediction results for each image
        """
        if not os.path.exists(directory_path):
            raise FileNotFoundError(f"Directory not found: {directory_path}")
        
        # Get all image files
        image_paths = []
        for filename in os.listdir(directory_path):
            if filename.lower().endswith(extensions):
                image_paths.append(os.path.join(directory_path, filename))
        
        if not image_paths:
            print(f"No images found in {directory_path}")
            return []
        
        print(f"Found {len(image_paths)} images in {directory_path}")
        
        # Predict for all images
        return self.predict_batch(image_paths)

def print_results(results):
    """Print prediction results in a formatted way"""
    if not results:
        print("No results to display")
        return
    
    print("\n" + "="*80)
    print("PREDICTION RESULTS")
    print("="*80)
    
    for i, result in enumerate(results, 1):
        if 'error' in result:
            print(f"\n{i}. {os.path.basename(result['image_path'])}")
            print(f"   ERROR: {result['error']}")
            continue
        
        print(f"\n{i}. {os.path.basename(result['image_path'])}")
        print(f"   Predicted Class: {result['predicted_class_name']}")
        print(f"   Confidence: {result['confidence']:.4f} ({result['confidence']*100:.2f}%)")
        print(f"   Probabilities:")
        for class_name, prob in result['probabilities'].items():
            print(f"     {class_name}: {prob:.4f} ({prob*100:.2f}%)")

def save_results_csv(results, output_path):
    """Save results to CSV file"""
    import csv
    
    with open(output_path, 'w', newline='') as csvfile:
        fieldnames = ['image_path', 'filename', 'predicted_class', 'predicted_class_name', 
                     'confidence', 'class_0_prob', 'class_1_prob', 'error']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        for result in results:
            if 'error' in result:
                writer.writerow({
                    'image_path': result['image_path'],
                    'filename': os.path.basename(result['image_path']),
                    'error': result['error']
                })
            else:
                writer.writerow({
                    'image_path': result['image_path'],
                    'filename': os.path.basename(result['image_path']),
                    'predicted_class': result['predicted_class'],
                    'predicted_class_name': result['predicted_class_name'],
                    'confidence': result['confidence'],
                    'class_0_prob': result['probabilities']['Class 0'],
                    'class_1_prob': result['probabilities']['Class 1+']
                })
    
    print(f"\nResults saved to: {output_path}")

def organize_images_by_prediction(results, output_dir, copy_files=True):
    """
    Organize images into directories based on their predicted classes
    
    Args:
        results (list): List of prediction results
        output_dir (str): Base directory to create class folders in
        copy_files (bool): If True, copy files; if False, move files
    
    Returns:
        dict: Statistics about the organization process
    """
    if not results:
        print("No results to organize")
        return {}
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Create class directories
    class_0_dir = os.path.join(output_dir, "class_0")
    class_1_dir = os.path.join(output_dir, "class_1+")
    
    os.makedirs(class_0_dir, exist_ok=True)
    os.makedirs(class_1_dir, exist_ok=True)
    
    # Statistics
    stats = {
        'total_processed': 0,
        'class_0_count': 0,
        'class_1_count': 0,
        'errors': 0,
        'error_files': []
    }
    
    operation = "Copying" if copy_files else "Moving"
    print(f"\n{operation} images to class directories...")
    print(f"Output directory: {output_dir}")
    
    for result in results:
        if 'error' in result:
            stats['errors'] += 1
            stats['error_files'].append(result['image_path'])
            continue
        
        try:
            source_path = result['image_path']
            filename = os.path.basename(source_path)
            predicted_class = result['predicted_class']
            
            # Determine destination directory
            if predicted_class == 0:
                dest_dir = class_0_dir
                stats['class_0_count'] += 1
            else:
                dest_dir = class_1_dir
                stats['class_1_count'] += 1
            
            dest_path = os.path.join(dest_dir, filename)
            
            # Handle filename conflicts
            counter = 1
            original_dest_path = dest_path
            while os.path.exists(dest_path):
                name, ext = os.path.splitext(filename)
                new_filename = f"{name}_{counter}{ext}"
                dest_path = os.path.join(dest_dir, new_filename)
                counter += 1
            
            # Copy or move the file
            if copy_files:
                shutil.copy2(source_path, dest_path)
            else:
                shutil.move(source_path, dest_path)
            
            stats['total_processed'] += 1
            
            if dest_path != original_dest_path:
                print(f"  {operation}: {filename} -> {os.path.basename(dest_path)} (renamed due to conflict)")
            
        except Exception as e:
            print(f"Error {operation.lower()} {result['image_path']}: {str(e)}")
            stats['errors'] += 1
            stats['error_files'].append(result['image_path'])
    
    # Print organization summary
    print(f"\n{operation} completed!")
    print(f"Total images processed: {stats['total_processed']}")
    print(f"Class 0 images: {stats['class_0_count']} -> {class_0_dir}")
    print(f"Class 1+ images: {stats['class_1_count']} -> {class_1_dir}")
    if stats['errors'] > 0:
        print(f"Errors: {stats['errors']}")
    
    return stats

def main():
    parser = argparse.ArgumentParser(description='Dental Binary Classification Inference')
    parser.add_argument('--model', '-m', required=True, 
                       help='Path to the trained model file')
    parser.add_argument('--image', '-i', 
                       help='Path to a single image file')
    parser.add_argument('--directory', '-d', 
                       help='Path to directory containing images')
    parser.add_argument('--output', '-o', 
                       help='Path to save CSV results (optional)')
    parser.add_argument('--organize', 
                       help='Base directory to organize images by prediction class')
    parser.add_argument('--move', action='store_true',
                       help='Move files instead of copying when organizing (use with --organize)')
    parser.add_argument('--device', 
                       choices=['cpu', 'cuda'], 
                       help='Device to use for inference (default: auto)')
    
    args = parser.parse_args()
    
    # Validate arguments
    if not args.image and not args.directory:
        parser.error("Must specify either --image or --directory")
    
    if args.image and args.directory:
        parser.error("Cannot specify both --image and --directory")
    
    try:
        # Initialize classifier
        classifier = DentalClassifier(args.model, args.device)
        
        # Perform inference
        if args.image:
            print(f"Performing inference on single image: {args.image}")
            results = [classifier.predict_single(args.image)]
        else:
            print(f"Performing inference on directory: {args.directory}")
            results = classifier.predict_directory(args.directory)
        
        # Display results
        print_results(results)
        
        # Save results if requested
        if args.output:
            save_results_csv(results, args.output)
        
        # Organize images by prediction if requested
        if args.organize:
            organize_images_by_prediction(results, args.organize, copy_files=not args.move)
        
        # Summary statistics
        if len(results) > 1:
            valid_results = [r for r in results if 'error' not in r]
            if valid_results:
                class_0_count = sum(1 for r in valid_results if r['predicted_class'] == 0)
                class_1_count = sum(1 for r in valid_results if r['predicted_class'] == 1)
                avg_confidence = np.mean([r['confidence'] for r in valid_results])
                
                print(f"\n" + "="*80)
                print("SUMMARY")
                print("="*80)
                print(f"Total images processed: {len(valid_results)}")
                print(f"Class 0 predictions: {class_0_count}")
                print(f"Class 1+ predictions: {class_1_count}")
                print(f"Average confidence: {avg_confidence:.4f} ({avg_confidence*100:.2f}%)")
                if len(results) > len(valid_results):
                    print(f"Errors: {len(results) - len(valid_results)}")
    
    except Exception as e:
        print(f"Error: {str(e)}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
