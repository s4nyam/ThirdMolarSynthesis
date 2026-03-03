#!/usr/bin/env python3
"""
Validation script for data augmentation study setup.
Checks datasets, configuration, and environment.
"""

import json
import sys
from pathlib import Path

def check_environment():
    """Check Python environment and dependencies"""
    print("="*80)
    print("ENVIRONMENT CHECK")
    print("="*80)
    
    # Python version
    import sys
    print(f"✓ Python version: {sys.version.split()[0]}")
    if sys.version_info < (3, 8):
        print("✗ ERROR: Python 3.8+ required")
        return False
    
    # PyTorch
    try:
        import torch
        print(f"✓ PyTorch version: {torch.__version__}")
        print(f"✓ CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"  Device: {torch.cuda.get_device_name(0)}")
    except ImportError:
        print("✗ ERROR: PyTorch not installed")
        return False
    
    # Other dependencies
    dependencies = {
        'torchvision': 'torchvision',
        'numpy': 'numpy',
        'pandas': 'pandas',
        'sklearn': 'scikit-learn',
        'PIL': 'Pillow',
        'tqdm': 'tqdm',
        'openpyxl': 'openpyxl',
    }
    
    for import_name, package_name in dependencies.items():
        try:
            __import__(import_name)
            print(f"✓ {package_name} installed")
        except ImportError:
            print(f"✗ ERROR: {package_name} not installed")
            return False
    
    return True

def check_configuration():
    """Check configuration file"""
    print("\n" + "="*80)
    print("CONFIGURATION CHECK")
    print("="*80)
    
    config_path = Path("config.json")
    if not config_path.exists():
        print(f"✗ ERROR: config.json not found in {Path.cwd()}")
        return False
    
    print(f"✓ Configuration file found: {config_path}")
    
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
    except json.JSONDecodeError as e:
        print(f"✗ ERROR: Invalid JSON in config.json: {e}")
        return False
    
    print(f"✓ Configuration valid JSON")
    
    # Check required fields
    required_fields = ["dataset_source1", "dataset_source2", "sample_sizes", 
                      "augmentation_percentages", "models", "output_dir"]
    for field in required_fields:
        if field not in config:
            print(f"✗ ERROR: Missing configuration field: {field}")
            return False
    
    print(f"✓ All required fields present")
    
    # Check models
    valid_models = [
        "resnet18", "resnet50", "vgg16", "mobilenet_v2", "densenet121",
        "efficientnet_b0", "inception_v3", "shufflenet_v2_x1_0",
        "squeezenet1_0", "wide_resnet50_2"
    ]
    
    for model in config["models"]:
        if model not in valid_models:
            print(f"✗ WARNING: Unknown model: {model}")
    
    print(f"✓ Models: {len(config['models'])} configured")
    
    # Print summary
    print(f"\n  Sample sizes: {config['sample_sizes']}")
    print(f"  Augmentation %: {config['augmentation_percentages']}")
    print(f"  Epochs: {config.get('epochs', 100)}")
    print(f"  Batch size: {config.get('batch_size', 32)}")
    
    return True

def check_datasets():
    """Check dataset availability and structure"""
    print("\n" + "="*80)
    print("DATASET CHECK")
    print("="*80)
    
    config_path = Path("config.json")
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    datasets = {
        'dataset_source1': config["dataset_source1"],
        'dataset_source2': config["dataset_source2"],
    }
    
    all_ok = True
    
    for name, path in datasets.items():
        print(f"\n{name}: {path}")
        dataset_path = Path(path)
        
        # Check if path exists
        if not dataset_path.exists():
            print(f"  ✗ ERROR: Path does not exist")
            all_ok = False
            continue
        
        print(f"  ✓ Path exists")
        
        # Check subdirectories
        for class_label in ['0', '1']:
            class_dir = dataset_path / class_label
            if not class_dir.exists():
                print(f"  ✗ ERROR: Missing class folder: {class_label}/")
                all_ok = False
                continue
            
            # Count images
            image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
            images = [f for f in class_dir.iterdir() if f.suffix.lower() in image_extensions]
            
            if len(images) == 0:
                print(f"  ✗ ERROR: No images in class {class_label}/")
                all_ok = False
            else:
                print(f"  ✓ Class {class_label}: {len(images)} images")
    
    return all_ok

def check_output_directory():
    """Check output directory structure"""
    print("\n" + "="*80)
    print("OUTPUT DIRECTORY CHECK")
    print("="*80)
    
    config_path = Path("config.json")
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    output_dir = Path(config["output_dir"])
    
    if not output_dir.exists():
        print(f"Output directory does not exist: {output_dir}")
        print(f"It will be created when training starts.")
        return True
    
    print(f"✓ Output directory exists: {output_dir}")
    
    # Check subdirectories
    subdirs = ['models', 'plots', 'logs']
    for subdir in subdirs:
        subdir_path = output_dir / subdir
        if subdir_path.exists():
            print(f"  ✓ {subdir}/ exists")
        else:
            print(f"  - {subdir}/ will be created")
    
    return True

def estimate_requirements():
    """Estimate disk space and time requirements"""
    print("\n" + "="*80)
    print("RESOURCE REQUIREMENTS ESTIMATE")
    print("="*80)
    
    config_path = Path("config.json")
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    num_models = len(config["models"])
    num_sample_sizes = len(config["sample_sizes"])
    num_aug_percents = len(config["augmentation_percentages"])
    
    total_experiments = num_models * num_sample_sizes * num_aug_percents
    epochs = config.get("epochs", 100)
    
    print(f"Models: {num_models}")
    print(f"Sample sizes: {num_sample_sizes}")
    print(f"Augmentation percentages: {num_aug_percents}")
    print(f"Epochs per model: {epochs}")
    print(f"\nTotal experiments: {total_experiments}")
    
    # Rough estimates
    avg_model_size = 50  # MB
    total_model_storage = (total_experiments * avg_model_size) / 1024  # GB
    
    time_per_epoch = 2  # seconds (rough estimate)
    total_time_seconds = total_experiments * epochs * time_per_epoch
    total_time_hours = total_time_seconds / 3600
    
    print(f"\nStorage estimate:")
    print(f"  Model files: ~{total_model_storage:.1f} GB")
    print(f"  Results file: ~10 MB")
    print(f"  Plots: ~50 MB")
    print(f"  Total: ~{total_model_storage + 1:.1f} GB")
    
    print(f"\nTime estimate (single GPU):")
    print(f"  Total: ~{total_time_hours:.1f} hours")
    print(f"  Per experiment: ~{total_time_seconds/total_experiments:.0f} seconds")
    
    if total_time_hours > 24:
        print(f"  ⚠ Warning: Estimated to run for {total_time_hours:.1f} hours")
        print(f"    Consider using GPU or reducing epochs for testing")

def main():
    print("\n" + "="*80)
    print("DATA AUGMENTATION STUDY - VALIDATION SCRIPT")
    print("="*80 + "\n")
    
    checks = [
        ("Environment", check_environment),
        ("Configuration", check_configuration),
        ("Datasets", check_datasets),
        ("Output Directory", check_output_directory),
    ]
    
    results = {}
    for check_name, check_func in checks:
        try:
            results[check_name] = check_func()
        except Exception as e:
            print(f"\n✗ ERROR in {check_name}: {e}")
            results[check_name] = False
    
    # Estimate requirements
    try:
        if results.get("Configuration", False):
            estimate_requirements()
    except Exception as e:
        print(f"Could not estimate requirements: {e}")
    
    # Summary
    print("\n" + "="*80)
    print("VALIDATION SUMMARY")
    print("="*80)
    
    all_ok = all(results.values())
    
    for check_name, result in results.items():
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {check_name}")
    
    print("\n" + "="*80)
    if all_ok:
        print("✓ All checks passed! Ready to start training.")
        print("\nRun: python train-bench.py")
    else:
        print("✗ Some checks failed. Fix the issues above before running training.")
    print("="*80 + "\n")
    
    return 0 if all_ok else 1

if __name__ == "__main__":
    sys.exit(main())
