#!/usr/bin/env python3
"""
Mini test script to validate the framework with a quick run.
Uses reduced settings for fast testing.
"""

import json
import sys
from pathlib import Path
import shutil

def create_test_config():
    """Create a test configuration with minimal settings"""
    test_config = {
        "dataset_source1": "/home/sj/working_dir/3m-project-128/3m-experimental-data-class0vsRest/original-dataset",
        "dataset_source2": "/home/sj/working_dir/3m-project-128/3m-experimental-data-class0vsRest/tg-manual-label-dataset",
        "sample_sizes": [300],
        "augmentation_percentages": [10, 30],
        "split_ratios": [0.7, 0.2, 0.1],
        "epochs": 10,
        "batch_size": 32,
        "learning_rate": 0.001,
        "weight_decay": 0.0001,
        "num_workers": 2,
        "models": [
            "resnet18",
            "mobilenet_v2"
        ],
        "output_dir": "/home/sj/working_dir/3m-project-128/classification-bench/test_results",
        "random_seed": 42
    }
    
    return test_config

def run_test():
    """Run a quick test"""
    print("\n" + "="*80)
    print("DATA AUGMENTATION STUDY - MINI TEST")
    print("="*80 + "\n")
    
    print("This will run a quick test with:")
    print("  - 1 sample size: 300")
    print("  - 2 augmentation percentages: 10%, 30%")
    print("  - 2 models: ResNet18, MobileNetV2")
    print("  - 10 epochs (instead of 100)")
    print("  - Results will be saved in: test_results/")
    
    response = input("\nProceed? (y/n): ").strip().lower()
    if response != 'y':
        print("Cancelled.")
        return
    
    # Create test config
    test_config = create_test_config()
    config_path = Path("test_config.json")
    
    with open(config_path, 'w') as f:
        json.dump(test_config, f, indent=2)
    
    print(f"\n✓ Test configuration saved to {config_path}")
    
    # Run the training script with test config
    print("\nStarting training script...")
    print("This is expected to take 5-10 minutes on GPU\n")
    
    try:
        # Import and run with test config
        import importlib.util
        spec = importlib.util.spec_from_file_location("train_bench", "train-bench.py")
        tb = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(tb)
        tb.CONFIG = test_config
        tb.main()
        
        print("\n" + "="*80)
        print("✓ TEST COMPLETED SUCCESSFULLY")
        print("="*80)
        print("\nResults saved in: test_results/")
        print("Check test_results/augmentation_study_results.xlsx for results")
        print("Check test_results/plots/ for visualizations")
        
    except Exception as e:
        print("\n" + "="*80)
        print("✗ TEST FAILED")
        print("="*80)
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Cleanup prompt
    print("\n" + "="*80)
    response = input("Clean up test results? (y/n): ").strip().lower()
    if response == 'y':
        test_results_path = Path("test_results")
        if test_results_path.exists():
            shutil.rmtree(test_results_path)
        if config_path.exists():
            config_path.unlink()
        print("✓ Cleaned up test files")

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == '--auto':
        # Run non-interactively
        test_config = create_test_config()
        config_path = Path("test_config.json")
        with open(config_path, 'w') as f:
            json.dump(test_config, f, indent=2)
        print("Test config created: test_config.json")
    else:
        run_test()
