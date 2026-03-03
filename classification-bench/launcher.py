#!/usr/bin/env python3
"""
Quick launcher script for data augmentation study experiments.
Allows easy customization of dataset paths and parameters.
"""

import json
import argparse
from pathlib import Path
import sys

def create_custom_config(output_file: str = "config.json"):
    """Create a custom configuration file interactively"""
    print("\n" + "="*80)
    print("DATA AUGMENTATION STUDY - Configuration Setup")
    print("="*80 + "\n")
    
    config = {}
    
    # Dataset paths
    print("DATASET CONFIGURATION")
    print("-" * 80)
    config["dataset_source1"] = input(
        "Enter path for primary dataset (original): "
    ).strip()
    config["dataset_source2"] = input(
        "Enter path for augmentation dataset: "
    ).strip()
    
    # Sample sizes
    print("\nSAMPLE SIZES")
    print("-" * 80)
    sample_input = input(
        "Enter sample sizes (comma-separated) [default: 300,500,800,1000,1500,1800]: "
    ).strip()
    if sample_input:
        config["sample_sizes"] = [int(x.strip()) for x in sample_input.split(",")]
    else:
        config["sample_sizes"] = [300, 500, 800, 1000, 1500, 1800]
    
    # Augmentation percentages
    print("\nAUGMENTATION PERCENTAGES")
    print("-" * 80)
    aug_input = input(
        "Enter augmentation percentages (comma-separated) [default: 10,30,50,100,200,500]: "
    ).strip()
    if aug_input:
        config["augmentation_percentages"] = [int(x.strip()) for x in aug_input.split(",")]
    else:
        config["augmentation_percentages"] = [10, 30, 50, 100, 200, 500]
    
    # Training parameters
    print("\nTRAINING PARAMETERS")
    print("-" * 80)
    config["epochs"] = int(input("Number of epochs [default: 100]: ").strip() or "100")
    config["batch_size"] = int(input("Batch size [default: 32]: ").strip() or "32")
    config["learning_rate"] = float(input("Learning rate [default: 0.001]: ").strip() or "0.001")
    config["weight_decay"] = float(input("Weight decay [default: 0.0001]: ").strip() or "0.0001")
    
    # Models
    print("\nMODELS")
    print("-" * 80)
    print("Available models: resnet18, resnet50, vgg16, mobilenet_v2, densenet121,")
    print("                 efficientnet_b0, inception_v3, shufflenet_v2_x1_0,")
    print("                 squeezenet1_0, wide_resnet50_2")
    models_input = input(
        "Enter models to use (comma-separated) [default: all]: "
    ).strip()
    if models_input:
        config["models"] = [x.strip() for x in models_input.split(",")]
    else:
        config["models"] = [
            "resnet18", "resnet50", "vgg16", "mobilenet_v2", "densenet121",
            "efficientnet_b0", "inception_v3", "shufflenet_v2_x1_0",
            "squeezenet1_0", "wide_resnet50_2"
        ]
    
    # Output directory
    print("\nOUTPUT SETTINGS")
    print("-" * 80)
    config["output_dir"] = input(
        "Output directory [default: ./results]: "
    ).strip() or "./results"
    config["random_seed"] = int(input("Random seed [default: 42]: ").strip() or "42")
    
    # Save configuration
    with open(output_file, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"\nConfiguration saved to {output_file}")
    return config

def print_config(config_file: str = "config.json"):
    """Print current configuration"""
    try:
        with open(config_file, 'r') as f:
            config = json.load(f)
        
        print("\n" + "="*80)
        print("CURRENT CONFIGURATION")
        print("="*80)
        print(json.dumps(config, indent=2))
    except FileNotFoundError:
        print(f"Configuration file {config_file} not found.")

def main():
    parser = argparse.ArgumentParser(
        description="Data Augmentation Study - Configuration and Launcher"
    )
    parser.add_argument(
        "--setup", action="store_true",
        help="Interactive setup to create custom configuration"
    )
    parser.add_argument(
        "--config", type=str, default="config.json",
        help="Configuration file to use"
    )
    parser.add_argument(
        "--show", action="store_true",
        help="Show current configuration"
    )
    parser.add_argument(
        "--run", action="store_true",
        help="Run the training script with current configuration"
    )
    
    args = parser.parse_args()
    
    if args.setup:
        create_custom_config(args.config)
    
    if args.show:
        print_config(args.config)
    
    if args.run:
        print(f"Running training with configuration: {args.config}")
        import subprocess
        result = subprocess.run(
            [sys.executable, "train-bench.py"],
            env={**dict(os.environ), "CONFIG_FILE": args.config}
        )
        sys.exit(result.returncode)
    
    if not any([args.setup, args.show, args.run]):
        # Default: show config and prompt for action
        print_config(args.config)
        print("\nUsage:")
        print("  python launcher.py --setup          # Interactive configuration")
        print("  python launcher.py --show           # Show current configuration")
        print("  python launcher.py --run            # Run training")

if __name__ == "__main__":
    import os
    main()
