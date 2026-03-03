"""
Comprehensive Data Augmentation Study Framework
This script trains multiple classification models with varying amounts of augmentation.
The test set is fixed across all experiments for each sample size.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.image as mpimg
from pathlib import Path
from datetime import datetime
from typing import List, Tuple, Dict
import json
import random
from collections import defaultdict
import logging

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms, models
from torchvision.datasets import ImageFolder
from PIL import Image
import tqdm
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    precision_recall_curve,
    roc_curve,
    auc,
)

# ============================================================================
# Configuration Loading
# ============================================================================

def load_config(config_path: str = None) -> Dict:
    """Load configuration from JSON file or use defaults"""
    default_config = {
        # Data paths - Easy to change for different experiments
        "dataset_source1": "/home/sj/working_dir/3m-project-128/3m-experimental-data-class0vsRest/original-dataset",
        "dataset_source2": "/home/sj/working_dir/3m-project-128/3m-experimental-data-class0vsRest/tg-manual-label-dataset",
        
        # Sample sizes for augmentation study
        # "sample_sizes": [300, 500, 800, 1000, 1500, 1800],
        "sample_sizes": [300],
        
        # Augmentation percentages (relative to sample size)
        # "augmentation_percentages": [10, 30, 50, 100, 200, 500],
        "augmentation_percentages": [10],
        
        # Data split ratios (train, val, test)
        "split_ratios": [0.7, 0.2, 0.1],
        
        # Training parameters
        "epochs": 2,
        "batch_size": 32,
        "learning_rate": 0.001,
        "weight_decay": 1e-4,
        "num_workers": 4,
        
        # Models to benchmark
        "models": [
            "resnet18",
            "resnet50",
            "vgg16",
            "mobilenet_v2",
            "densenet121",
            "efficientnet_b0",
            "shufflenet_v2_x1_0",
            "squeezenet1_0",
            "wide_resnet50_2",
        ],
        
        # Output settings
        "output_dir": "/home/sj/working_dir/3m-project-128/classification-bench/results",
        "random_seed": 2048,
    }
    
    # Try to load from config file
    if config_path is None:
        config_path = Path(__file__).parent / "config.json"
    
    if Path(config_path).exists():
        try:
            with open(config_path, 'r') as f:
                file_config = json.load(f)
                default_config.update(file_config)
                print(f"Loaded configuration from {config_path}")
        except Exception as e:
            print(f"Error loading config file {config_path}: {e}")
            print("Using default configuration")
    
    # Add device
    default_config["device"] = "cuda:3" if torch.cuda.is_available() else "cpu"
    
    return default_config

CONFIG = load_config()

# ============================================================================
# Utility Functions
# ============================================================================

def set_seed(seed: int):
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def create_output_dir():
    """Create output directory structure"""
    output_dir = Path(CONFIG["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    (output_dir / "models").mkdir(exist_ok=True)
    (output_dir / "plots").mkdir(exist_ok=True)
    (output_dir / "logs").mkdir(exist_ok=True)
    
    return output_dir


def _load_summary_df(summary_path: Path) -> pd.DataFrame:
    """Load Summary results from an Excel/CSV file."""
    if summary_path.suffix.lower() == ".xlsx":
        df = pd.read_excel(summary_path, sheet_name="Summary")
    else:
        df = pd.read_csv(summary_path)

    # Normalize common variants
    df = df.rename(columns={
        "Test Acc": "Test Accuracy",
    })
    return df


def _ordered_unique(values: List, preferred_order: List = None) -> List:
    """Return unique values, respecting a preferred ordering when provided."""
    seen = set()
    values_unique = []
    for v in values:
        if v not in seen:
            seen.add(v)
            values_unique.append(v)

    if not preferred_order:
        try:
            return sorted(values_unique)
        except Exception:
            return values_unique

    preferred = [v for v in preferred_order if v in seen]
    remainder = [v for v in values_unique if v not in set(preferred)]
    try:
        remainder = sorted(remainder)
    except Exception:
        pass
    return preferred + remainder


def create_accuracy_pdfs_and_mosaics_from_summary(
    summary_path: Path,
    output_dir: Path,
    logger: logging.Logger = None,
) -> None:
    """Create per-model accuracy PDFs and an all-model mosaic PDF.

    Produces two styles:
    - withgap: true numeric x-axis spacing for augmentation percentages
    - withoutgap: equidistant x-axis spacing (categorical positions)

    Output structure:
      results/plots/accuracy_pdfs/withgap/*.pdf
      results/plots/accuracy_pdfs/withoutgap/*.pdf
      results/plots/accuracy_pdfs/withgap/mosaics/_ALL_MODELS_accuracy_mosaic.pdf
      results/plots/accuracy_pdfs/withoutgap/mosaics/_ALL_MODELS_accuracy_mosaic.pdf
    """
    if logger is None:
        logger = logging.getLogger('ExperimentLogger')

    df = _load_summary_df(summary_path)
    required_cols = {"Model", "Sample Size", "Augmentation %", "Test Accuracy"}
    if not required_cols.issubset(set(df.columns)):
        logger.warning(
            f"Skipping accuracy PDF plots: summary missing columns {required_cols - set(df.columns)}"
        )
        return

    # Normalize types
    df["Model"] = df["Model"].astype(str)
    df["Sample Size"] = pd.to_numeric(df["Sample Size"], errors="coerce")
    df["Augmentation %"] = pd.to_numeric(df["Augmentation %"], errors="coerce")
    df["Test Accuracy"] = pd.to_numeric(df["Test Accuracy"], errors="coerce")
    df = df.dropna(subset=["Model", "Sample Size", "Augmentation %", "Test Accuracy"])
    if df.empty:
        logger.warning("Skipping accuracy PDF plots: summary is empty after cleaning.")
        return

    # Determine ordering
    models_in_summary = _ordered_unique(df["Model"].unique().tolist(), CONFIG.get("models"))
    sample_sizes_in_summary = sorted({int(s) for s in df["Sample Size"].unique().tolist()})
    aug_all = sorted({int(a) for a in df["Augmentation %"].unique().tolist()})

    base_out = output_dir / "plots" / "accuracy_pdfs"
    withgap_dir = base_out / "withgap"
    withoutgap_dir = base_out / "withoutgap"
    withgap_mosaic_dir = withgap_dir / "mosaics"
    withoutgap_mosaic_dir = withoutgap_dir / "mosaics"
    for d in [withgap_dir, withoutgap_dir, withgap_mosaic_dir, withoutgap_mosaic_dir]:
        d.mkdir(parents=True, exist_ok=True)

    aug_to_pos = {aug: i for i, aug in enumerate(aug_all)}

    def plot_model(ax, model_name: str, mode: str) -> None:
        data_m = df[df["Model"] == model_name]
        if data_m.empty:
            ax.set_axis_off()
            return

        for size in sorted({int(s) for s in data_m["Sample Size"].unique().tolist()}):
            d = data_m[data_m["Sample Size"] == size].sort_values("Augmentation %")
            if d.empty:
                continue
            x_aug = [int(a) for a in d["Augmentation %"].tolist()]
            y = d["Test Accuracy"].astype(float).tolist()
            if mode == "withgap":
                x = x_aug
            else:
                x = [aug_to_pos.get(a, None) for a in x_aug]
                # Filter any unmapped augs (shouldn't happen)
                xy = [(xx, yy) for xx, yy in zip(x, y) if xx is not None]
                if not xy:
                    continue
                x, y = zip(*xy)

            ax.plot(x, y, marker='o', linewidth=2, label=f"samples={int(size)}")

        ax.set_title(model_name)
        ax.set_ylabel("Test Accuracy")
        ax.grid(True, alpha=0.3)
        if mode == "withgap":
            ax.set_xlabel("Augmentation %")
            ax.set_xticks(aug_all)
        else:
            ax.set_xlabel("Augmentation %")
            ax.set_xticks(list(range(len(aug_all))))
            ax.set_xticklabels([str(a) for a in aug_all], rotation=0)
        ax.set_ylim(0.0, 1.0)
        ax.legend(fontsize=8, loc='best')

    # Individual per-model PDFs
    for model_name in models_in_summary:
        for mode, out_dir in [("withgap", withgap_dir), ("withoutgap", withoutgap_dir)]:
            fig, ax = plt.subplots(1, 1, figsize=(8.5, 5.0))
            plot_model(ax, model_name, mode)
            fig.tight_layout()
            out_path = out_dir / f"{model_name}_{mode}.pdf"
            fig.savefig(out_path, bbox_inches="tight")
            plt.close(fig)
            logger.info(f"Saved: {out_path}")

    # Mosaic PDFs (single page, grid)
    def save_mosaic(mode: str, mosaic_dir: Path) -> None:
        n = len(models_in_summary)
        if n == 0:
            return
        cols = 3
        rows = int(np.ceil(n / cols))
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 7.0, rows * 4.8))
        # Normalize axes to 2D array
        if rows == 1:
            axes = np.array([axes])
        axes_flat = axes.flatten()

        for i, model_name in enumerate(models_in_summary):
            plot_model(axes_flat[i], model_name, mode)

        # Turn off extra axes
        for j in range(n, len(axes_flat)):
            axes_flat[j].set_axis_off()

        fig.suptitle(f"Accuracy vs Augmentation — {mode}", fontsize=14)
        fig.tight_layout(rect=[0, 0.02, 1, 0.98])
        out_path = mosaic_dir / "_ALL_MODELS_accuracy_mosaic.pdf"
        fig.savefig(out_path, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Saved: {out_path}")

    save_mosaic("withgap", withgap_mosaic_dir)
    save_mosaic("withoutgap", withoutgap_mosaic_dir)


def create_pr_roc_mosaics_from_summary(summary_path: Path, output_dir: Path, logger: logging.Logger = None) -> None:
    """Create mosaics (one-row) of PR/ROC curve images per model.

    Uses the summary file to determine which models + sample sizes exist.
    It does NOT compute PR/ROC from Excel (not possible without per-sample scores);
    instead it stitches existing PNGs created during training runs.

    Mosaic rule: one output image per model, one row, columns = increasing sample sizes.
    This stitches existing PR/ROC PNGs (created during training runs). If some images
    are missing, the mosaic is still saved with placeholders.
    """
    if logger is None:
        logger = logging.getLogger('ExperimentLogger')

    df = _load_summary_df(summary_path)
    required_cols = {"Model", "Sample Size"}
    if not required_cols.issubset(set(df.columns)):
        logger.warning(f"Cannot build mosaics: summary missing columns {required_cols - set(df.columns)}")
        return

    df["Sample Size"] = pd.to_numeric(df["Sample Size"], errors="coerce")
    df = df.dropna(subset=["Model", "Sample Size"])

    # Build mosaics for ALL models in the summary
    models_in_summary = sorted(df["Model"].astype(str).unique().tolist())
    if not models_in_summary:
        logger.warning("No models found in summary; skipping PR/ROC mosaics.")
        return

    pr_dir = output_dir / "plots" / "pr_curves"
    roc_dir = output_dir / "plots" / "roc_curves"
    pr_mosaic_dir = pr_dir / "mosaics"
    roc_mosaic_dir = roc_dir / "mosaics"
    pr_mosaic_dir.mkdir(parents=True, exist_ok=True)
    roc_mosaic_dir.mkdir(parents=True, exist_ok=True)

    def make_row_mosaic(model_name: str, sizes: List[int], kind: str) -> None:
        if kind == "pr":
            base_dir = pr_dir
            out_dir = pr_mosaic_dir
            suffix = "PR"
        else:
            base_dir = roc_dir
            out_dir = roc_mosaic_dir
            suffix = "ROC"

        n = len(sizes)
        if n == 0:
            return

        fig, axes = plt.subplots(1, n, figsize=(max(4 * n, 6), 4), squeeze=False)
        axes = axes[0]

        any_found = False
        for i, size in enumerate(sizes):
            ax = axes[i]
            img_path = base_dir / f"{model_name}_samples-{int(size)}_{suffix}_curve.png"
            if img_path.exists():
                try:
                    img = mpimg.imread(str(img_path))
                    ax.imshow(img)
                    any_found = True
                except Exception:
                    ax.text(0.5, 0.5, "Failed to load\nimage", ha="center", va="center")
            else:
                ax.text(0.5, 0.5, "Missing\nimage", ha="center", va="center")

            ax.set_title(f"samples={int(size)}", fontsize=10)
            ax.axis("off")

        if not any_found:
            logger.warning(
                f"No {suffix} curve images found for {model_name}. "
                f"Mosaic will be saved with placeholders. Expected files like: "
                f"{base_dir}/{model_name}_samples-<size>_{suffix}_curve.png"
            )

        fig.suptitle(f"{model_name} — {suffix} curves across sample sizes", fontsize=12)
        fig.tight_layout()
        out_path = out_dir / f"{model_name}_{suffix}_mosaic.png"
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Saved: {out_path}")

    for model_name in models_in_summary:
        sizes = sorted({int(s) for s in df[df["Model"].astype(str) == model_name]["Sample Size"].unique().tolist()})
        make_row_mosaic(model_name, sizes, kind="pr")
        make_row_mosaic(model_name, sizes, kind="roc")

def get_all_images_from_dataset(dataset_path: str, class_label: str) -> List[str]:
    """Get all image paths from a specific class folder"""
    class_dir = Path(dataset_path) / class_label
    if not class_dir.exists():
        print(f"Warning: {class_dir} does not exist")
        return []
    
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
    images = [str(f) for f in class_dir.iterdir() 
              if f.suffix.lower() in image_extensions]
    return images

def select_samples(dataset_paths: List[str], sample_size: int, class_label: str = "0") -> List[str]:
    """
    Select samples from multiple dataset sources for a given class.
    Distribution: samples are taken from each dataset proportionally.
    """
    all_samples = []
    samples_per_source = sample_size // len(dataset_paths)
    remainder = sample_size % len(dataset_paths)
    
    selected_samples = []
    for idx, dataset_path in enumerate(dataset_paths):
        num_samples = samples_per_source + (1 if idx < remainder else 0)
        available_samples = get_all_images_from_dataset(dataset_path, class_label)
        
        if len(available_samples) < num_samples:
            print(f"Warning: Not enough samples in {dataset_path}/{class_label}. "
                  f"Available: {len(available_samples)}, Requested: {num_samples}")
            selected = available_samples
        else:
            selected = random.sample(available_samples, num_samples)
        
        selected_samples.extend(selected)
    
    return selected_samples

def prepare_data_for_experiment(base_sample_size: int, augmentation_percent: int) -> Tuple[Dict, str]:
    """
    Prepare training, validation, and test data with correct augmentation logic.
    
    Logic:
    1. Sample size (e.g., 300) = original samples from source 1
    2. Apply 70-20-10 split to original samples only:
       - Train: 70% of sample_size (e.g., 210)
       - Val: 20% of sample_size (e.g., 60)
       - Test: 10% of sample_size (e.g., 30)
    3. Augmentation %: Take this % of sample_size from source 2 and ADD to training only
       - If Aug%=10 and sample_size=300: Add 30 samples to train
       - Final train: 210 + 30 = 240
    
    Returns:
        Dict with keys: train_0, train_1, val_0, val_1, test_0, test_1
        str: log message
    """
    log_msg = ""
    
    # Get available samples
    available_base_0 = get_all_images_from_dataset(CONFIG["dataset_source1"], "0")
    available_base_1 = get_all_images_from_dataset(CONFIG["dataset_source1"], "1")
    available_aug_0 = get_all_images_from_dataset(CONFIG["dataset_source2"], "0")
    available_aug_1 = get_all_images_from_dataset(CONFIG["dataset_source2"], "1")
    
    # Check if we have enough base samples
    if len(available_base_0) < base_sample_size or len(available_base_1) < base_sample_size:
        log_msg = (f"ERROR: Not enough base samples. "
                  f"Available 0: {len(available_base_0)}, Available 1: {len(available_base_1)}, "
                  f"Requested: {base_sample_size}")
        return None, log_msg
    
    # Select base samples (will be split into train/val/test)
    base_samples_0 = random.sample(available_base_0, base_sample_size)
    base_samples_1 = random.sample(available_base_1, base_sample_size)
    
    # Calculate splits based on original sample size
    # Using only the split ratios (70-20-10) without augmentation
    train_count = int(base_sample_size * CONFIG["split_ratios"][0])  # 70%
    val_count = int(base_sample_size * CONFIG["split_ratios"][1])    # 20%
    test_count = base_sample_size - train_count - val_count          # Remaining (10%)
    
    # Split base samples into train/val/test
    base_0_indices = list(range(len(base_samples_0)))
    base_1_indices = list(range(len(base_samples_1)))
    
    random.shuffle(base_0_indices)
    random.shuffle(base_1_indices)
    
    # Class 0 splits
    train_base_0 = [base_samples_0[i] for i in base_0_indices[:train_count]]
    val_base_0 = [base_samples_0[i] for i in base_0_indices[train_count:train_count+val_count]]
    test_base_0 = [base_samples_0[i] for i in base_0_indices[train_count+val_count:]]
    
    # Class 1 splits
    train_base_1 = [base_samples_1[i] for i in base_1_indices[:train_count]]
    val_base_1 = [base_samples_1[i] for i in base_1_indices[train_count:train_count+val_count]]
    test_base_1 = [base_samples_1[i] for i in base_1_indices[train_count+val_count:]]
    
    # Calculate augmentation samples (only for training)
    aug_size = int(base_sample_size * augmentation_percent / 100)
    
    # Get augmentation samples from source 2 (only if needed)
    aug_samples_0 = []
    aug_samples_1 = []
    
    if aug_size > 0:
        if len(available_aug_0) < aug_size or len(available_aug_1) < aug_size:
            log_msg = (f"WARNING: Insufficient augmentation data. Aug%={augmentation_percent}: "
                      f"Need {aug_size} per class but have "
                      f"Class 0: {len(available_aug_0)}, Class 1: {len(available_aug_1)}")
            # Don't return None, continue with whatever we can get
            aug_samples_0 = random.sample(available_aug_0, min(aug_size, len(available_aug_0)))
            aug_samples_1 = random.sample(available_aug_1, min(aug_size, len(available_aug_1)))
        else:
            aug_samples_0 = random.sample(available_aug_0, aug_size)
            aug_samples_1 = random.sample(available_aug_1, aug_size)
    
    # Add augmentation samples to training set
    final_train_0 = train_base_0 + aug_samples_0
    final_train_1 = train_base_1 + aug_samples_1
    
    data_dict = {
        'train_0': final_train_0,
        'train_1': final_train_1,
        'val_0': val_base_0,
        'val_1': val_base_1,
        'test_0': test_base_0,
        'test_1': test_base_1,
        'train_count_base': train_count,
        'train_count_aug': len(aug_samples_0),
        'val_count': val_count,
        'test_count': test_count,
    }
    
    log_msg = (f"Data prepared: Train={len(final_train_0)}+{len(final_train_1)} "
              f"(base {train_count}+aug {len(aug_samples_0)}), "
              f"Val={val_count}, Test={test_count}")
    
    return data_dict, log_msg

# ============================================================================
# Custom Dataset Class
# ============================================================================

class CustomImageDataset(Dataset):
    """Custom dataset for loading images from file paths"""
    
    def __init__(self, image_paths: List[str], labels: List[int], transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image, self.labels[idx]

# ============================================================================
# Model Loading
# ============================================================================

def get_input_size(model_name: str) -> int:
    """Get the required input size for each model"""
    if model_name == "inception_v3":
        return 299  # Inception V3 requires 299x299
    else:
        return 224  # Most other models use 224x224

def load_model(model_name: str, num_classes: int = 2, pretrained: bool = True):
    """Load a pretrained model and adapt final layer"""
    weights = 'DEFAULT' if pretrained else None
    
    if model_name == "resnet18":
        model = models.resnet18(weights=weights)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)
    
    elif model_name == "resnet50":
        model = models.resnet50(weights=weights)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)
    
    elif model_name == "vgg16":
        model = models.vgg16(weights=weights)
        num_ftrs = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(num_ftrs, num_classes)
    
    elif model_name == "mobilenet_v2":
        model = models.mobilenet_v2(weights=weights)
        num_ftrs = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(num_ftrs, num_classes)
    
    elif model_name == "densenet121":
        model = models.densenet121(weights=weights)
        num_ftrs = model.classifier.in_features
        model.classifier = nn.Linear(num_ftrs, num_classes)
    
    elif model_name == "efficientnet_b0":
        model = models.efficientnet_b0(weights=weights)
        num_ftrs = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(num_ftrs, num_classes)
    
    elif model_name == "inception_v3":
        model = models.inception_v3(weights=weights)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)
    
    elif model_name == "shufflenet_v2_x1_0":
        model = models.shufflenet_v2_x1_0(weights=weights)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)
    
    elif model_name == "squeezenet1_0":
        model = models.squeezenet1_0(weights=weights)
        model.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=1)
    
    elif model_name == "wide_resnet50_2":
        model = models.wide_resnet50_2(weights=weights)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)
    
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    return model

# ============================================================================
# Training Function
# ============================================================================

def train_epoch(model, train_loader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    
    pbar = tqdm.tqdm(train_loader, desc="Training")
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        
        # Handle Inception V3 which returns named tuple with auxiliary outputs
        if isinstance(outputs, tuple):
            outputs = outputs[0]  # Use main output, ignore auxiliary
        
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        
        pbar.set_postfix(loss=running_loss / (pbar.n + 1))
    
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = accuracy_score(all_labels, all_preds)
    
    return epoch_loss, epoch_acc

def validate(model, val_loader, criterion, device):
    """Validate the model"""
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in tqdm.tqdm(val_loader, desc="Validation"):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            
            # Handle Inception V3 which returns named tuple with auxiliary outputs
            if isinstance(outputs, tuple):
                outputs = outputs[0]  # Use main output, ignore auxiliary
            
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    epoch_loss = running_loss / len(val_loader)
    epoch_acc = accuracy_score(all_labels, all_preds)
    
    return epoch_loss, epoch_acc, all_preds, all_labels

def test(model, test_loader, device):
    """Test the model and return detailed metrics"""
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for images, labels in tqdm.tqdm(test_loader, desc="Testing"):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            
            # Handle Inception V3 which returns named tuple with auxiliary outputs
            if isinstance(outputs, tuple):
                outputs = outputs[0]  # Use main output, ignore auxiliary
            
            # Predicted labels and probabilities for positive class
            _, preds = torch.max(outputs, 1)
            probs = torch.softmax(outputs, dim=1)[:, 1]
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'predictions': all_preds,
        'labels': all_labels,
        'probabilities': all_probs,
    }

def setup_logger(output_dir: Path) -> logging.Logger:
    """Setup logger for console and file output"""
    logger = logging.getLogger('ExperimentLogger')
    logger.setLevel(logging.DEBUG)
    
    # Create formatters
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    
    # File handler
    log_file = output_dir / "logs" / f"experiments_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    
    # Add handlers
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    
    return logger

def log_result_to_excel(result: Dict, excel_writer_obj) -> None:
    """
    Log result immediately to Excel as it's available.
    Appends new row to existing Excel file.
    """
    try:
        # Create dataframe from single result
        result_df = pd.DataFrame([{
            'Model': result['model'],
            'Sample Size': result['sample_size'],
            'Augmentation %': result['augmentation_percent'],
            'Train Samples': result['train_samples'],
            'Val Samples': result['val_samples'],
            'Test Samples': result['test_samples'],
            'Best Val Acc': f"{result['best_val_acc']:.4f}",
            'Test Accuracy': f"{result['test_accuracy']:.4f}",
            'Test Precision': f"{result['test_precision']:.4f}",
            'Test Recall': f"{result['test_recall']:.4f}",
            'Test F1-Score': f"{result['test_f1']:.4f}",
        }])
        
        # Use provided writer object to append
        if excel_writer_obj is not None:
            result_df.to_excel(excel_writer_obj, sheet_name='Summary', startrow=excel_writer_obj.sheets['Summary'].max_row, header=False, index=False)
            excel_writer_obj.sheets['Summary'].parent.save(excel_writer_obj.path)
    except Exception as e:
        print(f"Error logging to Excel: {e}")

def append_row_to_excel(excel_path: Path, row_data: Dict) -> None:
    """
    Append a single row to existing Excel file.
    Creates file if it doesn't exist.
    """
    try:
        # Check if file exists
        if excel_path.exists():
            # Read existing data
            existing_df = pd.read_excel(excel_path, sheet_name='Summary')
            # Create new row
            new_row_df = pd.DataFrame([{
                'Model': row_data['model'],
                'Sample Size': row_data['sample_size'],
                'Augmentation %': row_data['augmentation_percent'],
                'Train Samples': row_data['train_samples'],
                'Val Samples': row_data['val_samples'],
                'Test Samples': row_data['test_samples'],
                'Best Val Acc': f"{row_data['best_val_acc']:.4f}",
                'Test Accuracy': f"{row_data['test_accuracy']:.4f}",
                'Test Precision': f"{row_data['test_precision']:.4f}",
                'Test Recall': f"{row_data['test_recall']:.4f}",
                'Test F1-Score': f"{row_data['test_f1']:.4f}",
            }])
            # Append
            combined_df = pd.concat([existing_df, new_row_df], ignore_index=True)
        else:
            # Create new file
            combined_df = pd.DataFrame([{
                'Model': row_data['model'],
                'Sample Size': row_data['sample_size'],
                'Augmentation %': row_data['augmentation_percent'],
                'Train Samples': row_data['train_samples'],
                'Val Samples': row_data['val_samples'],
                'Test Samples': row_data['test_samples'],
                'Best Val Acc': f"{row_data['best_val_acc']:.4f}",
                'Test Accuracy': f"{row_data['test_accuracy']:.4f}",
                'Test Precision': f"{row_data['test_precision']:.4f}",
                'Test Recall': f"{row_data['test_recall']:.4f}",
                'Test F1-Score': f"{row_data['test_f1']:.4f}",
            }])
        
        # Write back
        combined_df.to_excel(excel_path, sheet_name='Summary', index=False)
    except Exception as e:
        print(f"Error appending row to Excel: {e}")

# ============================================================================
# Main Training Pipeline
# ============================================================================

def run_experiment(model_name: str, sample_size: int, augmentation_percent: int, 
                   output_dir: Path, device, logger, excel_path: Path) -> Dict:
    """
    Run a single experiment and log results immediately to Excel.
    
    Returns:
        Dict with all experiment results
    """
    
    logger.info(f"\n{'='*80}")
    logger.info(f"Experiment: Model={model_name}, Samples={sample_size}, Aug%={augmentation_percent}%")
    logger.info(f"{'='*80}")
    
    # Prepare training, validation, and test data
    data_dict, prep_log = prepare_data_for_experiment(sample_size, augmentation_percent)
    
    if data_dict is None:
        logger.error(f"Skipping: {prep_log}")
        return None
    
    logger.info(prep_log)
    
    # Combine all data
    train_images = data_dict['train_0'] + data_dict['train_1']
    train_labels = [0] * len(data_dict['train_0']) + [1] * len(data_dict['train_1'])
    
    val_images = data_dict['val_0'] + data_dict['val_1']
    val_labels = [0] * len(data_dict['val_0']) + [1] * len(data_dict['val_1'])
    
    test_images = data_dict['test_0'] + data_dict['test_1']
    test_labels = [0] * len(data_dict['test_0']) + [1] * len(data_dict['test_1'])
    
    # Get input size for this model
    input_size = get_input_size(model_name)
    
    # Data transforms
    train_transform = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Create datasets and dataloaders
    train_dataset = CustomImageDataset(train_images, train_labels, transform=train_transform)
    val_dataset = CustomImageDataset(val_images, val_labels, transform=val_transform)
    test_dataset = CustomImageDataset(test_images, test_labels, transform=val_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=CONFIG["batch_size"], 
                             shuffle=True, num_workers=CONFIG["num_workers"])
    val_loader = DataLoader(val_dataset, batch_size=CONFIG["batch_size"], 
                           shuffle=False, num_workers=CONFIG["num_workers"])
    test_loader = DataLoader(test_dataset, batch_size=CONFIG["batch_size"], 
                            shuffle=False, num_workers=CONFIG["num_workers"])
    
    logger.info(f"DataLoaders: Train={len(train_dataset)}, Val={len(val_dataset)}, Test={len(test_dataset)}")
    
    # Load model
    model = load_model(model_name, num_classes=2, pretrained=True)
    model = model.to(device)
    
    # Setup training
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=CONFIG["learning_rate"], 
                          weight_decay=CONFIG["weight_decay"])
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    
    # Training loop
    best_val_acc = 0.0
    best_model_state = None
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    
    for epoch in range(CONFIG["epochs"]):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, _, _ = validate(model, val_loader, criterion, device)
        scheduler.step()
        
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict().copy()
        
        if (epoch + 1) % max(1, CONFIG['epochs']//5) == 0 or epoch == 0:
            logger.info(f"Epoch {epoch+1}/{CONFIG['epochs']}: "
                      f"TrLoss={train_loss:.4f}, TrAcc={train_acc:.4f}, "
                      f"ValLoss={val_loss:.4f}, ValAcc={val_acc:.4f}")
    
    # Load best model and test
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    test_metrics = test(model, test_loader, device)
    
    # Compile results
    result = {
        'model': model_name,
        'sample_size': sample_size,
        'augmentation_percent': augmentation_percent,
        'train_samples': len(train_images),
        'val_samples': len(val_images),
        'test_samples': len(test_images),
        'best_val_acc': best_val_acc,
        'test_accuracy': test_metrics['accuracy'],
        'test_precision': test_metrics['precision'],
        'test_recall': test_metrics['recall'],
        'test_f1': test_metrics['f1'],
        'predictions': test_metrics['predictions'],
        'labels': test_metrics['labels'],
        'probabilities': test_metrics['probabilities'],
    }
    
    logger.info(f"Results: ValAcc={best_val_acc:.4f}, TestAcc={result['test_accuracy']:.4f}, "
               f"TestF1={result['test_f1']:.4f}")
    
    # Log to Excel immediately
    append_row_to_excel(excel_path, result)
    logger.info(f"Result logged to Excel: {excel_path}")
    
    return result

def main():
    """Main execution function"""
    
    print("Starting Data Augmentation Study...")
    print(f"Device: {CONFIG['device']}")
    print(f"Models: {len(CONFIG['models'])} models")
    print(f"Sample sizes: {CONFIG['sample_sizes']}")
    print(f"Augmentation percentages: {CONFIG['augmentation_percentages']}")
    
    # Set seed
    set_seed(CONFIG["random_seed"])
    
    # Create output directory
    output_dir = create_output_dir()
    print(f"Output directory: {output_dir}")
    
    # Setup logger
    logger = setup_logger(output_dir)
    logger.info("="*80)
    logger.info("Data Augmentation Study Started")
    logger.info(f"Device: {CONFIG['device']}")
    logger.info(f"Models: {len(CONFIG['models'])} models")
    logger.info(f"Sample sizes: {CONFIG['sample_sizes']}")
    logger.info(f"Augmentation percentages: {CONFIG['augmentation_percentages']}")
    logger.info("="*80)
    
    # Results paths
    excel_path = output_dir / "augmentation_study_results.xlsx"
    csv_path = output_dir / "augmentation_study_results.csv"

    # Helper: check if a summary file exists and is usable for plotting
    def summary_file_ready(path: Path) -> bool:
        if not path.exists():
            return False
        try:
            if path.suffix.lower() == ".xlsx":
                df_summary = pd.read_excel(path, sheet_name='Summary')
            else:
                df_summary = pd.read_csv(path)
        except Exception:
            return False
        required = {"Model", "Sample Size", "Augmentation %", "Test Accuracy"}
        return required.issubset(set(df_summary.columns)) and len(df_summary) > 0

    # If a summary file already exists, skip training and only generate plots
    if summary_file_ready(excel_path) or summary_file_ready(csv_path):
        existing_path = excel_path if excel_path.exists() else csv_path
        logger.info(f"Found existing results file: {existing_path}. Skipping training and generating plots.")
        print(f"Found existing results file: {existing_path}. Skipping training and generating plots.")
        try:
            create_plots_from_summary(existing_path, output_dir, logger)
            # Re-create PDF accuracy plots (withgap/withoutgap) + mosaic PDFs
            create_accuracy_pdfs_and_mosaics_from_summary(existing_path, output_dir, logger)
            # Also try to generate PR/ROC mosaics from existing PR/ROC curve images
            create_pr_roc_mosaics_from_summary(existing_path, output_dir, logger)
        except Exception as e:
            logger.error(f"Error creating plots from summary: {e}")
        print("Done.")
        return

    # Initialize Excel file with headers if not present
    if not excel_path.exists():
        headers_df = pd.DataFrame(columns=[
            'Model', 'Sample Size', 'Augmentation %', 'Train Samples', 'Val Samples', 'Test Samples',
            'Best Val Acc', 'Test Accuracy', 'Test Precision', 'Test Recall', 'Test F1-Score'
        ])
        headers_df.to_excel(excel_path, sheet_name='Summary', index=False)
        logger.info(f"Created Excel file: {excel_path}")
    
    # Run experiments
    all_results = []
    total_experiments = len(CONFIG["models"]) * len(CONFIG["sample_sizes"]) * len(CONFIG["augmentation_percentages"])
    completed = 0
    
    for sample_size in CONFIG["sample_sizes"]:
        for aug_percent in CONFIG["augmentation_percentages"]:
            for model_name in CONFIG["models"]:
                try:
                    result = run_experiment(
                        model_name, sample_size, aug_percent,
                        output_dir, CONFIG["device"], logger, excel_path
                    )
                    
                    if result is not None:
                        all_results.append(result)
                    
                    completed += 1
                    logger.info(f"Progress: {completed}/{total_experiments}")
                
                except Exception as e:
                    logger.error(f"Error in experiment: {e}")
                    import traceback
                    logger.error(traceback.format_exc())
    
    # Create plots
    if all_results:
        logger.info("Creating plots...")
        try:
            create_plots(all_results, output_dir, logger)
            # Also create accuracy PDF plots + mosaics based on the Excel summary we just wrote
            create_accuracy_pdfs_and_mosaics_from_summary(excel_path, output_dir, logger)
        except Exception as e:
            logger.error(f"Error creating plots: {e}")
    
    logger.info("="*80)
    logger.info("Experiment completed!")
    logger.info(f"Total experiments: {completed}")
    logger.info(f"Results saved to {excel_path}")
    logger.info(f"Logs saved to {output_dir / 'logs'}")
    logger.info("="*80)
    
    print("\nExperiment completed!")
    print(f"Results saved to {excel_path}")
    print(f"Logs saved to {output_dir / 'logs'}")

def create_plots(results: List[Dict], output_dir: Path, logger: logging.Logger = None):
    """Create visualization plots"""
    
    if logger is None:
        logger = logging.getLogger('ExperimentLogger')
    
    df = pd.DataFrame([
        {
            'Model': r['model'],
            'Sample Size': r['sample_size'],
            'Augmentation %': r['augmentation_percent'],
            'Test Accuracy': float(r['test_accuracy']),
            'Test F1-Score': float(r['test_f1']),
        }
        for r in results
    ])
    
    # Plot 1: Accuracy vs Augmentation % for each sample size
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    
    for idx, sample_size in enumerate(sorted(df['Sample Size'].unique())):
        ax = axes[idx]
        data = df[df['Sample Size'] == sample_size]
        
        for model in data['Model'].unique():
            model_data = data[data['Model'] == model].sort_values('Augmentation %')
            ax.plot(model_data['Augmentation %'], model_data['Test Accuracy'], 
                   marker='o', label=model, linewidth=2)
        
        ax.set_xlabel('Augmentation Percentage (%)')
        ax.set_ylabel('Test Accuracy')
        ax.set_title(f'Sample Size: {sample_size}')
        ax.legend(fontsize=8, loc='best')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_path = output_dir / "plots" / "accuracy_vs_augmentation.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    logger.info(f"Saved: {plot_path}")
    plt.close()

    # Plot 4: Precision-Recall curves per Model/Sample Size across augmentation percentages
    pr_dir = output_dir / "plots" / "pr_curves"
    pr_dir.mkdir(parents=True, exist_ok=True)

    # Build PR curves using raw probabilities from results
    models = sorted({r['model'] for r in results})
    for model in models:
        sizes = sorted({r['sample_size'] for r in results if r['model'] == model})
        for size in sizes:
            entries = [r for r in results if r['model'] == model and r['sample_size'] == size]
            if not entries:
                continue
            # sort by augmentation percent for consistent legend order
            entries = sorted(entries, key=lambda x: x['augmentation_percent'])

            plt.figure(figsize=(7.5, 5))
            for r in entries:
                y_true = r.get('labels')
                y_scores = r.get('probabilities')
                aug = r.get('augmentation_percent')
                if not y_true or not y_scores:
                    continue
                try:
                    precision, recall, _ = precision_recall_curve(y_true, y_scores)
                except Exception:
                    continue
                # Plot PR curve (recall on x-axis, precision on y-axis)
                plt.plot(recall, precision, linewidth=1.8, label=f"Aug {aug}% (n={len(y_true)})")

            plt.title(f"{model} — PR Curve (samples={size})")
            plt.xlabel("Recall")
            plt.ylabel("Precision")
            plt.xlim(0.0, 1.0)
            plt.ylim(0.0, 1.0)
            plt.grid(True, linestyle=":", linewidth=0.7, alpha=0.6)
            plt.legend(title="Augmentation %", fontsize=8)
            plt.tight_layout()

            pr_out = pr_dir / f"{model}_samples-{size}_PR_curve.png"
            plt.savefig(pr_out, dpi=300, bbox_inches='tight')
            plt.close()
            logger.info(f"Saved: {pr_out}")

    # Plot 5: ROC curves per Model/Sample Size across augmentation percentages
    roc_dir = output_dir / "plots" / "roc_curves"
    roc_dir.mkdir(parents=True, exist_ok=True)

    for model in models:
        sizes = sorted({r['sample_size'] for r in results if r['model'] == model})
        for size in sizes:
            entries = [r for r in results if r['model'] == model and r['sample_size'] == size]
            if not entries:
                continue
            # sort by augmentation percent for consistent legend order
            entries = sorted(entries, key=lambda x: x['augmentation_percent'])

            plt.figure(figsize=(7.5, 5))
            for r in entries:
                y_true = r.get('labels')
                y_scores = r.get('probabilities')
                aug = r.get('augmentation_percent')
                if not y_true or not y_scores:
                    continue
                try:
                    fpr, tpr, _ = roc_curve(y_true, y_scores)
                    roc_auc = auc(fpr, tpr)
                except Exception:
                    continue
                plt.plot(fpr, tpr, linewidth=1.8, label=f"Aug {aug}% (AUC={roc_auc:.3f})")

            # Diagonal baseline
            plt.plot([0, 1], [0, 1], linestyle='--', color='gray', linewidth=1)
            plt.title(f"{model} — ROC Curve (samples={size})")
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            plt.xlim(0.0, 1.0)
            plt.ylim(0.0, 1.0)
            plt.grid(True, linestyle=":", linewidth=0.7, alpha=0.6)
            plt.legend(title="Augmentation %", fontsize=8)
            plt.tight_layout()

            roc_out = roc_dir / f"{model}_samples-{size}_ROC_curve.png"
            plt.savefig(roc_out, dpi=300, bbox_inches='tight')
            plt.close()
            logger.info(f"Saved: {roc_out}")
    
    # Plot 2: Heatmap of accuracy across sample sizes and models
    fig, axes = plt.subplots(1, len(df['Augmentation %'].unique()), 
                            figsize=(20, 5))
    
    for idx, aug_percent in enumerate(sorted(df['Augmentation %'].unique())):
        data = df[df['Augmentation %'] == aug_percent]
        pivot = data.pivot_table(values='Test Accuracy', 
                                index='Model', 
                                columns='Sample Size')
        
        sns.heatmap(pivot, annot=True, fmt='.3f', cmap='RdYlGn', 
                   ax=axes[idx], vmin=0.5, vmax=1.0, cbar=True)
        axes[idx].set_title(f'Augmentation: {aug_percent}%')
    
    plt.tight_layout()
    plot_path = output_dir / "plots" / "accuracy_heatmap.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    logger.info(f"Saved: {plot_path}")
    plt.close()
    
    # Plot 3: F1-Score vs Augmentation %
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    
    for idx, sample_size in enumerate(sorted(df['Sample Size'].unique())):
        ax = axes[idx]
        data = df[df['Sample Size'] == sample_size]
        
        for model in data['Model'].unique():
            model_data = data[data['Model'] == model].sort_values('Augmentation %')
            ax.plot(model_data['Augmentation %'], model_data['Test F1-Score'], 
                   marker='s', label=model, linewidth=2)
        
        ax.set_xlabel('Augmentation Percentage (%)')
        ax.set_ylabel('Test F1-Score')
        ax.set_title(f'Sample Size: {sample_size}')
        ax.legend(fontsize=8, loc='best')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_path = output_dir / "plots" / "f1_vs_augmentation.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    logger.info(f"Saved: {plot_path}")
    plt.close()

def create_plots_from_summary(summary_path: Path, output_dir: Path, logger: logging.Logger = None):
    """Generate plots directly from existing summary Excel/CSV (skip training).
    Produces Plot 1, 2, 3 as PNG. PR/ROC curves require per-sample probabilities and
    are therefore generated only during training runs.
    """
    if logger is None:
        logger = logging.getLogger('ExperimentLogger')

    # Load summary file
    if summary_path.suffix.lower() == ".xlsx":
        df = pd.read_excel(summary_path, sheet_name='Summary')
    else:
        df = pd.read_csv(summary_path)

    # Normalize columns
    df = df.rename(columns={
        'Test Acc': 'Test Accuracy',
    })
    required_cols = ['Model', 'Sample Size', 'Augmentation %', 'Test Accuracy']
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in summary: {missing}")

    if 'Test F1-Score' not in df.columns:
        df['Test F1-Score'] = float('nan')

    # Convert types
    df['Sample Size'] = pd.to_numeric(df['Sample Size'], errors='coerce')
    df['Augmentation %'] = pd.to_numeric(df['Augmentation %'], errors='coerce')
    df['Test Accuracy'] = pd.to_numeric(df['Test Accuracy'], errors='coerce')
    df['Test F1-Score'] = pd.to_numeric(df['Test F1-Score'], errors='coerce')
    df = df.dropna(subset=['Model', 'Sample Size', 'Augmentation %', 'Test Accuracy'])

    # Plot 1: Accuracy vs Augmentation %
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    for idx, sample_size in enumerate(sorted(df['Sample Size'].unique())):
        ax = axes[idx]
        data = df[df['Sample Size'] == sample_size]
        for model in sorted(data['Model'].unique()):
            model_data = data[data['Model'] == model].sort_values('Augmentation %')
            ax.plot(model_data['Augmentation %'], model_data['Test Accuracy'], marker='o', label=model, linewidth=2)
        ax.set_xlabel('Augmentation Percentage (%)')
        ax.set_ylabel('Test Accuracy')
        ax.set_title(f'Sample Size: {sample_size}')
        ax.legend(fontsize=8, loc='best')
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plot_path = output_dir / 'plots' / 'accuracy_vs_augmentation.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    logger.info(f"Saved: {plot_path}")
    plt.close()

    # Plot 2: Heatmap of accuracy
    fig, axes = plt.subplots(1, len(sorted(df['Augmentation %'].unique())), figsize=(20, 5))
    for idx, aug_percent in enumerate(sorted(df['Augmentation %'].unique())):
        data = df[df['Augmentation %'] == aug_percent]
        pivot = data.pivot_table(values='Test Accuracy', index='Model', columns='Sample Size')
        sns.heatmap(pivot, annot=True, fmt='.3f', cmap='RdYlGn', ax=axes[idx], vmin=0.5, vmax=1.0, cbar=True)
        axes[idx].set_title(f'Augmentation: {aug_percent}%')
    plt.tight_layout()
    plot_path = output_dir / 'plots' / 'accuracy_heatmap.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    logger.info(f"Saved: {plot_path}")
    plt.close()

    # Plot 3: F1-Score vs Augmentation %
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    for idx, sample_size in enumerate(sorted(df['Sample Size'].unique())):
        ax = axes[idx]
        data = df[df['Sample Size'] == sample_size]
        for model in sorted(data['Model'].unique()):
            model_data = data[data['Model'] == model].sort_values('Augmentation %')
            ax.plot(model_data['Augmentation %'], model_data['Test F1-Score'], marker='s', label=model, linewidth=2)
        ax.set_xlabel('Augmentation Percentage (%)')
        ax.set_ylabel('Test F1-Score')
        ax.set_title(f'Sample Size: {sample_size}')
        ax.legend(fontsize=8, loc='best')
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plot_path = output_dir / 'plots' / 'f1_vs_augmentation.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    logger.info(f"Saved: {plot_path}")
    plt.close()

if __name__ == "__main__":
    main()
