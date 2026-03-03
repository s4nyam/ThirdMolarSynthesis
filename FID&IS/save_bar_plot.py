from pathlib import Path
import re
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def count_ls_entries(dir_path: Path) -> int:
    """Mimic `ls | wc -l`: count all non-hidden entries in a dir.

    This counts files, symlinks (including image shortcuts), and subdirectories,
    ignoring only entries starting with a dot, just like a plain `ls`.
    """
    if not dir_path.exists() or not dir_path.is_dir():
        return 0
    return sum(1 for p in dir_path.iterdir() if not p.name.startswith("."))


def count_diff_per_class(diff_dir: Path, num_classes: int = 7) -> List[int]:
    """Count Diff images per class using filename prefix like '0_...png'."""
    counts = [0] * num_classes
    if not diff_dir.exists() or not diff_dir.is_dir():
        return counts

    pattern = re.compile(r"^(\d+)_")

    for p in diff_dir.iterdir():
        if not p.is_file() or p.suffix.lower() not in IMAGE_EXTS:
            continue
        m = pattern.match(p.name)
        if not m:
            continue
        cls = int(m.group(1))
        if 0 <= cls < num_classes:
            counts[cls] += 1
    return counts


def count_cdiff_per_class(base_dir: Path, num_classes: int = 7) -> List[int]:
    """Count cDiff images per class using folders 3m-cDiff-128-class-<k>."""
    counts = [0] * num_classes
    for cls in range(num_classes):
        dir_path = base_dir / f"3m-cDiff-128-class-{cls}"
        counts[cls] = count_ls_entries(dir_path)
    return counts


def count_cgan_per_class(base_dir: Path, num_classes: int = 7) -> List[int]:
    """Count cGAN images per class using folders 3m-cGAN-128-class-<k>."""
    counts = [0] * num_classes
    for cls in range(num_classes):
        dir_path = base_dir / f"3m-cGAN-128-class-{cls}"
        counts[cls] = count_ls_entries(dir_path)
    return counts


def count_training_c_per_class(base_dir: Path, num_classes: int = 7) -> List[int]:
    """Count training-c images per class using folders 3m-training-c-128-class-<k>."""
    counts = [0] * num_classes
    for cls in range(num_classes):
        dir_path = base_dir / f"3m-training-c-128-class-{cls}"
        counts[cls] = count_ls_entries(dir_path)
    return counts


def build_and_save_plot(base_dir: Path, output_name: str = "class_distribution_barplot.pdf") -> None:
    num_classes = 7

    # Class-wise counts for classes 0-6
    cdiff_counts = count_cdiff_per_class(base_dir, num_classes=num_classes)
    cgan_counts = count_cgan_per_class(base_dir, num_classes=num_classes)
    training_c_counts = count_training_c_per_class(base_dir, num_classes=num_classes)

    # Unconditional totals for Diff, GAN and training
    diff_total = count_ls_entries(base_dir / "3m-Diff-128")
    gan_total = count_ls_entries(base_dir / "3m-GAN-128")
    training_total = count_ls_entries(base_dir / "3m-training-128")

    # We will have positions: 0..6 classes, and 7 for 'uc'
    num_positions = num_classes + 1
    x_positions = np.arange(num_positions, dtype=float)

    # Merge conditional (per class) and unconditional (uc) for each pair
    cdiff_diff = cdiff_counts + [diff_total]
    cgan_gan = cgan_counts + [gan_total]
    trainingc_training = training_c_counts + [training_total]

    # X-axis labels: 0c, 1c, ..., 6c, uc
    class_labels = [f"{i}c" for i in range(num_classes)] + ["uc"]

    # Total relative width of all bars within one x-position (0<group_width<=1)
    group_width = 0.8

    fig, ax = plt.subplots(figsize=(10, 4))

    # Multi-bar for each position: merged conditional/unconditional pairs
    sources = [
        ("cDiff / Diff", cdiff_diff),
        ("cGAN / GAN", cgan_gan),
        ("training-c / training", trainingc_training),
    ]

    num_sources = len(sources)
    # Individual bar width so that all bars together use `group_width` of the space
    width = group_width / num_sources
    offsets = [(i - (num_sources - 1) / 2) * width for i in range(num_sources)]

    for offset, (label, values) in zip(offsets, sources):
        ax.bar(x_positions + offset, values, width, label=label)

    # X-axis tick positions and labels
    ax.set_xticks(x_positions)
    ax.set_xticklabels(class_labels)
    ax.set_xlim(-0.5, num_positions - 0.5)

    ax.set_xlabel("Conditional classes (0c-6c) and uc (unconditional)")
    ax.set_ylabel("Number of samples (ls count)")
    ax.set_title("Class-wise counts for conditional and unconditional datasets")
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    fig.tight_layout()

    output_path = base_dir / output_name
    fig.savefig(output_path, format="pdf")
    print(f"Saved bar plot to: {output_path}")


def build_and_save_small_plot(base_dir: Path, output_name: str = "small_class_distribution_barplot.pdf") -> None:
    """Save a compact bar plot for the fixed distribution:

    1:3561, 2A:1469, 2B:102, 2C:106, 3A:62, 3B:44, 3C:12
    """

    labels = ["1", "2A", "2B", "2C", "3A", "3B", "3C"]
    values = [3561, 1469, 102, 106, 62, 44, 12]

    x = np.arange(len(labels), dtype=float)
    width = 0.4  # thin bars

    fig, ax = plt.subplots(figsize=(5, 3))
    ax.bar(x, values, width, color="tab:blue")

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Count")
    ax.set_title("Additional class distribution")
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    fig.tight_layout()

    output_path = base_dir / output_name
    fig.savefig(output_path, format="pdf")
    print(f"Saved small bar plot to: {output_path}")


if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parent
    build_and_save_plot(base_dir)
    build_and_save_small_plot(base_dir)
