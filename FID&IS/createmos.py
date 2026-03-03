#!/usr/bin/env python3
"""
Script to create mosaic images for conditional and unconditional third molar data.

Conditional mosaic: 7 columns (classes 0-6) x 3 rows (training, GAN, diffusion)
Unconditional mosaic: 3 columns x 3 rows (3 images each for train, GAN, diffusion)
"""

import os
import glob
import random
try:
    from PIL import Image, ImageDraw, ImageFont
except ImportError:
    print("PIL (Pillow) not found. Installing...")
    os.system("pip install Pillow")
    from PIL import Image, ImageDraw, ImageFont
import numpy as np

def extract_class_from_filename(filename):
    """Extract class label from training data filename pattern: something_something_classlabel.png"""
    basename = os.path.basename(filename)
    parts = basename.split('_')
    if len(parts) >= 3:
        class_label = parts[-1].replace('.png', '')
        try:
            return int(class_label)
        except ValueError:
            return None
    return None

def get_random_images_from_dir(directory, count=1, class_filter=None):
    """Get random images from a directory, optionally filtered by class."""
    if not os.path.exists(directory):
        print(f"Warning: Directory {directory} does not exist")
        return []
    
    png_files = glob.glob(os.path.join(directory, "*.png"))
    
    if class_filter is not None:
        # Filter by class label extracted from filename
        filtered_files = []
        for file in png_files:
            file_class = extract_class_from_filename(file)
            if file_class == class_filter:
                filtered_files.append(file)
        png_files = filtered_files
    
    if len(png_files) == 0:
        print(f"Warning: No PNG files found in {directory}" + 
              (f" for class {class_filter}" if class_filter is not None else ""))
        return []
    
    # Return random sample
    return random.sample(png_files, min(count, len(png_files)))

def create_text_label(text, size, font_size=14, bg_color=(240, 240, 240), text_color=(0, 0, 0), rotate=False):
    """Create a text label image."""
    if rotate:
        # For rotated text, swap width and height
        img = Image.new('RGB', (size[1], size[0]), bg_color)
    else:
        img = Image.new('RGB', size, bg_color)
    
    draw = ImageDraw.Draw(img)
    
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", font_size)
    except:
        try:
            font = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", font_size)
        except:
            font = ImageFont.load_default()
    
    # Get text dimensions
    bbox = draw.textbbox((0, 0), text, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    
    # Center the text
    x = (img.size[0] - text_width) // 2
    y = (img.size[1] - text_height) // 2
    
    draw.text((x, y), text, fill=text_color, font=font)
    
    if rotate:
        # Rotate 90 degrees counter-clockwise for row labels
        img = img.rotate(90, expand=True)
    
    return np.array(img)

def load_and_resize_image(image_path, target_size=(128, 128)):
    """Load and resize an image to target size."""
    try:
        img = Image.open(image_path)
        if img.mode != 'RGB':
            img = img.convert('RGB')
        img = img.resize(target_size, Image.Resampling.LANCZOS)
        return np.array(img)
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        # Return a black image as fallback
        return np.zeros((target_size[1], target_size[0], 3), dtype=np.uint8)

def create_conditional_mosaic(base_dir, output_path, img_size=(128, 128)):
    """
    Create conditional mosaic: 7 columns (classes 0-6) x 3 rows (training, GAN, diffusion)
    With proper header labels (not overlays)
    """
    print("Creating conditional mosaic...")
    
    # Directory paths
    training_dir = os.path.join(base_dir, "raw-3m-training-c-128")
    
    # Labels
    class_labels = ["1", "2A", "2B", "2C", "3A", "3B", "3C"]
    row_labels = ["Training", "cGAN", "cDiffusion"]
    
    # Dimensions
    data_rows = 3
    data_cols = 7
    label_height = 20  # Height for column headers
    label_width = 20   # Width for row headers
    
    # Create mosaic array with space for labels
    mosaic_height = label_height + data_rows * img_size[1]
    mosaic_width = label_width + data_cols * img_size[0]
    mosaic = np.ones((mosaic_height, mosaic_width, 3), dtype=np.uint8) * 255  # White background
    
    # Create column headers (class labels)
    for col_idx in range(data_cols):
        header_img = create_text_label(class_labels[col_idx], (img_size[0], label_height), font_size=12)
        start_x = label_width + col_idx * img_size[0]
        end_x = start_x + img_size[0]
        mosaic[0:label_height, start_x:end_x] = header_img
    
    # Create row headers (model labels) - rotated
    for row_idx in range(data_rows):
        header_img = create_text_label(row_labels[row_idx], (label_width, img_size[1]), font_size=12, rotate=True)
        start_y = label_height + row_idx * img_size[1]
        end_y = start_y + img_size[1]
        mosaic[start_y:end_y, 0:label_width] = header_img
    
    # Fill data cells
    for class_idx in range(7):  # classes 0-6
        print(f"Processing class {class_idx}...")
        
        # Row 0: Training data
        training_imgs = get_random_images_from_dir(training_dir, count=1, class_filter=class_idx)
        if training_imgs:
            img_array = load_and_resize_image(training_imgs[0], img_size)
            start_y = label_height
            end_y = start_y + img_size[1]
            start_x = label_width + class_idx * img_size[0]
            end_x = start_x + img_size[0]
            mosaic[start_y:end_y, start_x:end_x] = img_array
        
        # Row 1: GAN
        gan_dir = os.path.join(base_dir, f"3m-cGAN-128-class-{class_idx}")
        gan_imgs = get_random_images_from_dir(gan_dir, count=1)
        if gan_imgs:
            img_array = load_and_resize_image(gan_imgs[0], img_size)
            start_y = label_height + img_size[1]
            end_y = start_y + img_size[1]
            start_x = label_width + class_idx * img_size[0]
            end_x = start_x + img_size[0]
            mosaic[start_y:end_y, start_x:end_x] = img_array
        
        # Row 2: Diffusion
        diff_dir = os.path.join(base_dir, f"3m-cDiff-128-class-{class_idx}")
        diff_imgs = get_random_images_from_dir(diff_dir, count=1)
        if diff_imgs:
            img_array = load_and_resize_image(diff_imgs[0], img_size)
            start_y = label_height + 2 * img_size[1]
            end_y = start_y + img_size[1]
            start_x = label_width + class_idx * img_size[0]
            end_x = start_x + img_size[0]
            mosaic[start_y:end_y, start_x:end_x] = img_array
    
    # Save mosaic
    mosaic_img = Image.fromarray(mosaic)
    mosaic_img.save(output_path, 'PNG', quality=100, optimize=False)
    print(f"Conditional mosaic saved to: {output_path}")

def create_unconditional_mosaic(base_dir, output_path, img_size=(128, 128)):
    """
    Create unconditional mosaic: 3 columns x 3 rows (3 images each for train, GAN, diffusion)
    With proper header labels (not overlays)
    """
    print("Creating unconditional mosaic...")
    
    # Directory paths
    training_dir = os.path.join(base_dir, "raw-3m-training-c-128")
    gan_dir = os.path.join(base_dir, "3m-GAN-128")
    diff_dir = os.path.join(base_dir, "3m-Diff-128")
    
    # Labels
    col_labels = ["", "", ""]
    row_labels = ["Training", "GAN", "Diffusion"]
    
    # Dimensions
    data_rows = 3
    data_cols = 3
    label_height = 20  # Height for column headers
    label_width = 20   # Width for row headers

    # Create mosaic array with space for labels
    mosaic_height = label_height + data_rows * img_size[1]
    mosaic_width = label_width + data_cols * img_size[0]
    mosaic = np.ones((mosaic_height, mosaic_width, 3), dtype=np.uint8) * 255  # White background
    
    # Create column headers (sample labels)
    for col_idx in range(data_cols):
        header_img = create_text_label(col_labels[col_idx], (img_size[0], label_height), font_size=12)
        start_x = label_width + col_idx * img_size[0]
        end_x = start_x + img_size[0]
        mosaic[0:label_height, start_x:end_x] = header_img
    
    # Create row headers (model labels) - rotated
    for row_idx in range(data_rows):
        header_img = create_text_label(row_labels[row_idx], (label_width, img_size[1]), font_size=12, rotate=True)
        start_y = label_height + row_idx * img_size[1]
        end_y = start_y + img_size[1]
        mosaic[start_y:end_y, 0:label_width] = header_img
    
    # Row 0: Training data (3 random images)
    print("Processing training data...")
    training_imgs = get_random_images_from_dir(training_dir, count=3)
    for col_idx in range(min(3, len(training_imgs))):
        img_array = load_and_resize_image(training_imgs[col_idx], img_size)
        start_y = label_height
        end_y = start_y + img_size[1]
        start_x = label_width + col_idx * img_size[0]
        end_x = start_x + img_size[0]
        mosaic[start_y:end_y, start_x:end_x] = img_array
    
    # Row 1: GAN data (3 random images)
    print("Processing GAN data...")
    gan_imgs = get_random_images_from_dir(gan_dir, count=3)
    for col_idx in range(min(3, len(gan_imgs))):
        img_array = load_and_resize_image(gan_imgs[col_idx], img_size)
        start_y = label_height + img_size[1]
        end_y = start_y + img_size[1]
        start_x = label_width + col_idx * img_size[0]
        end_x = start_x + img_size[0]
        mosaic[start_y:end_y, start_x:end_x] = img_array
    
    # Row 2: Diffusion data (3 random images)
    print("Processing diffusion data...")
    diff_imgs = get_random_images_from_dir(diff_dir, count=3)
    for col_idx in range(min(3, len(diff_imgs))):
        img_array = load_and_resize_image(diff_imgs[col_idx], img_size)
        start_y = label_height + 2 * img_size[1]
        end_y = start_y + img_size[1]
        start_x = label_width + col_idx * img_size[0]
        end_x = start_x + img_size[0]
        mosaic[start_y:end_y, start_x:end_x] = img_array
    
    # Save mosaic
    mosaic_img = Image.fromarray(mosaic)
    mosaic_img.save(output_path, 'PNG', quality=100, optimize=False)
    print(f"Unconditional mosaic saved to: {output_path}")

def main():
    # Set random seed for reproducibility
    random.seed(42)
    np.random.seed(42)
    
    # Base directory containing all the image folders
    base_dir = "/home/sanyam/playground/3m-FIDandIS"
    
    # Create output directory
    output_dir = os.path.join(base_dir, "mosaics")
    os.makedirs(output_dir, exist_ok=True)
    
    # Output paths
    conditional_output = os.path.join(output_dir, "conditional.png")
    unconditional_output = os.path.join(output_dir, "unconditional.png")
    
    # Image size (assuming all images are 128x128)
    img_size = (128, 128)
    
    # Create mosaics
    create_conditional_mosaic(base_dir, conditional_output, img_size)
    create_unconditional_mosaic(base_dir, unconditional_output, img_size)
    
    print("\nMosaic creation completed!")
    print(f"Conditional mosaic: {conditional_output}")
    print(f"Unconditional mosaic: {unconditional_output}")

if __name__ == "__main__":
    main()