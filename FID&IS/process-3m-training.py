import os
import shutil
from collections import defaultdict, Counter
import random
import numpy as np
from PIL import Image, ImageEnhance
from tqdm import tqdm
import cv2

def extract_class_label(filename):
    """Extract class label from filename by splitting and taking the last element before .png"""
    return filename.split('_')[-1].split('.')[0]

def analyze_class_distribution(source_folder):
    """Analyze the distribution of classes in the dataset"""
    class_counts = Counter()
    class_files = defaultdict(list)
    
    for filename in os.listdir(source_folder):
        if filename.endswith('.png'):
            class_label = extract_class_label(filename)
            class_counts[class_label] += 1
            class_files[class_label].append(filename)
    
    print("Class distribution:")
    for class_label, count in sorted(class_counts.items()):
        print(f"Class {class_label}: {count} images")
    
    return class_files, class_counts

def rotate_image(image, angle):
    """Rotate image by given angle"""
    return image.rotate(angle, expand=True, fillcolor=(0, 0, 0))

def flip_image(image, direction):
    """Flip image horizontally or vertically"""
    if direction == 'horizontal':
        return image.transpose(Image.FLIP_LEFT_RIGHT)
    elif direction == 'vertical':
        return image.transpose(Image.FLIP_TOP_BOTTOM)
    return image

def adjust_brightness(image, factor):
    """Adjust brightness of image"""
    enhancer = ImageEnhance.Brightness(image)
    return enhancer.enhance(factor)

def adjust_contrast(image, factor):
    """Adjust contrast of image"""
    enhancer = ImageEnhance.Contrast(image)
    return enhancer.enhance(factor)

def gaussian_noise(image, mean=0, std=15):
    """Add gaussian noise to image"""
    img_array = np.array(image)
    noise = np.random.normal(mean, std, img_array.shape).astype(np.uint8)
    noisy_img = np.clip(img_array.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(noisy_img)

def random_crop_and_resize(image, crop_ratio=0.9):
    """Randomly crop and resize back to original size"""
    width, height = image.size
    crop_width = int(width * crop_ratio)
    crop_height = int(height * crop_ratio)
    
    left = random.randint(0, width - crop_width)
    top = random.randint(0, height - crop_height)
    
    cropped = image.crop((left, top, left + crop_width, top + crop_height))
    return cropped.resize((width, height), Image.Resampling.LANCZOS)

def apply_augmentation(image):
    """Apply a random augmentation to the image"""
    augmentations = [
        lambda img: rotate_image(img, random.choice([90, 180, 270])),
        lambda img: rotate_image(img, random.randint(-30, 30)),
        lambda img: flip_image(img, 'horizontal'),
        lambda img: flip_image(img, 'vertical'),
        lambda img: adjust_brightness(img, random.uniform(0.7, 1.3)),
        lambda img: adjust_contrast(img, random.uniform(0.8, 1.2)),
        lambda img: gaussian_noise(img, std=random.randint(5, 20)),
        lambda img: random_crop_and_resize(img, random.uniform(0.85, 0.95))
    ]
    
    # Apply 1-3 random augmentations
    num_augs = random.randint(1, 3)
    selected_augs = random.sample(augmentations, num_augs)
    
    augmented_img = image
    for aug in selected_augs:
        try:
            augmented_img = aug(augmented_img)
        except Exception as e:
            print(f"Warning: Augmentation failed: {e}")
            continue
    
    return augmented_img

def process_dataset(source_folder, dest_folder, target_count=550):
    """Process the dataset to balance classes to target_count images each"""
    
    # Create destination folder if it doesn't exist
    os.makedirs(dest_folder, exist_ok=True)
    
    # Analyze class distribution
    class_files, class_counts = analyze_class_distribution(source_folder)
    
    print(f"\nProcessing dataset to balance classes to {target_count} images each...")
    
    total_operations = 0
    for class_label in class_files:
        current_count = class_counts[class_label]
        if current_count < target_count:
            total_operations += target_count
        else:
            total_operations += target_count
    
    # Process each class
    with tqdm(total=total_operations, desc="Processing images") as pbar:
        for class_label in sorted(class_files.keys()):
            current_files = class_files[class_label]
            current_count = len(current_files)
            
            print(f"\nProcessing class {class_label}: {current_count} -> {target_count} images")
            
            if current_count >= target_count:
                # If we have enough images, randomly select target_count
                selected_files = random.sample(current_files, target_count)
                for i, filename in enumerate(selected_files):
                    src_path = os.path.join(source_folder, filename)
                    # Create new filename with sequential numbering
                    new_filename = f"class_{class_label}_{i+1:06d}.png"
                    dest_path = os.path.join(dest_folder, new_filename)
                    shutil.copy2(src_path, dest_path)
                    pbar.update(1)
            else:
                # Copy all original images first
                for i, filename in enumerate(current_files):
                    src_path = os.path.join(source_folder, filename)
                    new_filename = f"class_{class_label}_{i+1:06d}.png"
                    dest_path = os.path.join(dest_folder, new_filename)
                    shutil.copy2(src_path, dest_path)
                    pbar.update(1)
                
                # Generate additional images through augmentation
                images_needed = target_count - current_count
                print(f"  Generating {images_needed} additional images through augmentation...")
                
                for i in range(images_needed):
                    # Select a random original image to augment
                    source_filename = random.choice(current_files)
                    src_path = os.path.join(source_folder, source_filename)
                    
                    try:
                        # Load and augment the image
                        with Image.open(src_path) as img:
                            if img.mode != 'RGB':
                                img = img.convert('RGB')
                            
                            augmented_img = apply_augmentation(img)
                            
                            # Save augmented image
                            aug_filename = f"class_{class_label}_{current_count + i + 1:06d}_aug.png"
                            aug_path = os.path.join(dest_folder, aug_filename)
                            augmented_img.save(aug_path)
                            
                    except Exception as e:
                        print(f"  Error processing {source_filename}: {e}")
                        # Copy original instead of augmented if augmentation fails
                        shutil.copy2(src_path, os.path.join(dest_folder, aug_filename))
                    
                    pbar.update(1)
    
    print(f"\nDataset processing completed!")
    print(f"Processed dataset saved to: {dest_folder}")
    
    # Verify the final distribution
    verify_processed_dataset(dest_folder)

def verify_processed_dataset(dest_folder):
    """Verify the processed dataset has the correct distribution"""
    print(f"\nVerifying processed dataset in {dest_folder}...")
    
    class_counts = Counter()
    for filename in os.listdir(dest_folder):
        if filename.endswith('.png'):
            class_label = filename.split('_')[1]
            class_counts[class_label] += 1
    
    print("Final class distribution:")
    for class_label, count in sorted(class_counts.items()):
        print(f"Class {class_label}: {count} images")
    
    total_images = sum(class_counts.values())
    print(f"Total images: {total_images}")

if __name__ == "__main__":
    # Set random seed for reproducibility
    random.seed(42)
    np.random.seed(42)
    
    # Define source and destination folders
    source_folder = "/home/sanyam/playground/3m-FIDandIS/3m-training-c-128"
    dest_folder = "/home/sanyam/playground/3m-FIDandIS/3m-training-c-128-processed"
    
    # Process the dataset
    process_dataset(source_folder, dest_folder, target_count=550)