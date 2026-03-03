import cv2
import os
import numpy as np
from tqdm import tqdm
from skimage import exposure

def is_dark(image, threshold=0.3):
    """Check if image is too dark based on histogram"""
    hist = cv2.calcHist([image], [0], None, [256], [0,256])
    hist = hist / hist.sum()
    return np.sum(hist[:50]) > threshold  # More than threshold% pixels in dark range

def is_bright(image, threshold=0.3):
    """Check if image is too bright based on histogram"""
    hist = cv2.calcHist([image], [0], None, [256], [0,256])
    hist = hist / hist.sum()
    return np.sum(hist[200:]) > threshold  # More than threshold% pixels in bright range

def normalize_image(image):
    """Advanced normalization with CLAHE and gamma correction"""
    # Convert to float32 for processing
    img = image.astype('float32')
    img_normalized = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
    
    # Apply different processing based on brightness
    if is_dark(image):
        # For dark images - more aggressive CLAHE with gamma correction
        clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8,8))
        img_normalized = clahe.apply(img_normalized.astype('uint8'))
        img_normalized = exposure.adjust_gamma(img_normalized, gamma=0.7)
    elif is_bright(image):
        # For bright images - moderate CLAHE with inverse gamma
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(16,16))
        img_normalized = clahe.apply(img_normalized.astype('uint8'))
        img_normalized = exposure.adjust_gamma(img_normalized, gamma=1.3)
    else:
        # For normal images - mild CLAHE
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        img_normalized = clahe.apply(img_normalized.astype('uint8'))
    
    # Final contrast stretching
    p2, p98 = np.percentile(img_normalized, (2, 98))
    img_normalized = exposure.rescale_intensity(img_normalized, in_range=(p2, p98))
    
    return img_normalized

def create_comparison_mosaic(original, processed):
    """Create before/after comparison with histograms"""
    # Resize images to same dimensions
    h, w = original.shape
    processed = cv2.resize(processed, (w, h))
    
    # Create histograms
    hist_original = create_histogram_image(original)
    hist_processed = create_histogram_image(processed)
    
    # Combine images
    top = np.hstack([original, processed])
    bottom = np.hstack([hist_original, hist_processed])
    mosaic = np.vstack([top, bottom])
    
    # Add labels
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(mosaic, "Original", (10, 30), font, 1, 255, 2)
    cv2.putText(mosaic, "Processed", (w+10, 30), font, 1, 255, 2)
    cv2.putText(mosaic, "Original Hist", (10, h+30), font, 1, 255, 2)
    cv2.putText(mosaic, "Processed Hist", (w+10, h+30), font, 1, 255, 2)
    
    return mosaic

def create_histogram_image(image, height=200):
    """Create histogram visualization"""
    hist = cv2.calcHist([image], [0], None, [256], [0,256])
    hist = cv2.normalize(hist, hist, 0, height, cv2.NORM_MINMAX)
    hist_img = np.zeros((height, 256), dtype=np.uint8)
    
    for i in range(256):
        cv2.line(hist_img, (i, height), (i, height - int(hist[i])), 255, 1)
    
    return cv2.resize(hist_img, (image.shape[1], height))

def process_directory(source_dir, dest_dir):
    """Process all images in directory"""
    os.makedirs(dest_dir, exist_ok=True)
    
    image_files = [f for f in os.listdir(source_dir) 
                 if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif'))]
    
    for filename in tqdm(image_files, desc="Processing images"):
        src_path = os.path.join(source_dir, filename)
        dest_path = os.path.join(dest_dir, filename)
        
        # Read and process image
        original = cv2.imread(src_path, cv2.IMREAD_GRAYSCALE)
        if original is None:
            continue
            
        # Normalize image
        processed = normalize_image(original)
        
        # Create comparison mosaic
        mosaic = create_comparison_mosaic(original, processed)
        
        # Save result
        cv2.imwrite(dest_path, mosaic)

if __name__ == "__main__":
    source_dir = "thirdmolar_gen_greyadjusted"
    dest_dir = "thirdmolar_gen_greyadjusted_normalized"
    
    # Install scikit-image if needed
    try:
        from skimage import exposure
    except ImportError:
        print("Installing scikit-image...")
        import subprocess
        subprocess.check_call(["pip", "install", "scikit-image"])
        from skimage import exposure
    
    print(f"Processing images from {source_dir} to {dest_dir}")
    process_directory(source_dir, dest_dir)
    print("Image normalization complete!")