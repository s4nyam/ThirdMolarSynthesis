import os
import shutil
from pathlib import Path
from tqdm import tqdm
import glob

def import_images_with_suffix():
    """
    Import images from two source directories and copy them to destination
    with parent folder name as suffix in filename.
    """
    # Source directories
    source_dirs = [
        "/home/sanyam/playground/3m-FIDandIS/moredata/p1",
        "/home/sanyam/playground/3m-FIDandIS/moredata/p2"
    ]
    
    # Destination directory
    destination_dir = "/home/sanyam/playground/tg-dental-activity/syn2"
    
    # Create destination directory if it doesn't exist
    os.makedirs(destination_dir, exist_ok=True)
    
    # Common image extensions
    image_extensions = ['*.png', '*.jpg', '*.jpeg', '*.bmp', '*.tiff', '*.tif', '*.gif']
    
    # Collect all image files from both source directories
    all_files = []
    
    for source_dir in source_dirs:
        if not os.path.exists(source_dir):
            print(f"Warning: Source directory {source_dir} does not exist!")
            continue
            
        # Extract parent folder name for suffix
        parent_folder = os.path.basename(source_dir)
        
        # Find all image files
        for ext in image_extensions:
            files = glob.glob(os.path.join(source_dir, ext))
            for file_path in files:
                all_files.append((file_path, parent_folder))
    
    print(f"Found {len(all_files)} image files to process")
    
    # Process files with tqdm progress bar
    copied_count = 0
    skipped_count = 0
    
    for file_path, suffix in tqdm(all_files, desc="Copying images"):
        try:
            # Get original filename and extension
            original_filename = os.path.basename(file_path)
            name, ext = os.path.splitext(original_filename)
            
            # Create new filename with suffix
            new_filename = f"{name}_{suffix}{ext}"
            
            # Full destination path
            dest_path = os.path.join(destination_dir, new_filename)
            
            # Check if file already exists
            if os.path.exists(dest_path):
                print(f"Skipping {new_filename} - file already exists")
                skipped_count += 1
                continue
            
            # Copy file to destination
            shutil.copy2(file_path, dest_path)
            copied_count += 1
            
        except Exception as e:
            print(f"Error processing {file_path}: {str(e)}")
            continue
    
    print(f"\nProcessing complete!")
    print(f"Files copied: {copied_count}")
    print(f"Files skipped: {skipped_count}")
    print(f"Destination: {destination_dir}")

if __name__ == "__main__":
    import_images_with_suffix()