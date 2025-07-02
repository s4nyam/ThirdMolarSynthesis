import os
import random
import shutil
import string
from pathlib import Path

# Set your source and destination paths here
source_path = Path("uspforp/uspforp_cropped")
destination_path = Path("thirdmolar")

# Ensure destination directory exists
destination_path.mkdir(parents=True, exist_ok=True)

# Get all files in the source path
all_files = [f for f in source_path.iterdir() if f.is_file()]
selected_files = random.sample(all_files, min(300, len(all_files)))  # Select up to 300 files

def generate_random_filename():
    random_str = ''.join(random.choices(string.ascii_lowercase + string.digits, k=8))
    random_num = random.randint(0, 4)
    random_letter = random.choice(string.ascii_uppercase)
    return f"{random_str}_{random_num}{random_letter}.png"

# Copy and rename files
for file in selected_files:
    new_filename = generate_random_filename()
    destination_file = destination_path / new_filename
    shutil.copy(file, destination_file)

print(f"Copied and renamed {len(selected_files)} files.")
