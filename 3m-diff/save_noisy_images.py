import os
import torch
import torchvision.utils as vutils
from torchvision import transforms
from PIL import Image
from simple_diffusion.scheduler import DDIMScheduler
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm  # Import tqdm for progress bar

# Define number of timesteps
n_timesteps = 1000

def save_noisy_images(image_path, noise_scheduler, device, save_folder="noisy_images"):
    """Adds noise to a specific image and saves all noisy versions."""
    os.makedirs(save_folder, exist_ok=True)
    
    # Load and preprocess image
    transform = transforms.Compose([
        transforms.ToTensor(),  # Convert to [0,1]
        transforms.Lambda(lambda x: 2 * x - 1)  # Convert to [-1,1]
    ])
    
    image = Image.open(image_path).convert("RGB")
    image = transform(image).to(device).unsqueeze(0)  # Add batch dimension
    
    # Store average pixel values, standard deviation, and grey values for each timestep
    avg_pixel_values = []
    std_pixel_values = []
    grey_values = []
    
    # Iterate over all timesteps with tqdm for progress bar
    for t in tqdm(range(n_timesteps), desc="Adding noise", unit="step"):
        noise = torch.randn_like(image).to(device)
        noisy_image = noise_scheduler.add_noise(image, noise, torch.tensor([t], device=device))
        
        # Normalize to [0,1] for saving
        noisy_image = (noisy_image + 1) / 2  
        vutils.save_image(noisy_image, os.path.join(save_folder, f"noisy_t{t}.png"))
        
        # Calculate average pixel value and standard deviation for this timestep
        avg_pixel_value = torch.mean(noisy_image).item()
        std_pixel_value = torch.std(noisy_image).item()
        avg_pixel_values.append(avg_pixel_value)
        std_pixel_values.append(std_pixel_value)
        
        # Convert noisy image to grayscale and calculate grey value as a percentage (0% = white, 100% = black)
        grey_image = torch.mean(noisy_image, dim=1, keepdim=True)  # Convert to grayscale by averaging channels
        grey_value = 1 - torch.mean(grey_image).item()  # Invert to represent 0% as white and 100% as black
        grey_values.append(grey_value)  # Convert to percentage
    
    print(f"Saved {n_timesteps} noisy images in {save_folder}")
    
    # Plot the average pixel values, standard deviation, and grey values over time
    plt.figure(figsize=(10, 6))
    plt.plot(range(n_timesteps), avg_pixel_values, label="Average Pixel Value", color='blue')
    plt.plot(range(n_timesteps), std_pixel_values, label="Standard Deviation of Pixel Values", color='red')
    plt.plot(range(n_timesteps), grey_values, label="Grey Value", color='green')
    plt.xlabel("Timestep", fontsize=16)
    plt.ylabel("Value", fontsize=16)
    plt.title("Noise Addition Process Over Time", fontsize=16)
    plt.legend(fontsize=16, framealpha=0.5)
    plt.tick_params(axis='both', labelsize=16)
    plt.savefig(os.path.join(save_folder, "noise_addition_curve.pdf"), format="pdf")  # Save as PDF
    plt.close()
    print(f"Saved noise addition curve as PDF in {save_folder}")

if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    noise_scheduler = DDIMScheduler(num_train_timesteps=n_timesteps, beta_schedule="cosine")
    
    # Specify the image path from your dataset
    # image_path = "/home/s4nyam/playground/sd/images/train_00005012.png" fopr 2nd image
    image_path = "/home/s4nyam/playground/sd/images/train_00000046.png"
    
    save_noisy_images(image_path, noise_scheduler, device)