import os
import torch
import torchvision.utils as vutils
from simple_diffusion.scheduler import DDIMScheduler

def save_plain_noise(timestep, noise_scheduler, device, save_folder="plain_noise_images"):
    """Generates plain noise based on a given timestep and saves it as an image."""
    os.makedirs(save_folder, exist_ok=True)
    
    # Generate random noise
    noise = torch.randn((1, 3, 128, 256), device=device)  # Assuming image size is 128x256
    
    # Apply noise transformation based on timestep
    noisy_image = noise_scheduler.add_noise(torch.zeros_like(noise), noise, torch.tensor([timestep], device=device))
    
    # Normalize to [0,1] for saving
    noisy_image = (noisy_image + 1) / 2  
    vutils.save_image(noisy_image, os.path.join(save_folder, f"plain_noise_t{timestep}.png"))
    
    print(f"Saved plain noise image for timestep {timestep} in {save_folder}")

if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    noise_scheduler = DDIMScheduler(num_train_timesteps=1000, beta_schedule="cosine")
    
    # Specify the timestep
    timestep = 300  # Change this to any desired timestep
    
    save_plain_noise(timestep, noise_scheduler, device)