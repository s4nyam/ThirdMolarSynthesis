import os
from pathlib import Path
import torch
from torch import nn
from diffusers import DDPMScheduler, UNet2DConditionModel
import datetime
from torchvision import transforms
from PIL import Image
from tqdm.auto import tqdm

# Set device
device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Configuration (must match training config)
channels_to_keep = 3
num_classes = 7
class_emb_size = 128
image_size = 128

# Create timestamped output directory
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
output_dir = f"infer_{timestamp}"
os.makedirs(output_dir, exist_ok=True)
print(f"Output will be saved to: {output_dir}")

# Define the model (same as in training)
class ClassConditionedUnet(nn.Module):
    def __init__(self, num_classes=7, class_emb_size=128):
        super().__init__()
        self.class_emb = nn.Embedding(num_classes, class_emb_size)

        self.model = UNet2DConditionModel(
            sample_size=image_size,
            in_channels=channels_to_keep,
            out_channels=channels_to_keep,
            cross_attention_dim=class_emb_size,
            layers_per_block=3,
            block_out_channels=(32, 64, 128),
            down_block_types=(
                "DownBlock2D",
                "AttnDownBlock2D",
                "AttnDownBlock2D",
            ),
            up_block_types=(
                "AttnUpBlock2D",
                "AttnUpBlock2D",
                "UpBlock2D",
            ),
        )

    def forward(self, x, t, class_labels):
        class_cond = self.class_emb(class_labels).unsqueeze(1)
        return self.model(x, timestep=t, encoder_hidden_states=class_cond).sample

# Initialize model
print("\nInitializing model...")
net = ClassConditionedUnet(num_classes=num_classes, class_emb_size=class_emb_size).to(device)

# Load checkpoint
checkpoint_path = "checkpoints/checkpoint_epoch_360.pth"
print(f"Loading checkpoint from {checkpoint_path}...")
checkpoint = torch.load(checkpoint_path, map_location=device)
net.load_state_dict(checkpoint['model_state_dict'])
net.eval()
print("Model loaded successfully!")

# Create scheduler
noise_scheduler = DDPMScheduler(num_train_timesteps=1000, beta_schedule='squaredcos_cap_v2')
import matplotlib.pyplot as plt

def generate_images(number_of_images_to_generate=8):
    """Generate and save images for each class with progress tracking"""
    # Create class directories with progress bar
    print("\nCreating output directories...")
    for class_id in tqdm(range(num_classes), desc="Creating class folders"):
        class_dir = os.path.join(output_dir, str(class_id))
        os.makedirs(class_dir, exist_ok=True)

    # Prepare inputs
    print("\nPreparing generation inputs...")
    x = torch.randn(num_classes * number_of_images_to_generate, 
                   channels_to_keep, image_size, image_size).to(device)
    y = torch.tensor([[i] * number_of_images_to_generate 
                     for i in range(num_classes)]).flatten().to(device)

    # Generation process with progress bar
    print(f"\nGenerating {number_of_images_to_generate} images per class...")
    for t in tqdm(noise_scheduler.timesteps, desc="Denoising steps"):
        with torch.no_grad():
            residual = net(x, t, y)
        x = noise_scheduler.step(residual, t, x).prev_sample

    # Save images with progress bar
    print("\nSaving generated images...")
    for idx, (image, label) in enumerate(tqdm(zip(x, y), 
                                            total=len(y),
                                            desc="Saving images")):
        # Convert tensor to PIL Image
        img = image.detach().cpu().clip(-1, 1)        
        # Save image
        class_dir = os.path.join(output_dir, str(label.item()))
        img_num = idx % number_of_images_to_generate
        img_path = os.path.join(class_dir, f"{img_num}.png")
        plt.imsave(img_path, img[0], cmap='gray')

    print(f"\nGeneration complete! Saved {num_classes * number_of_images_to_generate} images to {output_dir}")

if __name__ == "__main__":
    number_of_images_to_generate = 100  # Adjust as needed
    generate_images(number_of_images_to_generate)