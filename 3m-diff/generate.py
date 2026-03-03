import argparse
from datetime import datetime

import torch
import torch.nn.functional as F
import imageio  # For creating GIFs

import os
from PIL import Image
from torchvision import utils

from simple_diffusion.scheduler import DDIMScheduler
from simple_diffusion.model import UNet
# generated_samples/
# └── 20231012_123456_seed0/
#     ├── intermediate/
#     │   ├── step_0000_sample_0000.jpeg
#     │   ├── step_0001_sample_0000.jpeg
#     │   └── ...
#     ├── 0.jpeg
#     ├── grid.jpeg
#     ├── sample_0000_animation.gif
#     ├── sample_0001_animation.gif
#     └── ...
n_timesteps = 1000
n_inference_timesteps = 250
seedvalue =  100 # This value decides how many samples are gomna be created, for example for seed value 500 it produces 500 (each seed) * 32 (batch of different) images 
n_samples = 50 # Number of samples to generate for each seed
def main(args, seedpass):
    print("Generating {} samples for seed: ".format(n_samples), seedpass)
    
    # Define current_date at the beginning of the function
    current_date = datetime.today().strftime('%Y%m%d_%H%M%S')
    
    model = UNet(3, image_size=(args.resolution, args.resolution), hidden_dims=[64, 128, 256, 512],
                 use_flash_attn=args.use_flash_attn)
    noise_scheduler = DDIMScheduler(num_train_timesteps=n_timesteps,
                                    beta_schedule="cosine")

    pretrained = torch.load(args.pretrained_model_path, map_location="cuda:0")["ema_model_state"]
    model.load_state_dict(pretrained, strict=False)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    model.eval()
    with torch.no_grad():
        # has to be instantiated every time, because of reproducibility
        generator = torch.manual_seed(seedpass)
        generated_images = noise_scheduler.generate(
            model,
            num_inference_steps=n_inference_timesteps,
            generator=generator,
            eta=1.0,
            batch_size=args.eval_batch_size,
            save_intermediate=False,  # Enable saving intermediate images
            output_dir=f"./{args.samples_dir}/{current_date}_seed{seedpass}/intermediate",  # Use current_date here
        )

        images = generated_images["sample"]
        images_processed = (images * 255).round().astype("uint8")

        out_dir = f"./{args.samples_dir}/{current_date}_seed{seedpass}/"
        os.makedirs(out_dir, exist_ok=True)  # Ensure the directory exists
        for idx, image in enumerate(images_processed):
            image = Image.fromarray(image)
            image.save(f"{out_dir}/{idx}.jpeg")

        utils.save_image(generated_images["sample_pt"],
                         f"{out_dir}/grid.jpeg",
                         nrow=args.eval_batch_size // 4)
        # # Create animations for each sample
        # intermediate_dir = f"./{args.samples_dir}/{current_date}_seed{seedpass}/intermediate/"
        # for sample_idx in range(args.eval_batch_size):
        #     # Collect all intermediate images for this sample
        #     intermediate_images = []
        #     for step in range(n_inference_timesteps):
        #         img_path = f"{intermediate_dir}/step_{step:04d}_sample_{sample_idx:04d}.jpeg"
        #         if os.path.exists(img_path):
        #             img = Image.open(img_path)
        #             intermediate_images.append(img)

        #     # Save as GIF
        #     if intermediate_images:
        #         gif_path = f"{out_dir}/sample_{sample_idx:04d}_animation.gif"
        #         imageio.mimsave(gif_path, intermediate_images, duration=0.001)  # Adjust duration as needed
        #         print(f"Saved animation for sample {sample_idx} at {gif_path}")

for each_seed in range(40,seedvalue):
    if __name__ == "__main__":
        parser = argparse.ArgumentParser(
            description="Simple script for image generation.")
        parser.add_argument("--samples_dir", type=str, default="generated_samples/")
        parser.add_argument("--resolution", type=int, default=128)
        parser.add_argument("--pretrained_model_path",
                            type=str,
                            default="trained_models/ddpm-model-ep118.pth", # Path to the trained model, change if needed
                            help="Path to pretrained model")
        parser.add_argument("--eval_batch_size", type=int, default=n_samples)
        parser.add_argument('--use_flash_attn', action='store_true')

        args = parser.parse_args()

        main(args,each_seed)
