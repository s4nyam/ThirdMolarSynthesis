import os
from pathlib import Path
import random
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from transformers import CLIPProcessor, CLIPModel, CLIPConfig

torch.manual_seed(42)
random.seed(42)
##### Preprocessing Module #######
channels_to_keep = 3  # Number of channels to keep in the image (1 for grayscale, 3 for RGB)
# Set device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("device: ",device)
# DataLoader setup
batch_size = 8
kwargs = {'num_workers': 0, 'pin_memory': True} if device.type == 'cuda:0' else {}
folder = "cuda0"
os.makedirs(folder, exist_ok=True)
train_data_path = 'thirdmolar'

# Function to get label from the file name
def get_label(path: Path) -> int:
    return int(path.stem.split("_")[-1][0])

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, data_path, transform=None):
        self.data_path = data_path
        self.transform = transform
        self.image_paths = list(Path(data_path).glob("*.png"))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = int(img_path.stem.split("_")[-1][0])

        with open(img_path, 'rb') as f:
            image = Image.open(f).convert('RGB')  # Keep the image in RGB format

        if self.transform:
            image = self.transform(image)
        return image, label

# Transformation

if (channels_to_keep == 1):
    transform = transforms.Compose([
        transforms.Resize((128, 128)),  # Resize the image to 128x128 pixels
        transforms.Grayscale(num_output_channels=1),  # Convert to grayscale
        transforms.ToTensor(),          # Convert the image to a PyTorch tensor
    ])
else:
    transform = transforms.Compose([
        transforms.Resize((128, 128)),  # Resize the image to 128x128 pixels
        transforms.ToTensor(),          # Convert the image to a PyTorch tensor
    ])

train_dataset = CustomDataset(train_data_path, transform=transform)
data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, **kwargs)

# %matplotlib inline
import matplotlib.pyplot as plt

# Function to show images with labels
def show_images(images, labels, num_images=8):
    fig, axes = plt.subplots(1, num_images, figsize=(12, 3))
    axes = axes.flatten()
    for img, label, ax in zip(images, labels, axes):
        # Convert image tensor to numpy and change layout from CxHxW to HxWxC
        img = img.permute(1, 2, 0).numpy()
        ax.imshow(img)
        ax.set_title(f'Label: {label}')
        ax.axis('off')
    plt.tight_layout()
    plt.show()
    plt.savefig("cuda0/data_loader1.png",dpi=600)
    plt.close()

# Display some images from the dataset
for images, labels in data_loader:
    show_images(images, labels.numpy(), num_images=8)
    break

import torch
import torchvision
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from diffusers import DDPMScheduler, UNet2DModel
from matplotlib import pyplot as plt
from tqdm.auto import tqdm
print(f'Using device: {device}')

# Feed it into a dataloader (batch size 8 here just for demo)
train_dataloader = data_loader

# View some examples
x, y = next(iter(train_dataloader))
print('Input shape:', x.shape)
print('Labels:', y)
plt.imshow(torchvision.utils.make_grid(x)[0], cmap='gray')
plt.savefig("cuda0/data_loader2.png",dpi=600)
plt.close()


##### Diffusion Module #######

class ClassConditionedUnet(nn.Module):
  def __init__(self, num_classes=7, class_emb_size=7):
    super().__init__()
    
    # The embedding layer will map the class label to a vector of size class_emb_size
    self.class_emb = nn.Embedding(num_classes, class_emb_size)

    # Self.model is an unconditional UNet with extra input channels to accept the conditioning information (the class embedding)
    self.model = UNet2DModel(
        sample_size=128,           # the target image resolution
        in_channels= channels_to_keep + class_emb_size, # Additional input channels for class cond.
        out_channels=channels_to_keep,           # the number of output channels
        layers_per_block=3,       # how many ResNet layers to use per UNet block
        block_out_channels=(32, 64, 128), 
        down_block_types=( 
            "DownBlock2D",        # a regular ResNet downsampling block
            "AttnDownBlock2D",    # a ResNet downsampling block with spatial self-attention
            "AttnDownBlock2D",
        ), 
        up_block_types=(
            "AttnUpBlock2D",
            "AttnUpBlock2D",      # a ResNet upsampling block with spatial self-attention
            "UpBlock2D",          # a regular ResNet upsampling block
          ),
    )

  # Our forward method now takes the class labels as an additional argument
  def forward(self, x, t, class_labels):
    # Shape of x:
    bs, ch, w, h = x.shape
    
    # class conditioning in right shape to add as additional input channels
    class_cond = self.class_emb(class_labels) # Map to embedding dimension
    class_cond = class_cond.view(bs, class_cond.shape[1], 1, 1).expand(bs, class_cond.shape[1], w, h)
    # x is shape (bs, 1, 28, 28) and class_cond is now (bs, 4, 28, 28)

    # Net input is now x and class cond concatenated together along dimension 1
    net_input = torch.cat((x, class_cond), 1) # (bs, 7, 28, 28)

    # Feed this to the UNet alongside the timestep and return the prediction
    return self.model(net_input, t).sample # (bs, 1, 28, 28)
     
# Create a scheduler
noise_scheduler = DDPMScheduler(num_train_timesteps=1000, beta_schedule='squaredcos_cap_v2')
     

# Redefining the dataloader to set the batch size higher than the demo of 8
bsbs = 128
train_dataloader = DataLoader(train_dataset, batch_size=bsbs, shuffle=True)

# How many runs through the data should we do?
n_epochs = 200

# Our network 
net = ClassConditionedUnet().to(device)

# Our loss function
loss_fn = nn.MSELoss()

# The optimizer
opt = torch.optim.Adam(net.parameters(), lr=1e-3) 

# Keeping a record of the losses for later viewing
losses = []

# The training loop
for epoch in range(n_epochs):
    for x, y in tqdm(train_dataloader):
        
        # Get some data and prepare the corrupted version
        x = x.to(device) * 2 - 1 # Data on the GPU (mapped to (-1, 1)) --------------------------------- Potentially need to change this 
        y = y.to(device)
        noise = torch.randn_like(x)
        timesteps = torch.randint(0, 999, (x.shape[0],)).long().to(device)
        noisy_x = noise_scheduler.add_noise(x, noise, timesteps)

        # Get the model prediction
        pred = net(noisy_x, timesteps, y) # Note that we pass in the labels y

        # Calculate the loss
        loss = loss_fn(pred, noise) # How close is the output to the noise

        # Backprop and update the params:
        opt.zero_grad()
        loss.backward()
        opt.step()

        # Store the loss for later
        losses.append(loss.item())

    # Print out the average of the last 100 loss values to get an idea of progress:
    avg_loss = sum(losses[-100:])/100
    print(f'Finished epoch {epoch}. Average of the last 100 loss values: {avg_loss:05f}')

    # View the loss curve
    plt.plot(losses)
    plt.savefig("cuda0/loss_plot"+str(epoch)+".png",dpi=600)
    plt.close()
    if epoch == 0:
        continue  # Skip the first epoch for sampling
    if epoch % 15 == 0:
        # Sampling and saving images
        # Prepare random x to start from, plus some desired labels y
        x = torch.randn(56, channels_to_keep, 128, 128).to(device)
        y = torch.tensor([[i]*8 for i in range(7)]).flatten().to(device)

        # Sampling loop
        for i, t in tqdm(enumerate(noise_scheduler.timesteps)):
            # Get model pred
            with torch.no_grad():
                residual = net(x, t, y)  # Again, note that we pass in our labels y

            # Update sample with step
            x = noise_scheduler.step(residual, t, x).prev_sample

        # Create the image grid
        grid = torchvision.utils.make_grid(x.detach().cpu().clip(-1, 1), nrow=8) # --------------------------------- Potentially need to change this 

        # Show the results with row labels
        fig, ax = plt.subplots(figsize=(12, 12))
        ax.imshow(grid[0], cmap='gray')
        ax.axis('off')


        plt.tight_layout()
        plt.savefig(f"cuda0/generated{epoch}.png", dpi=600, bbox_inches='tight')
        plt.close()

        # Create directory structure and save individual images
        epoch_dir = f"cuda0/epoch{epoch}"
        os.makedirs(epoch_dir, exist_ok=True)

        # Save individual images
        for idx, (image, label) in enumerate(zip(x, y)):
            # Create class directory if it doesn't exist
            class_dir = os.path.join(epoch_dir, str(label.item()))
            os.makedirs(class_dir, exist_ok=True)
            
            # Determine the sample number (0-7 since we have 8 samples per class)
            sample_num = idx % 8
            
            # Prepare the image (convert to PIL format)
            img = image.detach().cpu().squeeze()  # Remove channel dimension for grayscale
            img = (img + 1) / 2  # Scale from [-1, 1] to [0, 1] --------------------------------- Potentially need to change this 
            img = transforms.ToPILImage()(img)
            
            # Save the image
            img.save(os.path.join(class_dir, f"{sample_num}.png"))