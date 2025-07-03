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
from diffusers import UNet2DConditionModel  # Changed import

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
    def __init__(self, num_classes=7, class_emb_size=128):
        super().__init__()
        self.class_emb = nn.Embedding(num_classes, class_emb_size)

        self.model = UNet2DConditionModel(
            sample_size=128,
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
        class_cond = self.class_emb(class_labels).unsqueeze(1)  # (B, 1, class_emb_size)
        return self.model(x, timestep=t, encoder_hidden_states=class_cond).sample

    


def save_checkpoint(epoch, model, optimizer, losses, path="checkpoints"):
    os.makedirs(path, exist_ok=True)
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'losses': losses,
    }
    torch.save(checkpoint, f"{path}/checkpoint_epoch_{epoch}.pth")
    print(f"Checkpoint saved at epoch {epoch}")


def load_checkpoint(path, model, optimizer):
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    losses = checkpoint['losses']
    return epoch, losses

# Create a scheduler
noise_scheduler = DDPMScheduler(num_train_timesteps=1000, beta_schedule='squaredcos_cap_v2')
     

# Redefining the dataloader to set the batch size higher than the demo of 8
bsbs = 128
train_dataloader = DataLoader(train_dataset, batch_size=bsbs, shuffle=True)

# How many runs through the data should we do?
n_epochs = 1000

# Our network 
net = ClassConditionedUnet().to(device)
from torchinfo import summary
summary(
    net,
    input_size=[(2, channels_to_keep, 128, 128), (2,), (2,)],
    dtypes=[torch.float, torch.long, torch.long],
    device=device
)
# Our loss function
loss_fn = nn.MSELoss()

# The optimizer
opt = torch.optim.Adam(net.parameters(), lr=1e-3) 

# Keeping a record of the losses for later viewing
losses = []
# Checkpoint directory setup
checkpoint_dir = "checkpoints"
os.makedirs(checkpoint_dir, exist_ok=True)

# Optional: Load from checkpoint if resuming training
start_epoch = 0
last_epoch = 255
if os.path.exists("checkpoints/checkpoint_epoch_{}.pth".format(last_epoch)):
    start_epoch, losses = load_checkpoint("checkpoints/checkpoint_epoch_{}.pth".format(last_epoch), net, opt)
    print(f"Resuming training from epoch {start_epoch}")


# The training loop
for epoch in range(last_epoch+1, n_epochs):
    for x, y in tqdm(train_dataloader):
        # Get some data and prepare the corrupted version
        x = x.to(device) * 2 - 1 # Data on the GPU (mapped to (-1, 1))
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
    if epoch == 0:
        continue  # Skip saving the first epoch
    if epoch % 15 == 0 or epoch == n_epochs - 1:
        save_checkpoint(epoch, net, opt, losses)
    # View the loss curve
    plt.plot(losses)
    plt.savefig("cuda0/loss_plot"+str(epoch)+".png",dpi=600)
    plt.close()
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
        grid = torchvision.utils.make_grid(x.detach().cpu().clip(-1, 1), nrow=8) # 

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
            img = image.detach().cpu() # Remove channel dimension for grayscale
            img = (img + 1) / 2  # Scale from [-1, 1] to [0, 1] 
            img = transforms.ToPILImage()(img)
            
            # Save the image
            img.save(os.path.join(class_dir, f"{sample_num}.png"))