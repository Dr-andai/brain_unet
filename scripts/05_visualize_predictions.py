import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))

import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from models.unet import UNet
from datasets.dataset import BrainMRIDataset


# Paths
checkpoint_path = Path("checkpoints/unet_epoch20.pth")
image_dir = Path("data/processed_numpy")
mask_dir = Path("data/kmeans_masks")

# Load Validation set
dataset = BrainMRIDataset(image_dir, mask_dir)
val_loader = DataLoader(dataset, batch_size=1, shuffle=True)

# Load model
model = UNet(in_channels=1, out_channels=3)
model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
model.eval()

# Visualize a few prediction
n_samples = 5
for i, (image, mask) in enumerate(val_loader):
    if i>= n_samples:
        break

    with torch.no_grad():
        output = model(image)
        pred = torch.argmax(output, dim=1).squeeze().numpy()
    
    img_np = image.squeeze().numpy()
    mask_np = mask.squeeze().numpy()

    # Plot
    fig, axs = plt.subplots(1, 3, figsize=(12, 4))
    axs[0].imshow(img_np, cmap='gray')
    axs[0].set_title("Input Image")

    axs[1].imshow(mask_np, cmap='viridis')
    axs[1].set_title("K-Means Mask")

    axs[2].imshow(pred, cmap='viridis')
    axs[2].set_title("U-Net Prediction")

    for ax in axs:
        ax.axis('off')
    plt.tight_layout()
    plt.show()