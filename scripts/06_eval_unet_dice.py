import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))
import torch
import numpy as np
from torch.utils.data import DataLoader
from models.unet import UNet
from datasets.dataset import BrainMRIDataset
from tqdm import tqdm

# Set paths
checkpoint_path = Path("checkpoints/unet_epoch10.pth")
image_dir = Path("data/processed_numpy")
mask_dir = Path("data/kmeans_masks")

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Dataset & loader
dataset = BrainMRIDataset(image_dir, mask_dir)
val_loader = DataLoader(dataset, batch_size=1, shuffle=False)

# Load model
model = UNet(in_channels=1, out_channels=3).to(device)
model.load_state_dict(torch.load(checkpoint_path, map_location=device))
model.eval()

# Dice score function
"""
Dice score = how well the predicted segmentation matches the ground truth mask.
The Dice coefficient measures overlap between:
- the predicted mask (P) and
- the ground truth mask (G)
"""
def dice_score(pred, target, num_classes=3):
    dice = []
    for cls in range(num_classes):
        pred_cls = (pred == cls).float()
        target_cls = (target == cls).float()
        intersection = (pred_cls * target_cls).sum()
        denominator = pred_cls.sum() + target_cls.sum()
        if denominator == 0:
            dice.append(torch.tensor(1.0))
        else:
            dice.append((2. * intersection) / denominator)
    return dice

# Evaluation loop
dice_scores = []

for image, mask in tqdm(val_loader, desc="Evaluating"):
    image, mask = image.to(device), mask.to(device)

    with torch.no_grad():
        output = model(image)
        pred = torch.argmax(output, dim=1)
    
    dice = dice_score(pred.squeeze(), mask.squeeze(), num_classes=3)
    dice_scores.append(torch.stack(dice))

# Average Dice
dice_scores = torch.stack(dice_scores)
mean_dice = dice_scores.mean(dim=0)

for i, score in enumerate(mean_dice):
    print(f"Dice Score - Class {i}: {score.item()}:.4f")
print(f"\nMean Dice Score: {mean_dice.mean().item():.4f}")