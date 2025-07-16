import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))

import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from models.unet import UNet
from datasets.dataset import BrainMRIDataset
from tqdm import tqdm
import matplotlib.pyplot as plt

# Paths
checkpoints_dir = Path("checkpoints")
image_dir = Path("data/processed_numpy")
mask_dir = Path("data/kmeans_masks")
output_csv = Path("results/dice_scores.csv")
output_plot = Path("results/dice_plot.png")

# Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dataset = BrainMRIDataset(image_dir, mask_dir)
val_loader = DataLoader(dataset, batch_size=1, shuffle=False)

# Epoch and Dice relationship
"""
- Early Epochs - Dice starts low — the model doesn't yet understand the task
- Middle Epochs	Dice increases as the model learns to better segment regions
- Later Epochs	Dice plateaus — model can't improve further on validation data
"""
# Dice function
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

# Evaluate each checkpoint
results = []
checkpoints = sorted(checkpoints_dir.glob("unet_epoch*.pth"))

for ckpt in checkpoints:
    print(f"Evaluating {ckpt.name}")
    model = UNet(in_channels=1, out_channels=3).to(device)
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.eval()

    all_dice = []

    for image, mask in tqdm(val_loader, desc=f"  → {ckpt.name}"):
        image, mask = image.to(device), mask.to(device)
        with torch.no_grad():
            pred = model(image)
            pred = torch.argmax(pred, dim=1)
        dice = dice_score(pred.squeeze(), mask.squeeze(), num_classes=3)
        all_dice.append(torch.stack(dice))

    all_dice = torch.stack(all_dice)
    mean_dice = all_dice.mean(dim=0)
    row = {
        "epoch": int(ckpt.stem.split("epoch")[-1]),
        "dice_class0": mean_dice[0].item(),
        "dice_class1": mean_dice[1].item(),
        "dice_class2": mean_dice[2].item(),
        "dice_mean": mean_dice.mean().item()
    }
    results.append(row)

# Save to CSV
df = pd.DataFrame(results).sort_values("epoch")
output_csv.parent.mkdir(exist_ok=True)
df.to_csv(output_csv, index=False)
print(f"\n✅ Saved Dice scores to {output_csv}")

# Shwo plot
plt.figure(figsize=(10, 6))
plt.plot(df["epoch"], df["dice_class0"], label="Class 0")
plt.plot(df["epoch"], df["dice_class1"], label="Class 1")
plt.plot(df["epoch"], df["dice_class2"], label="Class 2")
plt.plot(df["epoch"], df["dice_mean"], label="Mean Dice", linestyle="--", color="black")
plt.xlabel("Epoch")
plt.ylabel("Dice Score")
plt.title("Dice Score over Epochs")
plt.legend()
plt.grid(True)
plt.savefig(output_plot)
print(f"📈 Saved plot to {output_plot}")
plt.show()