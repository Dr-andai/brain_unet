import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))
""""
Import PyTorch, model, dataset, optimizer
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from datasets.dataset import BrainMRIDataset
from models.unet import UNet

# Configs
EPOCHS = 20
BATCH_SIZE = 8
LEARNING_RATE = 1e-4
VALID_SPLIT = 0.2
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Paths
image_dir = Path("data/processed_numpy")
mask_dir = Path("data/kmeans_masks")
checkpoint_dir = Path("checkpoints")
checkpoint_dir.mkdir(exist_ok = True)

# Dataset
full_dataset = BrainMRIDataset(image_dir, mask_dir)
# Add this:
# print("🧪 mask_dir exists:", hasattr(full_dataset, "mask_dir"))
# print("🔎 mask_dir =", getattr(full_dataset, "mask_dir", "NOT FOUND"))

# Train/val split
val_size = int(len(full_dataset)*VALID_SPLIT)
train_size = len(full_dataset) - val_size
train_dataset, val_dataset = random_split(full_dataset,[train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

# Model
""""
Initialize model, from models.unet.py
Send to cuda/cpu
"""
model = UNet(in_channels=1, out_channels=3).to(DEVICE)

# Loss and Optimizer
""""
Defining loss function and optimizer
"""
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Training loop
"""
For each epoch: loop through batches, 
- load image and mask
- forward pass
- compute loss
- backward pass
- update weights
"""
for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0

    for images, masks in train_loader:
        images = images.to(DEVICE)
        masks = masks.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
    
    avg_train_loss = running_loss / len(train_loader)

    # Validation
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for images, masks in val_loader:
            images = images.to(DEVICE)
            masks = masks.to(DEVICE)
            outputs = model(images)
            loss = criterion(outputs, masks)
            val_loss += loss.item()
    
    avg_val_loss = val_loss / len(val_loader)

    print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

    # Save checkpoint
    torch.save(model.state_dict(), checkpoint_dir / f"unet_epoch{epoch+1}.pth")
