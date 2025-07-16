import torch
from torch.utils.data import Dataset
from pathlib import Path
import numpy as np

"""
- Acts like a smart list that returns (image, mask) pairs
- Handles loading, transforming, and returning tensors
- Is passed to a DataLoader to train the model in batches
"""
class BrainMRIDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None): # this method takes in paths to preprocessed and corresponding masked images
        print("✅ Initializing BrainMRIDataset")
        self.image_dir = Path(image_dir)
        self.mask_dir = Path(mask_dir)
        self.transform = transform

        # match image and mask
        self.image_paths = list(self.image_dir.rglob("*.npy"))
    
    def __len__(self):
        return len(self.image_paths) # Returns: The total number of image-mask pairs
    
    def __getitem__(self, index):
        """
        - Loads the image and mask at the given index
        - Converts them to PyTorch tensors (FloatTensor for image, LongTensor for mask)
        - Returns: (image_tensor, mask_tensor)
        """
        image_path = self.image_paths[index]
        relative_path = image_path.relative_to(self.image_dir)
        mask_path = self.mask_dir/relative_path

        # Load iamge and mask
        image = np.load(image_path).astype(np.float32)
        mask = np.load(mask_path).astype(np.int64)

        # Add channel dimension to image
        image = torch.from_numpy(image).unsqueeze(0)
        mask = torch.from_numpy(mask)

        if self.transform:
            image, mask = self.transform(image, mask)
        
        return image, mask