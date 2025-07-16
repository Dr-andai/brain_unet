import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from pathlib import Path
import random

# all .npy files
sample_path = Path("data/processed_numpy")
npy_files = list(sample_path.rglob("*.npy"))

# get random mri
random_file = random.choice(npy_files)

# Load and reshape image
img = np.load(random_file)  # shape: (128, 128)
pixels = img.reshape(-1, 1)

# Apply KMeans
kmeans = KMeans(n_clusters=3, random_state=42)
labels = kmeans.fit_predict(pixels)

# Reshape labels to 2D mask
seg_mask = labels.reshape(img.shape)

# Sort clusters by intensity mean (low → high)
cluster_means = [pixels[labels == i].mean() for i in range(3)]
sorted_indices = np.argsort(cluster_means)
sorted_mask = np.zeros_like(seg_mask)
for new_label, original_index in enumerate(sorted_indices):
    sorted_mask[seg_mask == original_index] = new_label

# Visualize
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.imshow(img, cmap="gray")
plt.title("Original Slice")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(sorted_mask, cmap="viridis")
plt.title("K-Means Segmentation (3 Clusters)")
plt.axis("off")

plt.tight_layout()
plt.show()