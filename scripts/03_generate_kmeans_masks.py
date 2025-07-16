from pathlib import Path
import numpy as np
from sklearn.cluster import KMeans
from tqdm import tqdm
import os

# PATHS
INPUT_DIR = Path("data/processed_numpy")
OUTPUT_DIR = Path("data/kmeans_masks")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Load all slices
npy_files = list(INPUT_DIR.rglob("*.npy"))

def segment_kmeans(img: np.ndarray, k=3) -> np.ndarray:
    pixels = img.reshape(-1, 1)
    kmeans = KMeans(n_clusters=k, random_state=42)
    labels = kmeans.fit_predict(pixels)
    mask = labels.reshape(img.shape)

    # Sort clusters by mean intensity
    cluster_means = [pixels[labels == i].mean() for i in range(k)]
    sorted_indices = np.argsort(cluster_means)
    sorted_mask = np.zeros_like(mask)
    for new_label, old_index in enumerate(sorted_indices):
        sorted_mask[mask == old_index] = new_label
    return sorted_mask

# Process and save masks
for npy_file in tqdm(npy_files, desc="Generating masks"):
    img = np.load(npy_file)
    mask = segment_kmeans(img)

    # Save using same relative path
    relative_path = npy_file.relative_to(INPUT_DIR).with_suffix(".npy")
    out_path = OUTPUT_DIR / relative_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_path, mask)