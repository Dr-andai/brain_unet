import os
import cv2
import numpy as np
from tqdm import tqdm
from pathlib import Path

# Config
INPUT_DIR = Path("data/jpegs")
OUTPUT_DIR = Path("data/processed_numpy")
IMAGE_SIZE = (128, 128)

os.makedirs(OUTPUT_DIR, exist_ok=True)

def preprocess_image(img_path, size=(128,128)):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    img_resized = cv2.resize(img, size)
    img_norm = img_resized.astype(np.float32)/255.0
    return img_norm

# Process all images
# image_paths = list(Path(INPUT_DIR).glob("*.jpg"))
image_paths = [p for p in INPUT_DIR.rglob("*") if p.suffix.lower() in [".jpg", ".jpeg"]]


for img_path in tqdm(image_paths, desc="Preprocessing images"):
    img_array = preprocess_image(str(img_path), IMAGE_SIZE)
    
    # Construct the output path, preserving relative subfolders
    relative_path = img_path.relative_to(INPUT_DIR).with_suffix(".npy")
    out_path = OUTPUT_DIR / relative_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save the processed numpy array
    np.save(out_path, img_array)
