# 🧠 Brain MRI Segmentation with Unsupervised Preprocessing (NINS 2022 dataset)

This repository handles **data preprocessing**, **unsupervised mask generation**, and **training a U-Net** model for brain tissue segmentation (Gray Matter, White Matter, Background) using the NINS 2022 dataset (https://figshare.com/articles/dataset/Brain_MRI_Dataset/14778750).

---

## 📂 Project Structure
brain_unet/  
├── checkpoints/# Saved model weights  
├── data/ # kmeans_masks and processed MRI images (raw datasest available in link above)  
├── datasets/ # Custom PyTorch dataset class  
├── models/ # U-Net model  
├── results  
├── scripts/  
│ ├── 01_preprocess.py  
│ ├── 02_cluster_kmeans_masks.py  
│ ├── 03_generate_kmeans_masks.py  
│ ├── 04_train_unet.py  
│ └── 05_visualize_predictions.py  
│ └── 06_eval_unet_dice.py  
│ └── 07_track_dice_over_epochs.py  

Each image is clustered into 3 tissue types:
Class 0: Background  
Class 1: Gray Matter  
Class 2: White Matter

- Run uv run scripts/03_generate_kmeans_masks.py
Train a U-Net on the images and K-means masks:
- uv run scripts/04_train_unet.py