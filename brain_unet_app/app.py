import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))

import gradio as gr
import torch
import numpy as np
from torchvision import transforms
from models.unet import UNet
from PIL import Image
import matplotlib.pyplot as plt
import io

# Load model
model_path = "checkpoints/unet_epoch10.pth"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = UNet(in_channels=1, out_channels=3).to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# Preprocessing
def preprocess_image(img):
    gray = img.convert("L").resize((128, 128))
    array = np.array(gray).astype(np.float32) / 255.0
    tensor = torch.from_numpy(array).unsqueeze(0).unsqueeze(0).to(device)
    return tensor

# Prediction
def segment(image):
    # Preprocess
    input_tensor = preprocess_image(image)
    with torch.no_grad():
        output = model(input_tensor)
        pred = torch.argmax(output.squeeze(), dim=0).cpu().numpy()
    
    # Class labels
    class_labels = {
        0: "Background",
        1: "Gray Matter",
        2: "White Matter"
    }

    # Stats
    total_pixels = pred.size
    stats = {}
    for i in range(3):
        count = (pred == i).sum()
        percent = count / total_pixels
        stats[f"{class_labels[i]}"] = round(percent, 2)
    
    # Convert original to grayscale numpy
    base_img = np.array(image.convert("L").resize((128, 128)))

    # Create color mask
    overlay = np.zeros((128, 128, 3), dtype=np.uint8)

    # Define RGB colors for each class
    colors = {
        1: [255, 0, 0],   # Red for gray matter
        2: [0, 255, 0]    # Green for white matter
    }

    for cls, color in colors.items():
        overlay[pred == cls] = color

    # Blend overlay with original image
    blended = np.stack([base_img]*3, axis=-1)
    alpha = 0.5  # transparency
    blended = (1 - alpha) * blended + alpha * overlay
    blended = blended.astype(np.uint8)

    # Convert to image
    output_image = Image.fromarray(blended)

    # # Plot side-by-side
    # fig, axes = plt.subplots(1, 2, figsize=(6, 3))
    # axes[0].imshow(image.convert("L").resize((128, 128)), cmap="gray")
    # axes[0].set_title("Original")
    # axes[1].imshow(pred, cmap="jet")
    # axes[1].set_title("Prediction")
    # for ax in axes:
    #     ax.axis("off")
    # buf = io.BytesIO()
    # plt.tight_layout()
    # plt.savefig(buf, format="png")
    # plt.close(fig)
    # buf.seek(0)
    # output_image = Image.open(buf)

    # Visualize only the predicted mask
    # fig, ax = plt.subplots(figsize=(3, 3))
    # ax.imshow(pred, cmap="jet")
    # ax.set_title("Segmentation")
    # ax.axis("off")

    # # Save to buffer
    # buf = io.BytesIO()
    # plt.tight_layout()
    # plt.savefig(buf, format="png")
    # plt.close(fig)
    # buf.seek(0)
    # output_image = Image.open(buf)

    return output_image, stats

# Gradio UI
demo = gr.Interface(
    fn=segment,
    inputs=gr.Image(type="pil", label="Upload Brain MRI (.jpg, .png)"),
    outputs=[
        gr.Image(label="Segmented Output"),
        gr.Label(label="Tissue Composition (%)")
    ],
    title="Brain MRI Segmentation (U-Net)",
    description="Upload a T1-weighted brain slice. The model will segment it into tissue classes and estimate area percentages.",
    allow_flagging="never"
)

if __name__ == "__main__":
    demo.launch()
