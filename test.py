import os
import re
import sys

import cv2
import numpy as np
import torch
from torchvision.utils import save_image

from net.model import DD_UIE


# ---------------------------------------------------------
# Device and GPU setup
# ---------------------------------------------------------
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2"
device_ids = [0, 1, 2]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dtype_np = "float32"


# ---------------------------------------------------------
# Utility: load all images from a folder
# ---------------------------------------------------------
def load_image_paths(folder):
    """
    List all image files from a folder and sort them numerically by filename.
    Assumes filenames are like '0001.png', '2.jpg', etc.
    """
    files = [f for f in os.listdir(folder) if not f.startswith(".")]
    # Try numeric sort based on the stem before the first dot
    try:
        files.sort(key=lambda x: int(os.path.splitext(x)[0]))
    except ValueError:
        files.sort()
    return [os.path.join(folder, f) for f in files]


# ---------------------------------------------------------
# Utility: find the latest DD_UIE checkpoint
# ---------------------------------------------------------
def find_latest_checkpoint(ckpt_dir="saved_models"):
    """
    Find the checkpoint with the largest epoch index matching 'DD_UIE_*.pth'.
    Returns the full path or raises RuntimeError if none is found.
    """
    if not os.path.isdir(ckpt_dir):
        raise RuntimeError(f"Checkpoint directory not found: {ckpt_dir}")

    pattern = re.compile(r"DD_UIE_(\d+)\.pth$")
    best_epoch = None
    best_path = None

    for fname in os.listdir(ckpt_dir):
        m = pattern.match(fname)
        if m is not None:
            epoch = int(m.group(1))
            if best_epoch is None or epoch > best_epoch:
                best_epoch = epoch
                best_path = os.path.join(ckpt_dir, fname)

    if best_path is None:
        raise RuntimeError("No checkpoint matching 'DD_UIE_*.pth' found in saved_models/.")

    print(f"Using checkpoint: {best_path} (epoch {best_epoch})")
    return best_path


# ---------------------------------------------------------
# Main testing routine
# ---------------------------------------------------------
def main():
    # Directories for test input and output images
    input_dir = "./data/Test_400/input/"
    output_dir = "./data/Test_400/output/"
    os.makedirs(output_dir, exist_ok=True)

    # Build model and wrap with DataParallel if multiple GPUs are visible
    model = DD_UIE(in_channels=3, channels=16, num_resblock=4, num_memblock=4)
    if len(device_ids) > 1 and torch.cuda.is_available():
        model = torch.nn.DataParallel(model, device_ids=device_ids)
    model = model.to(device)

    # Load checkpoint
    ckpt_path = find_latest_checkpoint("saved_models")
    state_dict = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    # Collect test image paths
    img_paths = load_image_paths(input_dir)
    if len(img_paths) == 0:
        print(f"No images found in {input_dir}.")
        return

    print(f"Found {len(img_paths)} test images in {input_dir}.")
    print(f"Saving enhanced images to {output_dir}.")

    # Inference loop
    with torch.no_grad():
        for idx, img_path in enumerate(img_paths):
            # Read and preprocess image
            img = cv2.imread(img_path)
            if img is None:
                print(f"[Warning] Failed to read image: {img_path}")
                continue

            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (256, 256))

            img_np = np.array(img).astype(dtype_np)
            img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0)  # (1, 3, H, W)
            img_tensor = img_tensor / 255.0
            img_tensor = img_tensor.to(device)

            # Forward pass
            out = model(img_tensor)
            out = out.clamp(0.0, 1.0)

            # Save result with the same filename in the output folder
            filename = os.path.basename(img_path)
            save_path = os.path.join(output_dir, filename)
            save_image(out, save_path, nrow=1, normalize=True)

            if (idx + 1) % 50 == 0 or (idx + 1) == len(img_paths):
                print(f"Processed {idx + 1}/{len(img_paths)} images")

    print("Testing finished.")


if __name__ == "__main__":
    main()
