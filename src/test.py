import os
import numpy as np
import torch
from PIL import ImageOps
from matplotlib import pyplot as plt

from data_loading import BlastocystDataset

# 1. Paths & dirs
img_path  = "../examples/images/Blast_PCRM_1201754 D5.BMP"
mask_path = "../examples/masks/Blast_PCRM_1201754 D5.png"
image_dir = os.path.dirname(img_path)
mask_dir  = os.path.dirname(mask_path)
fname     = os.path.basename(img_path)

# 2. Build dataset with augment=True and fixed seed
ds = BlastocystDataset(
    image_dir=image_dir,
    mask_dir=mask_dir,
    seed=42,
    augment=True
)

# 3. Find the index of our target file
idx = ds.image_filenames.index(fname)

# 4. Plot N different augmentations
N = 3
fig, axes = plt.subplots(N, 2, figsize=(6, 3*N))
for i in range(N):
    img_t, mask_t, _ = ds[idx]
    img_arr = img_t.cpu().numpy().squeeze()
    mask_arr = mask_t.cpu().numpy().astype("uint8")

    # show image
    H, W = img_arr.shape
    plt.figure(figsize=(W/100, H/100), dpi=100)
    plt.imshow(img_arr, cmap="gray", interpolation="nearest")
    plt.title(f"Aug {i+1} – image ({W}×{H})")
    plt.axis("off")
    plt.show()

    # show mask
    plt.figure(figsize=(W/100, H/100), dpi=100)
    plt.imshow(mask_arr, cmap="tab20", interpolation="nearest")
    plt.title(f"Aug {i+1} – mask")
    plt.axis("off")
    plt.show()

