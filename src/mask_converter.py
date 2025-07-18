import numpy as np
from PIL import Image
import os
from scipy.ndimage import binary_fill_holes

# ─── Colour mapping for each structure ────────────────────────────────────────
rgb_dict = {
    'ICM':        np.array([ 51, 221, 255], dtype=np.uint8),   # Blue
    'TE':         np.array([250,  50,  83], dtype=np.uint8),   # Red
    'ZP':         np.array([ 61, 245,  61], dtype=np.uint8),   # Green
    'BL': np.array([255, 245,  61], dtype=np.uint8),           # Yellow
    'background': np.array([  0,   0,   0], dtype=np.uint8)    # Black
}

# ─── Input / output directories ──────────────────────────────────────────────
icm_dir    = r"../data/GT_ICM"
te_dir     = r"../data/GT_TE"
zp_dir     = r"../data/GT_ZP"
output_dir = r"../data/full_masks"
os.makedirs(output_dir, exist_ok=True)

# ─── Process each mask pair ──────────────────────────────────────────────────
for file_name in os.listdir(icm_dir):
    if not file_name.endswith(".bmp"):
        continue

    base = file_name.replace(" ICM_Mask.bmp", "")

    # 1) Load binary masks (True = mask)
    icm_mask = (np.array(Image.open(os.path.join(icm_dir, f"{base} ICM_Mask.bmp")).convert("L")) == 255)
    te_mask  = (np.array(Image.open(os.path.join(te_dir,  f"{base} TE_Mask.bmp")).convert("L")) == 255)
    zp_mask  = (np.array(Image.open(os.path.join(zp_dir,  f"{base} ZP_Mask.bmp")).convert("L")) == 255)

    # 2) Compute blastocoel:
    #    a) fill holes in ZP to get full interior (ring + cavity)
    filled_zp = binary_fill_holes(zp_mask)
    #    b) isolate only the interior (subtract the ring itself)
    interior  = filled_zp & ~zp_mask
    #    c) subtract ICM and TE to leave just the blastocoel
    bc_mask   = interior & ~(icm_mask | te_mask)

    # 3) Paint onto an RGB canvas
    H, W = icm_mask.shape
    canvas = np.zeros((H, W, 3), dtype=np.uint8)
    canvas[icm_mask] = rgb_dict['ICM']
    canvas[te_mask]  = rgb_dict['TE']
    canvas[zp_mask]  = rgb_dict['ZP']
    canvas[bc_mask]  = rgb_dict['BL']
    # everything else remains black

    # 4) Save the result
    out_path = os.path.join(output_dir, f"{base}.png")
    Image.fromarray(canvas).save(out_path)

print("RGB segmentation masks generated successfully!")
