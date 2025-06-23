import numpy as np
from PIL import Image
import os

# Colour mapping for each structure
rgb_dict = {
    'ICM': np.array([51,221,255], dtype=np.uint8),     # Blue
    'TE':  np.array([250,50,83], dtype=np.uint8),      # Red
    'ZP':  np.array([61,245,61], dtype=np.uint8),      # Green
    'background': np.array([0, 0, 0], dtype=np.uint8)  # Black
}

# Input directories for grayscale masks
icm_dir = r"../data/GT_ICM"
te_dir  = r"../data/GT_TE"
zp_dir  = r"../data/GT_ZP"

# Output directory for RGB masks
output_dir = r"../data/masks"
os.makedirs(output_dir, exist_ok=True)

# Loop over mask filenames (using ICM as base reference)
for file_name in os.listdir(icm_dir):
    if not file_name.endswith(".bmp"):
        continue  # Skip non-mask files

    base_name = file_name.replace(" ICM_Mask.bmp", "")

    # Load grayscale masks as 2D arrays
    icm_mask = np.array(Image.open(os.path.join(icm_dir, f"{base_name} ICM_Mask.bmp")).convert('L'))
    te_mask  = np.array(Image.open(os.path.join(te_dir,  f"{base_name} TE_Mask.bmp")).convert('L'))
    zp_mask  = np.array(Image.open(os.path.join(zp_dir,  f"{base_name} ZP_Mask.bmp")).convert('L'))

    # Initialise an empty RGB image
    H, W = icm_mask.shape
    rgb_image = np.zeros((H, W, 3), dtype=np.uint8)

    # Apply RGB colours for each structure
    rgb_image[icm_mask == 255] = rgb_dict['ICM']
    rgb_image[te_mask  == 255] = rgb_dict['TE']
    rgb_image[zp_mask  == 255] = rgb_dict['ZP']
    # Remaining areas stay black (background)

    # Save the final RGB mask
    out_path = os.path.join(output_dir, f"{base_name}.png")
    Image.fromarray(rgb_image).save(out_path)

print("RGB segmentation masks successfully generated")
