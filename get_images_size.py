import os
from PIL import Image

# Path to your image folder
folder = "../data/masks"

widths = []
heights = []

for filename in os.listdir(folder):
    if filename.lower().endswith((".jpg", ".png", ".jpeg", ".bmp")):
        img_path = os.path.join(folder, filename)
        with Image.open(img_path) as img:
            w, h = img.size
            widths.append(w)
            heights.append(h)

print("Minimum width:", min(widths))
print("Maximum width:", max(widths))
print("Minimum height:", min(heights))
print("Maximum height:", max(heights))
