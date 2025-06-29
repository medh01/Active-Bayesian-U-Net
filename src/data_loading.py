import os
import random

from PIL import Image, ImageOps
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from albumentations import Compose, Affine
from albumentations.pytorch import ToTensorV2

Classes = {
    (51, 221, 255): 0,   # ICM
    (250,  50,  83): 1,  # TE
    ( 61, 245,  61): 2,  # ZP
    (  0,   0,   0): 3   # background
}

TARGET_SIZE = (624, 480)         # (W, H)

#Label Encoding
def mask_encoding(arr):
    h, w, _ = arr.shape
    class_mask = np.zeros((h, w), dtype=np.uint8)
    for rgb, idx in Classes.items():
        class_mask[np.all(arr == rgb, axis=-1)] = idx
    return class_mask


class BlastocystDataset(Dataset):
    def __init__(self,
                 image_dir,
                 mask_dir,
                 augment=False,
                 rotate_limit=30,
                 shift_limit=0.1,
                 scale_limit=0.2,
                 shear_limit=20,
                 ):
        self.image_dir = image_dir
        self.mask_dir  = mask_dir
        self.image_filenames = sorted(os.listdir(image_dir))
        self.augment = augment
        if augment:
            self.transform = Compose([
                Affine(
                    rotate=rotate_limit,
                    translate_percent=shift_limit,
                    scale=(1 - scale_limit, 1 + scale_limit),
                    shear=shear_limit,
                    p=1.0,
                ),
                ToTensorV2(),
            ], additional_targets={'mask': 'mask'})
        else:
            self.transform = Compose([
                ToTensorV2(),
            ], additional_targets={'mask': 'mask'})


    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        img_name  = self.image_filenames[idx]
        img_path  = os.path.join(self.image_dir, img_name)

        core      = os.path.splitext(img_name)[0]
        mask_path = os.path.join(self.mask_dir, core + ".png")

        # load / pad grayscale image
        img = Image.open(img_path).convert("L")
        img = ImageOps.pad(img, TARGET_SIZE, method=Image.BILINEAR, color=0)
        img = np.array(img, np.float32) / 255.0                    # [H,W] float
                                                                   # [1,H,W]

        # load / pad mask, then encode to classes
        mask_rgb = Image.open(mask_path).convert("RGB")
        mask_rgb = ImageOps.pad(mask_rgb, TARGET_SIZE, method=Image.NEAREST, color=(0, 0, 0))
        mask_arr = np.array(mask_rgb)
        mask = mask_encoding(mask_arr)                          # [H,W] uint8
                                                                    # int64 for CE-loss
        # Apply transformations (augmentations + ToTensorV2)
        transformed = self.transform(image=img, mask=mask)
        img = transformed['image']
        mask = transformed['mask'].long()  # Ensure mask is LongTensor for CrossEntropyLoss

        return img, mask, img_name   # img→[1,H,W], mask→[H,W]


class UnlabeledBlastocystDataset(Dataset):
    def __init__(self, image_dir):
        self.image_dir = image_dir
        self.image_filenames = sorted(os.listdir(image_dir))

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        img_name = self.image_filenames[idx]
        img_path = os.path.join(self.image_dir, img_name)

        img = Image.open(img_path).convert("L")
        img = ImageOps.pad(img, TARGET_SIZE, method=Image.BILINEAR, color=0)
        img = np.array(img, np.float32) / 255.0
        img = torch.from_numpy(img).unsqueeze(0)  # [1, H, W]

        return img, img_name

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def get_loaders_active(
        labeled_img_dir,
        labeled_mask_dir,
        unlabeled_img_dir,
        test_img_dir,
        test_mask_dir,
        batch_size,
        augment = False,
        generator=None,
        num_workers=4,
        pin_memory=True,
):
    # 1. Labeled Dataset (with masks)
    labeled_ds = BlastocystDataset(
        image_dir=labeled_img_dir,
        mask_dir=labeled_mask_dir,
        augment = augment
    )

    # 2. Unlabeled Dataset (images only)
    unlabeled_ds = UnlabeledBlastocystDataset(
        image_dir=unlabeled_img_dir
    )

    # 3. Test Dataset (with masks)
    test_ds = BlastocystDataset(
        image_dir=test_img_dir,
        mask_dir=test_mask_dir,
        augment = False
    )

    # Create loaders
    labeled_loader = DataLoader(
        labeled_ds,
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
        worker_init_fn=seed_worker,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True  # Helps with batch normalization
    )

    unlabeled_loader = DataLoader(
        unlabeled_ds,
        batch_size=batch_size,
        shuffle=False,  # Important for sample tracking
        worker_init_fn=seed_worker,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        worker_init_fn=seed_worker,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    return labeled_loader, unlabeled_loader, test_loader