import os
import random

from PIL import Image, ImageOps
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import cv2
from albumentations import Compose, Affine, HorizontalFlip, VerticalFlip
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
    """
    Encodes a 3-channel RGB mask to a 1-channel class mask.

    Args:
        arr (np.ndarray): A numpy array of shape (H, W, 3) representing the RGB mask.

    Returns:
        np.ndarray: A numpy array of shape (H, W) with integer class labels.
    """
    h, w, _ = arr.shape
    class_mask = np.zeros((h, w), dtype=np.uint8)
    for rgb, idx in Classes.items():
        class_mask[np.all(arr == rgb, axis=-1)] = idx
    return class_mask


class BlastocystDataset(Dataset):
    """
    A PyTorch Dataset for loading blastocyst images and their corresponding masks.
    """
    def __init__(self,
                 image_dir,
                 mask_dir,
                 seed=None,
                 augment=False,
                 
                 ):
        """
        Initializes the BlastocystDataset.

        Args:
            image_dir (str): The directory containing the images.
            mask_dir (str): The directory containing the masks.
            seed (int, optional): The random seed for augmentations. Defaults to None.
            augment (bool, optional): Whether to apply data augmentation. Defaults to False.
        """
        self.image_dir = image_dir
        self.mask_dir  = mask_dir
        self.image_filenames = sorted(os.listdir(image_dir))
        self.augment = augment
        if augment:
            self.transform = Compose([
                Affine(
                    rotate=(-90, 90),  # Random rotation between -90 and +90 degrees
                    translate_percent=(-0.1, 0.1),  # Shift by +/- 10%
                    scale=(0.8, 1.2),  # Zoom by +/- 20% (0.8 to 1.2)
                    p=1.0,
                    border_mode=cv2.BORDER_CONSTANT,
                    fill=0,  # image‐pad with 0 → black
                    fill_mask=3,  # mask‐pad with 3 →  background
                    mask_interpolation=cv2.INTER_NEAREST
                ),
                HorizontalFlip(p=0.5),  # 50% horizontal flip
                VerticalFlip(p=0.5),    # 50% vertical flip
                ToTensorV2(),
            ], additional_targets={'mask': 'mask'}, seed=seed)
        else:
            self.transform = Compose([
                ToTensorV2(),
            ], additional_targets={'mask': 'mask'}, seed=seed)


    def __len__(self):
        """
        Returns the number of images in the dataset.
        """
        return len(self.image_filenames)

    def __getitem__(self, idx):
        """
        Gets the image and mask at the given index.

        Args:
            idx (int): The index of the item to retrieve.

        Returns:
            tuple: A tuple containing the image, mask, and image name.
        """
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
    """
    A PyTorch Dataset for loading unlabeled blastocyst images.
    """
    def __init__(self, image_dir):
        """
        Initializes the UnlabeledBlastocystDataset.

        Args:
            image_dir (str): The directory containing the images.
        """
        self.image_dir = image_dir
        self.image_filenames = sorted(os.listdir(image_dir))

    def __len__(self):
        """
        Returns the number of images in the dataset.
        """
        return len(self.image_filenames)

    def __getitem__(self, idx):
        """
        Gets the image at the given index.

        Args:
            idx (int): The index of the item to retrieve.

        Returns:
            tuple: A tuple containing the image and image name.
        """
        img_name = self.image_filenames[idx]
        img_path = os.path.join(self.image_dir, img_name)

        img = Image.open(img_path).convert("L")
        img = ImageOps.pad(img, TARGET_SIZE, method=Image.BILINEAR, color=0)
        img = np.array(img, np.float32) / 255.0
        img = torch.from_numpy(img).unsqueeze(0)  # [1, H, W]

        return img, img_name

def seed_worker(worker_id):
    """
    Seeds the worker for reproducibility.
    """
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
        seed = None,
        augment = False,
        generator=None,
        num_workers=4,
        pin_memory=True,
):
    """
    Creates and returns data loaders for active learning.

    Args:
        labeled_img_dir (str): The directory for labeled images.
        labeled_mask_dir (str): The directory for labeled masks.
        unlabeled_img_dir (str): The directory for unlabeled images.
        test_img_dir (str): The directory for test images.
        test_mask_dir (str): The directory for test masks.
        batch_size (int): The batch size.
        seed (int, optional): The random seed. Defaults to None.
        augment (bool, optional): Whether to apply data augmentation. Defaults to False.
        generator (torch.Generator, optional): The random number generator for the data loader. Defaults to None.
        num_workers (int, optional): The number of worker threads. Defaults to 4.
        pin_memory (bool, optional): Whether to pin memory. Defaults to True.

    Returns:
        tuple: A tuple containing the labeled, unlabeled, and test data loaders.
    """
    # 1. Labeled Dataset (with masks)
    labeled_ds = BlastocystDataset(
        image_dir=labeled_img_dir,
        mask_dir=labeled_mask_dir,
        seed=seed,
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
        seed=seed,
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
