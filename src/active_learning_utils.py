import os
import shutil
import random
import torch
from tqdm import tqdm

def create_active_learning_pools(
        BASE_DIR,
        label_split_ratio=0.1,
        test_split_ratio=0.2,
        shuffle=True
):
    # Create directories
    dirs = {
        'labeled_img': os.path.join(BASE_DIR, "Labeled_pool", 'labeled_images'),
        'labeled_mask': os.path.join(BASE_DIR, "Labeled_pool", 'labeled_masks'),
        'unlabeled_img': os.path.join(BASE_DIR, "Unlabeled_pool", 'unlabeled_images'),
        'unlabeled_mask': os.path.join(BASE_DIR, "Unlabeled_pool", 'unlabeled_masks'),
        'test_img': os.path.join(BASE_DIR, "Test", 'test_images'),
        'test_mask': os.path.join(BASE_DIR, "Test", 'test_masks')
    }

    dirs["labeled_img_dir"] = dirs["labeled_img"]
    dirs["labeled_mask_dir"] = dirs["labeled_mask"]
    dirs["unlabeled_img_dir"] = dirs["unlabeled_img"]
    dirs["test_img_dir"] = dirs["test_img"]
    dirs["test_mask_dir"] = dirs["test_mask"]

    for path in dirs.values():
        os.makedirs(path, exist_ok=True)

    # Get image list
    img_dir = os.path.join(BASE_DIR, 'images')
    images = [f for f in os.listdir(img_dir) if f.lower().endswith("bmp")]

    if shuffle:
        random.shuffle(images)

    # Split images
    n_test = int(len(images) * test_split_ratio)
    n_labeled = int(len(images) * label_split_ratio)

    test_split = images[:n_test]
    labeled_split = images[n_test:n_test + n_labeled]
    unlabeled_split = images[n_test + n_labeled:]

    def copy_files(file_list, img_dest, mask_dest):

        for im in file_list:
            base_name = os.path.splitext(im)[0]

            # Copy image
            src_img = os.path.join(img_dir, im)
            dst_img = os.path.join(img_dest, im)
            shutil.copy(src_img, dst_img)

            # Copy mask
            mask_file = f"{base_name}.png"
            src_mask = os.path.join(BASE_DIR, 'masks', mask_file)
            dst_mask = os.path.join(mask_dest, mask_file)

            if os.path.exists(src_mask):
                shutil.copy(src_mask, dst_mask)
            else:
                print(f"Warning: Mask not found for {im} - {src_mask}")

    copy_files(test_split, dirs['test_img'], dirs['test_mask'])
    copy_files(labeled_split, dirs['labeled_img'], dirs['labeled_mask'])
    copy_files(unlabeled_split, dirs['unlabeled_img'], dirs['unlabeled_mask'])

    return dirs

def reset_data(base_dir):
    # Directories to remove
    dirs_to_remove = [
        os.path.join(base_dir, "Labeled_pool"),
        os.path.join(base_dir, "Unlabeled_pool"),
        os.path.join(base_dir, "Test")
    ]

    for dir_path in dirs_to_remove:
        if os.path.exists(dir_path):
            shutil.rmtree(dir_path)


def move_images_with_dict(
        base_dir: str,
        labeled_dir: str,
        unlabeled_dir: str,
        score_dict: dict,
        num_to_move: int = 10
):
    # Sort by descending uncertainty (most uncertain first)
    sorted_items = sorted(score_dict.items(), key=lambda x: x[1], reverse=True)

    moved = 0
    for im, score in sorted_items:
        if moved >= num_to_move:
            break

        # Clean filename and get base name
        im_clean = im.strip()
        base_name = os.path.splitext(im_clean)[0]

        # Image paths
        src_im = os.path.join(base_dir, unlabeled_dir, "unlabeled_images", im_clean)
        dst_im = os.path.join(base_dir, labeled_dir, "labeled_images", im_clean)

        # Mask paths
        mask_name = base_name + ".png"
        src_msk = os.path.join(base_dir, unlabeled_dir, "unlabeled_masks", mask_name)
        dst_msk = os.path.join(base_dir, labeled_dir, "labeled_masks", mask_name)

        # Verify image exists
        if not os.path.exists(src_im):
            print(f"[WARN] Image not found: {src_im}")
            continue

        # Move image
        shutil.copy(src_im, dst_im)
        os.remove(src_im)
        print(f"[MOVE] IMAGE {im_clean} (Uncertainty: {score:.4f})")

        # Move mask if exists
        if os.path.exists(src_msk):
            shutil.copy(src_msk, dst_msk)
            os.remove(src_msk)
            print(f"[MOVE]  MASK {mask_name}")
        else:
            print(f"[WARN] Mask not found: {src_msk}")

        moved += 1

    print(f"Moved {moved} most uncertain images from {unlabeled_dir} → {labeled_dir}.")

def score_unlabeled_pool(unlabeled_loader, model, score_fn, T=8, num_classes=4, device="cuda"):
    model.to(device).train()
    scores, fnames = [], []
    with torch.no_grad():
        for imgs, names in tqdm(unlabeled_loader, desc="Scoring", leave=False):
            imgs = imgs.to(device)
            s = score_fn(model, imgs, T=T, num_classes=num_classes)
            scores.extend(s.cpu().tolist())
            fnames.extend(names)
    return dict(zip(fnames, scores))