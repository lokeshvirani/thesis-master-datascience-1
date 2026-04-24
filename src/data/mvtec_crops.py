#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import glob
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import Dataset


class MVTecCrops(Dataset):
    """Dataset for pre-cropped defect regions from MVTec."""

    # Normalization as MVTec
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

    def __init__(self,
                 crop_root,
                 transform=None,
                 add_augmented=False,
                 augmented_crop_root=None,
                 output_size=(128, 128)):
        """
        Args:
            crop_root (str): Directory containing pre-cropped defect images (and masks)
            transform (callable, optional): Transform for images (default: ToTensor)
            add_augmented (bool): Whether to add augmented crops
            augmented_crop_root (str): Path to augmented crops root. Can be:
                (A) legacy: a folder with crops + *_GT.png masks in SAME folder
                (B) augmented: a folder containing imgs/ and masks/ subfolders
                    imgs:  00000.png
                    masks: 00000_GT.png   (IMPORTANT: masks have _GT suffix)
            output_size (tuple): Resize (W, H)
        """
        self.crop_root = crop_root
        self.add_augmented = add_augmented
        self.augmented_crop_root = augmented_crop_root
        self.output_size = output_size

        self.crop_files = []
        self.mask_files = []

        # Load original cropped defects
        self.load_crops(crop_root, self.crop_files, self.mask_files)

        # Load augmented if requested
        if add_augmented and augmented_crop_root and os.path.exists(augmented_crop_root):
            self.load_crops(augmented_crop_root, self.crop_files, self.mask_files)

        print(f"Loaded {len(self.crop_files)} total defect crops (including augmented={self.add_augmented})")

        self.transform = transform or self.get_default_transform()
        self.normalize = T.Normalize(self.mean, self.std)

    def load_crops(self, root_dir, crop_files, mask_files):
        """
        Load crop and mask filenames from a directory.

        Supports two layouts:

        (A) Legacy (single folder):
            root_dir/
                xxx.png
                xxx_GT.png

        (B) Augmented layout:
            root_dir/
                imgs/
                    00000.png
                masks/
                    00000_GT.png
        """
        # --- Case B: imgs/ + masks/ ---
        imgs_dir = os.path.join(root_dir, "imgs")
        masks_dir = os.path.join(root_dir, "masks")
        if os.path.isdir(imgs_dir):
            img_paths = sorted(glob.glob(os.path.join(imgs_dir, "*.png")))
            added = 0
            for img_path in img_paths:
                base = os.path.basename(img_path)  # e.g., 00000.png

                # IMPORTANT: your masks are named like 00000_GT.png
                mask_base = base.replace(".png", "_GT.png")  # e.g., 00000_GT.png
                mask_path = os.path.join(imgs_dir, mask_base)

                if os.path.exists(mask_path):
                    crop_files.append(img_path)
                    mask_files.append(mask_path)
                    added += 1

            print(f"[load_crops] {root_dir} (imgs/masks) -> found {len(img_paths)} imgs, paired {added}")
            return

        # --- Case A: legacy *_GT.png in same folder ---
        img_paths = sorted(glob.glob(os.path.join(root_dir, "*.png")))
        added = 0
        for img_path in img_paths:
            if img_path.endswith("_GT.png"):
                continue
            base = os.path.basename(img_path)              # e.g., abc.png
            mask_base = base.replace(".png", "_GT.png")    # e.g., abc_GT.png
            mask_path = os.path.join(root_dir, mask_base)
            if os.path.exists(mask_path):
                crop_files.append(img_path)
                mask_files.append(mask_path)
                added += 1

        print(f"[load_crops] {root_dir} (legacy) -> scanned {len(img_paths)} pngs, paired {added}")

    def __len__(self):
        return len(self.crop_files)

    def __getitem__(self, idx):
        crop_path = self.crop_files[idx]
        mask_path = self.mask_files[idx]

        crop = Image.open(crop_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        crop = crop.resize(self.output_size, Image.BILINEAR)
        mask = mask.resize(self.output_size, Image.NEAREST)

        # Default transform is ToTensor; safe for both.
        if self.transform:
            crop = self.transform(crop)
            mask = self.transform(mask)

        # Normalize only the RGB crop
        crop = self.normalize(crop)

        # Always defect class=1
        return crop, 1, mask, 0

    @staticmethod
    def get_default_transform():
        return T.Compose([T.ToTensor()])