#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import re
import torch
import pandas as pd

import torchvision.transforms as T

from PIL import Image
from torch.utils.data import Dataset


def c2chw(x):
    return x.unsqueeze(1).unsqueeze(2)


def inverse_list(items):
    """
    List to dict: element -> index
    """
    out = {}
    for idx, x in enumerate(items):
        out[x] = idx
    return out


class MVTec(Dataset):
    """
    MVTec/custom dataset loader.

    Supports two formats:

    1) CSV-based format
       dataroot/
           train/
           test/
           train.csv
           test.csv

       CSV must contain:
           - path
           - label   (negative / positive)

    2) Legacy format
       dataroot/
           train/
               000.png
               000_GT.png
               ...
           test/
               ...

    Args:
        dataroot (str): dataset root
        split (str): 'train' or 'test'
        negative_only (bool): keep only normal samples
        positive_only (bool): keep only anomalous samples
        add_augmented (bool): add augmented samples for training
        num_augmented (int): augmented folder naming support
        zero_shot (bool): preserved from original code
    """

    labels = ['ok', 'defect']

    # ImageNet stats
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

    def __init__(
        self,
        dataroot='/path/to/dataset/MVTec',
        split='train',
        negative_only=False,
        positive_only=False,
        add_augmented=False,
        num_augmented=100,
        zero_shot=False
    ):
        super(MVTec, self).__init__()

        self.fold = None
        self.dataroot = dataroot
        self.split = 'train' if split == 'val' else split
        self.split_path = os.path.join(self.dataroot, self.split)

        self.output_size = (632 // 2, 224 // 2)
        self.negative_only = negative_only
        self.positive_only = positive_only
        self.add_augmented = add_augmented
        self.num_augmented = num_augmented
        self.zero_shot = zero_shot

        if self.add_augmented:
            assert self.split == 'train', 'Augmented images are only for the training set!'
        if self.zero_shot:
            assert self.add_augmented, 'Zero-shot learning requires augmented images!'
            assert self.split == 'train', 'Zero-shot learning is only for the training set!'

        self.class_to_idx = inverse_list(self.labels)
        self.classes = self.labels
        self.transform = MVTec.get_transform(output_size=self.output_size)
        self.normalize = T.Normalize(MVTec.mean, MVTec.std)

        self.load_imgs()

    def load_imgs(self):
        csv_path = os.path.join(self.dataroot, f"{self.split}.csv")

        if os.path.isfile(csv_path):
            self._load_from_csv(csv_path)
        else:
            self._load_legacy()

    def _load_from_csv(self, csv_path):
        df = pd.read_csv(csv_path)

        if "path" not in df.columns or "label" not in df.columns:
            raise ValueError(
                f"{csv_path} must contain columns ['path', 'label'], found {list(df.columns)}"
            )

        df = df.copy()
        df["path"] = df["path"].astype(str).str.strip()
        df["label"] = df["label"].astype(str).str.strip().str.lower()

        # keep only files that actually exist
        abs_paths = [os.path.join(self.dataroot, p) for p in df["path"]]
        exists_mask = [os.path.isfile(p) for p in abs_paths]
        if not all(exists_mask):
            missing = [p for p, ok in zip(df["path"], exists_mask) if not ok][:10]
            print(f"[MVTec] Warning: {sum(not x for x in exists_mask)} missing files in {csv_path}. Examples: {missing}")
        df = df[exists_mask].reset_index(drop=True)

        # filter by label
        if self.negative_only:
            df = df[df["label"] == "negative"].reset_index(drop=True)
        elif self.positive_only:
            df = df[df["label"] == "positive"].reset_index(drop=True)

        if len(df) == 0:
            self.samples = torch.zeros((0, 3, *self.output_size), dtype=torch.float32)
            self.masks = torch.zeros((0, *self.output_size), dtype=torch.long)
            self.product_ids = []
            print(f"[MVTec] {csv_path} -> no samples after filtering.")
            return

        self.samples = torch.zeros((len(df), 3, *self.output_size), dtype=torch.float32)
        self.masks = torch.zeros((len(df), *self.output_size), dtype=torch.long)
        self.product_ids = []

        for idx, row in df.iterrows():
            rel_path = row["path"]
            label = row["label"]
            img_path = os.path.join(self.dataroot, rel_path)

            img = Image.open(img_path).convert("RGB")
            img = self.transform(img)
            self.samples[idx] = img

            product_id = os.path.splitext(os.path.basename(rel_path))[0]
            self.product_ids.append(product_id)

            mask = self._find_mask_for_image(img_path, rel_path, label)
            self.masks[idx] = mask

        print(f"[MVTec] Loaded {len(df)} samples from {csv_path}")

    def _find_mask_for_image(self, img_path, rel_path, label):
        """
        Try to locate a GT mask for the image.

        Priority:
        1) same directory, filename + _GT.png
        2) same directory, filename + _gt.png
        3) replace '/test/' with '/ground_truth/' and add _mask/_GT variants
        4) if label is negative -> zero mask
        5) if label is positive and no mask found -> full-image mask fallback
        """
        base_dir = os.path.dirname(img_path)
        stem = os.path.splitext(os.path.basename(img_path))[0]

        candidates = [
            os.path.join(base_dir, f"{stem}_GT.png"),
            os.path.join(base_dir, f"{stem}_gt.png"),
            os.path.join(base_dir, f"{stem}_mask.png"),
        ]

        # common mvtec-like alternative path
        if f"{os.sep}test{os.sep}" in img_path:
            gt_dir = base_dir.replace(f"{os.sep}test{os.sep}", f"{os.sep}ground_truth{os.sep}")
            candidates.extend([
                os.path.join(gt_dir, f"{stem}_mask.png"),
                os.path.join(gt_dir, f"{stem}_GT.png"),
                os.path.join(gt_dir, f"{stem}.png"),
            ])

        for cand in candidates:
            if os.path.isfile(cand):
                lab = Image.open(cand).convert('L')
                lab = self.transform(lab)
                return (lab > 0).long().squeeze(0)

        # No mask found
        if label == "negative":
            return torch.zeros(self.output_size, dtype=torch.long)

        # fallback for positive sample if no GT mask exists
        # this lets the pipeline run, but pixel AUROC will not be reliable
        return torch.ones(self.output_size, dtype=torch.long)

    def _load_legacy(self):
        # Determine augmented paths
        if self.num_augmented > 0:
            augmented_imgs_path = os.path.join(self.dataroot, f'augmented_{self.num_augmented}', 'imgs')
            augmented_masks_path = os.path.join(self.dataroot, f'augmented_{self.num_augmented}', 'masks')
        else:
            augmented_imgs_path = os.path.join(self.dataroot, 'augmented', 'imgs')
            augmented_masks_path = os.path.join(self.dataroot, 'augmented', 'masks')

        path = self.split_path
        if not os.path.isdir(path):
            raise ValueError(f"Split folder not found: {path}")

        image_list = sorted([
            f for f in os.listdir(path)
            if f.lower().endswith(".png") and not f.lower().endswith("_gt.png")
        ])
        assert len(image_list) > 0, f"No images found in {path}"

        cnt = 0
        for img_name in image_list:
            product_id = img_name[:-4]
            gt_path = os.path.join(path, product_id + '_GT.png')

            if not os.path.isfile(gt_path):
                continue

            lab = Image.open(gt_path).convert('L')
            lab_tensor = self.transform(lab)

            if self.negative_only and lab_tensor.sum() == 0:
                cnt += 1
            elif self.positive_only and lab_tensor.sum() > 0:
                cnt += 1
            elif not self.negative_only and not self.positive_only:
                cnt += 1

        if self.add_augmented and self.split == 'train' and os.path.isdir(augmented_imgs_path):
            aug_list = os.listdir(augmented_imgs_path)
            cnt += len(aug_list)

        self.samples = torch.zeros((cnt, 3, *self.output_size), dtype=torch.float32)
        self.masks = torch.zeros((cnt, *self.output_size), dtype=torch.long)
        self.product_ids = []

        cnt = 0
        for img_name in image_list:
            product_id = img_name[:-4]
            img_path = os.path.join(path, img_name)
            gt_path = os.path.join(path, product_id + '_GT.png')

            if not os.path.isfile(gt_path):
                continue

            img = self.transform(Image.open(img_path).convert("RGB"))
            lab = self.transform(Image.open(gt_path).convert('L'))
            lab_bin = (lab > 0).long().squeeze(0)

            if self.negative_only and lab_bin.sum() == 0:
                self.samples[cnt] = img
                self.masks[cnt] = lab_bin
                self.product_ids.append(product_id)
                cnt += 1
            elif self.positive_only and lab_bin.sum() > 0:
                self.samples[cnt] = img
                self.masks[cnt] = lab_bin
                self.product_ids.append(product_id)
                cnt += 1
            elif not self.negative_only and not self.positive_only:
                self.samples[cnt] = img
                self.masks[cnt] = lab_bin
                self.product_ids.append(product_id)
                cnt += 1

        if self.add_augmented and self.split == 'train' and os.path.isdir(augmented_imgs_path):
            aug_list = os.listdir(augmented_imgs_path)
            for img_name in aug_list:
                product_id = img_name[:-4]
                img = self.transform(Image.open(os.path.join(augmented_imgs_path, img_name)).convert("RGB"))
                lab_name = product_id + "_GT.png"
                lab = self.transform(Image.open(os.path.join(augmented_masks_path, lab_name)).convert('L'))
                lab_bin = (lab > 0).long().squeeze(0)

                self.samples[cnt] = img
                self.masks[cnt] = lab_bin
                self.product_ids.append(product_id)
                cnt += 1

        print(f"[MVTec] Loaded {cnt} samples from legacy layout at {path}")

    def __getitem__(self, index):
        x = self.samples[index]
        a = self.masks[index] > 0

        if self.normalize is not None:
            x = self.normalize(x)

        y = self.class_to_idx['defect'] if a.sum() > 0 else self.class_to_idx['ok']
        return x, y, a, 0

    def __len__(self):
        return self.samples.size(0)

    @staticmethod
    def get_transform(output_size=(632 // 2, 224 // 2)):
        return T.Compose([
            T.Resize(output_size),
            T.ToTensor()
        ])

    @staticmethod
    def denorm(x):
        return x * c2chw(torch.Tensor(MVTec.std)) + c2chw(torch.Tensor(MVTec.mean))


if __name__ == "__main__":
    dataroot = "data/mvtec_preprocessed"

    neg_dataset = MVTec(dataroot=dataroot, split='train', negative_only=True, add_augmented=False)
    print(f"Negative-only dataset shape: ({len(neg_dataset)}, 3, {neg_dataset.output_size[0]}, {neg_dataset.output_size[1]})")

    if len(neg_dataset) > 0:
        sample, label, mask, _ = neg_dataset[0]
        print(sample.shape, label, mask.shape)