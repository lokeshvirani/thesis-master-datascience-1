#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import random
import argparse
import numpy as np
import pandas as pd
from PIL import Image, ImageOps, ImageFilter

# ----------------------------
# Args
# ----------------------------
parser = argparse.ArgumentParser(description="Generate MemSeg-style positive samples for Dual PatchCore.")
parser.add_argument("--src_dir", type=str, required=True, help="Root preprocessed directory containing category folders.")
parser.add_argument("--category", type=str, default=None, help="If set, process only one category.")
parser.add_argument("--variants", type=int, default=10, help="Number of composition variants.")
parser.add_argument("--imgs_per_variant", type=int, default=50, help="Images to generate per variant.")
parser.add_argument("--out_dirname", type=str, default="memseg_10v50", help="Output folder name inside each category.")
parser.add_argument("--seed", type=int, default=42, help="Random seed.")
parser.add_argument("--target_size", type=int, default=256, help="Resize target size (square).")
args = parser.parse_args()

src_dir = os.path.abspath(args.src_dir)
base_seed = args.seed
TARGET = (args.target_size, args.target_size)

if args.variants <= 0:
    raise ValueError("--variants must be > 0")
if args.imgs_per_variant <= 0:
    raise ValueError("--imgs_per_variant must be > 0")

random.seed(base_seed)
np.random.seed(base_seed)

def resolve_path(cat_root, split, csv_path):
    csv_path = csv_path.lstrip("/").replace("\\", os.sep)
    parts = csv_path.split(os.sep)
    if split in parts:
        idx = parts.index(split)
        rel = os.path.join(*parts[idx:])
    else:
        rel = os.path.join(split, csv_path)
    return os.path.join(cat_root, rel)

def load_rgb(path, resize_to=None):
    img = Image.open(path).convert("RGB")
    if resize_to is not None:
        img = img.resize(resize_to, Image.BILINEAR)
    return img

def load_mask(path, resize_to=None):
    img = Image.open(path).convert("L")
    if resize_to is not None:
        img = img.resize(resize_to, Image.NEAREST)
    return img

def crop_to_mask_bbox(rgb_img, mask_img):
    mask_arr = np.array(mask_img, dtype=np.uint8)
    ys, xs = np.where(mask_arr > 10)
    if len(xs) == 0 or len(ys) == 0:
        return rgb_img, mask_img
    x0, x1 = xs.min(), xs.max() + 1
    y0, y1 = ys.min(), ys.max() + 1
    return rgb_img.crop((x0, y0, x1, y1)), mask_img.crop((x0, y0, x1, y1))

def scan_defect_crop_pairs(defect_root):
    pairs = []
    if not os.path.exists(defect_root):
        return pairs
    for root, _, files in os.walk(defect_root):
        file_set = set(files)
        for fn in files:
            if not fn.lower().endswith(".png") or fn.endswith("_GT.png"):
                continue
            stem = os.path.splitext(fn)[0]
            gt_name = stem + "_GT.png"
            if gt_name in file_set:
                pairs.append((os.path.join(root, fn), os.path.join(root, gt_name)))
    return sorted(pairs)

def choose_variant_specs(num_variants):
    base = [
        {"name": "soft_small",        "scale_range": (0.45, 0.70), "rot_range": (-8, 8),   "flip_prob": 0.0, "alpha_range": (0.42, 0.58), "blur_radius": 1.4, "jitter": 0.92},
        {"name": "soft_medium",       "scale_range": (0.65, 0.90), "rot_range": (-12, 12), "flip_prob": 0.2, "alpha_range": (0.48, 0.64), "blur_radius": 1.1, "jitter": 0.96},
        {"name": "rotated_small",     "scale_range": (0.60, 0.85), "rot_range": (-30, 30), "flip_prob": 0.3, "alpha_range": (0.55, 0.70), "blur_radius": 0.8, "jitter": 1.00},
        {"name": "scaled_up",         "scale_range": (0.90, 1.25), "rot_range": (-18, 18), "flip_prob": 0.3, "alpha_range": (0.60, 0.78), "blur_radius": 0.8, "jitter": 1.00},
        {"name": "hard_paste",        "scale_range": (0.75, 1.05), "rot_range": (-10, 10), "flip_prob": 0.2, "alpha_range": (0.80, 0.95), "blur_radius": 0.0, "jitter": 1.00},
        {"name": "flipped_rotated",   "scale_range": (0.65, 1.00), "rot_range": (-45, 45), "flip_prob": 0.8, "alpha_range": (0.55, 0.75), "blur_radius": 1.2, "jitter": 1.05},
        {"name": "large_soft",        "scale_range": (1.00, 1.35), "rot_range": (-20, 20), "flip_prob": 0.4, "alpha_range": (0.45, 0.62), "blur_radius": 1.6, "jitter": 0.95},
        {"name": "contrast_shifted",  "scale_range": (0.75, 1.05), "rot_range": (-25, 25), "flip_prob": 0.5, "alpha_range": (0.60, 0.80), "blur_radius": 0.6, "jitter": 1.12},
        {"name": "tiny_sharp",        "scale_range": (0.35, 0.55), "rot_range": (-12, 12), "flip_prob": 0.2, "alpha_range": (0.70, 0.90), "blur_radius": 0.2, "jitter": 1.00},
        {"name": "large_rotated_soft","scale_range": (1.05, 1.40), "rot_range": (-35, 35), "flip_prob": 0.6, "alpha_range": (0.40, 0.58), "blur_radius": 1.8, "jitter": 0.90},
    ]
    specs = []
    for i in range(num_variants):
        spec = base[i % len(base)].copy()
        spec["id"] = i
        specs.append(spec)
    return specs

def transform_crop(crop_img, crop_mask, spec, rng, bg_w, bg_h):
    if rng.random() < spec["flip_prob"]:
        crop_img = ImageOps.mirror(crop_img)
        crop_mask = ImageOps.mirror(crop_mask)

    scale = rng.uniform(*spec["scale_range"])
    new_w = max(8, int(crop_img.width * scale))
    new_h = max(8, int(crop_img.height * scale))

    max_w = max(8, int(bg_w * 0.6))
    max_h = max(8, int(bg_h * 0.6))
    ratio = min(max_w / new_w, max_h / new_h, 1.0)
    new_w = max(8, int(new_w * ratio))
    new_h = max(8, int(new_h * ratio))

    crop_img = crop_img.resize((new_w, new_h), Image.BILINEAR)
    crop_mask = crop_mask.resize((new_w, new_h), Image.NEAREST)

    angle = rng.uniform(*spec["rot_range"])
    crop_img = crop_img.rotate(angle, resample=Image.BILINEAR, expand=True, fillcolor=(0, 0, 0))
    crop_mask = crop_mask.rotate(angle, resample=Image.NEAREST, expand=True, fillcolor=0)

    return crop_img, crop_mask

def adjust_crop_intensity(crop_arr, factor):
    crop = crop_arr.astype(np.float32) * factor
    return np.clip(crop, 0, 255).astype(np.float32)

def choose_location(mask_img, bg_w, bg_h, rng):
    mw, mh = mask_img.size
    if mw >= bg_w or mh >= bg_h:
        return 0, 0
    x = rng.integers(0, bg_w - mw + 1)
    y = rng.integers(0, bg_h - mh + 1)
    return int(x), int(y)

def compose(bg_img, crop_img, crop_mask, spec, rng):
    bg = np.array(bg_img, dtype=np.float32)
    crop = np.array(crop_img, dtype=np.float32)
    mask = (np.array(crop_mask, dtype=np.uint8) > 10).astype(np.uint8)

    if mask.sum() == 0:
        return None, None

    crop = adjust_crop_intensity(crop, spec["jitter"])

    h, w, _ = bg.shape
    ch, cw, _ = crop.shape

    soft = Image.fromarray((mask * 255).astype(np.uint8))
    if spec["blur_radius"] > 0:
        soft = soft.filter(ImageFilter.GaussianBlur(radius=spec["blur_radius"]))
    soft = np.array(soft, dtype=np.float32) / 255.0

    alpha = rng.uniform(*spec["alpha_range"])
    soft = np.clip(soft * alpha, 0.0, 1.0)

    x, y = choose_location(Image.fromarray((mask * 255).astype(np.uint8)), w, h, rng)

    crop_canvas = np.zeros_like(bg, dtype=np.float32)
    soft_canvas = np.zeros((h, w), dtype=np.float32)
    hard_canvas = np.zeros((h, w), dtype=np.uint8)

    crop_canvas[y:y+ch, x:x+cw] = crop
    soft_canvas[y:y+ch, x:x+cw] = soft
    hard_canvas[y:y+ch, x:x+cw] = mask

    soft3 = soft_canvas[..., None]
    out = bg * (1.0 - soft3) + crop_canvas * soft3
    out = np.clip(out, 0, 255).astype(np.uint8)

    return out, hard_canvas

all_categories = sorted([
    d for d in os.listdir(src_dir)
    if os.path.isdir(os.path.join(src_dir, d))
])

categories = [args.category] if args.category else all_categories

print(f"[INFO] Found {len(all_categories)} categories under src_dir")
print(f"[INFO] Will generate for {len(categories)} categories: {categories}")
print(f"[INFO] variants={args.variants}, imgs_per_variant={args.imgs_per_variant}, total_per_category={args.variants * args.imgs_per_variant}")
print(f"[INFO] out_dirname={args.out_dirname}")

variant_specs = choose_variant_specs(args.variants)

for category in categories:
    cat_root = os.path.join(src_dir, category)
    if not os.path.isdir(cat_root):
        print(f"[WARN] Category folder not found: {cat_root} (skip)")
        continue

    print(f"\n[INFO] Category: {category}")

    defect_root = os.path.join(cat_root, "defect_crops")
    defect_pairs = scan_defect_crop_pairs(defect_root)
    if len(defect_pairs) == 0:
        print(f"[WARN] No defect crop pairs found in: {defect_root}")
        continue

    dst_root = os.path.join(cat_root, args.out_dirname)
    imgs_out = os.path.join(dst_root, "imgs")
    masks_out = os.path.join(dst_root, "imgs")
    os.makedirs(imgs_out, exist_ok=True)
    os.makedirs(masks_out, exist_ok=True)

    split = "train"
    csv_path = os.path.join(cat_root, f"{split}.csv")
    if not os.path.exists(csv_path):
        print(f"[WARN] Missing CSV: {csv_path}")
        continue

    df = pd.read_csv(csv_path)
    neg_rows = df[df["label"] == "negative"]["path"].tolist()

    if not neg_rows:
        print(f"[WARN] No negative rows found in: {csv_path}")
        continue

    print(f"[INFO] {split} | Neg={len(neg_rows)} | Defect crop pairs={len(defect_pairs)}")

    img_idx = 0

    for spec in variant_specs:
        print(f"[INFO] Variant {spec['id']+1}/{len(variant_specs)}: {spec['name']}")
        made = 0
        attempts = 0
        max_attempts = max(args.imgs_per_variant * 30, 300)

        while made < args.imgs_per_variant and attempts < max_attempts:
            attempts += 1
            try:
                neg_csv = random.choice(neg_rows)
                neg_path = resolve_path(cat_root, split, neg_csv)
                if not os.path.exists(neg_path):
                    continue

                crop_img_path, crop_mask_path = random.choice(defect_pairs)

                bg_img = load_rgb(neg_path, resize_to=TARGET)
                crop_img = load_rgb(crop_img_path, resize_to=None)
                crop_mask = load_mask(crop_mask_path, resize_to=None)

                crop_img, crop_mask = crop_to_mask_bbox(crop_img, crop_mask)

                rng = np.random.default_rng(base_seed + img_idx)

                crop_img, crop_mask = transform_crop(
                    crop_img, crop_mask, spec, rng,
                    bg_w=TARGET[0], bg_h=TARGET[1]
                )

                composed, hard_mask = compose(bg_img, crop_img, crop_mask, spec, rng)
                if composed is None or hard_mask is None or hard_mask.sum() == 0:
                    continue

                Image.fromarray(composed).save(os.path.join(imgs_out, f"{img_idx:05d}.png"))
                Image.fromarray((hard_mask * 255).astype(np.uint8)).save(os.path.join(masks_out, f"{img_idx:05d}_GT.png"))

                img_idx += 1
                made += 1

                if made % 10 == 0 or made == args.imgs_per_variant:
                    print(f"[INFO]   Generated {made}/{args.imgs_per_variant} for {spec['name']}")

            except Exception as e:
                print(f"[WARN] Skipping one sample due to error: {e}")
                continue

        if made < args.imgs_per_variant:
            print(f"[WARN] Only generated {made}/{args.imgs_per_variant} for {spec['name']}")

    print(f"[INFO] Finished category '{category}' with {img_idx} total generated images")

print("[INFO] Done.")
