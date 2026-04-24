#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import random
import argparse
import numpy as np
import pandas as pd
from PIL import Image

# ----------------------------
# Args
# ----------------------------
parser = argparse.ArgumentParser(description="Generate noise-based positive samples for Dual PatchCore.")
parser.add_argument("--src_dir", type=str, required=True, help="Root preprocessed directory containing category folders.")
parser.add_argument("--category", type=str, default=None, help="If set, process only one category.")
parser.add_argument("--variants", type=int, default=10, help="Number of noise variants to use.")
parser.add_argument("--imgs_per_variant", type=int, default=50, help="Images to generate per noise variant.")
parser.add_argument("--out_dirname", type=str, default="noise_10v50", help="Output folder name inside each category.")
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

def build_mask_path(cat_root, split, pos_csv_path):
    stem, _ = os.path.splitext(pos_csv_path)
    return resolve_path(cat_root, split, stem + "_GT.png")

def load_rgb(path):
    return Image.open(path).convert("RGB").resize(TARGET, Image.BILINEAR)

def load_mask(path):
    return Image.open(path).convert("L").resize(TARGET, Image.NEAREST)

def mask_to_binary(mask_img, threshold=10):
    arr = np.array(mask_img, dtype=np.uint8)
    return (arr > threshold).astype(np.uint8)

def ensure_nonempty_mask(mask_bin):
    if mask_bin.sum() == 0:
        h, w = mask_bin.shape
        cy, cx = h // 2, w // 2
        r = max(8, min(h, w) // 16)
        yy, xx = np.ogrid[:h, :w]
        circle = ((yy - cy) ** 2 + (xx - cx) ** 2) <= r * r
        mask_bin[circle] = 1
    return mask_bin

def dilate_binary_mask(mask_bin, iterations=1):
    out = mask_bin.copy()
    for _ in range(iterations):
        padded = np.pad(out, 1, mode="constant")
        h, w = out.shape
        new = np.zeros_like(out)
        for dy in range(3):
            for dx in range(3):
                new = np.maximum(new, padded[dy:dy+h, dx:dx+w])
        out = new
    return out

def blur_binary_mask(mask_bin, kernel_size=5):
    if kernel_size < 1:
        return mask_bin.astype(np.float32)
    pad = kernel_size // 2
    padded = np.pad(mask_bin.astype(np.float32), pad, mode="edge")
    h, w = mask_bin.shape
    out = np.zeros((h, w), dtype=np.float32)
    area = float(kernel_size * kernel_size)
    for y in range(h):
        for x in range(w):
            out[y, x] = padded[y:y+kernel_size, x:x+kernel_size].sum() / area
    return np.clip(out, 0.0, 1.0)

def choose_variant_specs(num_variants):
    base = [
        {"name": "gaussian_low",        "type": "gaussian",   "sigma": 8.0,  "expand": 0},
        {"name": "gaussian_medium",     "type": "gaussian",   "sigma": 15.0, "expand": 1},
        {"name": "gaussian_high",       "type": "gaussian",   "sigma": 28.0, "expand": 1},
        {"name": "speckle_low",         "type": "speckle",    "scale": 0.06, "expand": 1},
        {"name": "speckle_medium",      "type": "speckle",    "scale": 0.12, "expand": 1},
        {"name": "speckle_high",        "type": "speckle",    "scale": 0.18, "expand": 2},
        {"name": "salt_pepper_low",     "type": "saltpepper", "amount": 0.03, "expand": 1},
        {"name": "salt_pepper_medium",  "type": "saltpepper", "amount": 0.07, "expand": 1},
        {"name": "poisson",             "type": "poisson",    "lam_scale": 1.0, "expand": 1},
        {"name": "uniform",             "type": "uniform",    "width": 22.0, "expand": 1},
    ]
    specs = []
    for i in range(num_variants):
        spec = base[i % len(base)].copy()
        spec["id"] = i
        specs.append(spec)
    return specs

def apply_noise_variant(img_arr, mask_bin, spec, rng):
    h, w, _ = img_arr.shape
    out = img_arr.astype(np.float32).copy()

    if spec.get("expand", 0) > 0:
        mask_bin = dilate_binary_mask(mask_bin, iterations=spec["expand"])

    soft_mask = blur_binary_mask(mask_bin, kernel_size=5)[..., None]
    hard_mask = mask_bin[..., None].astype(bool)

    if spec["type"] == "gaussian":
        noise = rng.normal(loc=0.0, scale=spec["sigma"], size=out.shape).astype(np.float32)
        noisy = np.clip(out + noise, 0, 255)

    elif spec["type"] == "speckle":
        noise = rng.normal(loc=0.0, scale=spec["scale"], size=out.shape).astype(np.float32)
        noisy = np.clip(out + out * noise, 0, 255)

    elif spec["type"] == "saltpepper":
        noisy = out.copy()
        coords = rng.random((h, w))
        salt = (coords < spec["amount"] / 2.0) & mask_bin.astype(bool)
        pepper = (coords > 1.0 - spec["amount"] / 2.0) & mask_bin.astype(bool)
        for c in range(3):
            noisy[..., c][salt] = 255.0
            noisy[..., c][pepper] = 0.0

    elif spec["type"] == "poisson":
        norm = np.clip(out / 255.0, 0.0, 1.0)
        vals = np.maximum(norm * 255.0 * spec.get("lam_scale", 1.0), 1.0)
        noisy = rng.poisson(vals).astype(np.float32)
        noisy = np.clip(noisy / spec.get("lam_scale", 1.0), 0, 255)

    elif spec["type"] == "uniform":
        noise = rng.uniform(low=-spec["width"], high=spec["width"], size=out.shape).astype(np.float32)
        noisy = np.clip(out + noise, 0, 255)

    else:
        raise ValueError(f"Unknown variant type: {spec['type']}")

    blended = out * (1.0 - soft_mask) + noisy * soft_mask
    blended[~hard_mask.repeat(3, axis=2)] = out[~hard_mask.repeat(3, axis=2)]

    return np.clip(blended, 0, 255).astype(np.uint8), mask_bin.astype(np.uint8)

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
    pos_rows = df[df["label"] == "positive"]["path"].tolist()

    if not neg_rows or not pos_rows:
        print(f"[WARN] Empty split: {split} (neg={len(neg_rows)} pos={len(pos_rows)})")
        continue

    print(f"[INFO] {split} | Neg={len(neg_rows)} Pos={len(pos_rows)}")

    img_idx = 0

    for spec in variant_specs:
        print(f"[INFO] Variant {spec['id']+1}/{len(variant_specs)}: {spec['name']}")
        made = 0
        attempts = 0
        max_attempts = max(args.imgs_per_variant * 30, 300)

        while made < args.imgs_per_variant and attempts < max_attempts:
            attempts += 1

            neg_csv = random.choice(neg_rows)
            pos_csv = random.choice(pos_rows)

            neg_path = resolve_path(cat_root, split, neg_csv)
            mask_path = build_mask_path(cat_root, split, pos_csv)

            if not os.path.exists(neg_path) or not os.path.exists(mask_path):
                continue

            try:
                neg_img = load_rgb(neg_path)
                mask_img = load_mask(mask_path)

                neg_arr = np.array(neg_img, dtype=np.uint8)
                mask_bin = ensure_nonempty_mask(mask_to_binary(mask_img))

                rng = np.random.default_rng(base_seed + img_idx)
                out_arr, out_mask = apply_noise_variant(neg_arr, mask_bin, spec, rng)

                Image.fromarray(out_arr).save(os.path.join(imgs_out, f"{img_idx:05d}.png"))
                Image.fromarray((out_mask * 255).astype(np.uint8)).save(os.path.join(masks_out, f"{img_idx:05d}_GT.png"))

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
