"""Run the generation pipeline for one category.

Reads cached captions (made earlier by caption.py) paired with their real
defect images, then for each one inpaints the captioned defect onto a good
image at the real mask location and saves the result + a comparison strip.
"""
import argparse
import json
import os
import random

import pandas as pd

import regenerate


def find_mask(bad_path):
    """The GT mask sits next to the bad image: foo.png -> foo_GT.png."""
    stem, ext = os.path.splitext(bad_path)
    return stem + "_GT" + ext


def resolve_source(cat_dir, source_rel):
    """Find the real defect image. The JSON uses test/ paths, but some setups
    keep the same files under train/ instead, so try both folders."""
    candidates = [source_rel]
    for a, b in (("test/", "train/"), ("train/", "test/")):
        if source_rel.startswith(a):
            candidates.append(b + source_rel[len(a):])
    for rel in candidates:
        full = os.path.join(cat_dir, rel)
        if os.path.exists(full):
            return full
    return None


def load_captions(captions_json, category):
    """Return (caption, source_path) pairs for `category` from the JSON file."""
    with open(captions_json) as f:
        data = json.load(f)
    captions = data["captions"][category]   # list of caption strings
    sources = data["sources"][category]     # matching real-defect image paths
    return list(zip(captions, sources))     # pair caption[i] with source[i]


def list_good_images(cat_dir):
    """Return paths of normal (negative) images from train.csv."""
    df = pd.read_csv(os.path.join(cat_dir, "train.csv"))
    good_rows = df[df["label"] == "negative"]["path"].tolist()
    return [os.path.join(cat_dir, rel) for rel in good_rows
            if os.path.exists(os.path.join(cat_dir, rel))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src_dir", default="dataset/preprocessed")
    ap.add_argument("--category", required=True)
    ap.add_argument("--captions_json", required=True,
                    help="JSON with cached captions + sources (e.g. llm_captions_can_full.json)")
    ap.add_argument("--out_dirname", default="generated")
    ap.add_argument("--max_defects", type=int, default=None,
                    help="Only process the first N defects (default: all)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--low_vram", action="store_true",
                    help="Use less GPU memory (slower) - needed on uni")
    args = ap.parse_args()

    cat_dir = os.path.join(args.src_dir, args.category)
    out_dir = os.path.join(cat_dir, args.out_dirname)
    compare_dir = os.path.join(out_dir, "compare")
    masks_dir = os.path.join(out_dir, "masks")
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(compare_dir, exist_ok=True)
    os.makedirs(masks_dir, exist_ok=True)

    # Load the cached captions (each paired with its real-defect image) + good pool.
    pairs = load_captions(args.captions_json, args.category)
    if args.max_defects:
        pairs = pairs[:args.max_defects]
    good_images = list_good_images(cat_dir)

    rng = random.Random(args.seed)
    pipe = regenerate.load_model(low_vram=args.low_vram)  # the diffusion model

    for i, (caption_text, source_rel) in enumerate(pairs):
        bad_path = resolve_source(cat_dir, source_rel)  # real defect (test/ or train/)
        if bad_path is None:
            print(f"[skip] real defect not found: {source_rel}")
            continue
        mask_path = find_mask(bad_path)                 # its GT mask (foo -> foo_GT)
        if not os.path.exists(mask_path):
            print(f"[skip] mask not found for {os.path.basename(bad_path)}")
            continue

        good_path = rng.choice(good_images)            # random good base image
        good_image = regenerate.load_good_image(good_path)
        mask = regenerate.load_mask(mask_path)
        generated = regenerate.generate_defect(        # inpaint the captioned defect
            pipe, good_image, mask, caption_text, category=args.category, seed=args.seed + i)

        out_path = os.path.join(out_dir, f"{i:04d}.png")
        generated.save(out_path)
        mask.save(os.path.join(masks_dir, f"{i:04d}_GT.png"))  # save mask for later cropping
        regenerate.save_comparison(                    # good|real|gen strip
            os.path.join(compare_dir, f"{i:04d}.png"), good_image, bad_path, generated)
        print(f"[{i+1}/{len(pairs)}] {source_rel} -> {out_path}")
        print(f"      caption: {caption_text}")


if __name__ == "__main__":
    main()
