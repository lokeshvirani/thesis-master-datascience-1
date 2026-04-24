from diffusers.pipelines.stable_diffusion_xl.pipeline_stable_diffusion_xl_inpaint import (
    StableDiffusionXLInpaintPipeline
)
from diffusers.utils import load_image
import torch
import pandas as pd
import os
import random
import numpy as np
import argparse

# ----------------------------
# Args
# ----------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--src_dir", type=str, required=True)
parser.add_argument(
    "--category",
    type=str,
    default=None,
    help="If set, only generate for this category (e.g., can, sheet_metal)."
)
parser.add_argument(
    "--imgs_per_prompt",
    type=int,
    default=30,
    help="Number of generated images per prompt, per category."
)
parser.add_argument(
    "--out_dirname",
    type=str,
    default="augmented",
    help="Output folder name inside each category, e.g. augmented_10p30"
)
parser.add_argument("--seed", type=int, default=42)
args = parser.parse_args()

src_dir = os.path.abspath(args.src_dir)
imgs_per_prompt = args.imgs_per_prompt
base_seed = args.seed
out_dirname = args.out_dirname

if imgs_per_prompt <= 0:
    raise ValueError("--imgs_per_prompt must be > 0")

# ----------------------------
# Category-specific 10 prompts
# ----------------------------
CATEGORY_PROMPTS = {
    "can": [
        "a small dent on the metal can surface",
        "a fine scratch on the printed metal can surface",
        "a dark contamination stain on the can surface",
        "a chipped edge on the rim of the can",
        "a small puncture hole on the can body",
        "a rust-like corrosion spot on the metal can",
        "a crease-like deformation on the can wall",
        "a cracked coating area on the can surface",
        "a small abrasion mark on the can surface",
        "a damaged printed label region on the can body",
    ],
    "fabric": [
        "a small tear in the fabric texture",
        "a frayed thread region on the fabric surface",
        "a dark stain on the fabric surface",
        "a damaged fiber region on the fabric texture",
        "a small hole in the woven fabric",
        "a rubbed worn-out patch on the fabric",
        "a local contamination mark on the fabric surface",
        "a pulled thread defect in the fabric",
        "a scratched region on the fabric texture",
        "a discolored patch on the fabric surface",
    ],
    "fruit_jelly": [
        "a dark contamination spot on the fruit jelly surface",
        "a small puncture mark on the fruit jelly",
        "a broken surface region on the fruit jelly",
        "a deformed dent on the fruit jelly surface",
        "a cloudy impurity inside the fruit jelly",
        "a scratched outer layer on the fruit jelly",
        "a damaged edge on the fruit jelly surface",
        "a crack-like defect on the fruit jelly",
        "a discolored patch on the fruit jelly surface",
        "a collapsed surface area on the fruit jelly",
    ],
    "rice": [
        "a dark impurity cluster among the rice grains",
        "a contaminated patch in the rice grains",
        "a discolored region in the rice grains",
        "a small damaged grain cluster in the rice",
        "a foreign material spot in the rice grains",
        "a crushed grain region in the rice cluster",
        "a dark stain-like defect among the rice grains",
        "an irregular defective grain patch in the rice",
        "a broken rice grain group with contamination",
        "a clumped abnormal patch in the rice grains",
    ],
    "sheet_metal": [
        "a small dent on the sheet metal surface",
        "a fine scratch on the sheet metal",
        "a small rust spot on the metal surface",
        "a crack on the sheet metal surface",
        "a puncture hole in the sheet metal",
        "a bent deformation on the metal surface",
        "a chipped coating region on the sheet metal",
        "a dark contamination mark on the metal surface",
        "a scuff mark on the sheet metal",
        "a corroded patch on the sheet metal surface",
    ],
    "vial": [
        "a small crack on the glass vial surface",
        "a chipped edge on the glass vial",
        "a cloudy contamination spot on the vial surface",
        "a fine scratch on the glass vial",
        "a puncture-like mark on the vial body",
        "a broken rim defect on the glass vial",
        "a dark stain on the transparent vial surface",
        "a deformed surface region on the glass vial",
        "a chipped neck region on the vial",
        "a damaged transparent patch on the vial wall",
    ],
    "wallplugs": [
        "a small crack on the plastic wallplug surface",
        "a chipped edge on the wallplug",
        "a scratched region on the plastic wallplug",
        "a dark contamination spot on the wallplug surface",
        "a small deformation on the plastic wallplug",
        "a puncture-like defect on the wallplug body",
        "a broken notch on the wallplug",
        "a worn plastic patch on the wallplug surface",
        "a bent edge on the wallplug",
        "a damaged locking fin on the wallplug",
    ],
    "walnuts": [
        "a crack on the walnut shell surface",
        "a dark contamination stain on the walnut shell",
        "a small hole in the walnut shell",
        "a chipped shell edge on the walnut",
        "a broken shell patch on the walnut surface",
        "a scratched shell region on the walnut",
        "a deformed area on the walnut shell",
        "a damaged shell texture on the walnut surface",
        "a crushed shell patch on the walnut",
        "a dark mold-like spot on the walnut shell",
    ],
}

negative_prompt = (
    "text, watermark, logo, letters, numbers, label, brand name, "
    "cartoon, painting, illustration, unrealistic colors, "
    "extra objects, duplicate objects, background clutter, "
    "heavy blur, strong distortion, random noise, glare, reflections, "
    "oversized defect, large damage, unrealistic defect shape"
)

num_inference_steps = 30
guidance_scale = 10.0
strength = 0.7
padding_mask_crop = 2
TARGET = (256, 256)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.float16 if device.type == "cuda" else torch.float32
generator_device = "cuda" if device.type == "cuda" else "cpu"

print("[INFO] Device:", device)
print(f"[INFO] imgs_per_prompt={imgs_per_prompt}")
print(f"[INFO] out_dirname={out_dirname}")

# ----------------------------
# Seed
# ----------------------------
random.seed(base_seed)
np.random.seed(base_seed)
torch.manual_seed(base_seed)

# ----------------------------
# Path resolver
# ----------------------------
def resolve_path(cat_root, split, csv_path):
    """
    Ensures train/test appears exactly once.
    Works whether csv_path is:
      - img.png
      - train/img.png
      - category/train/img.png
    """
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

# ----------------------------
# Load model once
# ----------------------------
pipe_kwargs = {"torch_dtype": dtype}
if device.type == "cuda":
    pipe_kwargs["variant"] = "fp16"

pipe = StableDiffusionXLInpaintPipeline.from_pretrained(
    "diffusers/stable-diffusion-xl-1.0-inpainting-0.1",
    **pipe_kwargs
).to(device)

pipe.set_progress_bar_config(disable=True)
pipe.enable_attention_slicing()

# ----------------------------
# Categories
# ----------------------------
all_categories = sorted([
    d for d in os.listdir(src_dir)
    if os.path.isdir(os.path.join(src_dir, d))
])

if args.category:
    categories = [args.category]
else:
    categories = all_categories

print(f"[INFO] Found {len(all_categories)} categories under src_dir")
print(f"[INFO] Will generate for {len(categories)} categories: {categories}")

# ----------------------------
# Main loop
# ----------------------------
for category in categories:
    cat_root = os.path.join(src_dir, category)
    if not os.path.isdir(cat_root):
        print(f"[WARN] Category folder not found: {cat_root} (skip)")
        continue

    print(f"\n[INFO] Category: {category}")

    prompts = CATEGORY_PROMPTS.get(category, [])
    if len(prompts) == 0:
        print(f"[WARN] No prompts found for category: {category}")
        continue

    print(f"[INFO] Prompts={len(prompts)}")
    print(f"[INFO] Images per prompt={imgs_per_prompt}")
    print(f"[INFO] Total target images={len(prompts) * imgs_per_prompt}")

    # Keep this layout because your current loader expects augmented masks here too
    dst_root = os.path.join(cat_root, out_dirname)
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
        print("[WARN] This script needs both negative and positive rows in train.csv.")
        continue

    print(f"[INFO] {split} | Neg={len(neg_rows)} Pos={len(pos_rows)}")

    img_idx = 0

    for prompt in prompts:
        print(f"[INFO] Prompt: {prompt}")

        made = 0
        attempts = 0
        max_attempts = max(imgs_per_prompt * 25, 200)

        while made < imgs_per_prompt and attempts < max_attempts:
            attempts += 1

            neg_csv = random.choice(neg_rows)
            pos_csv = random.choice(pos_rows)

            neg_path = resolve_path(cat_root, split, neg_csv)
            mask_path = build_mask_path(cat_root, split, pos_csv)

            if not os.path.exists(neg_path):
                continue
            if not os.path.exists(mask_path):
                continue

            try:
                neg_img = load_image(neg_path).resize(TARGET)
                mask = load_image(mask_path).resize(TARGET)

                generator = torch.Generator(device=generator_device).manual_seed(base_seed + img_idx)

                out = pipe(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    image=neg_img,
                    mask_image=mask,
                    guidance_scale=guidance_scale,
                    num_inference_steps=num_inference_steps,
                    strength=strength,
                    generator=generator,
                    height=TARGET[1],
                    width=TARGET[0],
                    original_size=TARGET,
                    target_size=TARGET,
                    padding_mask_crop=padding_mask_crop
                ).images[0]

                out.save(os.path.join(imgs_out, f"{img_idx:05d}.png"))
                mask.save(os.path.join(masks_out, f"{img_idx:05d}_GT.png"))

                img_idx += 1
                made += 1

                if made % 10 == 0 or made == imgs_per_prompt:
                    print(f"[INFO]   Generated {made}/{imgs_per_prompt} for current prompt")

            except Exception as e:
                print(f"[WARN] Skipping one sample due to error: {e}")
                continue

        if made < imgs_per_prompt:
            print(f"[WARN] Only generated {made}/{imgs_per_prompt} for prompt: {prompt}")

    print(f"[INFO] Finished category '{category}' with {img_idx} total generated images")

print("[INFO] Done.")
