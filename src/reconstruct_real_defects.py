"""reconstruct_real_defects.py — grounded real->caption->regenerate->compare loop.

Professor's new task (2026-05-21): the previous real-vs-generated t-SNE used
random mask shapes + generic human prompts, so "real and synthetic don't
overlap" was not a fair test. This script runs the *grounded* version:

    real BAD image ──► [vision LLM] ──► caption (what the defect looks like)
          │
          └──► its own GT mask (where the defect is)
                       │
    good (normal) image + GT mask + caption ──► [SDXL inpaint] ──► generated defect
                                                          │
                                          goal: generated ≈ the original BAD image

i.e. we hand the generator the *actual* description and the *actual* location of
a real defect and ask it to reproduce that defect on a clean object. The output
folder plugs straight into `src/analysis/tsne_real_vs_gen.py --variant <name>`,
so the follow-up t-SNE compares REAL defects against these grounded
reconstructions (not random-prompt junk).

Captions come from a JSON with paired sources (each real defect is captioned by
a vision-language model), structured as:
    {"captions": {"can": [cap0, cap1, ...]},
     "sources":  {"can": [path0, path1, ...]}}     # caption[i] <-> source[i]

Usage (uni, from project root):
    source thesis_env/bin/activate
    python src/reconstruct_real_defects.py \
        --src_dir data/preprocessed \
        --category can \
        --captions_json llm_captions.json \
        --out_dirname recon_llmcaption \
        --imgs_per_caption 4 \
        --low_vram
Then:
    python src/analysis/tsne_real_vs_gen.py \
        --base data/preprocessed --variant recon_llmcaption \
        --memseg-variant "" --categories can --out-dir results/recon_tsne
"""
import argparse
import json
import os
import random as rand_mod

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageFilter
from diffusers.pipelines.stable_diffusion_xl.pipeline_stable_diffusion_xl_inpaint import (
    StableDiffusionXLInpaintPipeline
)
from diffusers.utils import load_image

# ------------------------------------------------------------------------
# Constants  (named block — see project conventions)
# ------------------------------------------------------------------------
TARGET = (256, 256)            # inpaint canvas (matches generate_augmented_images_v2)
NUM_INFERENCE_STEPS = 30
DEFAULT_GUIDANCE = 7.0
DEFAULT_STRENGTH = 0.92
PADDING_MASK_CROP = 2
MASK_BLUR_RADIUS = 2           # soften GT-mask edges so the inpaint blends in
MODEL_ID = "diffusers/stable-diffusion-xl-1.0-inpainting-0.1"

# Photoreal grounding suffix — without it SDXL drifts to stylised renderings.
PROMPT_TEMPLATE = ("photorealistic close-up macro photo of {caption}, "
                   "industrial inspection image, natural lighting, subtle defect")

# Per-category negative prompts (mirror generate_augmented_images_v2.py so the
# grounded run uses the same defect-type guidance that the honest fix relied on).
DEFAULT_NEGATIVE_PROMPT = (
    "text, watermark, logo, letters, numbers, label, brand name, "
    "qr code, barcode, digital pattern, circuit board, pixelated pattern, "
    "cartoon, painting, illustration, unrealistic colors, "
    "cartoon character, mascot, drawing, sticker, anime, emoji, smiley, "
    "rainbow, holographic, prismatic, hologram, iridescent, kaleidoscope, "
    "extra objects, duplicate objects, foreign object, toy, ornament, "
    "background clutter, "
    "heavy blur, strong distortion, random noise, glare, reflections, lens flare, "
    "oversized defect, large damage, unrealistic defect shape"
)
CATEGORY_NEGATIVE_PROMPTS = {
    "can": (
        "dent, impact dent, scratch, fine scratch, rust, corrosion, "
        "raw metal, exposed aluminium, exposed metal, unpainted metal, "
        "puncture hole, hole, chipped rim, jagged edge, crease, "
        "cracked coating, peeled coating, abrasion mark, scuff mark, "
        "watermark, qr code, barcode, circuit board, pixelated pattern, "
        "cartoon, painting, illustration, "
        "cartoon character, mascot, anime, emoji, smiley, "
        "extra objects, duplicate objects, foreign object, toy, ornament, "
        "background clutter, "
        "heavy blur, strong distortion, random noise, glare, lens flare, "
        "oversized defect, large damage, unrealistic defect shape"
    ),
    "fruit_jelly": (
        "puncture, puncture mark, hole, dent, deformed dent, scratch, "
        "scratched surface, crack, cracked, broken edge, chipped edge, "
        "collapsed surface, solid jelly slab, solid surface, "
        "opaque container, metal can, metal cup, ceramic cup, "
        "watermark, logo, letters, brand name, qr code, barcode, "
        "circuit board, pixelated pattern, "
        "cartoon, painting, illustration, "
        "cartoon character, mascot, anime, emoji, smiley, sticker, "
        "rainbow, holographic, prismatic, hologram, iridescent, "
        "extra objects, duplicate objects, toy, ornament, "
        "background clutter, "
        "heavy blur, strong distortion, random noise, lens flare, "
        "oversized defect, large damage, unrealistic defect shape"
    ),
    "walnuts": (
        "tiny scratch, fine surface scratch, small mould spot, small dark spot, "
        "small surface stain, intact walnut, perfect walnut, undamaged walnut, "
        "tiny defect, miniature defect, "
        "watermark, logo, letters, brand name, qr code, barcode, "
        "circuit board, pixelated pattern, "
        "cartoon, painting, illustration, "
        "cartoon character, mascot, anime, emoji, smiley, sticker, "
        "rainbow, holographic, prismatic, hologram, iridescent, "
        "duplicate objects, toy, ornament, "
        "background clutter, "
        "heavy blur, strong distortion, random noise, glare, lens flare, "
        "oversized defect, unrealistic defect shape"
    ),
}


# ------------------------------------------------------------------------
# Caption / source loading + real-defect resolution
# ------------------------------------------------------------------------
def load_caption_pairs(captions_json, category):
    """Return list of (caption, source_path) for `category`.

    Accepts either {"captions": {...}, "sources": {...}} or a bare
    {category: [caption,...]} (no sources -> raises, since we need the source
    image to recover the GT mask)."""
    with open(captions_json) as f:
        payload = json.load(f)
    if "captions" not in payload or "sources" not in payload:
        raise SystemExit(
            "captions_json must contain BOTH 'captions' and 'sources' "
            "(each maps category -> list, with caption[i] paired to source[i]).")
    caps = payload["captions"].get(category, [])
    srcs = payload["sources"].get(category, [])
    if not caps:
        raise SystemExit(f"no captions for category '{category}' in {captions_json}")
    if len(caps) != len(srcs):
        raise SystemExit(
            f"captions ({len(caps)}) and sources ({len(srcs)}) length mismatch "
            f"for '{category}'")
    return list(zip(caps, srcs))


def resolve_real_defect(cat_dir, source_path):
    """Map a caption's source path to the preprocessed BAD image + its GT mask.

    The captions JSON references the raw mvtec_ad_2 tree (e.g.
    `.../test/bad/000_regular.png`) while the preprocessed tree flattens the
    name to `test/bad000_regular.png` with mask `test/bad000_regular_GT.png`.
    Tries a few spellings and verifies the GT mask exists.

    Returns (bad_img_path, gt_mask_path) or (None, None) if not found.
    """
    base = os.path.basename(source_path)                 # e.g. 000_regular.png
    stem, ext = os.path.splitext(base)
    candidates = [
        f"bad{base}",        # 000_regular.png -> bad000_regular.png
        base,                # already bad000_regular.png
    ]
    for split in ("test", "train"):
        for cand in candidates:
            img = os.path.join(cat_dir, split, cand)
            gt = os.path.join(cat_dir, split,
                              f"bad{stem}_GT{ext}" if not cand.startswith("bad")
                              else f"{os.path.splitext(cand)[0]}_GT{ext}")
            if os.path.exists(img) and os.path.exists(gt):
                return img, gt
    return None, None


def list_good_images(cat_dir):
    """Normal (negative) image paths from train.csv."""
    df = pd.read_csv(os.path.join(cat_dir, "train.csv"))
    rel = df[df["label"] == "negative"]["path"].tolist()
    out = []
    for r in rel:
        p = r if os.path.isabs(r) else os.path.join(cat_dir, r)
        if os.path.exists(p):
            out.append(p)
    return out


def load_gt_mask(gt_path, target, blur_radius):
    """GT mask -> soft-edged L-mode inpaint mask at `target` size."""
    m = Image.open(gt_path).convert("L").resize(target, Image.NEAREST)
    if blur_radius > 0:
        m = m.filter(ImageFilter.GaussianBlur(radius=blur_radius))
    return m


def save_compare(path, good_img, real_bad_path, gen_img, target):
    """Side-by-side triptych: good base | real defect | generated reconstruction."""
    real = Image.open(real_bad_path).convert("RGB").resize(target)
    strip = Image.new("RGB", (target[0] * 3, target[1]), (255, 255, 255))
    strip.paste(good_img.resize(target), (0, 0))
    strip.paste(real, (target[0], 0))
    strip.paste(gen_img.resize(target), (target[0] * 2, 0))
    strip.save(path)


# ------------------------------------------------------------------------
# Pipeline
# ------------------------------------------------------------------------
def build_pipe(low_vram):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    kwargs = {"torch_dtype": dtype}
    if device.type == "cuda":
        kwargs["variant"] = "fp16"
    pipe = StableDiffusionXLInpaintPipeline.from_pretrained(MODEL_ID, **kwargs)
    if low_vram and device.type == "cuda":
        print("[INFO] low_vram: model CPU offload + VAE slicing/tiling")
        pipe.enable_model_cpu_offload()
        pipe.enable_vae_slicing()
        pipe.enable_vae_tiling()
    else:
        pipe = pipe.to(device)
    pipe.set_progress_bar_config(disable=True)
    pipe.enable_attention_slicing()
    gen_device = "cuda" if device.type == "cuda" else "cpu"
    return pipe, gen_device


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src_dir", default="data/preprocessed",
                    help="Root with <category>/ subfolders")
    ap.add_argument("--category", required=True)
    ap.add_argument("--captions_json", required=True,
                    help="JSON with paired 'captions' + 'sources' (per category)")
    ap.add_argument("--out_dirname", default="recon_llmcaption",
                    help="Output folder name inside the category")
    ap.add_argument("--imgs_per_caption", type=int, default=4,
                    help="Reconstructions per real defect (different good bases)")
    ap.add_argument("--max_captions", type=int, default=None,
                    help="Cap number of real defects used (default: all)")
    ap.add_argument("--guidance", type=float, default=DEFAULT_GUIDANCE)
    ap.add_argument("--strength", type=float, default=DEFAULT_STRENGTH)
    ap.add_argument("--mask_blur", type=float, default=MASK_BLUR_RADIUS)
    ap.add_argument("--base_seed", type=int, default=42)
    ap.add_argument("--low_vram", action="store_true")
    ap.add_argument("--no_compare", action="store_true",
                    help="Skip writing the good|real|gen comparison triptychs")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    cat_dir = os.path.join(os.path.abspath(args.src_dir), args.category)
    if not os.path.isdir(cat_dir):
        raise SystemExit(f"missing category dir: {cat_dir}")

    pairs = load_caption_pairs(args.captions_json, args.category)
    if args.max_captions:
        pairs = pairs[:args.max_captions]

    good_imgs = list_good_images(cat_dir)
    if not good_imgs:
        raise SystemExit(f"no normal/good images for {args.category}")

    neg_prompt = CATEGORY_NEGATIVE_PROMPTS.get(args.category, DEFAULT_NEGATIVE_PROMPT)

    out_root = os.path.join(cat_dir, args.out_dirname)
    imgs_out = os.path.join(out_root, "imgs")
    masks_out = os.path.join(out_root, "masks")
    cmp_out = os.path.join(out_root, "compare")
    for d in (imgs_out, masks_out, cmp_out):
        os.makedirs(d, exist_ok=True)

    existing = [f for f in os.listdir(imgs_out) if f.endswith(".png") and "_GT" not in f]
    expected = len(pairs) * args.imgs_per_caption
    if len(existing) >= expected and not args.overwrite:
        print(f"[INFO] {args.category}: already has {len(existing)} (>= {expected}); "
              f"use --overwrite to regenerate. Skipping.")
        return

    rng = rand_mod.Random(args.base_seed + sum(ord(c) for c in args.category))
    pipe, gen_device = build_pipe(args.low_vram)

    print(f"[INFO] category={args.category}  defects={len(pairs)}  "
          f"imgs_per_caption={args.imgs_per_caption}  -> {expected} images")
    print(f"[INFO] out={out_root}")

    manifest = []
    img_idx = 0
    n_unresolved = 0
    for ci, (caption, source) in enumerate(pairs):
        bad_path, gt_path = resolve_real_defect(cat_dir, source)
        if bad_path is None:
            print(f"[WARN] could not resolve real defect for source={source}; skip")
            n_unresolved += 1
            continue
        mask = load_gt_mask(gt_path, TARGET, args.mask_blur)
        full_prompt = PROMPT_TEMPLATE.format(caption=caption)
        print(f"[INFO] [{ci+1}/{len(pairs)}] {os.path.basename(bad_path)}")
        print(f"        caption: {caption[:90]}")
        for j in range(args.imgs_per_caption):
            good_path = rng.choice(good_imgs)
            good_img = load_image(good_path).resize(TARGET)
            generator = torch.Generator(device=gen_device).manual_seed(
                args.base_seed * 1000 + img_idx)
            try:
                gen = pipe(
                    prompt=full_prompt,
                    negative_prompt=neg_prompt,
                    image=good_img,
                    mask_image=mask,
                    guidance_scale=args.guidance,
                    num_inference_steps=NUM_INFERENCE_STEPS,
                    strength=args.strength,
                    generator=generator,
                    height=TARGET[1], width=TARGET[0],
                    original_size=TARGET, target_size=TARGET,
                    padding_mask_crop=PADDING_MASK_CROP,
                ).images[0]
            except Exception as e:
                print(f"[WARN]   gen error: {e}")
                continue
            name = f"{img_idx:05d}.png"
            gen.save(os.path.join(imgs_out, name))
            mask.save(os.path.join(masks_out, f"{img_idx:05d}_GT.png"))
            if not args.no_compare:
                save_compare(os.path.join(cmp_out, name),
                             good_img, bad_path, gen, TARGET)
            manifest.append({
                "idx": img_idx, "caption": caption,
                "real_defect": os.path.relpath(bad_path, cat_dir),
                "gt_mask": os.path.relpath(gt_path, cat_dir),
                "good_base": os.path.relpath(good_path, cat_dir),
            })
            img_idx += 1
        print(f"        -> {img_idx} total")

    with open(os.path.join(out_root, "manifest.json"), "w") as f:
        json.dump({"category": args.category,
                   "model": MODEL_ID,
                   "guidance": args.guidance, "strength": args.strength,
                   "imgs_per_caption": args.imgs_per_caption,
                   "unresolved_sources": n_unresolved,
                   "items": manifest}, f, indent=2)
    print(f"\n[INFO] done: {img_idx} reconstructions written to {imgs_out}")
    if n_unresolved:
        print(f"[INFO] {n_unresolved} caption source(s) could not be resolved "
              f"to a preprocessed defect+GT pair.")


if __name__ == "__main__":
    main()
