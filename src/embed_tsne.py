"""Compare real-bad vs regenerated vs normal images with t-SNE (step 3).

Turn each image into a feature vector (embedding) with a pretrained network,
then squash all vectors to 2-D with t-SNE and plot the three groups in
different colours. If the regenerated points overlap the real-bad points,
the generation closed the gap; if they sit apart, it did not.
"""
import argparse
import os
import random

import matplotlib
matplotlib.use("Agg")           # no display on the server; save to file instead
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torchvision
from PIL import Image
from sklearn.manifold import TSNE
from torchvision import transforms


# How every image must be prepared before going into ResNet50 (ImageNet recipe).
TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),                 # ResNet50's expected input size
    transforms.ToTensor(),                         # PIL image -> tensor, scales to 0..1
    transforms.Normalize(mean=[0.485, 0.456, 0.406],   # ImageNet channel means
                         std=[0.229, 0.224, 0.225]),   # ImageNet channel std-devs
])


def load_feature_extractor():
    """Load a pretrained ResNet50 with its classifier removed -> outputs a vector."""
    weights = torchvision.models.ResNet50_Weights.IMAGENET1K_V2
    model = torchvision.models.resnet50(weights=weights)
    model.fc = torch.nn.Identity()   # drop the 1000-class head -> keep the 2048-d features
    model.eval()                     # inference mode (no training / no dropout)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    return model.to(device)


def find_mask(image_path):
    """GT mask next to a real defect image: foo.png -> foo_GT.png."""
    stem, ext = os.path.splitext(image_path)
    return stem + "_GT" + ext


def gen_mask_for(gen_path):
    """Saved mask of a generated image: .../generated/0000.png -> .../masks/0000_GT.png."""
    folder = os.path.dirname(gen_path)
    name = os.path.splitext(os.path.basename(gen_path))[0]
    return os.path.join(folder, "masks", name + "_GT.png")


def crop_to_mask(img, mask_path, pad=8):
    """Crop img to the bounding box of the defect (white area of the mask)."""
    if mask_path is None or not os.path.exists(mask_path):
        return img                                 # no mask -> keep whole image
    mask = np.array(Image.open(mask_path).convert("L").resize(img.size, Image.NEAREST))
    ys, xs = np.where(mask > 10)                   # pixels that are part of the defect
    if len(xs) == 0:
        return img                                 # empty mask -> keep whole image
    x0, y0, x1, y1 = xs.min(), ys.min(), xs.max(), ys.max()
    x0, y0 = max(0, x0 - pad), max(0, y0 - pad)    # add a little padding around it
    x1, y1 = min(img.width, x1 + pad), min(img.height, y1 + pad)
    return img.crop((x0, y0, x1, y1))


@torch.no_grad()
def embed_images(model, paths, masks=None):
    """Turn image paths into an (N, 2048) array; if masks given, crop to the defect first."""
    device = next(model.parameters()).device   # where the model lives (GPU/CPU)
    vectors = []
    for i, path in enumerate(paths):
        img = Image.open(path).convert("RGB")
        if masks is not None:
            img = crop_to_mask(img, masks[i])            # focus on the defect region
        tensor = TRANSFORM(img).unsqueeze(0).to(device)  # (1, 3, 224, 224) batch of 1
        feat = model(tensor)                             # (1, 2048) feature vector
        vectors.append(feat.cpu().numpy()[0])           # back to CPU, store the 2048 numbers
    return np.array(vectors)                             # stack into (N, 2048)


def list_by_label(cat_dir, csv_name, label):
    """Image paths from a csv (test.csv / train.csv) filtered by label, skipping masks."""
    df = pd.read_csv(os.path.join(cat_dir, csv_name))
    rows = df[df["label"] == label]["path"].tolist()
    paths = []
    for rel in rows:
        if "_GT" in rel:                       # never include mask images
            continue
        full = os.path.join(cat_dir, rel)
        if os.path.exists(full):
            paths.append(full)
    return paths


def list_generated(cat_dir, generated_dirname):
    """The generated defect images (top-level pngs, not the compare/ strips)."""
    gen_dir = os.path.join(cat_dir, generated_dirname)
    if not os.path.isdir(gen_dir):            # not generated yet -> no images
        return []
    return [os.path.join(gen_dir, f) for f in sorted(os.listdir(gen_dir))
            if f.endswith(".png")]            # the compare/ folder is skipped (not a .png)


def overlap_score(gen_feats, real_feats, normal_feats, trials=20, seed=0):
    """Fraction of generated points closer to a real defect than to a normal.

    Computed on the raw 2048-d features (not the t-SNE coords). Normals are
    subsampled to match the number of real defects so the comparison is fair
    (otherwise the far more numerous normals win on density alone); averaged
    over several random subsamples. ~1.0 = generated look like real defects;
    ~0.0 = they look like normal images."""
    rng = np.random.default_rng(seed)
    n = min(len(real_feats), len(normal_feats))
    dist_to_real = np.array([np.linalg.norm(real_feats - g, axis=1).min()
                             for g in gen_feats])
    scores = []
    for _ in range(trials):
        sample = normal_feats[rng.choice(len(normal_feats), n, replace=False)]
        dist_to_normal = np.array([np.linalg.norm(sample - g, axis=1).min()
                                   for g in gen_feats])
        scores.append(float(np.mean(dist_to_real < dist_to_normal)))
    return float(np.mean(scores))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src_dir", default="dataset/preprocessed")
    ap.add_argument("--category", required=True)
    ap.add_argument("--generated_dirname", default="generated")
    ap.add_argument("--out", default=None, help="output PNG path (default: in category dir)")
    ap.add_argument("--max_per_group", type=int, default=None,
                    help="cap images per group for speed (default: all)")
    ap.add_argument("--crop_to_mask", action="store_true",
                    help="crop each image to its defect region before embedding")
    args = ap.parse_args()

    cat_dir = os.path.join(args.src_dir, args.category)

    # 1. collect paths for the three groups (use test.csv if present, else train.csv)
    csv_name = "test.csv" if os.path.exists(os.path.join(cat_dir, "test.csv")) else "train.csv"
    real_paths = list_by_label(cat_dir, csv_name, "positive")
    normal_paths = list_by_label(cat_dir, csv_name, "negative")
    gen_paths = list_generated(cat_dir, args.generated_dirname)
    if args.max_per_group:
        real_paths = real_paths[:args.max_per_group]
        normal_paths = normal_paths[:args.max_per_group]
        gen_paths = gen_paths[:args.max_per_group]
    print(f"real={len(real_paths)}  generated={len(gen_paths)}  normal={len(normal_paths)}")

    if not (real_paths and gen_paths and normal_paths):
        raise SystemExit(
            "need images in all 3 groups - did you run run_pipeline.py first?")

    # build per-image masks if cropping (real -> own GT, generated -> saved mask,
    # normal -> a borrowed real mask so the crops are comparable in size)
    real_masks = gen_masks = normal_masks = None
    if args.crop_to_mask:
        rng = random.Random(0)
        real_masks = [find_mask(p) for p in real_paths]
        gen_masks = [gen_mask_for(p) for p in gen_paths]
        usable = [m for m in real_masks if os.path.exists(m)]
        normal_masks = [rng.choice(usable) for _ in normal_paths]

    # 2. embed each group into feature vectors
    model = load_feature_extractor()
    real_feats = embed_images(model, real_paths, real_masks)
    gen_feats = embed_images(model, gen_paths, gen_masks)
    normal_feats = embed_images(model, normal_paths, normal_masks)

    # how often is a generated defect closer to a real defect than to a normal?
    score = overlap_score(gen_feats, real_feats, normal_feats)
    print(f"OVERLAP_SCORE {args.category}: {score:.2f}  "
          f"(1.0 = looks like real defects, 0.0 = looks like normals)")

    # 3. run t-SNE on all vectors together (so they share one 2-D space)
    all_feats = np.concatenate([real_feats, gen_feats, normal_feats])
    perplexity = min(30, len(all_feats) - 1)        # must be < number of samples
    coords = TSNE(n_components=2, perplexity=perplexity,
                  random_state=42).fit_transform(all_feats)

    # split the 2-D coords back into the three groups (same order we concatenated)
    n_real, n_gen = len(real_feats), len(gen_feats)
    real_xy = coords[:n_real]
    gen_xy = coords[n_real:n_real + n_gen]
    normal_xy = coords[n_real + n_gen:]

    # 4. plot the three groups in three colours
    plt.figure(figsize=(8, 8))
    plt.scatter(normal_xy[:, 0], normal_xy[:, 1], c="blue", label="normal", alpha=0.6)
    plt.scatter(real_xy[:, 0], real_xy[:, 1], c="green", label="real defect", alpha=0.7)
    plt.scatter(gen_xy[:, 0], gen_xy[:, 1], c="orange", label="generated", alpha=0.7)
    plt.legend()
    suffix = "_crop" if args.crop_to_mask else ""
    plt.title(f"t-SNE: {args.category}" + (" (defect crops)" if args.crop_to_mask else ""))

    out = args.out or os.path.join(cat_dir, f"tsne_{args.category}{suffix}.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"saved {out}")


if __name__ == "__main__":
    main()
