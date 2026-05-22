"""Compare real-bad vs regenerated vs normal images with t-SNE (step 3).

Turn each image into a feature vector (embedding) with a pretrained network,
then squash all vectors to 2-D with t-SNE and plot the three groups in
different colours. If the regenerated points overlap the real-bad points,
the generation closed the gap; if they sit apart, it did not.
"""
import argparse
import os

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


@torch.no_grad()
def embed_images(model, paths):
    """Turn a list of image paths into an (N, 2048) array of feature vectors."""
    device = next(model.parameters()).device   # where the model lives (GPU/CPU)
    vectors = []
    for path in paths:
        img = Image.open(path).convert("RGB")
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src_dir", default="dataset/preprocessed")
    ap.add_argument("--category", required=True)
    ap.add_argument("--generated_dirname", default="generated")
    ap.add_argument("--out", default=None, help="output PNG path (default: in category dir)")
    ap.add_argument("--max_per_group", type=int, default=None,
                    help="cap images per group for speed (default: all)")
    args = ap.parse_args()

    cat_dir = os.path.join(args.src_dir, args.category)

    # 1. collect paths for the three groups
    real_paths = list_by_label(cat_dir, "test.csv", "positive")
    normal_paths = list_by_label(cat_dir, "test.csv", "negative")
    gen_paths = list_generated(cat_dir, args.generated_dirname)
    if args.max_per_group:
        real_paths = real_paths[:args.max_per_group]
        normal_paths = normal_paths[:args.max_per_group]
        gen_paths = gen_paths[:args.max_per_group]
    print(f"real={len(real_paths)}  generated={len(gen_paths)}  normal={len(normal_paths)}")

    if not (real_paths and gen_paths and normal_paths):
        raise SystemExit(
            "need images in all 3 groups - did you run run_pipeline.py first?")

    # 2. embed each group into feature vectors
    model = load_feature_extractor()
    real_feats = embed_images(model, real_paths)
    gen_feats = embed_images(model, gen_paths)
    normal_feats = embed_images(model, normal_paths)

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
    plt.title(f"t-SNE: {args.category}")

    out = args.out or os.path.join(cat_dir, f"tsne_{args.category}.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"saved {out}")


if __name__ == "__main__":
    main()
