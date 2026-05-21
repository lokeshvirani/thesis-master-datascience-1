"""t-SNE of PatchCore patch descriptors: real vs generated anomalies, per category.

For every MVTec AD 2 category, two panels sharing one t-SNE space:
  Left  (Ours)  : normal patches + REAL anomaly patches + OURS generated anomaly patches
  Right (MemSeg): normal patches + REAL anomaly patches + MemSeg generated anomaly patches

Why patch-level: every point is a PatchCore patch descriptor (wide_resnet50_2
layer2+layer3, the exact extraction src/patchcore.py uses), so all four groups live
in one feature space — no crop-vs-full-image or image-size artefact, which is what
broke the earlier image-level t-SNE. A patch counts as "anomaly" when its feature-grid
cell falls inside the GT mask.

Real anomalies come from the labelled *test* split (full images + GT masks), NOT the
legacy defect_crops/ folder — those are degenerate (many are 3x3 px) and resize to
meaningless colour blobs.

Run from the project root on the uni server:
    source thesis_env/bin/activate
    python src/analysis/tsne_real_vs_gen.py
"""

import os
import csv
import random
import argparse

import numpy as np
from PIL import Image
from scipy.ndimage import binary_dilation

import torch
import torch.nn as nn
import torchvision.transforms as T
import torchvision.models as models
from torchvision.models import Wide_ResNet50_2_Weights

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ----------------------------- config -----------------------------
# Defaults; can be overridden via CLI flags (see argparse block at bottom).
BASE = "/media/data/Lokesh/thesis-master-datascience/data/preprocessed"
OURS_VARIANT = "augmented_8p30"
MEMSEG_VARIANT = "memseg_8v30"
CATEGORIES = ["can", "fabric", "fruit_jelly", "rice",
              "sheet_metal", "vial", "wallplugs", "walnuts"]

INPUT_SIZE = 384             # backbone input; finer grid (~48x48) than the 256 default
                             # so small real defects yield more than ~1 patch each
BATCH = 10                   # backbone forward batch (fits a GTX 1080 at 384px)
MASK_DILATE = 1              # grid-cell dilation of defect masks — catch boundary
                             # patches whose receptive field still sees the defect
N_NORMAL_IMGS = 200          # normal images to sample patches from
PATCHES_PER_NORMAL = 12      # random patches kept per normal image
MAX_POS_IMGS = 9999          # use every available positive image (real defects scarce)
MAX_PATCHES_PER_IMG = 50     # per-image cap for anomaly patches — stop one large
                             # defect image from dominating / forming a tight island
MAX_PATCHES_PER_GROUP = 2200 # cap t-SNE points per group
NEG_PATCHES_CAP = 1800       # normal patches kept a bit lower so they don't swamp
PCA_DIM = 50
PERPLEXITY = 30
SEED = 42
OUT_DIR = "results/tsne_real_vs_gen"

# size-tiered styling: normal small & faint (background), generated medium,
# real large & opaque & on top — so scarce real patches stay visible.
GROUP_STYLE = {
    "negative": dict(color="#4878c4", label="normal patches",
                     s=6, alpha=0.30, zorder=1, edge="none"),
    "gen":      dict(color="#d95f02", label="generated anomaly patches",
                     s=15, alpha=0.55, zorder=2, edge="none"),
    "real":     dict(color="#1b9e77", label="real anomaly patches",
                     s=30, alpha=0.80, zorder=3, edge="white"),
}

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
device = "cuda" if torch.cuda.is_available() else "cpu"
_normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])


# ----------------------------- backbone ---------------------------
class PatchExtractor(nn.Module):
    """wide_resnet50_2 layer2+layer3 patch descriptors — mirrors src/patchcore.py."""

    def __init__(self):
        super().__init__()
        self.model = models.wide_resnet50_2(weights=Wide_ResNet50_2_Weights.IMAGENET1K_V1)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False
        self._feats = []
        self.model.layer2[-1].register_forward_hook(lambda m, i, o: self._feats.append(o))
        self.model.layer3[-1].register_forward_hook(lambda m, i, o: self._feats.append(o))
        self.avg = nn.AvgPool2d(3, stride=1)

    @torch.no_grad()
    def forward(self, x):
        """(B,3,H,W) -> (patches (B,gh,gw,C), gh, gw)."""
        self._feats = []
        _ = self.model(x)
        f2, f3 = self._feats
        gh, gw = f2.shape[-2], f2.shape[-1]
        resize = nn.AdaptiveAvgPool2d((gh, gw))
        maps = [resize(self.avg(f)) for f in (f2, f3)]
        cat = torch.cat(maps, dim=1)                       # (B, C, gh, gw)
        patches = cat.permute(0, 2, 3, 1).contiguous()     # (B, gh, gw, C)
        return patches, gh, gw


# ----------------------------- data lists -------------------------
def list_negatives(cat_dir, limit):
    """Normal images: 'negative' rows of train.csv."""
    paths = []
    with open(os.path.join(cat_dir, "train.csv"), newline="") as fh:
        for r in csv.DictReader(fh):
            if r["label"].strip() == "negative":
                paths.append(os.path.join(cat_dir, r["path"]))
    random.shuffle(paths)
    return paths[:limit]


def list_test_reals(cat_dir, limit):
    """Real anomalies: 'positive' rows of test.csv (or train.csv as fallback,
    for categories that ship positives in the train split — e.g. `can`),
    paired with their GT masks."""
    csv_name = "test.csv" if os.path.exists(os.path.join(cat_dir, "test.csv")) else "train.csv"
    pairs = []
    with open(os.path.join(cat_dir, csv_name), newline="") as fh:
        for r in csv.DictReader(fh):
            if r["label"].strip() == "positive":
                img = os.path.join(cat_dir, r["path"])
                gt = img.replace(".png", "_GT.png")
                if os.path.exists(gt):
                    pairs.append((img, gt))
    random.shuffle(pairs)
    return pairs[:limit]


def list_variant(cat_dir, variant, limit):
    """Generated anomalies: (img, mask) pairs from <variant>/imgs + <variant>/masks."""
    imgs_dir = os.path.join(cat_dir, variant, "imgs")
    masks_dir = os.path.join(cat_dir, variant, "masks")
    if not os.path.isdir(imgs_dir):
        return []
    pairs = []
    for f in os.listdir(imgs_dir):
        if f.endswith(".png") and "_GT" not in f:
            gt = os.path.join(masks_dir, f.replace(".png", "_GT.png"))
            if os.path.exists(gt):
                pairs.append((os.path.join(imgs_dir, f), gt))
    random.shuffle(pairs)
    return pairs[:limit]


# ----------------------------- loading ----------------------------
def load_img_tensor(path):
    img = Image.open(path).convert("RGB").resize((INPUT_SIZE, INPUT_SIZE), Image.BILINEAR)
    return _normalize(T.functional.to_tensor(img))


def mask_on_grid(mask_path, gh, gw):
    """Binary (gh,gw) defect mask on the feature grid, dilated by MASK_DILATE
    cells. Falls back to the defect centroid cell when the defect is smaller
    than one grid cell."""
    raw = Image.open(mask_path).convert("L")
    grid = np.array(raw.resize((gw, gh), Image.NEAREST)) > 127
    if not grid.any():
        soft = np.array(raw.resize((gw, gh), Image.BILINEAR))
        if soft.max() == 0:
            return grid  # genuinely empty mask
        yy, xx = np.unravel_index(int(np.argmax(soft)), soft.shape)
        grid = np.zeros((gh, gw), dtype=bool)
        grid[yy, xx] = True
    if MASK_DILATE > 0:
        grid = binary_dilation(grid, iterations=MASK_DILATE)
    return grid


def cap(arr, n):
    if len(arr) > n:
        arr = arr[np.random.choice(len(arr), n, replace=False)]
    return arr


@torch.no_grad()
def extract_pos_patches(extractor, pairs, n_cap):
    """Descriptors of patches whose grid cell is inside the GT mask, capped per
    image so a single large defect can't dominate the group."""
    out = []
    for i in range(0, len(pairs), BATCH):
        chunk = pairs[i:i + BATCH]
        imgs = torch.stack([load_img_tensor(p) for p, _ in chunk]).to(device)
        patches, gh, gw = extractor(imgs)
        for b, (_, mpath) in enumerate(chunk):
            mk = mask_on_grid(mpath, gh, gw)
            if not mk.any():
                continue
            sel = patches[b][torch.from_numpy(mk).to(device)].cpu().numpy()
            if len(sel) > MAX_PATCHES_PER_IMG:
                sel = sel[np.random.choice(len(sel), MAX_PATCHES_PER_IMG, replace=False)]
            out.append(sel)
    if not out:
        return np.empty((0, 1536), dtype=np.float32)
    return cap(np.concatenate(out, 0), n_cap)


@torch.no_grad()
def extract_neg_patches(extractor, paths, per_img, n_cap):
    """Random patch descriptors from normal images."""
    out = []
    for i in range(0, len(paths), BATCH):
        chunk = paths[i:i + BATCH]
        imgs = torch.stack([load_img_tensor(p) for p in chunk]).to(device)
        patches, gh, gw = extractor(imgs)
        flat = patches.reshape(len(chunk), gh * gw, -1)
        for b in range(len(chunk)):
            idx = np.random.choice(gh * gw, min(per_img, gh * gw), replace=False)
            out.append(flat[b][idx].cpu().numpy())
    if not out:
        return np.empty((0, 1536), dtype=np.float32)
    return cap(np.concatenate(out, 0), n_cap)


# ----------------------------- run --------------------------------
def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    extractor = PatchExtractor().to(device).eval()

    for cat in CATEGORIES:
        cat_dir = os.path.join(BASE, cat)
        print(f"\n=== {cat} ===")

        neg = extract_neg_patches(
            extractor, list_negatives(cat_dir, N_NORMAL_IMGS),
            PATCHES_PER_NORMAL, NEG_PATCHES_CAP)
        real = extract_pos_patches(
            extractor, list_test_reals(cat_dir, MAX_POS_IMGS), MAX_PATCHES_PER_GROUP)
        ours = extract_pos_patches(
            extractor, list_variant(cat_dir, OURS_VARIANT, MAX_POS_IMGS), MAX_PATCHES_PER_GROUP)

        memseg_dir_exists = MEMSEG_VARIANT and os.path.isdir(
            os.path.join(cat_dir, MEMSEG_VARIANT, "imgs"))
        if memseg_dir_exists:
            memseg = extract_pos_patches(
                extractor, list_variant(cat_dir, MEMSEG_VARIANT, MAX_POS_IMGS),
                MAX_PATCHES_PER_GROUP)
        else:
            memseg = np.empty((0, 1536), dtype=np.float32)
            print(f"  (no memseg variant for {cat} — single-panel figure)")

        print(f"  patches: neg={len(neg)} real={len(real)} "
              f"ours={len(ours)} memseg={len(memseg)}")
        if min(len(real), len(ours), len(neg)) == 0:
            print(f"  !! skipping {cat}: an empty group")
            continue

        groups = [(neg, "negative"), (real, "real"), (ours, "ours")]
        if len(memseg) > 0:
            groups.append((memseg, "memseg"))
        feats = np.concatenate([g for g, _ in groups], axis=0)
        kinds = np.array(sum(
            ([name] * len(g) for g, name in groups), []))

        pca_dim = min(PCA_DIM, feats.shape[0], feats.shape[1])
        feats_pca = PCA(n_components=pca_dim, random_state=SEED).fit_transform(feats)
        perp = min(PERPLEXITY, max(5, (len(feats) - 1) // 3))
        coords = TSNE(n_components=2, perplexity=perp, learning_rate="auto",
                      init="pca", random_state=SEED).fit_transform(feats_pca)

        counts = {"negative": len(neg), "ours": len(ours),
                  "memseg": len(memseg), "real": len(real)}
        panels = [
            (f"Ours ({OURS_VARIANT})",
             {"negative": "negative", "ours": "gen", "real": "real"}),
        ]
        if len(memseg) > 0:
            panels.append((f"MemSeg ({MEMSEG_VARIANT})",
                           {"negative": "negative", "memseg": "gen", "real": "real"}))
        n_panels = len(panels)
        fig, axes = plt.subplots(1, n_panels, figsize=(7.5 * n_panels, 8),
                                 sharex=True, sharey=True, squeeze=False)
        axes = axes[0]
        for ax, (title, kmap) in zip(axes, panels):
            # draw order: normal (background) -> generated -> real (on top)
            for kind in ["negative", "ours", "memseg", "real"]:
                if kind not in kmap:
                    continue
                idx = kinds == kind
                if not idx.any():
                    continue
                style = GROUP_STYLE[kmap[kind]]
                ax.scatter(coords[idx, 0], coords[idx, 1],
                           s=style["s"], alpha=style["alpha"], zorder=style["zorder"],
                           color=style["color"],
                           label=f"{style['label']} (n={counts[kind]})",
                           linewidths=0.3, edgecolors=style["edge"])
            ax.set_title(title, fontsize=12)
            ax.legend(loc="best", fontsize=9, markerscale=2.0, framealpha=0.9)
            ax.set_xticks([])
            ax.set_yticks([])

        fig.suptitle(f"t-SNE of PatchCore patch descriptors — {cat}\n"
                     f"real vs generated anomalies", fontsize=13)
        fig.tight_layout(rect=[0, 0, 1, 0.92])
        out_file = os.path.join(OUT_DIR, f"tsne_{cat}.png")
        fig.savefig(out_file, dpi=200)
        plt.close(fig)
        print(f"  saved {out_file}")

    print("\nDone.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=BASE,
                    help="Root of preprocessed/<cat>/ folders")
    ap.add_argument("--variant", default=OURS_VARIANT,
                    help="Generated-anomaly variant folder name (Ours panel)")
    ap.add_argument("--memseg-variant", default=MEMSEG_VARIANT,
                    help="MemSeg variant folder name (MemSeg panel)")
    ap.add_argument("--categories", nargs="+", default=CATEGORIES,
                    help="Subset of categories to process")
    ap.add_argument("--out-dir", default=OUT_DIR,
                    help="Output directory for figures")
    cli = ap.parse_args()
    BASE = cli.base
    OURS_VARIANT = cli.variant
    MEMSEG_VARIANT = cli.memseg_variant
    CATEGORIES = cli.categories
    OUT_DIR = cli.out_dir
    main()
