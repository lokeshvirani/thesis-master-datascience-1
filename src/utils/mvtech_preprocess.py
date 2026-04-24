import os
import cv2
import argparse
import pandas as pd
from tqdm import tqdm


def resize_category(src_cat_dir, dst_cat_dir, res):
    """
    Resize images and GT masks for a single category.
    """
    for split in ["train", "test"]:
        src_split = os.path.join(src_cat_dir, split)
        dst_split = os.path.join(dst_cat_dir, split)

        if not os.path.exists(src_split):
            continue

        os.makedirs(dst_split, exist_ok=True)

        files = sorted(f for f in os.listdir(src_split) if f.lower().endswith(".png"))

        for fname in tqdm(files, desc=f"{os.path.basename(src_cat_dir)} | {split}", unit="img"):
            src_path = os.path.join(src_split, fname)
            dst_path = os.path.join(dst_split, fname)

            if "_GT" in fname:
                img = cv2.imread(src_path, cv2.IMREAD_GRAYSCALE)
                if img is None:
                    print(f"[WARN] Could not read GT mask: {src_path}")
                    continue
                img = cv2.resize(img, res, interpolation=cv2.INTER_NEAREST)
            else:
                img = cv2.imread(src_path)
                if img is None:
                    print(f"[WARN] Could not read image: {src_path}")
                    continue
                img = cv2.resize(img, res, interpolation=cv2.INTER_LINEAR)

            cv2.imwrite(dst_path, img)


def create_csv(cat_dst_dir):
    """
    Create train.csv and test.csv for a category.
    """
    for split in ["train", "test"]:
        split_dir = os.path.join(cat_dst_dir, split)
        if not os.path.exists(split_dir):
            continue

        gt_files = sorted(f for f in os.listdir(split_dir) if "_GT" in f)

        rows = {"path": [], "label": []}

        for gt in tqdm(gt_files, desc=f"CSV | {os.path.basename(cat_dst_dir)} | {split}", unit="img"):
            gt_path = os.path.join(split_dir, gt)
            mask = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)

            if mask is None:
                print(f"[WARN] Could not read GT mask for CSV: {gt_path}")
                continue

            label = "positive" if (mask > 0).any() else "negative"
            base, ext = os.path.splitext(gt)
            img_name = base.replace("_GT", "") + ext
            img_path = os.path.join(split_dir, img_name)
            if not os.path.exists(img_path):
                print(f"[WARN] Corresponding image not found: {img_path}")
                continue

            rows["path"].append(os.path.join(split, img_name))
            rows["label"].append(label)

        df = pd.DataFrame(rows)
        df.to_csv(os.path.join(cat_dst_dir, f"{split}.csv"), index=False)


def process_all_categories(src_dir, dst_dir, res):
    categories = [
        d for d in os.listdir(src_dir)
        if os.path.isdir(os.path.join(src_dir, d))
    ]

    print(f"[INFO] Found {len(categories)} categories")

    for category in categories:
        print(f"\n[INFO] Processing category: {category}")

        src_cat_dir = os.path.join(src_dir, category)
        dst_cat_dir = os.path.join(dst_dir, category)
        os.makedirs(dst_cat_dir, exist_ok=True)

        resize_category(src_cat_dir, dst_cat_dir, res)
        create_csv(dst_cat_dir)

        print(f"[INFO] Finished category: {category}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src_dir", type=str, required=True)
    parser.add_argument("--dst_dir", type=str, required=True)
    parser.add_argument("--res", type=int, nargs=2, default=[256, 256])  # w h

    args = parser.parse_args()
    res = tuple(args.res)

    os.makedirs(args.dst_dir, exist_ok=True)

    process_all_categories(args.src_dir, args.dst_dir, res)


if __name__ == "__main__":
    main()
