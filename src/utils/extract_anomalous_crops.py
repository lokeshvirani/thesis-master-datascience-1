import cv2
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import argparse
import glob

def reshape_image(image, target_size):
    return cv2.resize(image, (target_size[1], target_size[0]), interpolation=cv2.INTER_LINEAR)

def resolve_image_path(img_value_from_csv: str, train_dir: str, cat_dir: str = None):
    """
    Resolve a CSV 'path' value into an actual image filepath.

    Resolution order:
    1) If absolute and exists -> use it
    2) train_dir / csv_path
    3) cat_dir / csv_path (if cat_dir provided)
    4) train_dir / basename(csv_path)

    Returns:
        (resolved_path or None, tried_paths list)
    """
    tried = []

    if not isinstance(img_value_from_csv, str):
        return None, tried

    p = img_value_from_csv.strip().replace("\\", "/")

    # absolute path
    if os.path.isabs(p):
        tried.append(p)
        if os.path.exists(p):
            return p, tried

    # relative to train_dir
    cand1 = os.path.normpath(os.path.join(train_dir, p))
    tried.append(cand1)
    if os.path.exists(cand1):
        return cand1, tried

    # relative to category dir (multi-category)
    if cat_dir:
        cand2 = os.path.normpath(os.path.join(cat_dir, p))
        tried.append(cand2)
        if os.path.exists(cand2):
            return cand2, tried

    # basename fallback
    base = os.path.basename(p)
    cand3 = os.path.normpath(os.path.join(train_dir, base))
    tried.append(cand3)
    if os.path.exists(cand3):
        return cand3, tried

    return None, tried

def extract_anomalous_crops(train_dir: str, output_dir: str, csv_file: str = None, padding: int = 1,
                             min_area: int = 16, reshape_size: tuple = None, mask_suffix: str = "_GT",
                             cat_dir: str = None, verbose: bool = False, max_missing_logs: int = 50):
    os.makedirs(output_dir, exist_ok=True)
    processed_count = 0
    missing_logged = 0

    # Determine samples to process
    if csv_file:
        df = pd.read_csv(csv_file)
        if 'path' not in df.columns or 'label' not in df.columns:
            raise ValueError(f"CSV must have 'path' and 'label' columns. Found: {list(df.columns)}")
        samples = df[df['label'] == 'positive']['path'].tolist()
        if not samples:
            print(f"[WARN] No positive samples found in CSV: {csv_file}")
            return 0
    else:
        all_files = glob.glob(os.path.join(train_dir, "*.png"))
        samples = [
            os.path.basename(f) for f in all_files
            if not f.endswith(f"{mask_suffix}.png")
        ]
        if not samples:
            print(f"[WARN] No .png files found in {train_dir}.")
            return 0

    if verbose:
        print(f"\n[INFO] extract_anomalous_crops()")
        print(f"  train_dir   = {train_dir}")
        print(f"  output_dir  = {output_dir}")
        print(f"  csv_file    = {csv_file}")
        print(f"  cat_dir     = {cat_dir}")
        print(f"  samples     = {len(samples)}")

    for item in tqdm(samples, desc="Extracting anomaly patches"):
        # item is either filename (no CSV) OR a CSV path value
        if csv_file:
            img_path, tried = resolve_image_path(item, train_dir=train_dir, cat_dir=cat_dir)
            if img_path is None:
                if missing_logged < max_missing_logs:
                    print(f"\n[MISS] Image not found for CSV path: {item}")
                    for t in tried:
                        print(f"       tried: {t}")
                missing_logged += 1
                continue
            img_filename = os.path.basename(img_path)
        else:
            img_filename = item
            img_path = os.path.join(train_dir, img_filename)
            tried = [img_path]
            if not os.path.exists(img_path):
                if missing_logged < max_missing_logs:
                    print(f"\n[MISS] Image not found: {img_filename}")
                    print(f"       tried: {img_path}")
                missing_logged += 1
                continue

        base_name = os.path.splitext(os.path.basename(img_filename))[0]
        mask_filename = f"{base_name}{mask_suffix}.png"
        mask_path = os.path.join(os.path.dirname(img_path), mask_filename)

        if verbose:
            print(f"\n[FILE] item={item}")
            print(f"       img_path  = {img_path}")
            print(f"       mask_path = {mask_path}")

        if not os.path.exists(mask_path):
            if missing_logged < max_missing_logs:
                print(f"\n[MISS] Mask not found for image: {img_path}")
                print(f"       expected mask: {mask_path}")
            missing_logged += 1
            continue

        img_np = cv2.imread(img_path, cv2.IMREAD_COLOR)
        mask_np = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        if img_np is None or mask_np is None:
            print(f"[WARN] Failed to load img/mask: {img_path} / {mask_path}")
            continue

        img_np = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)

        mask_np = (mask_np > 0).astype(np.uint8)
        mask_uint8 = mask_np * 255

        contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Fallback: If no contours but mask has positive pixels
        if not contours and mask_np.sum() > 0:
            y_coords, x_coords = np.where(mask_np > 0)
            if len(y_coords) > 0:
                x_min = max(0, np.min(x_coords) - padding)
                x_max = min(mask_np.shape[1], np.max(x_coords) + padding + 1)
                y_min = max(0, np.min(y_coords) - padding)
                y_max = min(mask_np.shape[0], np.max(y_coords) + padding + 1)

                if (y_max - y_min) >= min_area and (x_max - x_min) >= min_area:
                    img_patch = img_np[y_min:y_max, x_min:x_max]
                    mask_patch = mask_uint8[y_min:y_max, x_min:x_max]

                    if reshape_size:
                        img_patch = reshape_image(img_patch, reshape_size)
                        mask_patch = reshape_image(mask_patch, reshape_size)

                    patch_filename = f"{base_name}_anomaly_full.png"
                    mask_out_filename = f"{base_name}_anomaly_full_mask.png"
                    cv2.imwrite(os.path.join(output_dir, patch_filename), cv2.cvtColor(img_patch, cv2.COLOR_RGB2BGR))
                    cv2.imwrite(os.path.join(output_dir, mask_out_filename), mask_patch)
                    processed_count += 1

        # Process each contour
        for i, contour in enumerate(contours):
            x, y, w, h = cv2.boundingRect(contour)

            x_min = max(0, x - padding)
            y_min = max(0, y - padding)
            x_max = min(img_np.shape[1], x + w + padding)
            y_max = min(img_np.shape[0], y + h + padding)

            if (y_max - y_min) < min_area or (x_max - x_min) < min_area:
                continue

            img_patch = img_np[y_min:y_max, x_min:x_max]
            mask_patch = mask_uint8[y_min:y_max, x_min:x_max]

            if reshape_size:
                img_patch = reshape_image(img_patch, reshape_size)
                mask_patch = reshape_image(mask_patch, reshape_size)

            patch_filename = f"{base_name}.png"
            mask_out_filename = f"{base_name}{mask_suffix}.png"
            cv2.imwrite(os.path.join(output_dir, patch_filename), cv2.cvtColor(img_patch, cv2.COLOR_RGB2BGR))
            cv2.imwrite(os.path.join(output_dir, mask_out_filename), mask_patch)
            processed_count += 1

    if missing_logged > 0:
        print(f"\n[SUMMARY] Missing files logged: {min(missing_logged, max_missing_logs)}/{missing_logged} (capped at {max_missing_logs})")

    print(f"Extracted and saved {processed_count} anomalous patches to {output_dir}")
    return processed_count

def analyze_results(output_dir):
    extracted_images = [f for f in os.listdir(output_dir) if not f.endswith('_GT.png')]
    if not extracted_images:
        print("No extracted patches found for analysis.")
        return

    sizes = []
    total_pixels = 0

    for img_name in extracted_images:
        img_path = os.path.join(output_dir, img_name)
        img = cv2.imread(img_path)
        if img is not None:
            h, w = img.shape[:2]
            sizes.append((h, w))
            total_pixels += h * w

    if sizes:
        avg_size = total_pixels / len(sizes)
        min_h = min(h for h, w in sizes)
        min_w = min(w for h, w in sizes)
        max_h = max(h for h, w in sizes)
        max_w = max(w for h, w in sizes)

        print("\nPatch Statistics:")
        print(f"Total patches: {len(sizes)}")
        print(f"Average patch area: {avg_size:.1f} pixels")
        print(f"Size range: ({min_h}×{min_w}) to ({max_h}×{max_w})")

        unique_sizes = set(sizes)
        if len(unique_sizes) == 1:
            print(f"All patches have identical dimensions: {min_h}×{min_w} (reshaped)")

def list_category_dirs(preprocessed_dir: str, train_subdir: str):
    cats = []
    for name in os.listdir(preprocessed_dir):
        p = os.path.join(preprocessed_dir, name)
        if not os.path.isdir(p):
            continue
        train_path = os.path.join(p, train_subdir)
        if os.path.isdir(train_path):
            cats.append(p)
    return sorted(cats)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract and save anomalous patches from images using their masks."
    )

    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--train_dir", type=str, help="(Old mode) Directory containing the images and masks.")
    mode.add_argument("--preprocessed_dir", type=str, help="(New mode) Root directory containing category folders.")

    parser.add_argument("--output_dir", type=str, default=None,
                        help="(Old mode only) Output directory. Ignored in multi-category mode.")
    parser.add_argument("--defect_crops_dirname", type=str, default="defect_crops",
                        help="(New mode) Folder name created inside each category (default: defect_crops).")

    parser.add_argument("--csv_file", type=str, default=None,
                        help="(Old mode) Path to CSV file. (New mode) CSV filename or relative path inside each category, e.g. 'train.csv'. If not found, category is processed without CSV filtering.")
    parser.add_argument("--train_subdir", type=str, default="train",
                        help="(New mode) Train folder name inside each category (default: train).")

    parser.add_argument("--padding", type=int, default=1)
    parser.add_argument("--min_area", type=int, default=1)
    parser.add_argument("--reshape", type=str, default=None)
    parser.add_argument("--analyze", action="store_true")

    # NEW: logging controls
    parser.add_argument("--verbose", action="store_true",
                        help="Print detailed per-file path resolution and loading logs.")
    parser.add_argument("--max_missing_logs", type=int, default=50,
                        help="Cap how many missing-file entries are printed.")

    args = parser.parse_args()

    reshape_size = None
    if args.reshape:
        try:
            h, w = map(int, args.reshape.split(','))
            reshape_size = (h, w)
            print(f"- Reshaping output to: {h}×{w} pixels")
        except ValueError:
            print(f"Error: Invalid reshape format '{args.reshape}'. Expected: 'HEIGHT,WIDTH'")
            exit(1)

    # OLD MODE
    if args.train_dir:
        if not args.output_dir:
            print("Error: --output_dir is required when using --train_dir")
            exit(1)

        print("Starting anomaly patch extraction (single folder):")
        print(f"- Source directory: {args.train_dir}")
        print(f"- Output directory: {args.output_dir}")
        print(f"- CSV file: {args.csv_file if args.csv_file else 'None (processing all images)'}")

        count = extract_anomalous_crops(
            train_dir=args.train_dir,
            output_dir=args.output_dir,
            csv_file=args.csv_file,
            padding=args.padding,
            min_area=args.min_area,
            reshape_size=reshape_size,
            verbose=args.verbose,
            max_missing_logs=args.max_missing_logs
        )

        if args.analyze and count > 0:
            analyze_results(args.output_dir)

    # NEW MODE
    else:
        pre_dir = args.preprocessed_dir
        categories = list_category_dirs(pre_dir, args.train_subdir)
        if not categories:
            print(f"No category folders found in {pre_dir} containing '{args.train_subdir}/'")
            exit(1)

        print("Starting anomaly patch extraction (multi-category):")
        print(f"- Preprocessed root: {pre_dir}")
        print(f"- Categories found: {len(categories)}")
        print(f"- Train subdir: {args.train_subdir}")
        print(f"- Output dirname (per category): {args.defect_crops_dirname}")
        print(f"- CSV: {args.csv_file if args.csv_file else 'None'}")

        grand_total = 0
        for cat_dir in categories:
            cat_name = os.path.basename(cat_dir)
            train_dir = os.path.join(cat_dir, args.train_subdir)
            out_dir = os.path.join(cat_dir, args.defect_crops_dirname)

            csv_path = None
            if args.csv_file:
                candidate = os.path.join(cat_dir, args.csv_file)
                csv_path = candidate if os.path.exists(candidate) else None

            print(f"\nCategory: {cat_name}")
            print(f"- Train: {train_dir}")
            print(f"- Output: {out_dir}")
            print(f"- CSV: {csv_path if csv_path else 'None'}")

            count = extract_anomalous_crops(
                train_dir=train_dir,
                output_dir=out_dir,
                csv_file=csv_path,
                padding=args.padding,
                min_area=args.min_area,
                reshape_size=reshape_size,
                cat_dir=cat_dir,  # important for resolving CSV paths
                verbose=args.verbose,
                max_missing_logs=args.max_missing_logs
            )
            grand_total += count

            if args.analyze and count > 0:
                analyze_results(out_dir)

        print(f"\nDone. Total patches across all categories: {grand_total}")
