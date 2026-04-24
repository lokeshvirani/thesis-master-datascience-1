import argparse
import os
import torch
import numpy as np
from torch.utils.data import DataLoader

from patchcore import PatchCoreDual, PatchCoreSingle
from data.mvtec import MVTec
from data.mvtec_crops import MVTecCrops


def list_category_dirs(preprocessed_dir: str):
    """
    Accepts either:
    1) a root directory containing category folders with train/ and test/
    2) a single category directory that itself contains train/ and test/

    Returns a list of category directories.
    """

    # Case 1: preprocessed_dir itself is already one category
    if (
        os.path.isdir(os.path.join(preprocessed_dir, "train")) and
        os.path.isdir(os.path.join(preprocessed_dir, "test"))
    ):
        return [preprocessed_dir]

    # Case 2: preprocessed_dir contains multiple category folders
    cats = []
    for name in os.listdir(preprocessed_dir):
        p = os.path.join(preprocessed_dir, name)
        if not os.path.isdir(p):
            continue
        if (
            os.path.isdir(os.path.join(p, "train")) and
            os.path.isdir(os.path.join(p, "test"))
        ):
            cats.append(p)

    return sorted(cats)


def run_one_category(args, category_name: str, dataset_path: str, crops_path: str, augmented_path: str, out_dir: str, device: str):
    # Determinism (same behavior, per category)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    os.makedirs(out_dir, exist_ok=True)

    print("\n==============================")
    print(f"Category       : {category_name}")
    print(f"dataset_path   : {dataset_path}")
    print(f"crops_path     : {crops_path}")
    print(f"add_augmented  : {args.add_augmented}")
    print(f"augmented_path : {augmented_path}")
    print(f"output_dir     : {out_dir}")
    print(f"device         : {device}")
    print(f"backbone       : {args.backbone}")
    print(f"batch_size     : {args.batch_size}  (NOTE: loaders still use batch_size=1 as before)")
    print(f"neg_sub        : {args.neg_subsampling}")
    print(f"pos_sub        : {args.pos_subsampling}")
    print(f"seed           : {args.seed}")
    print("==============================")

    # Normal dataset (defect-free)
    normal_dataset = MVTec(dataroot=dataset_path, split="train", negative_only=True)

    # Defect crops dataset
    anomalous_dataset = MVTecCrops(
        crop_root=crops_path,
        add_augmented=args.add_augmented,
        augmented_crop_root=augmented_path
    )

    # Test dataset
    test_dataset = MVTec(dataroot=dataset_path, split="test")

    print(f"Normal samples: {len(normal_dataset)}")
    print(f"Anomalous samples (Defect Cropped): {len(anomalous_dataset)}")
    print(f"Test samples: {len(test_dataset)}")

    # KEEP SAME FUNCTIONALITY AS YOUR CURRENT SCRIPT:
    # you define --batch_size but you used batch_size=1 in loaders. Not changing that.
    normal_loader = DataLoader(normal_dataset, batch_size=1, shuffle=True, num_workers=0)
    positive_loader = DataLoader(anomalous_dataset, batch_size=1, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)

    results = {}

    # # Experiment 1: Standard PatchCore (negative only)
    # print("\n===== Experiment 1: Standard PatchCore (Negative Only) =====")
    # negative_model = PatchCoreSingle(
    #     device=device,
    #     backbone=args.backbone,
    #     memory_type='negative',
    #     subsampling_share=args.neg_subsampling
    # )
    # negative_model.fit(normal_loader)

    # neg_image_auc, neg_pixel_auc = negative_model.evaluate_single(test_loader)

    # print(f"Standard PatchCore Results:")
    # print(f"Image-level AUROC: {neg_image_auc:.4f}")
    # print(f"Pixel-level AUROC: {neg_pixel_auc:.4f}")

    # results["standard_patchcore"] = {
    #     "image_auc": neg_image_auc,
    #     "pixel_auc": neg_pixel_auc
    # }

    # Experiment Dual PatchCore with pre-cropped defects
    print("\n===== Dual PatchCore with Pre-Cropped Defects =====")
    dual_model = PatchCoreDual(
        device=device,
        backbone=args.backbone,
        negative_subsampling=args.neg_subsampling,
        positive_subsampling=args.pos_subsampling
    )

    dual_model.fit(negative_dataloader=normal_loader, positive_dataloader=positive_loader)
    dual_image_auc, dual_pixel_auc = dual_model.evaluate(test_loader)

    print("Dual PatchCore Results:")
    print(f"Image-level AUROC: {dual_image_auc:.4f}")
    print(f"Pixel-level AUROC: {dual_pixel_auc:.4f}")

    results = {
        "dual_patchcore": {
            "image_auc": float(dual_image_auc),
            "pixel_auc": float(dual_pixel_auc),
        }
    }

    out_path = os.path.join(out_dir, "results.npy")
    np.save(out_path, results)
    print(f"Results saved to {out_path}")

    return results


def main():
    parser = argparse.ArgumentParser(description="PatchCore Dual Experiments")

    # OLD params (keep exactly)
    parser.add_argument("--dataset_path", type=str, default=None, help="Path to dataset (old single mode)")
    parser.add_argument("--crops_path", type=str, default=None, help="Path to pre-cropped defects (old single mode)")
    parser.add_argument("--output_dir", type=str, default="./results", help="Directory to save results")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for normal training (kept for compatibility)")
    parser.add_argument("--backbone", type=str, default="resnet50",
                        choices=["resnet50", "wide_resnet50_2"], help="Backbone network")
    parser.add_argument("--add_augmented", default=False, action="store_true", help="Use augmented defects")
    parser.add_argument("--augmented_path", type=str, default=None, help="Path to augmented images (old single mode)")
    parser.add_argument("--neg_subsampling", type=float, default=0.01, help="Negative memory bank subsampling rate")
    parser.add_argument("--pos_subsampling", type=float, default=0.10, help="Positive memory bank subsampling rate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    # NEW param: category-wise mode
    parser.add_argument("--preprocessed_dir", type=str, default=None,
                        help="If set, runs category-wise. This can be the root containing category folders OR a single category folder.")

    # NEW: folder names inside each category
    parser.add_argument("--train_dirname", type=str, default="train")
    parser.add_argument("--test_dirname", type=str, default="test")
    parser.add_argument("--defect_crops_dirname", type=str, default="defect_crops")

    # NEW: augmented folder name inside each category
    parser.add_argument("--augmented_dirname", type=str, default="augmented",
                        help="Folder name inside each category to use as augmented_path when --add_augmented is set.")

    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # CATEGORY-WISE MODE
    if args.preprocessed_dir:
        cats = list_category_dirs(args.preprocessed_dir)
        if not cats:
            raise ValueError(f"No categories found under {args.preprocessed_dir} that contain train/ and test/")

        all_results = {}

        for cat_dir in cats:
            cat_name = os.path.basename(cat_dir)

            dataset_path = cat_dir
            crops_path = os.path.join(cat_dir, args.defect_crops_dirname)

            if not os.path.isdir(os.path.join(cat_dir, args.train_dirname)):
                print(f"\n[SKIP] {cat_name}: missing {args.train_dirname}/ in {cat_dir}")
                continue
            if not os.path.isdir(os.path.join(cat_dir, args.test_dirname)):
                print(f"\n[SKIP] {cat_name}: missing {args.test_dirname}/ in {cat_dir}")
                continue
            if not os.path.isdir(crops_path):
                print(f"\n[SKIP] {cat_name}: missing {args.defect_crops_dirname}/ in {cat_dir}")
                continue

            augmented_path = None
            if args.add_augmented:
                augmented_path = os.path.join(cat_dir, args.augmented_dirname)

            out_dir = os.path.join(args.output_dir, cat_name)

            res = run_one_category(
                args=args,
                category_name=cat_name,
                dataset_path=dataset_path,
                crops_path=crops_path,
                augmented_path=augmented_path,
                out_dir=out_dir,
                device=device
            )
            all_results[cat_name] = res

        os.makedirs(args.output_dir, exist_ok=True)
        combined_path = os.path.join(args.output_dir, "all_results.npy")
        np.save(combined_path, all_results)
        print(f"\nSaved combined results to {combined_path}")
        return

    # OLD SINGLE MODE
    if not args.dataset_path or not args.crops_path:
        raise ValueError("Use either --preprocessed_dir (category-wise) OR (--dataset_path and --crops_path) (single).")

    cat_name = os.path.basename(os.path.normpath(args.dataset_path))

    run_one_category(
        args=args,
        category_name=cat_name,
        dataset_path=args.dataset_path,
        crops_path=args.crops_path,
        augmented_path=args.augmented_path,
        out_dir=args.output_dir,
        device=device
    )


if __name__ == "__main__":
    main()
