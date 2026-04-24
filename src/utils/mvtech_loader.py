import os
import shutil
import numpy as np
from PIL import Image
from tqdm import tqdm

def generate_black_image(width=256, height=256, channels=3, save_path=None):
    """
    Generate a completely black image.

    Parameters:
        width (int): Width of the image in pixels (default 256)
        height (int): Height of the image in pixels (default 256)
        channels (int): Number of color channels (default 3 for RGB)
        save_path (str, optional): Path to save the image

    Returns:
        np.ndarray: Black image as a NumPy array
    """
    black_image = np.zeros((height, width, channels), dtype=np.uint8)
    if save_path:
        img = Image.fromarray(black_image)
        img.save(save_path)
        print(f"[INFO] Black GT image saved: {save_path}")
    return black_image


def process_mvtec_dataset(src_root, dst_root, image_exts=(".png", ".jpg", ".jpeg")):
    """
    Process MVTec dataset: copies images and corresponding ground truth masks to a clean structure.
    
    Naming convention:
        - Copied image: {folder_name}{original_filename}.{ext}
        - Ground truth: {folder_name}{original_filename}_GT.{ext}
    
    Additional behavior:
        - Copies one defect category from test into train to add positive samples.
    """
    os.makedirs(dst_root, exist_ok=True)

    categories = [d for d in os.listdir(src_root) if os.path.isdir(os.path.join(src_root, d))]
    # categories = [c for c in categories if c == "cable"]
    print(f"[INFO] Found {len(categories)} categories: {categories}")

    for category in categories:
        cat_src = os.path.join(src_root, category)
        cat_dst = os.path.join(dst_root, category)

        os.makedirs(os.path.join(cat_dst, "train"), exist_ok=True)
        os.makedirs(os.path.join(cat_dst, "test"), exist_ok=True)

        print(f"\n[INFO] Processing category: {category}")

        # ==========================
        # Process TRAIN images
        # ==========================
        train_src = os.path.join(cat_src, "train")
        print(f"[INFO] Processing TRAIN folder: {train_src}")
        
        for cls in os.listdir(train_src):
            cls_path = os.path.join(train_src, cls)
            if not os.path.isdir(cls_path):
                continue

            print(f"[INFO] Processing class '{cls}' in TRAIN...")
            for fname in tqdm(os.listdir(cls_path), desc=f"TRAIN/{cls}", unit="img"):
                if fname.lower().endswith(image_exts):
                    src_path = os.path.join(cls_path, fname)
                    base_name, ext = os.path.splitext(fname)
                    
                    dst_path = os.path.join(cat_dst, "train", f"{cls}{fname}")
                    shutil.copy(src_path, dst_path)
                    print(f"  Copied TRAIN image: {dst_path}")

                    if cls.lower() == "good":
                        gt_name = f"{cls}{base_name}_GT{ext}"
                        gt_path = os.path.join(cat_dst, "train", gt_name)
                        generate_black_image(save_path=gt_path)

        # ==========================
        # Process TEST images
        # ==========================
        test_src = os.path.join(cat_src, "test")
        gt_src = os.path.join(cat_src, "ground_truth")
        print(f"[INFO] Processing TEST folder: {test_src}")

        defect_categories = [d for d in os.listdir(test_src) if os.path.isdir(os.path.join(test_src, d))]

        for defect_type in defect_categories:
            defect_path = os.path.join(test_src, defect_type)

            print(f"[INFO] Processing defect type '{defect_type}' in TEST...")
            for fname in tqdm(os.listdir(defect_path), desc=f"TEST/{defect_type}", unit="img"):
                if not fname.lower().endswith(image_exts):
                    continue

                src_path = os.path.join(defect_path, fname)
                base_name, ext = os.path.splitext(fname)
                
                dst_img_path = os.path.join(cat_dst, "test", f"{defect_type}{fname}")
                shutil.copy(src_path, dst_img_path)
                print(f"  Copied TEST image: {dst_img_path}")

                if defect_type.lower() == "good":
                    gt_name = f"{defect_type}{base_name}_GT{ext}"
                    gt_path = os.path.join(cat_dst, "test", gt_name)
                    generate_black_image(save_path=gt_path)
                else:
                    gt_folder = os.path.join(gt_src, defect_type)
                    if os.path.exists(gt_folder):
                        for gt_file in os.listdir(gt_folder):
                            if base_name in gt_file and gt_file.lower().endswith((".png", ".jpg", ".jpeg")):
                                src_gt_path = os.path.join(gt_folder, gt_file)
                                dst_gt_name = f"{defect_type}{base_name}_GT{ext}"
                                dst_gt_path = os.path.join(cat_dst, "test", dst_gt_name)
                                shutil.copy(src_gt_path, dst_gt_path)
                                print(f"  Copied GT mask: {dst_gt_path}")
                                break

        # ==========================
        # Add one defect category to TRAIN
        # ==========================
        defect_for_train = None
        for d in defect_categories:
            if d.lower() != "good":
                defect_for_train = d
                break

        if defect_for_train:
            print(f"[INFO] Adding defect category '{defect_for_train}' to TRAIN for positives...")
            defect_path = os.path.join(test_src, defect_for_train)
            gt_folder = os.path.join(gt_src, defect_for_train)

            for fname in tqdm(os.listdir(defect_path), desc=f"TRAIN_ADD/{defect_for_train}", unit="img"):
                if not fname.lower().endswith(image_exts):
                    continue

                src_path = os.path.join(defect_path, fname)
                base_name, ext = os.path.splitext(fname)

                dst_path = os.path.join(cat_dst, "train", f"{defect_for_train}{fname}")
                shutil.copy(src_path, dst_path)
                print(f"  Added TRAIN image: {dst_path}")

                if os.path.exists(gt_folder):
                    for gt_file in os.listdir(gt_folder):
                        if base_name in gt_file and gt_file.lower().endswith((".png", ".jpg", ".jpeg")):
                            src_gt_path = os.path.join(gt_folder, gt_file)
                            dst_gt_name = f"{defect_for_train}{base_name}_GT{ext}"
                            dst_gt_path = os.path.join(cat_dst, "train", dst_gt_name)
                            shutil.copy(src_gt_path, dst_gt_path)
                            print(f"  Added TRAIN GT mask: {dst_gt_path}")
                            break

        print(f"[INFO] Finished processing category: {category}")


if __name__ == "__main__":
    src_root = "/media/data/Lokesh/thesis-master-datascience/data/mvtec_ad_2"
    dst_root = "/media/data/Lokesh/thesis-master-datascience/data/mvtec_preprocessed_ad_2"

    print("[INFO] Starting MVTec dataset preprocessing...")
    process_mvtec_dataset(src_root, dst_root)
    print("[INFO] Dataset preprocessing completed!")

