import shutil
from pathlib import Path
import random

def organize_pet_dataset(images_dir, masks_dir, output_dir):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    images_path = Path(images_dir)
    image_files = list(images_path.glob('*.jpg'))

    print(f"Found {len(image_files)} pictures")

    processed = 0
    skipped = 0

    for image_file in image_files:
        image_name = image_file.stem

        mask_path = Path(masks_dir) / f"{image_name}.png"
        mask_file = None

        if mask_path.exists():
            mask_file = mask_path

        if mask_file is None:
            print(f"Mask not found: {image_name}")
            skipped += 1
            continue

        pair_folder = output_path / image_name
        pair_folder.mkdir(exist_ok=True)

        shutil.copy2(image_file, pair_folder / f"image{image_file.suffix}")
        shutil.copy2(mask_file, pair_folder / f"mask{mask_file.suffix}")

        processed += 1

    print(f"\nTotal processed pairs: {processed}")
    print(f"Skipped pairs: {skipped}")


def split_train_test(organized_dir):
    train_ratio = 0.7
    val_ratio = 0.15

    organized_path = Path(organized_dir)
    all_folders = [f for f in organized_path.iterdir() if f.is_dir()]

    random.seed(42)
    random.shuffle(all_folders)

    total_samples = len(all_folders)
    train_idx = int(total_samples * train_ratio)
    val_idx = train_idx + int(total_samples * val_ratio)

    train_dir = organized_path.parent / f"Train"
    val_dir = organized_path.parent / f"Validation"
    test_dir = organized_path.parent / f"Test"

    train_dir.mkdir(exist_ok=True)
    val_dir.mkdir(exist_ok=True)
    test_dir.mkdir(exist_ok=True)

    for folder in all_folders[:train_idx]:
        shutil.copytree(folder, train_dir / folder.name)

    for folder in all_folders[train_idx:val_idx]:
        shutil.copytree(folder, val_dir / folder.name)

    for folder in all_folders[val_idx:]:
        shutil.copytree(folder, test_dir / folder.name)

if __name__ == "__main__":
    IMAGES_DIR = "Data/images"
    MASKS_DIR = "Data/masks"
    OUTPUT_DIR = "Organized_Data"

    print("Pet Dataset organisation\n")

    organize_pet_dataset(IMAGES_DIR, MASKS_DIR, OUTPUT_DIR)
    split_train_test(OUTPUT_DIR)
