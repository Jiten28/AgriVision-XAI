# =============================================================================
# AgriVision-XAI  |  dataset.py
#
# What this file does:
#   1. Reads all images from the V1 train/ + valid/ folders (combined pool).
#   2. Performs a stratified 70 / 15 / 15 re-split so each class has exactly
#      the right proportion in every subset — fixing the V1 val=test overlap.
#   3. Copies (hard-links) the images into:
#         dataset_split/train/  dataset_split/val/  dataset_split/test/
#   4. Builds tf.data pipelines with augmentation for training and
#      clean pipelines for val/test.
#
# Run once:
#   python dataset.py
# Then train.py uses the loaders directly without re-splitting.
# =============================================================================

import os
import shutil
import random
from pathlib import Path
from collections import defaultdict

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

import config


# ---------------------------------------------------------------------------
# PART 1 — RE-SPLIT THE DATASET
# ---------------------------------------------------------------------------

def _collect_image_paths(dataset_root: Path) -> dict[str, list[Path]]:
    """
    Walk train/ and valid/ inside dataset_root.
    Return a dict:  { class_name: [path1, path2, ...], ... }
    All classes found in train/ are also expected in valid/.
    """
    images_by_class: dict[str, list[Path]] = defaultdict(list)
    VALID_EXTS = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}

    for split_folder in ["train", "valid"]:
        split_path = dataset_root / split_folder
        if not split_path.exists():
            raise FileNotFoundError(
                f"Expected folder not found: {split_path}\n"
                f"Check V1_DATASET_ROOT in config.py"
            )
        for class_dir in sorted(split_path.iterdir()):
            if not class_dir.is_dir():
                continue
            class_name = class_dir.name
            for img_path in class_dir.iterdir():
                if img_path.suffix in VALID_EXTS:
                    images_by_class[class_name].append(img_path)

    return dict(images_by_class)


def _copy_images(paths: list[Path], dest_class_dir: Path) -> None:
    """Copy a list of image paths into dest_class_dir."""
    dest_class_dir.mkdir(parents=True, exist_ok=True)
    for src in paths:
        dst = dest_class_dir / src.name
        # Avoid collisions when the same filename appears in train/ and valid/
        if dst.exists():
            stem, suffix = src.stem, src.suffix
            dst = dest_class_dir / f"{stem}_{src.parent.parent.name}{suffix}"
        shutil.copy2(src, dst)


def build_split(force: bool = False) -> None:
    """
    Creates dataset_split/{train,val,test}/<class>/ from the V1 dataset.

    Args:
        force: if True, delete and recreate the split even if it already exists.
               If False (default), skip if split already exists.
    """
    split_dir = config.SPLIT_DIR

    if split_dir.exists() and not force:
        # Count how many images are already there
        total = sum(1 for _ in split_dir.rglob("*.jpg")) + \
                sum(1 for _ in split_dir.rglob("*.JPG")) + \
                sum(1 for _ in split_dir.rglob("*.jpeg")) + \
                sum(1 for _ in split_dir.rglob("*.png"))
        if total > 0:
            print(f"[dataset] Split already exists at {split_dir}  ({total:,} images).")
            print("[dataset] Pass force=True to rebuild it.")
            return

    if split_dir.exists():
        print(f"[dataset] Removing existing split at {split_dir} ...")
        shutil.rmtree(split_dir)

    print(f"[dataset] Collecting images from {config.V1_DATASET_ROOT} ...")
    images_by_class = _collect_image_paths(config.V1_DATASET_ROOT)

    # Verify all expected classes are present
    found_classes   = set(images_by_class.keys())
    expected_classes = set(config.CLASS_NAMES)
    missing = expected_classes - found_classes
    extra   = found_classes - expected_classes
    if missing:
        print(f"[dataset] WARNING: {len(missing)} expected classes not found: {missing}")
    if extra:
        print(f"[dataset] WARNING: {len(extra)} unexpected classes found: {extra}")

    train_count = val_count = test_count = 0

    print("[dataset] Performing stratified 70/15/15 split ...")
    for class_name, all_paths in images_by_class.items():
        random.seed(config.RANDOM_SEED)
        random.shuffle(all_paths)

        # First cut: 85% (train+val) vs 15% (test)
        train_val, test_paths = train_test_split(
            all_paths,
            test_size=config.TEST_RATIO,
            random_state=config.RANDOM_SEED
        )
        # Second cut: split the 85% into 70% train and 15% val
        # 15% of total = 15/85 of the train+val portion
        val_fraction_of_trainval = config.VAL_RATIO / (config.TRAIN_RATIO + config.VAL_RATIO)
        train_paths, val_paths = train_test_split(
            train_val,
            test_size=val_fraction_of_trainval,
            random_state=config.RANDOM_SEED
        )

        _copy_images(train_paths, config.TRAIN_DIR / class_name)
        _copy_images(val_paths,   config.VAL_DIR   / class_name)
        _copy_images(test_paths,  config.TEST_DIR  / class_name)

        train_count += len(train_paths)
        val_count   += len(val_paths)
        test_count  += len(test_paths)

    total = train_count + val_count + test_count
    print(f"\n[dataset] Split complete:")
    print(f"  Total images : {total:>7,}")
    print(f"  Train        : {train_count:>7,}  ({train_count/total*100:.1f}%)")
    print(f"  Validation   : {val_count:>7,}  ({val_count/total*100:.1f}%)")
    print(f"  Test         : {test_count:>7,}  ({test_count/total*100:.1f}%)")
    print(f"  Classes      : {len(images_by_class)}")


# ---------------------------------------------------------------------------
# PART 2 — DATA AUGMENTATION PIPELINE  (training only)
# ---------------------------------------------------------------------------

def _augmentation_layer() -> tf.keras.Sequential:
    """
    Returns a Keras Sequential model of random augmentation layers.
    Applied only during training (not at val/test time).

    These augmentations simulate real-world variation:
      - Rotation / shifts / zoom  → different camera angles and distances
      - Horizontal flip           → mirror-image leaves (valid for disease patterns)
      - Brightness / contrast     → different lighting conditions (outdoor, cloudy, etc.)
    """
    return tf.keras.Sequential([
        tf.keras.layers.RandomRotation(
            factor=config.AUG_ROTATION / 360.0,
            fill_mode=config.AUG_FILL_MODE
        ),
        tf.keras.layers.RandomTranslation(
            height_factor=config.AUG_HEIGHT_SHIFT,
            width_factor=config.AUG_WIDTH_SHIFT,
            fill_mode=config.AUG_FILL_MODE
        ),
        tf.keras.layers.RandomZoom(
            height_factor=config.AUG_ZOOM,
            fill_mode=config.AUG_FILL_MODE
        ),
        tf.keras.layers.RandomFlip("horizontal"),   # horizontal only — see config
        tf.keras.layers.RandomBrightness(
            factor=(config.AUG_BRIGHTNESS[0] - 1.0, config.AUG_BRIGHTNESS[1] - 1.0)
        ),
        tf.keras.layers.RandomContrast(factor=0.2),
    ], name="augmentation")


# ---------------------------------------------------------------------------
# PART 3 — tf.data LOADERS
# ---------------------------------------------------------------------------

def _build_dataset(
    directory: Path,
    batch_size: int,
    augment: bool,
    shuffle: bool,
    class_names: list[str] | None = None,
) -> tuple[tf.data.Dataset, list[str]]:
    """
    Builds a tf.data.Dataset from a directory of class sub-folders.

    Returns:
        (dataset, class_names)
        dataset yields (image_batch, label_batch) where:
          - images are float32 in [0, 1]  (normalisation happens inside EfficientNet)
          - labels are one-hot encoded vectors of length NUM_CLASSES
    """
    raw_ds = tf.keras.utils.image_dataset_from_directory(
        directory=str(directory),
        labels="inferred",
        label_mode="categorical",     # one-hot encoded
        class_names=class_names,      # enforces a fixed ordering
        color_mode="rgb",
        batch_size=batch_size,
        image_size=config.IMAGE_SIZE,
        shuffle=shuffle,
        seed=config.RANDOM_SEED if shuffle else None,
        interpolation="bilinear",
        crop_to_aspect_ratio=False,   # we resize, not crop, to keep full leaf
    )
    found_classes = raw_ds.class_names

    # ---- Preprocessing ----
    # EfficientNetB0 (and all EfficientNet variants) includes its own
    # tf.keras.applications.efficientnet.preprocess_input internally,
    # so we only need to cast to float32; the model itself handles scaling.
    def preprocess(image, label):
        image = tf.cast(image, tf.float32)
        return image, label

    # ---- Augmentation (training only) ----
    augmenter = _augmentation_layer()

    def preprocess_and_augment(image, label):
        image = tf.cast(image, tf.float32)
        image = augmenter(image, training=True)
        return image, label

    if augment:
        ds = raw_ds.map(preprocess_and_augment, num_parallel_calls=tf.data.AUTOTUNE)
    else:
        ds = raw_ds.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)

    # ---- Performance tuning ----
    ds = ds.cache()          # cache after first epoch (fits in RAM for PlantVillage)
    ds = ds.prefetch(tf.data.AUTOTUNE)

    return ds, found_classes


def get_train_dataset(
    batch_size: int = config.BATCH_SIZE,
) -> tuple[tf.data.Dataset, list[str]]:
    """Training dataset: shuffled + augmented."""
    return _build_dataset(
        directory=config.TRAIN_DIR,
        batch_size=batch_size,
        augment=True,
        shuffle=True,
        class_names=config.CLASS_NAMES,
    )


def get_val_dataset(
    batch_size: int = config.BATCH_SIZE,
) -> tuple[tf.data.Dataset, list[str]]:
    """Validation dataset: no augmentation, no shuffle."""
    return _build_dataset(
        directory=config.VAL_DIR,
        batch_size=batch_size,
        augment=False,
        shuffle=False,
        class_names=config.CLASS_NAMES,
    )


def get_test_dataset(
    batch_size: int = config.BATCH_SIZE,
    directory: Path | None = None,
    class_names: list[str] | None = None,
) -> tuple[tf.data.Dataset, list[str]]:
    """
    Test dataset: no augmentation, no shuffle, no caching (used exactly once).

    Args:
        directory: defaults to config.TEST_DIR (in-domain held-out test set).
                   Pass config.PLANTDOC_DIR for cross-domain evaluation.
        class_names: if None, uses config.CLASS_NAMES.
                     For PlantDoc, pass its own class list.
    """
    directory   = directory   or config.TEST_DIR
    class_names = class_names or config.CLASS_NAMES

    raw_ds = tf.keras.utils.image_dataset_from_directory(
        directory=str(directory),
        labels="inferred",
        label_mode="categorical",
        class_names=class_names,
        color_mode="rgb",
        batch_size=batch_size,
        image_size=config.IMAGE_SIZE,
        shuffle=False,
        interpolation="bilinear",
        crop_to_aspect_ratio=False,
    )
    found_classes = raw_ds.class_names

    def preprocess(image, label):
        image = tf.cast(image, tf.float32)
        return image, label

    ds = raw_ds.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.prefetch(tf.data.AUTOTUNE)

    return ds, found_classes


# ---------------------------------------------------------------------------
# PART 4 — CLASS WEIGHT COMPUTATION  (handles class imbalance)
# ---------------------------------------------------------------------------

def compute_class_weights(train_dataset: tf.data.Dataset) -> dict[int, float]:
    """
    Computes per-class weights to give more importance to under-represented
    classes during training. Passed to model.fit(class_weight=...).

    Formula: weight[i] = total_samples / (num_classes * count[i])
    This is the sklearn balanced strategy.
    """
    class_counts = np.zeros(config.NUM_CLASSES, dtype=np.int64)

    for _, labels in train_dataset:
        # labels shape: (batch_size, num_classes) — one-hot encoded
        indices = np.argmax(labels.numpy(), axis=1)
        for idx in indices:
            class_counts[idx] += 1

    total = class_counts.sum()
    weights = {}
    for i, count in enumerate(class_counts):
        if count > 0:
            weights[i] = total / (config.NUM_CLASSES * count)
        else:
            weights[i] = 1.0  # class not in training set — no penalty

    print("\n[dataset] Class weight stats:")
    print(f"  Min weight : {min(weights.values()):.4f}")
    print(f"  Max weight : {max(weights.values()):.4f}")
    print(f"  Mean weight: {np.mean(list(weights.values())):.4f}")

    return weights


# ---------------------------------------------------------------------------
# PART 5 — DATASET SUMMARY  (diagnostic print)
# ---------------------------------------------------------------------------

def print_dataset_summary(train_ds, val_ds, test_ds) -> None:
    """Print image and batch counts for quick sanity check."""
    def count_images(ds):
        return sum(b[0].shape[0] for b in ds)

    n_train = count_images(train_ds)
    n_val   = count_images(val_ds)
    n_test  = count_images(test_ds)
    total   = n_train + n_val + n_test

    print("\n[dataset] Summary:")
    print(f"  Train      : {n_train:>7,} images")
    print(f"  Validation : {n_val:>7,} images")
    print(f"  Test       : {n_test:>7,} images")
    print(f"  Total      : {total:>7,} images")
    print(f"  Classes    : {config.NUM_CLASSES}")
    print(f"  Image size : {config.IMAGE_SIZE}")
    print(f"  Batch size : {config.BATCH_SIZE}")


# ---------------------------------------------------------------------------
# STANDALONE: run this file directly to build + verify the split
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 60)
    print("  AgriVision-XAI | dataset.py")
    print("=" * 60)

    # Step A: build the split (skips if already done)
    build_split(force=False)

    # Step B: load the three datasets to verify they work
    print("\n[dataset] Loading datasets for verification ...")
    train_ds, train_classes = get_train_dataset()
    val_ds,   val_classes   = get_val_dataset()
    test_ds,  test_classes  = get_test_dataset()

    # Verify class ordering is consistent across all three splits
    assert train_classes == val_classes == test_classes, (
        "Class ordering mismatch between splits! "
        "This should never happen — check class_names in config.py."
    )
    print(f"[dataset] Class ordering consistent across all splits. ({len(train_classes)} classes)")

    # Print summary
    print_dataset_summary(train_ds, val_ds, test_ds)

    # Step C: compute and display class weights
    print("\n[dataset] Computing class weights ...")
    class_weights = compute_class_weights(train_ds)

    # Show the 5 most and least weighted classes
    sorted_weights = sorted(class_weights.items(), key=lambda x: x[1], reverse=True)
    print("\n  Top 5 most weighted (most underrepresented) classes:")
    for idx, w in sorted_weights[:5]:
        print(f"    [{idx:02d}] {config.CLASS_NAMES[idx]:<55} weight={w:.4f}")
    print("\n  Bottom 5 least weighted (most overrepresented) classes:")
    for idx, w in sorted_weights[-5:]:
        print(f"    [{idx:02d}] {config.CLASS_NAMES[idx]:<55} weight={w:.4f}")

    # Step D: inspect one batch
    print("\n[dataset] Inspecting one training batch ...")
    for images, labels in train_ds.take(1):
        print(f"  Image batch shape : {images.shape}")   # (32, 224, 224, 3)
        print(f"  Label batch shape : {labels.shape}")   # (32, 38)
        print(f"  Image dtype       : {images.dtype}")
        print(f"  Image min/max     : {images.numpy().min():.2f} / {images.numpy().max():.2f}")
        print(f"  Classes in batch  : {np.unique(np.argmax(labels.numpy(), axis=1))}")

    print("\n[dataset] All checks passed. Ready for training.")
