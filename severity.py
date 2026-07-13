# =============================================================================
# AgriVision-XAI  |  severity.py
#
# Estimates disease severity from a leaf image using:
#
#   Method A — Grad-CAM-based (default, no extra model needed):
#       Uses the Grad-CAM heatmap from the classifier to estimate which
#       fraction of the leaf area is "activated" (diseased).
#       Activated pixel fraction → severity percentage → severity label.
#
#   Method B — HSV colour segmentation (classical baseline, no DL needed):
#       Segments diseased pixels by their colour deviation from healthy
#       green leaf tissue in HSV colour space.
#       Useful as a simple sanity-check / comparison baseline.
#
#   Method C — U-Net segmentation (future upgrade, stub provided):
#       Slot for a dedicated segmentation model if you want to train one.
#       Not implemented in the core V2 — flagged as future scope.
#
# Outputs per image (saved to config.SEVERITY_DIR):
#   {sample}_{disease}_severity.png   ← side-by-side: original | mask | result
#   severity_results.json             ← all scores in one file
#
# Usage:
#   python severity.py
#   python severity.py --method hsv --samples 20
# =============================================================================

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

import config
import dataset as ds_module
import model as model_module  # registers CBAMBlock before load_model
from explainability import gradcam


# ---------------------------------------------------------------------------
# LEAF ISOLATION (common pre-step for both methods)
# ---------------------------------------------------------------------------

def _isolate_leaf(image: np.ndarray) -> np.ndarray:
    """
    Creates a binary mask of the leaf region by:
      1. Converting to HSV.
      2. Thresholding on Saturation and Value channels to exclude white/grey
         backgrounds (common in PlantVillage images).
      3. Applying morphological cleanup.

    Returns:
        leaf_mask : (H, W) uint8 binary mask  (255 = leaf, 0 = background)
    """
    hsv = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_RGB2HSV)
    S = hsv[:, :, 1]
    V = hsv[:, :, 2]

    # Keep pixels that are saturated enough (not grey/white) and not too dark
    leaf_mask = cv2.inRange(S, 30, 255) & cv2.inRange(V, 40, 240)

    # Morphological cleanup: remove small noise, fill holes
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    leaf_mask = cv2.morphologyEx(
        leaf_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    leaf_mask = cv2.morphologyEx(
        leaf_mask, cv2.MORPH_OPEN,  kernel, iterations=1)

    return leaf_mask


# ---------------------------------------------------------------------------
# METHOD A — GRAD-CAM-BASED SEVERITY ESTIMATION
# ---------------------------------------------------------------------------

def severity_from_gradcam(
    model: tf.keras.Model,
    image: np.ndarray,
    pred_class_index: int,
    activation_threshold: float = 0.40,
) -> tuple[float, np.ndarray]:
    """
    Estimates the fraction of the leaf area that shows disease activation
    using the Grad-CAM heatmap from the classifier.

    Args:
        model               : trained classifier
        image               : (H, W, 3) float32 in [0, 255]
        pred_class_index    : predicted disease class index
        activation_threshold: Grad-CAM heatmap values above this are
                              considered "activated / diseased"

    Returns:
        severity_pct   : float in [0, 1] — fraction of leaf area that is activated
        disease_mask   : (H, W) uint8 binary mask of the diseased region
    """
    heatmap = gradcam(
        model, image, class_index=pred_class_index, method="gradcam++")
    leaf_mask = _isolate_leaf(image)

    # Binarise the heatmap: high activation → disease region
    disease_mask = ((heatmap >= activation_threshold) &
                    (leaf_mask > 0)).astype(np.uint8) * 255

    # Morphological cleanup on the disease mask
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    disease_mask = cv2.morphologyEx(
        disease_mask, cv2.MORPH_CLOSE, k, iterations=2)

    # Severity = (diseased pixels) / (total leaf pixels)
    leaf_pixels = np.sum(leaf_mask > 0)
    disease_pixels = np.sum(disease_mask > 0)
    severity_pct = float(
        disease_pixels / leaf_pixels) if leaf_pixels > 0 else 0.0

    return severity_pct, disease_mask


# ---------------------------------------------------------------------------
# METHOD B — HSV COLOUR SEGMENTATION BASELINE
# ---------------------------------------------------------------------------

def severity_from_hsv(image: np.ndarray) -> tuple[float, np.ndarray]:
    """
    Estimates disease severity by identifying pixels that deviate from
    healthy green leaf colour in HSV space.

    Healthy leaf: Hue ≈ 60–160°  (green range in OpenCV 0–180 scale: 30–80)
    Diseased pixels: yellow/brown/black lesions fall outside this range.

    Args:
        image : (H, W, 3) float32 RGB in [0, 255]

    Returns:
        severity_pct  : float in [0, 1]
        disease_mask  : (H, W) uint8 binary mask
    """
    img_uint8 = image.astype(np.uint8)
    hsv = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2HSV)
    H, S, V = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]

    # Leaf region (exclude background)
    leaf_mask = _isolate_leaf(image)

    # Healthy green: H in [25, 80], S > 40, V > 40  (OpenCV: H is 0-180)
    healthy_mask = (
        (H >= 25) & (H <= 80) &
        (S >= 40) &
        (V >= 40)
    ).astype(np.uint8) * 255

    # Diseased = leaf pixels that are NOT healthy green
    disease_mask = (
        (leaf_mask > 0) & (healthy_mask == 0)
    ).astype(np.uint8) * 255

    # Morphological cleanup
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    disease_mask = cv2.morphologyEx(
        disease_mask, cv2.MORPH_CLOSE, k, iterations=2)
    disease_mask = cv2.morphologyEx(
        disease_mask, cv2.MORPH_OPEN,  k, iterations=1)

    leaf_pixels = np.sum(leaf_mask > 0)
    disease_pixels = np.sum(disease_mask > 0)
    severity_pct = float(
        disease_pixels / leaf_pixels) if leaf_pixels > 0 else 0.0

    return severity_pct, disease_mask


# ---------------------------------------------------------------------------
# SEVERITY LABEL FROM PERCENTAGE
# ---------------------------------------------------------------------------

def severity_label(severity_pct: float, is_healthy_class: bool = False) -> str:
    """
    Maps a severity percentage to a categorical label using config thresholds.

    Args:
        severity_pct      : float in [0, 1]
        is_healthy_class  : if the classifier predicted a healthy class,
                            override and return "Healthy" regardless.
    """
    if is_healthy_class:
        return "Healthy"

    for label, (lo, hi) in config.SEVERITY_THRESHOLDS.items():
        if lo <= severity_pct < hi:
            return label

    return "Severe"  # fallback for >= 1.0


# ---------------------------------------------------------------------------
# VISUALISATION
# ---------------------------------------------------------------------------

def _visualise_severity(
    image: np.ndarray,
    disease_mask: np.ndarray,
    pred_name: str,
    severity_pct: float,
    label: str,
    method: str,
    sample_idx: int,
    save_dir: Path,
) -> None:
    """Saves a 3-panel figure: original | disease mask | overlay with label."""

    # Overlay: colour the disease mask red on the original image
    overlay = image.astype(np.uint8).copy()
    overlay[disease_mask > 0] = [255, 80, 80]

    # Colour the label box by severity
    label_colours = {
        "Healthy": "#4CAF50",
        "Mild": "#FFC107",
        "Moderate": "#FF9800",
        "Severe": "#F44336",
    }
    lc = label_colours.get(label, "#9E9E9E")

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    titles = ["Original", "Disease Mask", f"Overlay — {method.upper()}"]

    axes[0].imshow(image.astype(np.uint8))
    axes[0].set_title("Original Leaf")
    axes[0].axis("off")

    axes[1].imshow(disease_mask, cmap="hot")
    axes[1].set_title("Disease Region Mask")
    axes[1].axis("off")

    axes[2].imshow(overlay)
    axes[2].set_title(f"Severity Overlay\n{severity_pct*100:.1f}% infected")
    axes[2].axis("off")

    # Severity label banner
    fig.text(
        0.5, 0.01,
        f"Class: {pred_name}  |  Severity: {label}  ({severity_pct*100:.1f}% of leaf area)",
        ha="center", fontsize=11, fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.4", facecolor=lc,
                  alpha=0.85, edgecolor="none"),
        color="white" if label in ("Moderate", "Severe") else "black"
    )
    fig.suptitle(
        f"Sample {sample_idx:03d} — AgriVision-XAI Disease Severity Estimation", fontsize=12)
    plt.tight_layout(rect=[0, 0.06, 1, 1])

    safe_name = pred_name.replace("/", "_").replace(" ", "_")[:40]
    out_path = save_dir / f"{sample_idx:03d}_{safe_name}_severity.png"
    plt.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[severity] Saved to {out_path}")


# ---------------------------------------------------------------------------
# MAIN PIPELINE
# ---------------------------------------------------------------------------

def run_severity(
    model_path: str | None = None,
    method: str = "gradcam",
    num_samples: int = 20,
) -> None:
    print("=" * 65)
    print("  AgriVision-XAI | severity.py")
    print("=" * 65)
    print(f"[severity] Method: {method.upper()}")

    config.SEVERITY_DIR.mkdir(parents=True, exist_ok=True)

    path = Path(model_path) if model_path else config.CHECKPOINTS_DIR / \
        "best_model.keras"
    print(f"[severity] Loading model from {path} ...")
    model = tf.keras.models.load_model(str(path))

    test_ds, class_names = ds_module.get_test_dataset(batch_size=1)

    results = []
    sample_counter = 0
    step = max(1, sum(1 for _ in test_ds) // num_samples)

    for i, (image_batch, label_batch) in enumerate(test_ds):
        if i % step != 0:
            continue
        if sample_counter >= num_samples:
            break

        image = image_batch[0].numpy()
        true_idx = int(np.argmax(label_batch[0].numpy()))

        # Get model prediction
        probs = model(tf.cast(image[np.newaxis],
                      tf.float32), training=False).numpy()[0]
        pred_idx = int(np.argmax(probs))
        pred_name = class_names[pred_idx]
        confidence = float(probs[pred_idx])

        is_healthy = pred_idx in config.HEALTHY_CLASS_INDICES

        # Compute severity
        if is_healthy:
            severity_pct = 0.0
            disease_mask = np.zeros(image.shape[:2], dtype=np.uint8)
        elif method == "gradcam":
            severity_pct, disease_mask = severity_from_gradcam(
                model, image, pred_idx)
        elif method == "hsv":
            severity_pct, disease_mask = severity_from_hsv(image)
        else:
            raise ValueError(
                f"Unknown severity method: '{method}'. Use 'gradcam' or 'hsv'.")

        label = severity_label(severity_pct, is_healthy_class=is_healthy)

        print(f"\n[severity] Sample {sample_counter:03d}: {pred_name}")
        print(f"           Confidence  : {confidence:.3f}")
        print(f"           Infected    : {severity_pct*100:.1f}%")
        print(f"           Severity    : {label}")

        _visualise_severity(
            image, disease_mask, pred_name,
            severity_pct, label, method,
            sample_counter, config.SEVERITY_DIR
        )

        results.append({
            "sample_idx": sample_counter,
            "true_class": class_names[true_idx],
            "pred_class": pred_name,
            "confidence": round(confidence, 4),
            "severity_pct": round(severity_pct * 100, 2),
            "severity_label": label,
            "correct": pred_idx == true_idx,
        })
        sample_counter += 1

    # Save results JSON
    out_json = config.SEVERITY_DIR / "severity_results.json"
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[severity] Results saved to {out_json}")

    # Summary statistics
    pcts = [r["severity_pct"]
            for r in results if not r["pred_class"].endswith("healthy")]
    if pcts:
        print(f"\n[severity] Summary (diseased samples only):")
        print(f"  Mean infected area : {np.mean(pcts):.1f}%")
        print(f"  Max infected area  : {np.max(pcts):.1f}%")
        label_dist = {}
        for r in results:
            lb = r["severity_label"]
            label_dist[lb] = label_dist.get(lb, 0) + 1
        print(f"\n  Severity distribution:")
        for lb, cnt in sorted(label_dist.items()):
            print(f"    {lb:<10} : {cnt}")


# ---------------------------------------------------------------------------
# ENTRY POINT
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="AgriVision-XAI Severity Estimation")
    parser.add_argument("--model",   type=str,  default=None)
    parser.add_argument("--method",  type=str,
                        default="gradcam", choices=["gradcam", "hsv"])
    parser.add_argument("--samples", type=int,  default=20)
    args = parser.parse_args()

    run_severity(
        model_path=args.model,
        method=args.method,
        num_samples=args.samples,
    )
