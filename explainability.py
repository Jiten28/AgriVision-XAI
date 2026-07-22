# =============================================================================
# AgriVision-XAI  |  explainability.py
#
# Generates visual explanations for model predictions using three methods:
#
#   1. Grad-CAM      — heatmap of which spatial regions contributed most
#                      to the prediction. Fast; works per-class.
#   2. Grad-CAM++    — improved version; better localisation when multiple
#                      instances of a disease pattern are present.
#   3. LIME          — model-agnostic; explains predictions by perturbing
#                      superpixels in the input image. Slower but intuitive.
#
# Outputs (saved to config.GRADCAM_DIR):
#   {class_name}_{img_index}_gradcam.png      ← Grad-CAM overlay
#   {class_name}_{img_index}_gradcam_pp.png   ← Grad-CAM++ overlay
#   {class_name}_{img_index}_lime.png         ← LIME superpixel explanation
#
# Usage:
#   python explainability.py
#   python explainability.py --model checkpoints/best_model.keras --samples 10
# =============================================================================

import argparse
from pathlib import Path

import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as mpl_cm
import tensorflow as tf

import config
import dataset as ds_module
import model as model_module  # registers CBAMBlock before load_model


# ---------------------------------------------------------------------------
# GRAD-CAM and GRAD-CAM++ IMPLEMENTATION
# ---------------------------------------------------------------------------
# Both methods work by:
#   1. Creating a sub-model that outputs (a) the last conv feature maps
#      and (b) the final class scores — simultaneously.
#   2. Computing gradients of the target class score w.r.t. those feature maps.
#   3. Weighting the feature maps by those gradients (different weighting
#      rules for standard Grad-CAM vs Grad-CAM++).
#   4. Taking a ReLU of the weighted sum → only positive influences shown.
#   5. Upsampling the resulting heatmap to the input image size.

def _get_gradcam_model(model: tf.keras.Model) -> tf.keras.Model:
    """
    Builds a grad model that outputs (last_conv_feature_maps, predictions).

    In Keras 3, the backbone is a nested sub-model.  We cannot directly wire
    an internal sublayer's output into the outer model's graph.  Instead we:
      1. Build a backbone sub-model: input → last conv layer output
      2. Build the full grad model:  input → [backbone_conv_out, full_pred]
    Both sub-models share the same input tensor so gradients flow correctly.
    """
    from model import BACKBONE_LAYER_NAME

    # --- Step 1: find the backbone sub-model inside the main model ---
    backbone_layer_name = BACKBONE_LAYER_NAME.get(
        config.BACKBONE, config.BACKBONE.lower()
    )
    backbone = model.get_layer(backbone_layer_name)

    # --- Step 2: find the last Conv layer INSIDE the backbone ---
    last_conv_layer = None
    for layer in reversed(backbone.layers):
        if isinstance(layer, (tf.keras.layers.Conv2D,
                              tf.keras.layers.DepthwiseConv2D)):
            last_conv_layer = layer
            break

    if last_conv_layer is None:
        raise ValueError(
            f"Could not find a Conv2D layer in backbone '{backbone_layer_name}'."
        )

    print(f"[explainability] Grad-CAM target layer: '{last_conv_layer.name}'")

    # --- Step 3: build a backbone sub-model that exposes the conv output ---
    backbone_grad = tf.keras.Model(
        inputs=backbone.input,
        outputs=last_conv_layer.output,
        name="backbone_grad"
    )

    # --- Step 4: build the full grad model using the OUTER model's input ---
    img_input = model.input                          # outer model input
    conv_out = backbone_grad(img_input)             # conv feature maps
    predictions = model.output                         # final softmax output

    grad_model = tf.keras.Model(
        inputs=img_input,
        outputs=[conv_out, predictions],
        name="gradcam_model"
    )
    return grad_model


def gradcam(
    model: tf.keras.Model,
    image: np.ndarray,
    class_index: int | None = None,
    method: str = "gradcam",   # "gradcam" or "gradcam++"
) -> np.ndarray:
    """
    Computes a Grad-CAM or Grad-CAM++ heatmap for a single image.

    Args:
        model       : trained Keras model
        image       : (H, W, 3) numpy array, float32 in [0, 255]
        class_index : which class to explain. None = use the predicted class.
        method      : "gradcam" for standard, "gradcam++" for improved version.

    Returns:
        heatmap : (H, W) float32 array in [0, 1] — upsampled to input size
    """
    grad_model = _get_gradcam_model(model)

    # Add batch dimension
    img_tensor = tf.cast(image[np.newaxis, ...], tf.float32)

    with tf.GradientTape() as tape:
        tape.watch(img_tensor)
        conv_outputs, predictions = grad_model(img_tensor, training=False)
        if class_index is None:
            class_index = tf.argmax(predictions[0]).numpy()
        class_score = predictions[:, class_index]

    # Gradient of the class score w.r.t. the conv feature maps
    grads = tape.gradient(class_score, conv_outputs)   # (1, h, w, C)

    # Pool gradients (different rules for Grad-CAM vs Grad-CAM++)
    if method == "gradcam":
        # Standard Grad-CAM: global average pool the gradients
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))   # (C,)
        conv_out = conv_outputs[0]                               # (h, w, C)
        heatmap = conv_out @ pooled_grads[..., tf.newaxis]      # (h, w, 1)

    elif method == "gradcam++":
        # Grad-CAM++: weight gradients by gradient² / (2×gradient² + sum(conv×gradient³))
        grads_sq = grads ** 2
        grads_cu = grads ** 3
        conv_out_exp = conv_outputs  # (1, h, w, C)
        # Per-pixel normalization factor (sum over spatial dimensions)
        denom = 2.0 * grads_sq + tf.reduce_sum(
            conv_out_exp * grads_cu, axis=(1, 2), keepdims=True
        )
        alpha = grads_sq / (denom + 1e-8)
        # Weight by ReLU(gradient) — only positive gradients
        weights = tf.reduce_sum(
            alpha * tf.nn.relu(grads), axis=(1, 2))  # (1, C)
        # (h, w, C)
        conv_out = conv_outputs[0]
        heatmap = conv_out @ weights[0, ...,
                                     tf.newaxis]                  # (h, w, 1)

    else:
        raise ValueError(
            f"Unknown method: '{method}'. Use 'gradcam' or 'gradcam++'.")

    # ReLU: keep only features that positively impact the class score
    heatmap = tf.nn.relu(heatmap)
    heatmap = heatmap.numpy().squeeze()                    # (h, w)

    # Normalise to [0, 1]
    if heatmap.max() > 0:
        heatmap = heatmap / heatmap.max()

    # Upsample to original image size
    heatmap_resized = cv2.resize(heatmap, (image.shape[1], image.shape[0]))

    return heatmap_resized


def overlay_heatmap(
    image: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = 0.45,
    colormap: str = "jet",
) -> np.ndarray:
    """
    Blends the Grad-CAM heatmap over the original image.

    Args:
        image    : (H, W, 3) float32 in [0, 255]
        heatmap  : (H, W) float32 in [0, 1]
        alpha    : heatmap opacity
        colormap : matplotlib colormap name

    Returns:
        blended  : (H, W, 3) uint8 BGR image suitable for cv2.imwrite
    """
    cmap = mpl_cm.get_cmap(colormap)
    heat_rgb = (cmap(heatmap)[:, :, :3] *
                255).astype(np.uint8)  # (H, W, 3) RGB
    heat_bgr = cv2.cvtColor(heat_rgb, cv2.COLOR_RGB2BGR)

    img_bgr = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_RGB2BGR)

    blended = cv2.addWeighted(img_bgr, 1 - alpha, heat_bgr, alpha, 0)
    return blended


# ---------------------------------------------------------------------------
# LIME IMPLEMENTATION
# ---------------------------------------------------------------------------

def lime_explanation(
    model: tf.keras.Model,
    image: np.ndarray,
    num_samples: int = config.LIME_NUM_SAMPLES,
    num_features: int = config.LIME_NUM_FEATURES,
) -> np.ndarray:
    """
    Generates a LIME explanation for a single image using superpixel segmentation.

    How LIME works:
      1. Segment the image into superpixels (coherent regions).
      2. Randomly turn superpixels on/off to create N perturbed images.
      3. Run the model on each perturbed image.
      4. Fit a linear model to predict the class score from superpixel on/off state.
      5. The linear weights tell us which superpixels most influence the prediction.

    Args:
        model        : trained Keras model
        image        : (H, W, 3) float32 in [0, 255]
        num_samples  : number of perturbations (more = more accurate, slower)
        num_features : number of top superpixels to highlight in the output

    Returns:
        explanation_overlay : (H, W, 3) uint8 image showing influential superpixels
    """
    try:
        from skimage.segmentation import slic
        from skimage.color import label2rgb
    except ImportError:
        raise ImportError(
            "scikit-image is required for LIME. Install it with:\n"
            "    pip install scikit-image"
        )

    img_uint8 = image.astype(np.uint8)

    # Step 1: Superpixel segmentation (SLIC algorithm)
    segments = slic(img_uint8, n_segments=50,
                    compactness=10, sigma=1, start_label=0)
    num_superpixels = segments.max() + 1

    # Step 2: Generate perturbed images (randomly mask superpixels)
    rng = np.random.RandomState(config.RANDOM_SEED)
    perturbations = rng.randint(
        0, 2, (num_samples, num_superpixels)).astype(np.float32)

    def perturb_image(mask_row: np.ndarray) -> np.ndarray:
        """Apply a binary superpixel mask to the image (off = grey)."""
        perturbed = img_uint8.copy().astype(np.float32)
        for sp_idx in range(num_superpixels):
            if mask_row[sp_idx] == 0:
                perturbed[segments == sp_idx] = 128.0  # grey out
        return perturbed

    print(f"[explainability] LIME: generating {num_samples} perturbations ...")
    perturbed_imgs = np.array([
        perturb_image(perturbations[i]) for i in range(num_samples)
    ])  # (num_samples, H, W, 3)

    # Step 3: Get model predictions on perturbed images
    # Process in batches of 64 to avoid OOM
    batch_sz = 64
    scores = []
    pred_class = int(tf.argmax(
        model(tf.cast(image[np.newaxis], tf.float32), training=False)[0]).numpy())

    for i in range(0, num_samples, batch_sz):
        batch = tf.cast(perturbed_imgs[i:i + batch_sz], tf.float32)
        # (batch, NUM_CLASSES)
        preds = model(batch, training=False).numpy()
        scores.extend(preds[:, pred_class])

    scores = np.array(scores, dtype=np.float32)   # (num_samples,)

    # Step 4: Fit a weighted linear model
    # Distance weight: samples closer to the original image get more weight
    distances = np.sum((perturbations - 1.0) ** 2, axis=1) ** 0.5
    kernel_width = 0.25
    weights = np.exp(-distances ** 2 / (2 * kernel_width ** 2))

    # Weighted least-squares: X'WX β = X'Wy
    W = np.diag(weights)
    XtW = perturbations.T @ W
    try:
        beta = np.linalg.solve(XtW @ perturbations + 1e-5 * np.eye(num_superpixels),
                               XtW @ scores)
    except np.linalg.LinAlgError:
        beta = np.zeros(num_superpixels)

    # Step 5: Build explanation overlay
    # Highlight the top-k superpixels with positive contributions
    top_indices = np.argsort(beta)[-num_features:]
    mask = np.zeros_like(segments, dtype=bool)
    for idx in top_indices:
        mask[segments == idx] = True

    # Green overlay on supporting regions, darkened background
    overlay = img_uint8.copy().astype(np.float32)
    overlay[~mask] = overlay[~mask] * \
        0.3                        # dim background
    overlay[mask, 1] = np.minimum(overlay[mask, 1] * 1.5, 255)  # green tint

    return overlay.astype(np.uint8)


# ---------------------------------------------------------------------------
# VISUALISATION PIPELINE
# ---------------------------------------------------------------------------

def explain_sample(
    model: tf.keras.Model,
    image: np.ndarray,
    true_label: int,
    class_names: list[str],
    sample_idx: int,
    save_dir: Path,
    run_lime: bool = True,
) -> None:
    """
    Generates and saves Grad-CAM, Grad-CAM++, and (optionally) LIME for
    a single image.
    """
    img_f32 = image.astype(np.float32)
    probs = model(tf.cast(img_f32[np.newaxis],
                  tf.float32), training=False).numpy()[0]
    pred_idx = int(np.argmax(probs))
    conf = float(probs[pred_idx])

    pred_name = class_names[pred_idx]
    true_name = class_names[true_label]
    correct = "✓" if pred_idx == true_label else "✗"

    print(f"\n[explainability] Sample {sample_idx:03d}: "
          f"True={true_name}  |  Pred={pred_name}  |  Conf={conf:.3f} {correct}")

    # --- Grad-CAM ---
    heatmap_gc = gradcam(
        model, img_f32, class_index=pred_idx, method="gradcam")
    overlay_gc = overlay_heatmap(img_f32, heatmap_gc)

    # --- Grad-CAM++ ---
    heatmap_gcpp = gradcam(
        model, img_f32, class_index=pred_idx, method="gradcam++")
    overlay_gcpp = overlay_heatmap(img_f32, heatmap_gcpp)

    # --- LIME ---
    if run_lime:
        lime_overlay = lime_explanation(model, img_f32)

    # --- Plot and save ---
    n_cols = 4 if run_lime else 3
    fig, axes = plt.subplots(1, n_cols, figsize=(5 * n_cols, 5))
    suptitle = (f"Sample {sample_idx:03d} | True: {true_name} | "
                f"Pred: {pred_name} | Conf: {conf:.3f} {correct}")
    fig.suptitle(suptitle, fontsize=9, fontweight="bold")

    axes[0].imshow(img_f32.astype(np.uint8))
    axes[0].set_title("Original")
    axes[0].axis("off")

    axes[1].imshow(cv2.cvtColor(overlay_gc, cv2.COLOR_BGR2RGB))
    axes[1].set_title("Grad-CAM")
    axes[1].axis("off")

    axes[2].imshow(cv2.cvtColor(overlay_gcpp, cv2.COLOR_BGR2RGB))
    axes[2].set_title("Grad-CAM++")
    axes[2].axis("off")

    if run_lime:
        axes[3].imshow(lime_overlay)
        axes[3].set_title("LIME")
        axes[3].axis("off")

    plt.tight_layout()
    safe_name = pred_name.replace("/", "_").replace(" ", "_")[:40]
    out_path = save_dir / f"{sample_idx:03d}_{safe_name}_explanation.png"
    plt.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[explainability] Saved explanation to {out_path}")


def run_explainability(
    model_path: str | None = None,
    num_samples: int = config.NUM_GRADCAM_SAMPLES,
    run_lime: bool = True,
) -> None:
    """
    Main function: loads model + test set, runs Grad-CAM / Grad-CAM++ / LIME
    on a sample of test images and saves all visualisations.
    """
    print("=" * 65)
    print("  AgriVision-XAI | explainability.py")
    print("=" * 65)

    config.GRADCAM_DIR.mkdir(parents=True, exist_ok=True)

    # Load model
    path = Path(model_path) if model_path else config.CHECKPOINTS_DIR / \
        "best_model.keras"
    print(f"[explainability] Loading model from {path} ...")
    model = tf.keras.models.load_model(str(path))

    # Load test set (no batching needed — we process image by image)
    test_ds, class_names = ds_module.get_test_dataset(batch_size=1)

    # Sample `num_samples` images spread across the test set
    sample_counter = 0
    step = max(1, sum(1 for _ in test_ds) // num_samples)

    for i, (image_batch, label_batch) in enumerate(test_ds):
        if i % step != 0:
            continue
        if sample_counter >= num_samples:
            break

        image = image_batch[0].numpy()        # (H, W, 3)
        label = int(np.argmax(label_batch[0].numpy()))

        explain_sample(
            model=model,
            image=image,
            true_label=label,
            class_names=class_names,
            sample_idx=sample_counter,
            save_dir=config.GRADCAM_DIR,
            run_lime=run_lime,
        )
        sample_counter += 1

    print(
        f"\n[explainability] Done. {sample_counter} explanations saved to {config.GRADCAM_DIR}")


# ---------------------------------------------------------------------------
# ENTRY POINT
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="AgriVision-XAI Explainability")
    parser.add_argument("--model",   type=str, default=None)
    parser.add_argument("--samples", type=int,
                        default=config.NUM_GRADCAM_SAMPLES)
    parser.add_argument("--no-lime", action="store_true",
                        help="Skip LIME (faster)")
    args = parser.parse_args()

    run_explainability(
        model_path=args.model,
        num_samples=args.samples,
        run_lime=not args.no_lime,
    )
