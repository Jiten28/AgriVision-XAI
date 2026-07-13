# =============================================================================
# AgriVision-XAI  |  evaluate.py
#
# Runs the final, definitive evaluation on the held-out test sets.
# This file should be run EXACTLY ONCE after training — not during
# hyperparameter search (use the val set for that).
#
# Generates:
#   outputs/confusion_matrix_indomain.png   ← in-domain (PlantVillage split)
#   outputs/confusion_matrix_crossdomain.png← cross-domain (PlantDoc, if set)
#   outputs/classification_report.txt       ← per-class precision/recall/F1
#   outputs/evaluation_summary.json         ← all numbers in one JSON file
#
# Usage:
#   python evaluate.py
#   python evaluate.py --model checkpoints/best_model.keras  (to pick a specific checkpoint)
# =============================================================================

import json
import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    top_k_accuracy_score,
)

import config
import dataset as ds_module
import model as model_module  # registers CBAMBlock before load_model


# ---------------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------------

def _load_model(model_path: str | None) -> tf.keras.Model:
    """Load the trained model. Defaults to best_model.keras if no path given."""
    path = Path(model_path) if model_path else config.CHECKPOINTS_DIR / \
        "best_model.keras"
    if not path.exists():
        raise FileNotFoundError(
            f"Model not found at {path}\n"
            f"Run train.py first, or pass --model <path> explicitly."
        )
    print(f"[evaluate] Loading model from {path} ...")
    model = tf.keras.models.load_model(str(path))
    print(f"[evaluate] Model loaded.")
    return model


def _predict(model: tf.keras.Model, dataset: tf.data.Dataset) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Run inference over a dataset and return:
        y_true     : (N,)   integer class indices
        y_pred     : (N,)   predicted class indices (argmax of softmax)
        y_probs    : (N, C) softmax probabilities for all classes
    """
    all_true = []
    all_probs = []

    for images, labels in dataset:
        probs = model(images, training=False).numpy()  # (batch, C)
        true = np.argmax(labels.numpy(), axis=1)      # (batch,)
        all_true.append(true)
        all_probs.append(probs)

    y_true = np.concatenate(all_true,  axis=0)
    y_probs = np.concatenate(all_probs, axis=0)
    y_pred = np.argmax(y_probs, axis=1)

    return y_true, y_pred, y_probs


def _plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list[str],
    title: str,
    save_path: Path,
) -> None:
    """
    Plots and saves a confusion matrix heatmap.
    Uses shortened class names (e.g. "Tomato_EarlyBlight") for readability.
    """
    cm = confusion_matrix(y_true, y_pred)

    # Normalise to percentages per row (true class)
    cm_norm = cm.astype("float") / (cm.sum(axis=1, keepdims=True) + 1e-9)

    # Shorten class names for the axis labels
    short_names = [
        name.replace("(including_sour)___", "___")
            .replace(",_bell", "")
            .replace("_(", "_")
            .replace(")", "")
            .replace(" ", "_")
        for name in class_names
    ]

    n = len(class_names)
    fig_size = max(16, n * 0.55)
    fig, ax = plt.subplots(figsize=(fig_size, fig_size))

    sns.heatmap(
        cm_norm,
        annot=(n <= 38),    # show numbers only if the matrix is not too large
        fmt=".2f",
        cmap="Blues",
        xticklabels=short_names,
        yticklabels=short_names,
        linewidths=0.3,
        linecolor="gray",
        ax=ax,
        cbar_kws={"shrink": 0.6},
    )

    ax.set_title(title, fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("Predicted Class", fontsize=11)
    ax.set_ylabel("True Class",      fontsize=11)
    ax.tick_params(axis="x", rotation=90, labelsize=7)
    ax.tick_params(axis="y", rotation=0,  labelsize=7)

    plt.tight_layout()
    plt.savefig(str(save_path), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[evaluate] Confusion matrix saved to {save_path}")


def _per_class_stats(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list[str],
) -> dict:
    """
    Returns per-class precision, recall, f1, support as a dict.
    Also prints the worst-5 and best-5 classes by F1.
    """
    report_dict = classification_report(
        y_true, y_pred,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )

    # Sort by F1 score
    class_f1 = [
        (name, report_dict[name]["f1-score"])
        for name in class_names
        if name in report_dict
    ]
    class_f1.sort(key=lambda x: x[1])

    print("\n  5 Worst classes by F1 score:")
    for name, f1 in class_f1[:5]:
        print(f"    {name:<55} F1={f1:.4f}")

    print("\n  5 Best classes by F1 score:")
    for name, f1 in class_f1[-5:]:
        print(f"    {name:<55} F1={f1:.4f}")

    return report_dict


# ---------------------------------------------------------------------------
# MAIN EVALUATION FUNCTION
# ---------------------------------------------------------------------------

def evaluate(model_path: str | None = None) -> None:
    print("=" * 65)
    print("  AgriVision-XAI | evaluate.py")
    print("=" * 65)

    config.OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    model = _load_model(model_path)
    summary = {}  # will be serialised to JSON

    # ------------------------------------------------------------------
    # A. IN-DOMAIN TEST SET  (PlantVillage 15% held-out)
    # ------------------------------------------------------------------
    print(f"\n{'─' * 65}")
    print("  A. In-Domain Evaluation (PlantVillage held-out test set)")
    print(f"{'─' * 65}")

    if not config.TEST_DIR.exists():
        print(f"[evaluate] ERROR: Test directory not found: {config.TEST_DIR}")
        print("[evaluate] Run dataset.py first to create the split.")
        return

    test_ds, class_names = ds_module.get_test_dataset(
        batch_size=config.BATCH_SIZE)

    print("[evaluate] Running inference on test set ...")
    y_true, y_pred, y_probs = _predict(model, test_ds)

    # --- Core metrics ---
    acc = accuracy_score(y_true, y_pred)
    top3acc = top_k_accuracy_score(
        y_true, y_probs, k=3, labels=np.arange(config.NUM_CLASSES))
    report_str = classification_report(
        y_true, y_pred,
        target_names=class_names,
        zero_division=0
    )

    print(f"\n  In-Domain Test Accuracy   : {acc:.4f}  ({acc*100:.2f}%)")
    print(f"  In-Domain Top-3 Accuracy  : {top3acc:.4f}  ({top3acc*100:.2f}%)")
    print(f"\n  Classification Report:\n{report_str}")

    # --- Confusion Matrix ---
    _plot_confusion_matrix(
        y_true, y_pred, class_names,
        title="AgriVision-XAI — In-Domain Confusion Matrix (PlantVillage Test Set)",
        save_path=config.OUTPUTS_DIR / "confusion_matrix_indomain.png",
    )

    # --- Per-class deep dive ---
    report_dict = _per_class_stats(y_true, y_pred, class_names)

    # Save classification report as text
    report_path = config.OUTPUTS_DIR / "classification_report.txt"
    with open(report_path, "w") as f:
        f.write("AgriVision-XAI — Classification Report (In-Domain Test Set)\n")
        f.write("=" * 70 + "\n")
        f.write(report_str)
    print(f"\n[evaluate] Classification report saved to {report_path}")

    summary["in_domain"] = {
        "accuracy":      round(float(acc), 4),
        "top3_accuracy": round(float(top3acc), 4),
        "num_samples":   int(len(y_true)),
        "num_classes":   int(config.NUM_CLASSES),
        "per_class":     report_dict,
    }

    # ------------------------------------------------------------------
    # B. CROSS-DOMAIN TEST SET  (PlantDoc — if configured)
    # ------------------------------------------------------------------
    if config.PLANTDOC_DIR and Path(config.PLANTDOC_DIR).exists():
        print(f"\n{'─' * 65}")
        print("  B. Cross-Domain Evaluation (PlantDoc field-condition images)")
        print(f"{'─' * 65}")
        print("[evaluate] NOTE: PlantDoc uses different class names from PlantVillage.")
        print("[evaluate] Only overlapping classes will be evaluated.")

        # PlantDoc has its own class structure — load without enforcing class_names
        plantdoc_ds_raw = tf.keras.utils.image_dataset_from_directory(
            directory=str(config.PLANTDOC_DIR),
            labels="inferred",
            label_mode="categorical",
            color_mode="rgb",
            batch_size=config.BATCH_SIZE,
            image_size=config.IMAGE_SIZE,
            shuffle=False,
        )
        plantdoc_classes = plantdoc_ds_raw.class_names
        print(f"[evaluate] PlantDoc classes found: {len(plantdoc_classes)}")

        # Find overlapping classes with PlantVillage
        pv_set = set(config.CLASS_NAMES)
        pd_set = set(plantdoc_classes)
        overlap = sorted(pv_set & pd_set)
        print(f"[evaluate] Overlapping classes: {len(overlap)}")

        if len(overlap) == 0:
            print(
                "[evaluate] No overlapping class names found between PlantVillage and PlantDoc.")
            print(
                "[evaluate] This is expected — PlantDoc uses a different naming convention.")
            print(
                "[evaluate] Skipping cross-domain evaluation. You'll need to manually map class names.")
            summary["cross_domain"] = {
                "status": "skipped", "reason": "no_class_overlap"}
        else:
            def preprocess_pd(img, lbl):
                return tf.cast(img, tf.float32), lbl

            plantdoc_ds = plantdoc_ds_raw.map(
                preprocess_pd, num_parallel_calls=tf.data.AUTOTUNE
            ).prefetch(tf.data.AUTOTUNE)

            print("[evaluate] Running inference on PlantDoc ...")
            y_true_pd, y_pred_pd, y_probs_pd = _predict(model, plantdoc_ds)

            acc_pd = accuracy_score(y_true_pd, y_pred_pd)
            print(
                f"\n  Cross-Domain Accuracy (PlantDoc) : {acc_pd:.4f}  ({acc_pd*100:.2f}%)")
            print(
                f"  (Compare to In-Domain: {acc*100:.2f}% — gap = {(acc - acc_pd)*100:.2f}%)")

            _plot_confusion_matrix(
                y_true_pd, y_pred_pd, plantdoc_classes,
                title="AgriVision-XAI — Cross-Domain Confusion Matrix (PlantDoc)",
                save_path=config.OUTPUTS_DIR / "confusion_matrix_crossdomain.png",
            )

            summary["cross_domain"] = {
                "accuracy":    round(float(acc_pd), 4),
                "num_samples": int(len(y_true_pd)),
                "generalization_gap": round(float(acc - acc_pd), 4),
            }
    else:
        print("\n[evaluate] PlantDoc directory not configured or not found.")
        print("[evaluate] Set PLANTDOC_DIR in config.py to run cross-domain evaluation.")
        summary["cross_domain"] = {
            "status": "skipped", "reason": "dataset_not_configured"}

    # ------------------------------------------------------------------
    # C. Save summary
    # ------------------------------------------------------------------
    summary_path = config.OUTPUTS_DIR / "evaluation_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[evaluate] Full evaluation summary saved to {summary_path}")

    # ------------------------------------------------------------------
    # D. Final print
    # ------------------------------------------------------------------
    print(f"\n{'=' * 65}")
    print("  EVALUATION COMPLETE")
    print(f"{'=' * 65}")
    print(f"  In-Domain Test Accuracy  : {acc*100:.2f}%")
    if "accuracy" in summary.get("cross_domain", {}):
        print(
            f"  Cross-Domain Accuracy    : {summary['cross_domain']['accuracy']*100:.2f}%")
        print(
            f"  Generalisation Gap       : {summary['cross_domain']['generalization_gap']*100:.2f}%")
    print(f"\n  All outputs saved to: {config.OUTPUTS_DIR}")
    print(f"  Run explainability.py next for Grad-CAM visualisations.")


# ---------------------------------------------------------------------------
# ENTRY POINT
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AgriVision-XAI Evaluation")
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Path to .keras model file. Defaults to checkpoints/best_model.keras"
    )
    args = parser.parse_args()
    evaluate(model_path=args.model)
