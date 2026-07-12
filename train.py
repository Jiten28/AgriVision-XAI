# =============================================================================
# AgriVision-XAI  |  train.py
#
# Two-phase training strategy:
#
#   Phase 1 — "Head Training" (EPOCHS_FROZEN epochs)
#       Backbone is frozen (ImageNet weights preserved).
#       Only the CBAM block and classification head are trained.
#       Higher learning rate (LR_FROZEN = 1e-3) is safe here because
#       only randomly initialised layers receive gradients.
#
#   Phase 2 — "Fine-Tuning" (EPOCHS_FINETUNE epochs, or until early stop)
#       Top FINETUNE_LAYERS of the backbone are unfrozen.
#       Very low learning rate (LR_FINETUNE = 1e-5) to avoid destroying
#       the pretrained representations ("catastrophic forgetting").
#
# Usage:
#   python train.py
#
# Outputs saved to config.CHECKPOINTS_DIR:
#   best_model.keras          ← best val_accuracy checkpoint
#   final_model.keras         ← weights after all epochs
#   training_history.json     ← full loss/accuracy curves for both phases
#   training_history.png      ← plotted curves
# =============================================================================

import json
import time
from pathlib import Path

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

import config
import dataset as ds_module
import model as model_module


# ---------------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------------

def _make_dirs() -> None:
    """Create all output directories if they don't exist."""
    for d in [config.CHECKPOINTS_DIR, config.LOGS_DIR, config.OUTPUTS_DIR,
              config.GRADCAM_DIR, config.SEVERITY_DIR, config.EXPORT_DIR]:
        d.mkdir(parents=True, exist_ok=True)


def _build_callbacks(phase: int) -> list:
    """
    Returns the callback list for a given training phase.

    Phase 1 callbacks:
      - ModelCheckpoint  : saves the best model by val_accuracy
      - ReduceLROnPlateau: reduces LR if val_loss stagnates
      - TensorBoard      : logs for TensorBoard visualisation

    Phase 2 callbacks add:
      - EarlyStopping    : stops training if val_loss doesn't improve
                           (not used in Phase 1 because 10 epochs is short)
    """
    checkpoint_path = str(config.CHECKPOINTS_DIR / "best_model.keras")

    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path,
        monitor="val_accuracy",
        save_best_only=True,
        save_weights_only=False,
        mode="max",
        verbose=1,
    )

    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=config.LR_REDUCE_FACTOR,
        patience=config.LR_REDUCE_PATIENCE,
        min_lr=config.LR_REDUCE_MIN,
        verbose=1,
    )

    tensorboard = tf.keras.callbacks.TensorBoard(
        log_dir=str(config.LOGS_DIR / f"phase{phase}"),
        histogram_freq=1,
        update_freq="epoch",
    )

    callbacks = [checkpoint, reduce_lr, tensorboard]

    if phase == 2:
        early_stop = tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=config.EARLY_STOPPING_PATIENCE,
            restore_best_weights=True,  # reverts to best epoch automatically
            verbose=1,
        )
        callbacks.append(early_stop)

    return callbacks


def _compile(model: tf.keras.Model, learning_rate: float) -> None:
    """Compile the model. Called separately for Phase 1 and Phase 2."""
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="categorical_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.TopKCategoricalAccuracy(k=3, name="top3_accuracy"),
        ]
    )


def _plot_history(history_p1: dict, history_p2: dict) -> None:
    """
    Plots combined training curves for both phases side-by-side.
    Saves to config.OUTPUTS_DIR / training_history.png.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("AgriVision-XAI — Training History", fontsize=14, fontweight="bold")

    ep1 = range(1, len(history_p1["accuracy"]) + 1)
    ep2 = range(len(ep1) + 1, len(ep1) + len(history_p2["accuracy"]) + 1)

    colors = {"train": "#2196F3", "val": "#E91E63", "p2_train": "#4CAF50", "p2_val": "#FF9800"}

    for ax, metric, title in zip(
        axes,
        ["accuracy", "loss"],
        ["Accuracy", "Loss"]
    ):
        # Phase 1
        ax.plot(ep1, history_p1[metric],          color=colors["train"],    label="P1 Train",  linewidth=2)
        ax.plot(ep1, history_p1[f"val_{metric}"], color=colors["val"],      label="P1 Val",    linewidth=2, linestyle="--")
        # Phase 2
        ax.plot(ep2, history_p2[metric],          color=colors["p2_train"], label="P2 Train",  linewidth=2)
        ax.plot(ep2, history_p2[f"val_{metric}"], color=colors["p2_val"],   label="P2 Val",    linewidth=2, linestyle="--")
        # Phase boundary line
        ax.axvline(x=len(ep1) + 0.5, color="gray", linestyle=":", linewidth=1.5, label="Fine-tune start")

        ax.set_title(f"{title}", fontsize=12)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(title)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = config.OUTPUTS_DIR / "training_history.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n[train] Training curve saved to {out_path}")


def _save_history(history_p1: dict, history_p2: dict) -> None:
    """Saves combined training history to JSON for later analysis."""
    combined = {
        "phase1": history_p1,
        "phase2": history_p2,
        "config": {
            "backbone":       config.BACKBONE,
            "use_cbam":       config.USE_CBAM,
            "image_size":     list(config.IMAGE_SIZE),
            "batch_size":     config.BATCH_SIZE,
            "epochs_frozen":  config.EPOCHS_FROZEN,
            "epochs_finetune":config.EPOCHS_FINETUNE,
            "lr_frozen":      config.LR_FROZEN,
            "lr_finetune":    config.LR_FINETUNE,
            "random_seed":    config.RANDOM_SEED,
        }
    }
    out_path = config.OUTPUTS_DIR / "training_history.json"
    with open(out_path, "w") as f:
        json.dump(combined, f, indent=2)
    print(f"[train] Training history saved to {out_path}")


# ---------------------------------------------------------------------------
# MAIN TRAINING FUNCTION
# ---------------------------------------------------------------------------

def train() -> tf.keras.Model:
    """
    Full two-phase training pipeline. Returns the best trained model.
    """
    print("=" * 65)
    print("  AgriVision-XAI | train.py")
    print("=" * 65)
    _make_dirs()

    # ------------------------------------------------------------------
    # GPU Check
    # ------------------------------------------------------------------
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        print(f"\n[train] GPU detected: {[g.name for g in gpus]}")
        # Allow GPU memory to grow incrementally (prevents OOM on 8 GB VRAM)
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    else:
        print("[train] WARNING: No GPU detected. Training on CPU will be very slow.")

    # ------------------------------------------------------------------
    # 1. Build datasets
    # ------------------------------------------------------------------
    print("\n[train] Loading datasets ...")
    train_ds, _ = ds_module.get_train_dataset(batch_size=config.BATCH_SIZE)
    val_ds,   _ = ds_module.get_val_dataset(batch_size=config.BATCH_SIZE)
    ds_module.print_dataset_summary(train_ds, val_ds, ds_module.get_test_dataset()[0])

    # Compute class weights to handle any remaining class imbalance
    print("\n[train] Computing class weights ...")
    class_weights = ds_module.compute_class_weights(train_ds)

    # ------------------------------------------------------------------
    # 2. Build model (backbone frozen → Phase 1)
    # ------------------------------------------------------------------
    print(f"\n[train] Building model ({config.BACKBONE} + CBAM={config.USE_CBAM}) ...")
    model = model_module.build_model(freeze_backbone=True)
    model_module.model_summary(model)

    # ------------------------------------------------------------------
    # PHASE 1 — Train the head only
    # ------------------------------------------------------------------
    print(f"\n{'─' * 65}")
    print(f"  PHASE 1: Head Training  ({config.EPOCHS_FROZEN} epochs, LR={config.LR_FROZEN})")
    print(f"{'─' * 65}")

    _compile(model, learning_rate=config.LR_FROZEN)

    t0 = time.time()
    history_p1 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=config.EPOCHS_FROZEN,
        callbacks=_build_callbacks(phase=1),
        class_weight=class_weights,
        verbose=1,
    )
    elapsed_p1 = time.time() - t0
    print(f"\n[train] Phase 1 complete in {elapsed_p1/60:.1f} min.")
    print(f"[train] Best val_accuracy so far: {max(history_p1.history['val_accuracy']):.4f}")

    # ------------------------------------------------------------------
    # PHASE 2 — Unfreeze top layers and fine-tune
    # ------------------------------------------------------------------
    print(f"\n{'─' * 65}")
    print(f"  PHASE 2: Fine-Tuning  (up to {config.EPOCHS_FINETUNE} epochs, LR={config.LR_FINETUNE})")
    print(f"{'─' * 65}")

    model_module.unfreeze_top_layers(model, n_layers=config.FINETUNE_LAYERS)
    _compile(model, learning_rate=config.LR_FINETUNE)

    t0 = time.time()
    history_p2 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=config.EPOCHS_FINETUNE,
        callbacks=_build_callbacks(phase=2),
        class_weight=class_weights,
        verbose=1,
    )
    elapsed_p2 = time.time() - t0
    print(f"\n[train] Phase 2 complete in {elapsed_p2/60:.1f} min.")
    print(f"[train] Best val_accuracy (overall): {max(history_p2.history['val_accuracy']):.4f}")

    # ------------------------------------------------------------------
    # 3. Save final model and history
    # ------------------------------------------------------------------
    final_path = config.CHECKPOINTS_DIR / "final_model.keras"
    model.save(str(final_path))
    print(f"[train] Final model saved to {final_path}")

    _save_history(history_p1.history, history_p2.history)
    _plot_history(history_p1.history, history_p2.history)

    # ------------------------------------------------------------------
    # 4. Quick validation report
    # ------------------------------------------------------------------
    print("\n[train] Running final evaluation on validation set ...")
    val_results = model.evaluate(val_ds, verbose=0)
    metric_names = model.metrics_names
    print("\n  Validation results (final model):")
    for name, val in zip(metric_names, val_results):
        print(f"    {name:<20}: {val:.4f}")

    print("\n[train] Training complete. Run evaluate.py for full test-set metrics.")
    return model


# ---------------------------------------------------------------------------
# ENTRY POINT
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    trained_model = train()
