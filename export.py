# =============================================================================
# AgriVision-XAI  |  export.py
#
# Converts the trained Keras model to deployment formats:
#
#   A. TensorFlow Lite (.tflite)
#      — Float32: same accuracy as the Keras model, smaller file
#      — INT8 Quantised: ~4× smaller, ~2× faster on mobile CPU, tiny accuracy drop
#      Both are benchmarked for latency and file size.
#
#   B. ONNX (.onnx)  [optional — requires tf2onnx]
#      — Cross-platform format for deployment on Windows/Linux/edge devices,
#        and for running in ONNX Runtime (often faster than TF on CPU).
#
# Outputs saved to config.EXPORT_DIR:
#   agrivision_xai_fp32.tflite   ← full-precision TFLite
#   agrivision_xai_int8.tflite   ← INT8 quantised TFLite
#   agrivision_xai.onnx          ← ONNX (if tf2onnx installed)
#   export_report.json           ← file sizes + latency benchmark
#
# Usage:
#   python export.py
#   python export.py --model checkpoints/best_model.keras --no-onnx
# =============================================================================

import argparse
import json
import time
from pathlib import Path

import numpy as np
import tensorflow as tf

import config
import dataset as ds_module


# ---------------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------------

def _load_model(model_path: str | None) -> tf.keras.Model:
    path = Path(model_path) if model_path else config.CHECKPOINTS_DIR / "best_model.keras"
    if not path.exists():
        raise FileNotFoundError(f"Model not found at {path}. Run train.py first.")
    print(f"[export] Loading model from {path} ...")
    return tf.keras.models.load_model(str(path))


def _benchmark_tflite(
    tflite_path: Path,
    num_runs: int = 50,
) -> dict:
    """
    Loads a TFLite model and benchmarks single-image inference latency.
    Returns dict with mean/min/max latency in milliseconds and file size in KB.
    """
    interpreter = tf.lite.Interpreter(model_path=str(tflite_path))
    interpreter.allocate_tensors()

    input_details  = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    input_shape    = input_details[0]["shape"]     # e.g. (1, 224, 224, 3)
    input_dtype    = input_details[0]["dtype"]

    # Create a random input image
    dummy = np.random.rand(*input_shape).astype(np.float32) * 255.0
    if input_dtype == np.uint8:
        dummy = dummy.astype(np.uint8)
    elif input_dtype == np.int8:
        dummy = (dummy - 128.0).astype(np.int8)

    # Warm-up
    for _ in range(5):
        interpreter.set_tensor(input_details[0]["index"], dummy)
        interpreter.invoke()

    # Timed runs
    latencies = []
    for _ in range(num_runs):
        t0 = time.perf_counter()
        interpreter.set_tensor(input_details[0]["index"], dummy)
        interpreter.invoke()
        latencies.append((time.perf_counter() - t0) * 1000)  # ms

    file_size_kb = tflite_path.stat().st_size / 1024

    return {
        "file_size_kb"   : round(file_size_kb, 1),
        "latency_mean_ms": round(np.mean(latencies), 2),
        "latency_min_ms" : round(np.min(latencies),  2),
        "latency_max_ms" : round(np.max(latencies),  2),
    }


# ---------------------------------------------------------------------------
# A. TF LITE EXPORT
# ---------------------------------------------------------------------------

def export_tflite_fp32(model: tf.keras.Model, out_dir: Path) -> Path:
    """
    Converts to TFLite with float32 weights.
    Same numerical accuracy as the original Keras model.
    """
    print("\n[export] Converting to TFLite (float32) ...")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    # Default: float32, no quantisation
    tflite_model = converter.convert()

    out_path = out_dir / "agrivision_xai_fp32.tflite"
    out_path.write_bytes(tflite_model)
    print(f"[export] Saved: {out_path}  ({out_path.stat().st_size / 1024 / 1024:.1f} MB)")
    return out_path


def export_tflite_int8(
    model: tf.keras.Model,
    representative_dataset_fn,
    out_dir: Path,
) -> Path:
    """
    Converts to TFLite with INT8 quantisation via post-training quantisation (PTQ).

    INT8 quantisation:
      - Compresses weights from float32 → int8 (~4× smaller model file).
      - Also quantises activations using a representative dataset for calibration.
      - Typically causes <1% accuracy drop on PlantVillage.
      - Provides ~2× speedup on mobile CPUs and runs on microcontrollers.

    Args:
        representative_dataset_fn: a generator that yields batches of representative
                                   images from the training set — used to calibrate
                                   the activation quantisation thresholds.
    """
    print("\n[export] Converting to TFLite (INT8 quantised) ...")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)

    converter.optimizations            = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset   = representative_dataset_fn
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type     = tf.float32   # keep float input for ease of use
    converter.inference_output_type    = tf.float32   # keep float output

    tflite_model = converter.convert()

    out_path = out_dir / "agrivision_xai_int8.tflite"
    out_path.write_bytes(tflite_model)
    print(f"[export] Saved: {out_path}  ({out_path.stat().st_size / 1024 / 1024:.1f} MB)")
    return out_path


def _representative_dataset_generator():
    """
    Yields 200 representative images from the training set for INT8 calibration.
    Each yield must be a list of numpy arrays matching the model's input signature.
    """
    train_ds, _ = ds_module.get_train_dataset(batch_size=1)
    count = 0
    for image, _ in train_ds.take(200):
        yield [tf.cast(image, tf.float32)]
        count += 1
    print(f"[export] INT8 calibration used {count} representative samples.")


# ---------------------------------------------------------------------------
# B. ONNX EXPORT  (optional)
# ---------------------------------------------------------------------------

def export_onnx(model: tf.keras.Model, out_dir: Path) -> Path | None:
    """
    Converts the model to ONNX format using tf2onnx.
    Returns the output path, or None if tf2onnx is not installed.
    """
    try:
        import tf2onnx
        import onnx
    except ImportError:
        print("\n[export] tf2onnx / onnx not installed — skipping ONNX export.")
        print("[export] Install with: pip install tf2onnx onnx")
        return None

    print("\n[export] Converting to ONNX ...")
    out_path = out_dir / "agrivision_xai.onnx"

    input_signature = [
        tf.TensorSpec(
            shape=(None, *config.IMAGE_SHAPE),
            dtype=tf.float32,
            name="input_image"
        )
    ]

    model_proto, _ = tf2onnx.convert.from_keras(
        model,
        input_signature=input_signature,
        opset=13,
        output_path=str(out_path),
    )

    print(f"[export] Saved: {out_path}  ({out_path.stat().st_size / 1024 / 1024:.1f} MB)")
    return out_path


# ---------------------------------------------------------------------------
# ACCURACY CHECK  (quick sanity check that quantisation didn't degrade badly)
# ---------------------------------------------------------------------------

def _tflite_accuracy(tflite_path: Path, num_batches: int = 20) -> float:
    """
    Runs the TFLite model on num_batches of the test set and returns accuracy.
    Used to verify INT8 quantisation hasn't caused a severe accuracy drop.
    """
    interpreter = tf.lite.Interpreter(model_path=str(tflite_path))
    interpreter.allocate_tensors()
    input_details  = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    test_ds, _ = ds_module.get_test_dataset(batch_size=1)

    correct = 0
    total   = 0

    for images, labels in test_ds.take(num_batches * config.BATCH_SIZE):
        img = tf.cast(images, tf.float32).numpy()
        if len(img.shape) == 3:
            img = img[np.newaxis, ...]

        interpreter.set_tensor(input_details[0]["index"], img)
        interpreter.invoke()
        probs = interpreter.get_tensor(output_details[0]["index"])

        pred = np.argmax(probs, axis=1)[0]
        true = np.argmax(labels.numpy())
        correct += int(pred == true)
        total   += 1

    return correct / total if total > 0 else 0.0


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def export(model_path: str | None = None, skip_onnx: bool = False) -> None:
    print("=" * 65)
    print("  AgriVision-XAI | export.py")
    print("=" * 65)

    config.EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    model = _load_model(model_path)

    report = {
        "backbone": config.BACKBONE,
        "use_cbam": config.USE_CBAM,
        "image_size": list(config.IMAGE_SIZE),
        "num_classes": config.NUM_CLASSES,
    }

    # -- A1. Float32 TFLite --
    fp32_path = export_tflite_fp32(model, config.EXPORT_DIR)
    fp32_stats = _benchmark_tflite(fp32_path)
    print(f"\n[export] FP32 TFLite benchmark:")
    print(f"  File size     : {fp32_stats['file_size_kb']:.0f} KB")
    print(f"  Mean latency  : {fp32_stats['latency_mean_ms']:.2f} ms")
    report["tflite_fp32"] = fp32_stats

    # -- A2. INT8 Quantised TFLite --
    int8_path = export_tflite_int8(model, _representative_dataset_generator, config.EXPORT_DIR)
    int8_stats = _benchmark_tflite(int8_path)
    print(f"\n[export] INT8 TFLite benchmark:")
    print(f"  File size     : {int8_stats['file_size_kb']:.0f} KB")
    print(f"  Mean latency  : {int8_stats['latency_mean_ms']:.2f} ms")
    print(f"  Speedup vs FP32: {fp32_stats['latency_mean_ms'] / int8_stats['latency_mean_ms']:.2f}×")

    # Quick accuracy check on INT8
    print("\n[export] Checking INT8 accuracy on a subset of test set ...")
    int8_acc = _tflite_accuracy(int8_path, num_batches=20)
    print(f"[export] INT8 sample accuracy: {int8_acc*100:.2f}%")
    int8_stats["sample_accuracy"] = round(int8_acc, 4)
    report["tflite_int8"] = int8_stats

    # -- B. ONNX --
    if not skip_onnx:
        onnx_path = export_onnx(model, config.EXPORT_DIR)
        if onnx_path:
            report["onnx"] = {
                "file_size_kb": round(onnx_path.stat().st_size / 1024, 1)
            }

    # -- Save report --
    report_path = config.EXPORT_DIR / "export_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n[export] Export report saved to {report_path}")

    # -- Summary --
    print(f"\n{'=' * 65}")
    print("  EXPORT COMPLETE")
    print(f"{'=' * 65}")
    print(f"  FP32  TFLite : {fp32_stats['file_size_kb']:.0f} KB  |  {fp32_stats['latency_mean_ms']:.1f} ms")
    print(f"  INT8  TFLite : {int8_stats['file_size_kb']:.0f} KB  |  {int8_stats['latency_mean_ms']:.1f} ms  |  {int8_acc*100:.1f}% acc")
    print(f"\n  Files saved to: {config.EXPORT_DIR}")
    print(f"  Run app.py next to launch the web interface.")


# ---------------------------------------------------------------------------
# ENTRY POINT
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AgriVision-XAI Model Export")
    parser.add_argument("--model",    type=str,           default=None)
    parser.add_argument("--no-onnx",  action="store_true", help="Skip ONNX export")
    args = parser.parse_args()
    export(model_path=args.model, skip_onnx=args.no_onnx)
