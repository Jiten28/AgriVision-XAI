# =============================================================================
# AgriVision-XAI  |  model.py
# Compatible with: TF 2.20 / Keras 3 | Python 3.13 (CPU) + Python 3.11 (GPU)
# =============================================================================

import tensorflow as tf
from tensorflow import keras

import config


# ---------------------------------------------------------------------------
# CBAM — Keras 3 compatible, serializable custom layers
# ---------------------------------------------------------------------------

@tf.keras.utils.register_keras_serializable(package="AgriVision")
class ChannelAttention(keras.Layer):
    """Which feature channels matter most?"""

    def __init__(self, reduction_ratio: int = 16, **kwargs):
        super().__init__(**kwargs)
        self.reduction_ratio = reduction_ratio

    def build(self, input_shape):
        channels = input_shape[-1]
        units = max(channels // self.reduction_ratio, 1)
        self.dense1 = keras.layers.Dense(
            units, activation="relu", use_bias=False)
        self.dense2 = keras.layers.Dense(channels, use_bias=False)
        super().build(input_shape)

    def call(self, x):
        avg = keras.ops.mean(x, axis=[1, 2], keepdims=True)
        mx = keras.ops.max(x,  axis=[1, 2], keepdims=True)
        avg_out = self.dense2(self.dense1(avg))
        max_out = self.dense2(self.dense1(mx))
        return x * keras.ops.sigmoid(avg_out + max_out)

    def get_config(self):
        base = super().get_config()
        base["reduction_ratio"] = self.reduction_ratio
        return base


@tf.keras.utils.register_keras_serializable(package="AgriVision")
class SpatialAttention(keras.Layer):
    """Where in the image are the important features?"""

    def __init__(self, kernel_size: int = 7, **kwargs):
        super().__init__(**kwargs)
        self.kernel_size = kernel_size
        self.conv = keras.layers.Conv2D(
            1, kernel_size, padding="same", activation="sigmoid", use_bias=False
        )

    def call(self, x):
        avg = keras.ops.mean(x, axis=-1, keepdims=True)
        mx = keras.ops.max(x,  axis=-1, keepdims=True)
        concat = keras.ops.concatenate([avg, mx], axis=-1)
        return x * self.conv(concat)

    def get_config(self):
        base = super().get_config()
        base["kernel_size"] = self.kernel_size
        return base


@tf.keras.utils.register_keras_serializable(package="AgriVision")
class CBAMBlock(keras.Layer):
    """Channel attention → Spatial attention, sequentially."""

    def __init__(self, reduction_ratio: int = 16, **kwargs):
        super().__init__(**kwargs)
        self.reduction_ratio = reduction_ratio
        self.channel = ChannelAttention(reduction_ratio)
        self.spatial = SpatialAttention()

    def call(self, x):
        return self.spatial(self.channel(x))

    def get_config(self):
        base = super().get_config()
        base["reduction_ratio"] = self.reduction_ratio
        return base


# ---------------------------------------------------------------------------
# BACKBONE FACTORY
# ---------------------------------------------------------------------------

# Keras stores the backbone with a lowercased name internally.
# e.g. "EfficientNetB0" → layer name "efficientnetb0"
BACKBONE_LAYER_NAME = {
    "EfficientNetB0": "efficientnetb0",
    "ResNet50": "resnet50",
    "MobileNetV3Large": "mobilenetv3large",
}


def _get_backbone(name: str, input_shape: tuple) -> keras.Model:
    kwargs = dict(weights="imagenet", include_top=False,
                  input_shape=input_shape)
    if name == "EfficientNetB0":
        return keras.applications.EfficientNetB0(**kwargs)
    elif name == "ResNet50":
        return keras.applications.ResNet50(**kwargs)
    elif name == "MobileNetV3Large":
        return keras.applications.MobileNetV3Large(**kwargs)
    else:
        raise ValueError(f"Unknown backbone: '{name}'")


# ---------------------------------------------------------------------------
# FULL MODEL BUILDER
# ---------------------------------------------------------------------------

def build_model(
    backbone_name:   str = config.BACKBONE,
    num_classes:     int = config.NUM_CLASSES,
    input_shape:     tuple = config.IMAGE_SHAPE,
    use_cbam:        bool = config.USE_CBAM,
    dropout_rate:    float = config.DROPOUT_RATE,
    dense_units:     int = config.DENSE_UNITS,
    freeze_backbone: bool = True,
) -> keras.Model:

    inputs = keras.Input(shape=input_shape, name="input_image")
    backbone = _get_backbone(backbone_name, input_shape)
    backbone.trainable = not freeze_backbone

    x = backbone(inputs, training=not freeze_backbone)

    if use_cbam:
        x = CBAMBlock(reduction_ratio=16, name="cbam")(x)

    x = keras.layers.GlobalAveragePooling2D(name="gap")(x)
    x = keras.layers.Dense(dense_units, name="head_dense")(x)
    x = keras.layers.BatchNormalization(name="head_bn")(x)
    x = keras.layers.Activation("relu", name="head_relu")(x)
    x = keras.layers.Dropout(dropout_rate, name="head_dropout")(x)
    outputs = keras.layers.Dense(
        num_classes, activation="softmax", name="predictions")(x)

    return keras.Model(inputs=inputs, outputs=outputs, name="AgriVision_XAI")


# ---------------------------------------------------------------------------
# PHASE 2 — UNFREEZE TOP LAYERS
# ---------------------------------------------------------------------------

def unfreeze_top_layers(model: keras.Model, n_layers: int = config.FINETUNE_LAYERS) -> None:
    # Use the actual lowercase layer name Keras assigns internally
    layer_name = BACKBONE_LAYER_NAME.get(
        config.BACKBONE, config.BACKBONE.lower())
    backbone = model.get_layer(layer_name)
    backbone.trainable = True

    for layer in backbone.layers[:-n_layers]:
        layer.trainable = False

    trainable = sum(1 for l in backbone.layers if l.trainable)
    total = len(backbone.layers)
    print(
        f"\n[model] Fine-tune: {trainable}/{total} backbone layers now trainable.")
    print(f"[model] ({total - trainable} bottom layers remain frozen)")


# ---------------------------------------------------------------------------
# DIAGNOSTICS
# ---------------------------------------------------------------------------

def model_summary(model: keras.Model) -> None:
    model.summary(line_length=90, expand_nested=False)
    total = model.count_params()
    trainable = sum(tf.size(w).numpy() for w in model.trainable_weights)
    print(f"\n  Total params     : {total:>10,}")
    print(f"  Trainable params : {trainable:>10,}")
    print(f"  Frozen params    : {total - trainable:>10,}")


# ---------------------------------------------------------------------------
# STANDALONE TEST
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 60)
    print("  AgriVision-XAI | model.py")
    print("=" * 60)
    print(f"\n[model] Building {config.BACKBONE} + CBAM={config.USE_CBAM} ...")
    model = build_model(freeze_backbone=True)
    model_summary(model)

    # Test save + reload (verifies serialization decorators work)
    import tempfile
    import os
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "test_model.keras")
        model.save(path)
        reloaded = keras.models.load_model(path)
        print("\n[model] Save/reload test: PASSED")

    import numpy as np
    dummy = np.zeros((2, *config.IMAGE_SHAPE), dtype="float32")
    out = model(dummy, training=False)
    assert out.shape == (2, config.NUM_CLASSES)
    print(f"[model] Output shape {out.shape}: PASSED")
    print("[model] All checks passed.")
