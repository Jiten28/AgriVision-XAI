# =============================================================================
# AgriVision-XAI  |  model.py
#
# Architecture overview:
#
#   Input (224, 224, 3)
#       │
#       ▼
#   EfficientNetB0 backbone (pretrained on ImageNet, top removed)
#       │  ← frozen in Phase 1, top layers unfrozen in Phase 2
#       ▼
#   CBAM Block (Channel + Spatial Attention)   ← optional, config.USE_CBAM
#       │  focuses the model on disease-relevant regions of the feature map
#       ▼
#   Global Average Pooling 2D
#       │
#       ▼
#   Dense(512, ReLU)  +  BatchNormalization  +  Dropout(0.4)
#       │
#       ▼
#   Dense(NUM_CLASSES, softmax)   ← 38 outputs
#
# Two training phases (handled in train.py):
#   Phase 1 — only the head is trained (backbone.trainable = False)
#   Phase 2 — top FINETUNE_LAYERS of backbone are unfrozen for fine-tuning
# =============================================================================

import tensorflow as tf
from tensorflow import keras

import config


# ---------------------------------------------------------------------------
# CBAM — Convolutional Block Attention Module
# ---------------------------------------------------------------------------
# CBAM applies two sequential attention operations to the feature map:
#   1. Channel Attention: "which feature channels are important?"
#   2. Spatial Attention: "where in the image are the important features?"
#
# Paper: Woo et al. (2018) "CBAM: Convolutional Block Attention Module"
# ---------------------------------------------------------------------------

def channel_attention(feature_map: tf.Tensor, reduction_ratio: int = 16) -> tf.Tensor:
    """
    Channel attention: squeeze spatial dimensions → excite channel weights.

    Args:
        feature_map   : (batch, H, W, C)
        reduction_ratio: bottleneck factor for the MLP (C → C/r → C)

    Returns:
        channel_refined : (batch, H, W, C)  — feature map scaled by channel weights
    """
    _, H, W, C = feature_map.shape   # C = number of channels

    # Global Average Pool: captures broad context per channel
    avg_pool = tf.reduce_mean(feature_map, axis=[1, 2], keepdims=True)  # (B, 1, 1, C)
    # Global Max Pool: captures the most salient activation per channel
    max_pool = tf.reduce_max(feature_map, axis=[1, 2], keepdims=True)   # (B, 1, 1, C)

    # Shared MLP — same weights applied to both pooled features
    dense_units = max(C // reduction_ratio, 1)
    shared_dense1 = keras.layers.Dense(dense_units, activation="relu",  use_bias=False)
    shared_dense2 = keras.layers.Dense(C,           activation=None,    use_bias=False)

    avg_out = shared_dense2(shared_dense1(avg_pool))  # (B, 1, 1, C)
    max_out = shared_dense2(shared_dense1(max_pool))  # (B, 1, 1, C)

    # Channel attention map: sigmoid squashes to (0, 1) per channel
    channel_weights = tf.sigmoid(avg_out + max_out)   # (B, 1, 1, C)

    return feature_map * channel_weights               # broadcast × (B, H, W, C)


def spatial_attention(feature_map: tf.Tensor, kernel_size: int = 7) -> tf.Tensor:
    """
    Spatial attention: highlight which spatial locations matter.

    Args:
        feature_map : (batch, H, W, C)  — after channel attention
        kernel_size : conv kernel size (7 is standard from the paper)

    Returns:
        spatial_refined : (batch, H, W, C)
    """
    # Pool across channels to summarise each spatial location
    avg_pool = tf.reduce_mean(feature_map, axis=-1, keepdims=True)  # (B, H, W, 1)
    max_pool = tf.reduce_max(feature_map,  axis=-1, keepdims=True)  # (B, H, W, 1)

    # Concatenate and convolve to produce a single spatial attention map
    concat = tf.concat([avg_pool, max_pool], axis=-1)               # (B, H, W, 2)
    spatial_weights = keras.layers.Conv2D(
        filters=1,
        kernel_size=kernel_size,
        padding="same",
        activation="sigmoid",
        use_bias=False,
        name="spatial_attention_conv"
    )(concat)                                                        # (B, H, W, 1)

    return feature_map * spatial_weights                             # (B, H, W, C)


def cbam_block(feature_map: tf.Tensor, reduction_ratio: int = 16) -> tf.Tensor:
    """
    Full CBAM: channel attention → spatial attention, applied sequentially.
    """
    refined = channel_attention(feature_map, reduction_ratio)
    refined = spatial_attention(refined)
    return refined


# ---------------------------------------------------------------------------
# BACKBONE FACTORY
# ---------------------------------------------------------------------------

def _get_backbone(name: str, input_shape: tuple) -> keras.Model:
    """
    Returns the chosen backbone with ImageNet weights, top layer removed.
    input_shape: (H, W, C) without batch dimension.
    """
    kwargs = dict(
        weights="imagenet",
        include_top=False,
        input_shape=input_shape,
    )

    if name == "EfficientNetB0":
        backbone = keras.applications.EfficientNetB0(**kwargs)
    elif name == "ResNet50":
        backbone = keras.applications.ResNet50(**kwargs)
    elif name == "MobileNetV3Large":
        backbone = keras.applications.MobileNetV3Large(**kwargs)
    else:
        raise ValueError(
            f"Unknown backbone: '{name}'. "
            f"Choose from: EfficientNetB0, ResNet50, MobileNetV3Large"
        )

    return backbone


# ---------------------------------------------------------------------------
# FULL MODEL BUILDER
# ---------------------------------------------------------------------------

def build_model(
    backbone_name: str = config.BACKBONE,
    num_classes:   int = config.NUM_CLASSES,
    input_shape: tuple = config.IMAGE_SHAPE,
    use_cbam:     bool = config.USE_CBAM,
    dropout_rate: float = config.DROPOUT_RATE,
    dense_units:   int = config.DENSE_UNITS,
    freeze_backbone: bool = True,
) -> keras.Model:
    """
    Builds and returns the AgriVision-XAI model.

    Args:
        backbone_name   : which pretrained backbone to use (see config.BACKBONE)
        num_classes     : number of output classes (38 for PlantVillage)
        input_shape     : (H, W, C)
        use_cbam        : whether to insert the CBAM attention block after the backbone
        dropout_rate    : dropout probability before the final Dense layer
        dense_units     : size of the intermediate Dense layer in the classification head
        freeze_backbone : if True, backbone weights are frozen (Phase 1 training)

    Returns:
        Keras Model (not yet compiled — compilation happens in train.py)
    """

    # ---- Inputs ----
    inputs = keras.Input(shape=input_shape, name="input_image")

    # ---- EfficientNetB0 backbone ----
    # EfficientNetB0.preprocess_input expects pixel values in [0, 255].
    # Our dataset.py returns float32 values still in [0, 255] (cast but not /255),
    # so no additional rescaling is needed here.
    backbone = _get_backbone(backbone_name, input_shape)
    backbone.trainable = not freeze_backbone

    # Pass inputs through backbone
    x = backbone(inputs, training=not freeze_backbone)
    # x shape: (batch, 7, 7, 1280) for EfficientNetB0 with 224x224 input

    # ---- CBAM Attention Block ----
    if use_cbam:
        x = cbam_block(x, reduction_ratio=16)

    # ---- Classification Head ----
    x = keras.layers.GlobalAveragePooling2D(name="gap")(x)
    # Reduces (batch, 7, 7, 1280) → (batch, 1280)

    x = keras.layers.Dense(dense_units, name="head_dense")(x)
    x = keras.layers.BatchNormalization(name="head_bn")(x)
    x = keras.layers.Activation("relu", name="head_relu")(x)
    x = keras.layers.Dropout(dropout_rate, name="head_dropout")(x)

    # ---- Output Layer ----
    outputs = keras.layers.Dense(
        num_classes,
        activation="softmax",
        name="predictions"
    )(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name="AgriVision_XAI")

    return model


# ---------------------------------------------------------------------------
# PHASE 2 — UNFREEZE TOP LAYERS FOR FINE-TUNING
# ---------------------------------------------------------------------------

def unfreeze_top_layers(model: keras.Model, n_layers: int = config.FINETUNE_LAYERS) -> None:
    """
    Unfreezes the top `n_layers` of the backbone for fine-tuning (Phase 2).
    Called by train.py after Phase 1 completes.

    Why unfreeze only the top layers?
    The bottom layers of EfficientNetB0 learn generic features (edges, textures)
    that transfer well from ImageNet. The top layers learn task-specific features.
    We unfreeze only those to avoid destroying the generic low-level representations.

    Args:
        model   : the full AgriVision-XAI model
        n_layers: how many layers from the end of the backbone to unfreeze
    """
    # Locate the backbone by name
    backbone = model.get_layer(config.BACKBONE)
    backbone.trainable = True   # enable gradient flow through backbone

    # Freeze all backbone layers, then selectively unfreeze the last n_layers
    for layer in backbone.layers[:-n_layers]:
        layer.trainable = False

    trainable_count = sum(1 for l in backbone.layers if l.trainable)
    total_count     = len(backbone.layers)
    print(f"\n[model] Fine-tune: {trainable_count}/{total_count} backbone layers now trainable.")
    print(f"[model] ({total_count - trainable_count} bottom layers remain frozen)")


# ---------------------------------------------------------------------------
# DIAGNOSTICS
# ---------------------------------------------------------------------------

def model_summary(model: keras.Model) -> None:
    """Print model summary and parameter counts."""
    model.summary(line_length=100, expand_nested=False)

    total     = model.count_params()
    trainable = sum(tf.size(w).numpy() for w in model.trainable_weights)
    frozen    = total - trainable

    print(f"\n  Total parameters     : {total:>12,}")
    print(f"  Trainable parameters : {trainable:>12,}")
    print(f"  Frozen parameters    : {frozen:>12,}")


# ---------------------------------------------------------------------------
# STANDALONE: run this file directly to verify the model builds correctly
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 60)
    print("  AgriVision-XAI | model.py")
    print("=" * 60)

    print(f"\n[model] Building {config.BACKBONE} + CBAM={config.USE_CBAM} ...")
    model = build_model(freeze_backbone=True)
    model_summary(model)

    # Quick forward pass to check shapes
    print("\n[model] Running a dummy forward pass ...")
    dummy = tf.zeros((2, *config.IMAGE_SHAPE))   # batch of 2
    out = model(dummy, training=False)
    print(f"  Input shape  : {dummy.shape}")
    print(f"  Output shape : {out.shape}")   # should be (2, 38)
    assert out.shape == (2, config.NUM_CLASSES), \
        f"Output shape mismatch: expected (2, {config.NUM_CLASSES}), got {out.shape}"
    print(f"\n[model] Output shape correct: {out.shape}")
    print("[model] Model build verified.")
