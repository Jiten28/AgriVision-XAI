# =============================================================================
# AgriVision-XAI  |  config.py
# Central configuration — edit paths here, everything else reads from here.
# =============================================================================

from pathlib import Path

# ---------------------------------------------------------------------------
# 1. ROOT PATHS  (edit these to match your machine)
# ---------------------------------------------------------------------------
# Where the original V1 dataset lives (the folder that contains train/ and valid/)
V1_DATASET_ROOT = Path(r"D:\Coding\GitHub\AgriVision\V1_DATASET_ROOT")

# Where AgriVision-XAI will write its own re-split dataset and all outputs.
# Everything below is auto-derived — you only need to change V1_DATASET_ROOT above.
PROJECT_ROOT = Path(r"D:\Coding\GitHub\AgriVision-XAI")

# Sub-directories (auto-created by dataset.py)
SPLIT_DIR = PROJECT_ROOT / "dataset_split"   # re-split train/val/test
TRAIN_DIR = SPLIT_DIR / "train"
VAL_DIR = SPLIT_DIR / "val"
TEST_DIR = SPLIT_DIR / "test"

# Optional: PlantDoc dataset root (for cross-domain evaluation in Step 5)
# Leave as None if you haven't downloaded it yet.
PLANTDOC_DIR = None  # e.g. Path(r"D:\Datasets\PlantDoc")

# Where models, logs, and outputs are saved
CHECKPOINTS_DIR = PROJECT_ROOT / "checkpoints"
LOGS_DIR = PROJECT_ROOT / "logs"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
GRADCAM_DIR = OUTPUTS_DIR / "gradcam"
SEVERITY_DIR = OUTPUTS_DIR / "severity"
EXPORT_DIR = PROJECT_ROOT / "export"

# ---------------------------------------------------------------------------
# 2. DATASET SPLIT RATIOS
# ---------------------------------------------------------------------------
TRAIN_RATIO = 0.70   # 70 % of all images → training
# 15 % → validation  (used during training for early stopping)
VAL_RATIO = 0.15
TEST_RATIO = 0.15   # 15 % → test        (touched ONLY at final evaluation)
RANDOM_SEED = 42     # fixed seed for full reproducibility

# ---------------------------------------------------------------------------
# 3. IMAGE SETTINGS
# ---------------------------------------------------------------------------
# EfficientNetB0 native input (better than V1's 128x128)
IMAGE_SIZE = (224, 224)
IMAGE_SHAPE = (224, 224, 3)
NUM_CHANNELS = 3

# ---------------------------------------------------------------------------
# 4. TRAINING HYPERPARAMETERS
# ---------------------------------------------------------------------------
BATCH_SIZE = 32      # fits comfortably in 8 GB VRAM with EfficientNetB0
EPOCHS_FROZEN = 10      # Phase 1: train only the new head (backbone frozen)
EPOCHS_FINETUNE = 30      # Phase 2: unfreeze top layers and fine-tune
TOTAL_EPOCHS = EPOCHS_FROZEN + EPOCHS_FINETUNE   # 40 total

LR_FROZEN = 1e-3    # higher LR while backbone is frozen
LR_FINETUNE = 1e-5    # very low LR during fine-tuning to avoid catastrophic forgetting

# Early stopping: stop if val_loss doesn't improve for this many epochs
EARLY_STOPPING_PATIENCE = 7

# ReduceLROnPlateau: halve LR if val_loss stagnates for this many epochs
LR_REDUCE_PATIENCE = 3
LR_REDUCE_FACTOR = 0.5
LR_REDUCE_MIN = 1e-7

# ---------------------------------------------------------------------------
# 5. MODEL SETTINGS
# ---------------------------------------------------------------------------
BACKBONE = "EfficientNetB0"   # options: EfficientNetB0 | ResNet50 | MobileNetV3Large
DROPOUT_RATE = 0.4
DENSE_UNITS = 512       # units in the classification head dense layer
USE_CBAM = True      # set False to ablate attention module

# Fine-tuning: how many layers from the top of the backbone to unfreeze
FINETUNE_LAYERS = 20        # unfreeze top 20 layers of EfficientNetB0

# ---------------------------------------------------------------------------
# 6. CLASS INFORMATION  (38 PlantVillage classes — matches V1)
# ---------------------------------------------------------------------------
CLASS_NAMES = [
    "Apple___Apple_scab",
    "Apple___Black_rot",
    "Apple___Cedar_apple_rust",
    "Apple___healthy",
    "Blueberry___healthy",
    "Cherry_(including_sour)___Powdery_mildew",
    "Cherry_(including_sour)___healthy",
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
    "Corn_(maize)___Common_rust_",
    "Corn_(maize)___Northern_Leaf_Blight",
    "Corn_(maize)___healthy",
    "Grape___Black_rot",
    "Grape___Esca_(Black_Measles)",
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",
    "Grape___healthy",
    "Orange___Haunglongbing_(Citrus_greening)",
    "Peach___Bacterial_spot",
    "Peach___healthy",
    "Pepper,_bell___Bacterial_spot",
    "Pepper,_bell___healthy",
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy",
    "Raspberry___healthy",
    "Soybean___healthy",
    "Squash___Powdery_mildew",
    "Strawberry___Leaf_scorch",
    "Strawberry___healthy",
    "Tomato___Bacterial_spot",
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites Two-spotted_spider_mite",
    "Tomato___Target_Spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato___Tomato_mosaic_virus",
    "Tomato___healthy",
]
NUM_CLASSES = len(CLASS_NAMES)   # 38

# Healthy class indices — used by severity.py to skip severity estimation
# (no point estimating infection severity on a healthy leaf)
HEALTHY_CLASS_INDICES = [
    CLASS_NAMES.index(c) for c in CLASS_NAMES if c.endswith("___healthy")
]

# ---------------------------------------------------------------------------
# 7. AUGMENTATION SETTINGS  (applied only to training images)
# ---------------------------------------------------------------------------
AUG_ROTATION = 30        # degrees (±30)
AUG_WIDTH_SHIFT = 0.15
AUG_HEIGHT_SHIFT = 0.15
AUG_ZOOM = 0.15
AUG_HORIZONTAL_FLIP = True
AUG_VERTICAL_FLIP = False     # leaf orientation matters; vertical flip is unnatural
AUG_BRIGHTNESS = [0.7, 1.3]   # simulate different lighting conditions
AUG_FILL_MODE = "reflect"

# ---------------------------------------------------------------------------
# 8. EXPLAINABILITY SETTINGS  (Step 6)
# ---------------------------------------------------------------------------
GRADCAM_LAYER = "top_conv"   # last conv layer name in EfficientNetB0
NUM_GRADCAM_SAMPLES = 20           # how many sample images to visualise per run
# LIME perturbation samples (higher = slower but better)
LIME_NUM_SAMPLES = 1000
LIME_NUM_FEATURES = 10           # number of superpixels to highlight

# ---------------------------------------------------------------------------
# 9. SEVERITY THRESHOLDS  (Step 7)
# ---------------------------------------------------------------------------
# These define what % infected area maps to which severity label.
SEVERITY_THRESHOLDS = {
    "Healthy": (0.00, 0.05),  # 0 – 5 %  infected area
    "Mild": (0.05, 0.25),  # 5 – 25 %
    "Moderate": (0.25, 0.50),   # 25 – 50 %
    "Severe": (0.50, 1.00),   # 50 – 100 %
}
