# =============================================================================
# AgriVision-XAI  |  config.py
#
# All settings are read from the .env file first.
# If a variable is not set in .env, the default value defined here is used.
# Edit .env for machine-specific paths — never hardcode paths here.
# =============================================================================

import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env from the project root (the folder containing this file)
_HERE = Path(__file__).resolve().parent
load_dotenv(_HERE / ".env")


# ---------------------------------------------------------------------------
# HELPER — read env var with a typed default
# ---------------------------------------------------------------------------
def _env(key: str, default, cast=str):
    val = os.environ.get(key)
    if val is None or val.strip() == "":
        return default
    if cast == bool:
        return val.strip().lower() in ("1", "true", "yes")
    return cast(val.strip())


# ---------------------------------------------------------------------------
# 1. ROOT PATHS
# ---------------------------------------------------------------------------
# Where the original V1 dataset lives (folder containing train/ and valid/).
V1_DATASET_ROOT = Path(_env(
    "V1_DATASET_ROOT",
    r"C:\Users\jiten\Downloads\New Plant Diseases Dataset(Augmented)"
))

# Where AgriVision-XAI writes its split dataset and all outputs.
PROJECT_ROOT = Path(_env(
    "PROJECT_ROOT",
    r"D:\Coding\GitHub\AgriVision-XAI"
))

# Sub-directories (all auto-derived from PROJECT_ROOT)
SPLIT_DIR = PROJECT_ROOT / "dataset_split"
TRAIN_DIR = SPLIT_DIR / "train"
VAL_DIR = SPLIT_DIR / "val"
TEST_DIR = SPLIT_DIR / "test"
CHECKPOINTS_DIR = PROJECT_ROOT / "checkpoints"
LOGS_DIR = PROJECT_ROOT / "logs"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
GRADCAM_DIR = OUTPUTS_DIR / "gradcam"
SEVERITY_DIR = OUTPUTS_DIR / "severity"
EXPORT_DIR = PROJECT_ROOT / "export"

# Optional: PlantDoc dataset for cross-domain evaluation.
# Leave PLANTDOC_DIR blank in .env to skip this step.
_plantdoc_raw = _env("PLANTDOC_DIR", "")
PLANTDOC_DIR = Path(_plantdoc_raw) if _plantdoc_raw else None


# ---------------------------------------------------------------------------
# 2. DATASET SPLIT RATIOS
# ---------------------------------------------------------------------------
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15
RANDOM_SEED = _env("RANDOM_SEED", 42, int)


# ---------------------------------------------------------------------------
# 3. IMAGE SETTINGS
# ---------------------------------------------------------------------------
# IMAGE_SIZE   = (128, 128)     # CPU: 128 saves ~3x memory vs 224; restore to (224,224) on GPU
# IMAGE_SHAPE  = (128, 128, 3)
# NUM_CHANNELS = 3

# GPU: 224 is standard for EfficientNet; restore to (128,128) on CPU
IMAGE_SIZE = (224, 224)
IMAGE_SHAPE = (224, 224, 3)
DENSE_UNITS = 512


# ---------------------------------------------------------------------------
# 4. TRAINING HYPERPARAMETERS
# ---------------------------------------------------------------------------
BATCH_SIZE = _env("BATCH_SIZE",      32,     int)
EPOCHS_FROZEN = _env("EPOCHS_FROZEN",   10,     int)
EPOCHS_FINETUNE = _env("EPOCHS_FINETUNE", 30,     int)
TOTAL_EPOCHS = EPOCHS_FROZEN + EPOCHS_FINETUNE

LR_FROZEN = _env("LR_FROZEN",    1e-3,  float)
LR_FINETUNE = _env("LR_FINETUNE",  1e-5,  float)

EARLY_STOPPING_PATIENCE = 7
LR_REDUCE_PATIENCE = 3
LR_REDUCE_FACTOR = 0.5
LR_REDUCE_MIN = 1e-7


# ---------------------------------------------------------------------------
# 5. MODEL SETTINGS
# ---------------------------------------------------------------------------
BACKBONE = _env("BACKBONE",       "EfficientNetB0")
DROPOUT_RATE = 0.4
DENSE_UNITS = 256       # CPU: 256 is enough for smoke test; restore to 512 on GPU
USE_CBAM = _env("USE_CBAM",       True,  bool)
FINETUNE_LAYERS = _env("FINETUNE_LAYERS", 20,   int)


# ---------------------------------------------------------------------------
# 6. CLASS INFORMATION  (38 PlantVillage classes)
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

HEALTHY_CLASS_INDICES = [
    i for i, c in enumerate(CLASS_NAMES) if c.endswith("___healthy")
]


# ---------------------------------------------------------------------------
# 7. AUGMENTATION SETTINGS
# ---------------------------------------------------------------------------
AUG_ROTATION = 30
AUG_WIDTH_SHIFT = 0.15
AUG_HEIGHT_SHIFT = 0.15
AUG_ZOOM = 0.15
AUG_HORIZONTAL_FLIP = True
AUG_VERTICAL_FLIP = False
AUG_BRIGHTNESS = [0.7, 1.3]
AUG_FILL_MODE = "reflect"


# ---------------------------------------------------------------------------
# 8. EXPLAINABILITY SETTINGS
# ---------------------------------------------------------------------------
GRADCAM_LAYER = "top_conv"
NUM_GRADCAM_SAMPLES = 20
LIME_NUM_SAMPLES = 1000
LIME_NUM_FEATURES = 10


# ---------------------------------------------------------------------------
# 9. SEVERITY THRESHOLDS
# ---------------------------------------------------------------------------
SEVERITY_THRESHOLDS = {
    "Healthy": (0.00, 0.05),
    "Mild": (0.05, 0.25),
    "Moderate": (0.25, 0.50),
    "Severe": (0.50, 1.00),
}


# ---------------------------------------------------------------------------
# 10. DIAGNOSTICS — print active config on import (only when run directly)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 60)
    print("  AgriVision-XAI | Active Configuration")
    print("=" * 60)
    print(f"\n  .env loaded from : {_HERE / '.env'}")
    print(f"\n  Paths:")
    print(f"    V1_DATASET_ROOT : {V1_DATASET_ROOT}")
    print(f"    PROJECT_ROOT    : {PROJECT_ROOT}")
    print(f"    SPLIT_DIR       : {SPLIT_DIR}")
    print(f"    PLANTDOC_DIR    : {PLANTDOC_DIR or '(not set)'}")
    print(f"\n  Dataset:")
    print(f"    Split ratio     : {TRAIN_RATIO}/{VAL_RATIO}/{TEST_RATIO}")
    print(f"    Random seed     : {RANDOM_SEED}")
    print(f"    Image size      : {IMAGE_SIZE}")
    print(f"    Num classes     : {NUM_CLASSES}")
    print(f"\n  Training:")
    print(f"    Backbone        : {BACKBONE}")
    print(f"    Use CBAM        : {USE_CBAM}")
    print(f"    Batch size      : {BATCH_SIZE}")
    print(f"    Epochs (frozen) : {EPOCHS_FROZEN}")
    print(f"    Epochs (finetune): {EPOCHS_FINETUNE}")
    print(f"    LR frozen       : {LR_FROZEN}")
    print(f"    LR finetune     : {LR_FINETUNE}")
    print(f"    Finetune layers : {FINETUNE_LAYERS}")
