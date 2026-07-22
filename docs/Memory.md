# AgriVision-XAI — Memory (AI Context File)

> This file is updated continuously as the project progresses.
> When starting a new chat, share this file first so the AI has full context
> without re-reading the entire codebase.

---

## Project Identity

- **Name:** AgriVision-XAI
- **Developer:** Jiten Kumar (B.Tech CSE AI/ML, JNTU Hyderabad)
- **Purpose:** AI/ML Research Internship at IIIT Ranchi — V2 upgrade of Plant Disease Detection V1
- **GitHub:** github.com/Jiten28/AgriVision-XAI
- **Local path (laptop):** `D:\Coding\AgriVision-XAI`

---

## Hardware

| Machine | GPU           | RAM   | OS         | Python |
| ------- | ------------- | ----- | ---------- | ------ |
| Laptop  | RTX 4060 8 GB | 16 GB | Windows 11 | 3.13   |
| Desktop | RTX 4070      | 32 GB | Windows    | —      |

**Current mode:** CPU only (Windows native, no WSL2 yet)
**Framework:** TensorFlow 2.20 / Keras 3

---

## What Has Been Built (Phase 1 ✅)

| File                | Purpose                                          | Status     |
| ------------------- | ------------------------------------------------ | ---------- |
| `config.py`         | All settings via .env (python-dotenv)            | ✅ Working |
| `dataset.py`        | Stratified 70/15/15 split + tf.data pipelines    | ✅ Working |
| `model.py`          | EfficientNetB0 + CBAM + classification head      | ✅ Working |
| `train.py`          | Two-phase training with callbacks                | ✅ Working |
| `evaluate.py`       | Metrics, confusion matrix, cross-domain eval     | ✅ Working |
| `explainability.py` | Grad-CAM, Grad-CAM++, LIME                       | ✅ Built   |
| `severity.py`       | HSV + Grad-CAM severity estimation               | ✅ Built   |
| `export.py`         | TFLite FP32/INT8 + ONNX export                   | ✅ Built   |
| `docs/`             | PRD, Architecture, Rules, Phases, Design, Memory | ✅ Done    |
| `README.md`         | Project readme                                   | ✅ Done    |
| `.env`              | Local machine paths                              | ✅ Set     |
| `.gitignore`        | Git ignore rules                                 | ✅ Done    |
| `requirements.txt`  | Python dependencies                              | ✅ Done    |

**Not yet built:**

- `app.py` — Streamlit frontend (Phase 3, next)
- `.streamlit/config.toml` — theme config

---

## Dataset Status

- **Source:** PlantVillage (V1 augmented) at `D:\Coding\Plant Disease Detection\Plant-Disease-Detection-V1\Data`
- **Split:** 37,985 train / 8,158 val / 8,162 test (54,305 total, 38 classes)
- **PlantDoc:** Downloaded to `D:\Coding\AgriVision-XAI\PlantDoc-Dataset` (cross-domain eval pending)
- **Image size (CPU):** 128×128 | **(GPU later):** 224×224

---

## Training Status

- **Smoke test run:** ✅ Complete (2 frozen + 3 finetune epochs, CPU)
- **Phase 1 result:** val_accuracy = **84.5%**, top-3 = **96.7%**
- **Best model saved:** `checkpoints/best_model.keras`
- **Full training:** ⏳ Pending (needs GPU / WSL2)

---

## Critical Bugs Fixed (don't repeat these)

| Bug                                               | Fix                                                                                                   |
| ------------------------------------------------- | ----------------------------------------------------------------------------------------------------- |
| `ds.cache()` crashes 16 GB RAM                    | Removed cache, prefetch(2), parallel=2                                                                |
| `keras.saving` AttributeError on TF 2.20          | Use `tf.keras.utils.register_keras_serializable`                                                      |
| CBAM uses `tf.*` ops in Functional graph          | Rewrote as `keras.Layer` subclass with `keras.ops.*`                                                  |
| `unfreeze_top_layers` — layer not found           | Keras uses lowercase name: `"efficientnetb0"` not `"EfficientNetB0"` — use `BACKBONE_LAYER_NAME` dict |
| CBAMBlock not found on model load                 | Added `@tf.keras.utils.register_keras_serializable` + `get_config()` to all three CBAM classes        |
| evaluate/explainability/severity can't load model | Added `import model as model_module` at top of each file to trigger decorator registration            |

---

## Next Step

**Build `app.py`** — Streamlit frontend.
Refer to `docs/Design.md` for layout, colours, and component specs.
The app wraps: `model.py` + `explainability.py` + `severity.py`

```bash
pip install streamlit
python -m streamlit run app.py
```

---

## Key Config Values (current CPU mode)

```
IMAGE_SIZE    = (128, 128)
BATCH_SIZE    = 16
DENSE_UNITS   = 256
EPOCHS_FROZEN = 2   (smoke test; full = 10)
EPOCHS_FINETUNE = 3 (smoke test; full = 30)
```

## Restore for GPU

```
IMAGE_SIZE    = (224, 224)
BATCH_SIZE    = 32
DENSE_UNITS   = 512
EPOCHS_FROZEN = 10
EPOCHS_FINETUNE = 30
+ restore ds.cache() and AUTOTUNE in dataset.py
```
