# AgriVision-XAI — Development Phases

## Phase 1 — ML Backend ✅ COMPLETE

All core Python modules built and verified.

| File                | Status  | Notes                                                |
| ------------------- | ------- | ---------------------------------------------------- |
| `config.py`         | ✅ Done | Reads from .env, all settings centralised            |
| `dataset.py`        | ✅ Done | 70/15/15 stratified split, 54,305 images, 38 classes |
| `model.py`          | ✅ Done | EfficientNetB0 + CBAM, Keras 3 serializable          |
| `train.py`          | ✅ Done | Two-phase training, CPU-safe memory settings         |
| `evaluate.py`       | ✅ Done | In-domain + cross-domain evaluation                  |
| `explainability.py` | ✅ Done | Grad-CAM, Grad-CAM++, LIME                           |
| `severity.py`       | ✅ Done | HSV + Grad-CAM severity estimation                   |
| `export.py`         | ✅ Done | TFLite FP32 + INT8, ONNX optional                    |

**Phase 1 Results (CPU smoke test — 2+3 epochs):**

- Phase 1 val accuracy: **84.5%**
- Phase 1 top-3 accuracy: **96.7%**
- Training time per epoch: ~2.5 min (CPU, batch=16, image=128×128)

---

## Phase 2 — Full GPU Training ⏳ PENDING

**Prerequisite:** Switch to WSL2 on laptop OR use desktop PC (RTX 4070, 32 GB RAM)

**Steps when ready:**

1. Update `.env`:
   ```
   BATCH_SIZE=32
   EPOCHS_FROZEN=10
   EPOCHS_FINETUNE=30
   ```
2. Update `config.py`:
   ```python
   IMAGE_SIZE  = (224, 224)
   IMAGE_SHAPE = (224, 224, 3)
   DENSE_UNITS = 512
   ```
3. Update `dataset.py`: restore `ds.cache()`, `AUTOTUNE`, `prefetch(AUTOTUNE)`
4. Install GPU dependencies: `tensorflow[and-cuda]==2.20.0` in WSL2
5. Run: `python train.py`
6. Expected: **≥ 95% val accuracy** after full 40 epochs

---

## Phase 3 — Streamlit Frontend ⏳ NEXT

Build `app.py` — web interface wrapping all backend modules.

**Features to build:**

- [ ] Image upload widget
- [ ] Disease prediction display (class + confidence bar)
- [ ] Grad-CAM heatmap overlay on uploaded image
- [ ] Severity gauge (mild / moderate / severe + % infected)
- [ ] Top-3 predictions table
- [ ] Model info sidebar (backbone, accuracy, dataset)
- [ ] Download button for explanation image

**Design:** See `Design.md`

---

## Phase 4 — Cross-Domain Evaluation ⏳ PENDING

**Prerequisite:** PlantDoc dataset downloaded from Kaggle
(`kaggle.com/datasets/nirmalsankalana/plantdoc-dataset`)

**Steps:**

1. Set `PLANTDOC_DIR` in `.env`
2. Run: `python evaluate.py`
3. Report in-domain vs cross-domain accuracy gap
4. Document generalisation gap in research report

---

## Phase 5 — Export & Deployment ⏳ PENDING

1. Run `python export.py` after full GPU training
2. Benchmark FP32 vs INT8 TFLite latency
3. Document model size + inference speed
4. Integrate TFLite model into Streamlit app

---

## Phase 6 — Research Report Update ⏳ PENDING

Update `AgriVision-XAI_Internship_Research_Report.docx` with:

- [ ] Final accuracy numbers (in-domain + cross-domain)
- [ ] Confusion matrix figures
- [ ] Grad-CAM sample visualisations
- [ ] Severity estimation examples
- [ ] Version comparison table (V1 vs V2 final numbers)
