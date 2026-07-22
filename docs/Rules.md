# AgriVision-XAI — Rules

Rules for any AI assistant or developer working on this codebase.
Read this before writing or modifying any file.

---

## 1. Environment Rules

- Always use `config.py` for settings — never hardcode paths, batch sizes, or hyperparameters inside other files
- All paths come from `.env` via `python-dotenv` — never hardcode Windows paths like `D:\...`
- `.env` is never committed to Git — `.env.example` is the committed template
- The venv must be activated before running any script: `.\venv\Scripts\activate`

---

## 2. Framework Rules

- Framework: **TensorFlow 2.20 / Keras 3** — no PyTorch
- All custom Keras layers **must** use `@tf.keras.utils.register_keras_serializable(package="AgriVision")` decorator and implement `get_config()` — required for model save/load
- Never use raw `tf.*` ops directly inside Functional model graph construction — wrap them in `keras.Layer` subclasses using `keras.ops.*`
- Import order in every file: stdlib → numpy → tensorflow → project modules (config, dataset, model)

---

## 3. File Rules

- One responsibility per file — do not mix training logic into model.py or dataset logic into train.py
- Every file must run standalone as `python <file>.py` for testing
- All output files go to subdirectories of `config.PROJECT_ROOT` — never to the source folder
- Never delete `dataset_split/` automatically — only on explicit `force=True`

---

## 4. Memory & RAM Rules (CPU mode)

- **Never use `ds.cache()`** on the training dataset — crashes 16 GB RAM machines
- Keep `prefetch(2)` not `prefetch(tf.data.AUTOTUNE)` on CPU
- Keep `num_parallel_calls=2` not `AUTOTUNE` on CPU
- CPU thread limit: `set_intra_op_parallelism_threads(2)` and `set_inter_op_parallelism_threads(2)`
- When switching to GPU (WSL2): restore cache(), AUTOTUNE, IMAGE_SIZE=(224,224), DENSE_UNITS=512, BATCH_SIZE=32

---

## 5. Model Rules

- Backbone layer name in Keras is **lowercase**: `"efficientnetb0"` not `"EfficientNetB0"` — always use `BACKBONE_LAYER_NAME` dict from `model.py`
- Never train from scratch — always use pretrained ImageNet weights
- Phase 1 must complete before Phase 2 — never skip frozen training
- Always delete old `checkpoints/` before retraining with a new architecture

---

## 6. Dataset Rules

- The test set (`dataset_split/test/`) is touched **exactly once** — only in `evaluate.py`
- Never use the test set for hyperparameter tuning or architecture decisions
- PlantDoc is used only as a **cross-domain evaluation** set — never mixed into training data
- Class names must match `config.CLASS_NAMES` exactly in all three splits

---

## 7. XAI Rules

- Grad-CAM target layer: last Conv2D or DepthwiseConv2D in the backbone
- LIME: minimum 1000 perturbation samples for reliable superpixel weights
- Never run LIME on CPU for more than 20 samples — very slow
- Use `--no-lime` flag when running `explainability.py` on CPU for speed

---

## 8. What NOT to Do

- Do not add new dependencies without updating `requirements.txt`
- Do not use `model.fit()` with `validation_data=test_ds` — validation is for val set only
- Do not commit `.keras` model files to Git — they can be 20–100 MB
- Do not use `tf.placeholder` (deprecated) — use `keras.Input` instead
- Do not mix `tf.keras` and standalone `keras` imports in the same file — pick one
- Do not use `ds.cache()` on CPU (see Memory Rules above)
