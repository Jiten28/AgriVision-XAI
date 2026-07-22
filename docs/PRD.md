# AgriVision-XAI — Project Requirements Document (PRD)

## 1. Project Overview

**Project Name:** AgriVision-XAI
**Version:** 2.0
**Type:** AI/ML Research Internship Project — IIIT Ranchi
**Developer:** Jiten Kumar (B.Tech CSE, AI & ML — JNTU Hyderabad)
**GitHub:** github.com/Jiten28/AgriVision-XAI

### What We Are Building

A research-grade plant disease detection system that goes beyond a simple image classifier. It detects the disease, explains _why_ it made that prediction (XAI), estimates how _severe_ the infection is, and is optimised for real-world field images — not just controlled lab images.

### Why This Exists (Problem Statement)

Existing plant disease classifiers (including V1) are trained and tested on PlantVillage — a lab-condition dataset where accuracy drops from 99% to ~31% on real field images. They also produce only a disease label with no explanation and no severity estimate, making them impractical for actual farmer use.

---

## 2. Target Users

| User                             | Need                                                             |
| -------------------------------- | ---------------------------------------------------------------- |
| Agricultural researchers         | Accurate, explainable disease diagnosis for field studies        |
| Farmers / agronomists            | Quick identification + severity of crop infection                |
| IIIT Ranchi internship reviewers | Research-level novelty: XAI + severity + cross-domain evaluation |
| Future mobile app users          | Lightweight TFLite model for on-device inference                 |

---

## 3. Core Features (Must Have)

- [x] Stratified 70/15/15 dataset split (fixing V1 validation/test overlap bug)
- [x] EfficientNetB0 backbone with ImageNet transfer learning
- [x] CBAM attention module (channel + spatial)
- [x] Two-phase training (frozen head → fine-tuning)
- [x] Class-weight balancing for imbalanced dataset
- [x] Grad-CAM and Grad-CAM++ visual explanations
- [x] LIME superpixel explanations
- [x] Disease severity estimation (HSV + Grad-CAM thresholding)
- [x] In-domain evaluation (PlantVillage held-out test set)
- [ ] Cross-domain evaluation (PlantDoc field images) — pending PlantDoc download
- [x] TFLite export (FP32 + INT8 quantised)
- [ ] Streamlit web application (frontend — next phase)

## 4. Research Features (Differentiators from V1)

- Cross-domain generalisation study (lab → field accuracy gap)
- Explainability layer — no black-box predictions
- Severity quantification — mild / moderate / severe label per image
- Methodologically correct evaluation (true held-out test set)
- Portable, environment-variable-driven configuration

## 5. Out of Scope (V2)

- Federated learning (future scope)
- Multi-plant / whole-field detection (YOLO-based, future scope)
- Real-time video stream inference
- IoT sensor integration

---

## 6. Dataset

| Dataset                     | Purpose                        | Size                      | Status        |
| --------------------------- | ------------------------------ | ------------------------- | ------------- |
| PlantVillage (V1 augmented) | Train + val + test (in-domain) | 54,305 images, 38 classes | ✅ Split done |
| PlantDoc                    | Cross-domain evaluation only   | ~2,600 images, 17 classes | ⏳ Pending    |
| Self-collected field images | Additional real-world eval     | TBD                       | Future        |

## 7. Performance Targets

| Metric                   | Target   | Current (2 epochs CPU) |
| ------------------------ | -------- | ---------------------- |
| In-domain val accuracy   | ≥ 95%    | 84.5% (Phase 1 only)   |
| In-domain top-3 accuracy | ≥ 98%    | 96.7%                  |
| Cross-domain accuracy    | ≥ 60%    | Not yet evaluated      |
| TFLite INT8 size         | ≤ 20 MB  | Not yet exported       |
| Mobile inference latency | ≤ 200 ms | Not yet benchmarked    |
