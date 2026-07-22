# 🌿 AgriVision-XAI

**Explainable AI System for Plant Disease Detection**
_AI/ML Research Internship Project — IIIT Ranchi_

[![Python](https://img.shields.io/badge/Python-3.11%2F3.13-blue)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.20-orange)](https://tensorflow.org)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

---

## Overview

AgriVision-XAI is a research-grade plant disease detection system that goes beyond a simple image classifier. It not only identifies the disease but also **explains why** it made that prediction and **estimates how severe** the infection is — making it practical for real-world agricultural use.

This is Version 2 of the original [Plant Disease Detection V1](https://github.com/Jiten28/Plant-Disease-Detection) project, upgraded with research-level techniques for an AI/ML Research Internship at IIIT Ranchi.

---

## What Makes V2 Different from V1

| Feature        | V1                                         | V2 (AgriVision-XAI)                        |
| -------------- | ------------------------------------------ | ------------------------------------------ |
| Model          | Custom CNN (scratch)                       | EfficientNetB0 + CBAM (transfer learning)  |
| Dataset split  | Val set reused as test (inflated accuracy) | True 70/15/15 stratified split             |
| Explainability | None                                       | Grad-CAM, Grad-CAM++, LIME                 |
| Severity       | None                                       | Mild / Moderate / Severe + % infected area |
| Cross-domain   | Not evaluated                              | PlantDoc field-image evaluation            |
| Deployment     | Streamlit (hardcoded paths)                | TFLite + portable config via .env          |

---

## Architecture

```
Input Image → EfficientNetB0 → CBAM Attention → Classification Head
                                                        │
                          ┌─────────────────────────────┤
                          │                             │
                    XAI Module                  Severity Module
              (Grad-CAM / LIME)           (HSV + Grad-CAM thresholding)
                          │                             │
                   Heatmap overlay            Mild / Moderate / Severe
```

---

## Results

| Metric                                | Value                |
| ------------------------------------- | -------------------- |
| Val Accuracy (Phase 1, 2 epochs, CPU) | 84.5%                |
| Top-3 Accuracy                        | 96.7%                |
| Classes                               | 38 (14 crop species) |
| Training Images                       | 37,985               |
| Val Images                            | 8,158                |
| Test Images                           | 8,162                |

_Full GPU training (10+30 epochs, 224×224) expected to reach ≥ 95% val accuracy._

---

## Quick Start

### 1. Clone and set up environment

```bash
git clone https://github.com/Jiten28/AgriVision-XAI.git
cd AgriVision-XAI
python -m venv venv
venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

### 2. Configure paths

```bash
copy .env.example .env
# Edit .env — set V1_DATASET_ROOT and PROJECT_ROOT
```

### 3. Run the pipeline

```bash
python config.py          # verify config
python dataset.py         # build 70/15/15 split (run once)
python model.py           # verify model builds
python train.py           # train the model
python evaluate.py        # evaluate on test set
python explainability.py  # generate Grad-CAM visualisations
python severity.py        # estimate disease severity
python export.py          # export to TFLite
```

### 4. Launch web app

```bash
pip install streamlit
streamlit run app.py
```

---

## Dataset

- **Training:** [New Plant Diseases Dataset (Augmented)](https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset) — 54,305 images, 38 classes
- **Cross-domain evaluation:** [PlantDoc](https://www.kaggle.com/datasets/nirmalsankalana/plantdoc-dataset) — ~2,600 real field images

---

## Project Structure

```
AgriVision-XAI/
├── docs/               ← PRD, Architecture, Design, Phases, Memory
├── config.py           ← All settings (reads from .env)
├── dataset.py          ← Dataset split + data pipelines
├── model.py            ← EfficientNetB0 + CBAM architecture
├── train.py            ← Two-phase training
├── evaluate.py         ← Evaluation + confusion matrix
├── explainability.py   ← Grad-CAM, Grad-CAM++, LIME
├── severity.py         ← Disease severity estimation
├── export.py           ← TFLite / ONNX export
├── app.py              ← Streamlit web interface
├── .env.example        ← Environment variable template
└── requirements.txt
```

---

## Research Papers

1. Mohanty et al. (2016) — [Using Deep Learning for Image-Based Plant Disease Detection](https://arxiv.org/pdf/1604.03169)
2. Singh et al. (2020) — [PlantDoc: A Dataset for Visual Plant Disease Detection](https://arxiv.org/pdf/1911.10317)
3. CBAM + Grad-CAM for Leaf Disease (2025) — [arXiv:2509.26484](https://arxiv.org/pdf/2509.26484)
4. Plant Disease Multi-Dataset EfficientNet (MDPI 2025) — [mdpi.com](https://www.mdpi.com/2571-8800/8/1/4/pdf)

---

## Tech Stack

`Python` `TensorFlow 2.20` `Keras 3` `EfficientNetB0` `OpenCV`
`scikit-learn` `Grad-CAM` `LIME` `TFLite` `Streamlit`

---

## Developer

**Jiten Kumar**
B.Tech Computer Science (AI & ML) — JNTU Hyderabad
GitHub: [Jiten28](https://github.com/Jiten28)

_Built as part of AI/ML Research Internship at IIIT Ranchi_
