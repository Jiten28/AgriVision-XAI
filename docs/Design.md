# AgriVision-XAI — Design Specification

## 1. Theme & Personality

**Feel:** Clean, scientific, agriculture-inspired — not a generic dark dashboard.
Professional enough for a research internship demo, simple enough to show a farmer.

**Primary colour:** `#1F6F43` — deep agricultural green
**Accent colour:** `#4CAF50` — bright leaf green
**Warning colour:** `#FF9800` — orange (moderate severity)
**Danger colour:** `#F44336` — red (severe severity)
**Success colour:** `#4CAF50` — green (healthy / mild)
**Background:** `#F5F9F6` — very light green-tinted white
**Card background:** `#FFFFFF`
**Text primary:** `#1A1A1A`
**Text secondary:** `#666666`
**Border:** `#D4E6D9`

---

## 2. Typography

| Use             | Font       | Size    | Weight   |
| --------------- | ---------- | ------- | -------- |
| App title       | sans-serif | 2rem    | Bold     |
| Section headers | sans-serif | 1.2rem  | SemiBold |
| Body text       | sans-serif | 0.95rem | Regular  |
| Labels / badges | sans-serif | 0.8rem  | Medium   |
| Code / paths    | monospace  | 0.85rem | Regular  |

Streamlit uses system sans-serif — no custom font import needed.

---

## 3. Page Layout

```
┌─────────────────────────────────────────────────────┐
│  🌿 AgriVision-XAI          [sidebar: model info]  │
│  Explainable Plant Disease Detection                 │
├─────────────────────────────────────────────────────┤
│                                                      │
│  ┌──────────────────┐   ┌────────────────────────┐  │
│  │  Upload leaf     │   │  Prediction result     │  │
│  │  image here      │   │  ● Apple___Black_rot   │  │
│  │  [drag & drop]   │   │  Confidence: 94.2%     │  │
│  └──────────────────┘   │  ████████████░░ 94%    │  │
│                          └────────────────────────┘  │
│                                                      │
│  ┌──────────────────────────────────────────────┐   │
│  │  Explanation (Grad-CAM)                      │   │
│  │  [Original] [Grad-CAM] [Grad-CAM++] [LIME]  │   │
│  └──────────────────────────────────────────────┘   │
│                                                      │
│  ┌────────────────────┐  ┌───────────────────────┐  │
│  │  Severity          │  │  Top-3 Predictions    │  │
│  │  ████ MODERATE     │  │  1. Apple Black rot   │  │
│  │  38% infected area │  │  2. Apple Apple scab  │  │
│  └────────────────────┘  │  3. Apple Cedar rust  │  │
│                           └───────────────────────┘  │
└─────────────────────────────────────────────────────┘
```

---

## 4. Sidebar Content

```
Model Info
──────────
Backbone    : EfficientNetB0
Attention   : CBAM
Classes     : 38
Val Accuracy: 95.x%
Dataset     : PlantVillage

──────────
About
──────────
AgriVision-XAI V2
IIIT Ranchi Research Internship
Developer: Jiten Kumar
GitHub: Jiten28/AgriVision-XAI
```

---

## 5. Component Specs

### Upload Area

- Accepts: JPG, PNG, JPEG
- Max size: 10 MB
- Shows thumbnail preview after upload
- Border: dashed, `#1F6F43`, 2px

### Prediction Badge

- Healthy: green badge `#4CAF50`
- Diseased: red badge `#F44336`
- Font: bold, white text

### Confidence Bar

- Streamlit `st.progress()` styled with custom CSS
- Colour: green if > 80%, orange if 50–80%, red if < 50%

### Severity Gauge

- Healthy (0–5%): 🟢 Healthy
- Mild (5–25%): 🟡 Mild
- Moderate (25–50%): 🟠 Moderate
- Severe (50–100%): 🔴 Severe
- Shows % infected area below label

### Grad-CAM Display

- Four tab panels: Original | Grad-CAM | Grad-CAM++ | LIME
- Each image displayed at 300×300 px minimum
- Caption below each showing method name

---

## 6. Streamlit Config (`/.streamlit/config.toml`)

```toml
[theme]
primaryColor = "#1F6F43"
backgroundColor = "#F5F9F6"
secondaryBackgroundColor = "#E7F0EA"
textColor = "#1A1A1A"
font = "sans serif"

[server]
maxUploadSize = 10
```
