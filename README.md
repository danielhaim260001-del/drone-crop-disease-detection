# Crop Health Vision

**Drone-Based Smart Vision System for Detection of Plant Diseases in Agricultural Fields**

Final B.Sc. Electrical Engineering project developed at Afeka College of Engineering.

The system uses RGB drone imagery and classical computer-vision techniques to detect visual signs of crop disease and present the results using color-coded field maps and heatmaps.

## Supported Crops

- Potato
- Sugar Beet

## System Pipeline

```text
RGB Image Acquisition
        ↓
Image Preprocessing
        ↓
Plant Segmentation
        ↓
Feature Extraction
        ↓
Disease Score
        ↓
Classification
        ↓
Healthy / Suspicious / Sick
        ↓
Field Grid + Heatmap
```

The preprocessing stage includes:

- Resize to 800×600
- Gaussian Blur
- LAB color-space conversion
- CLAHE contrast enhancement
- HSV conversion
- Brightness clipping using the 99th percentile

The system operates offline and does not require an Internet connection during image analysis.

---

## Classification

The final output contains four possible system states:

- `Healthy`
- `Suspicious`
- `Sick`
- `No Plant`

For binary performance evaluation:

```text
Healthy = Negative
Suspicious + Sick = Positive
```

This reflects the purpose of the system as a screening tool: both suspicious and sick areas require user attention.

---

# Field Experiments

Three field experiments were used to evaluate the final system.

## Experiment 1 — Potato

| Metric | Result |
|---|---:|
| Images | 37 |
| TP | 7 |
| TN | 24 |
| FP | 6 |
| FN | 0 |
| Accuracy | **83.8%** |
| Precision | **53.8%** |
| Recall | **100.0%** |
| F1 | **70.0%** |

The final algorithm detected all diseased samples in this experiment, with no False Negatives.

---

## Experiment 2 — Potato

| Metric | Result |
|---|---:|
| Images | 37 |
| TP | 9 |
| TN | 27 |
| FP | 1 |
| FN | 0 |
| Accuracy | **97.3%** |
| Precision | **90.0%** |
| Recall | **100.0%** |
| F1 | **94.7%** |

Experiment 2 achieved the highest classification performance and no diseased samples were missed.

---

## Experiment 3 — Potato + Sugar Beet

The Potato subset contained 12 images and retained **100% classification accuracy**.

The Sugar Beet classifier was re-evaluated using 29 images.

### Sugar Beet Results

| Metric | Result |
|---|---:|
| Images | 29 |
| TP | 14 |
| TN | 7 |
| FP | 7 |
| FN | 1 |
| Accuracy | **72.4%** |
| Precision | **66.7%** |
| Recall | **93.3%** |
| F1 | **77.8%** |

### Combined Experiment 3

| Metric | Result |
|---|---:|
| Images | 41 |
| TP | 17 |
| TN | 16 |
| FP | 7 |
| FN | 1 |
| Accuracy | **80.5%** |
| Precision | **70.8%** |
| Recall | **94.4%** |
| F1 | **81.0%** |

---

# Overall Classification Performance

Across all three updated field experiments:

```text
Total images = 115
TP = 33
TN = 67
FP = 14
FN = 1
```

| Metric | Overall Result |
|---|---:|
| Accuracy | **87.0%** |
| Precision | **70.2%** |
| Recall | **97.1%** |
| F1 | **81.5%** |

The project design target was approximately:

```text
Accuracy ≥ 85%
```

The final system therefore met the overall classification-accuracy target.

The high Recall indicates that the system detected nearly all diseased samples, while most remaining errors were False Positives rather than missed disease cases.

---

# Illumination Robustness Validation

A separate engineering validation experiment was performed to evaluate sensitivity to illumination changes.

The experiment used:

- 13 field images
- 7 Potato images
- 6 Sugar Beet images
- 5 illumination levels
- 2 processing branches
- 130 classifier executions

Illumination levels:

```text
-30%
-20%
Original
+20%
+30%
```

Two processing branches were compared:

```text
Without Photometric Normalization
vs.
Full Preprocessing
```

## Main Validation Results

| Metric | Without Preprocessing | Full Preprocessing |
|---|---:|---:|
| Classification Stability | 82.69% | **84.62%** |
| Mean Score Deviation | 1.27 | **1.02** |
| Brightness Range | 68.68 | **23.66** |
| Mean Overexposed Pixels | 5.04% | **3.34%** |

The preprocessing pipeline reduced the brightness range by approximately:

```text
65.55%
```

and reduced Disease Score sensitivity by approximately:

```text
19.70%
```

The strongest robustness improvement was observed for Sugar Beet.

---

# User Interface

The Streamlit interface provides:

- Crop selection
- Single-image analysis
- Batch image analysis
- Field-grid visualization
- Disease heatmap
- Segmentation-quality assessment
- Ground Truth validation
- CSV export
- Illumination robustness testing

---

# Installation

Install the required Python packages:

```bash
pip install streamlit opencv-python-headless matplotlib numpy pandas
```

Run the application:

```bash
python -m streamlit run Crop_Health_Vision_FINAL.py
```

---

# Main Technologies

- Python
- OpenCV
- NumPy
- Pandas
- Matplotlib
- Streamlit

---

# Project Team

**Daniel Haim**  
**Ofri Azar**  
**Ido Dayan**

Afeka College of Engineering  
B.Sc. Electrical Engineering

---

## Disclaimer

The system is an engineering computer-vision prototype intended for crop screening and field monitoring.

A `Suspicious` or `Sick` result indicates an area requiring further attention and does not replace professional agronomic diagnosis.
