# Crop Health Vision

**Drone-Based Smart Vision System for Detection of Plant Diseases in Agricultural Fields**

Final B.Sc. Electrical Engineering project developed at Afeka College of Engineering.

Crop Health Vision is an offline computer-vision system that analyzes RGB drone and field images to identify visual signs associated with crop disease.

The system uses **classical image-processing techniques and a rule-based classifier**, without a trained Machine Learning or Deep Learning model.

---

## Supported Crops

- Potato
- Sugar Beet

---

## System Pipeline

```text
Drone + RGB Camera
        ↓
Image Acquisition
        ↓
Local Image Storage
        ↓
Transfer to Ground Computer
        ↓
Image Quality Check
        ↓
Image Preprocessing
        ↓
Plant Segmentation
        ↓
Crop-Specific Feature Extraction
        ↓
Disease Score
        ↓
Classification
        ↓
Healthy / Suspicious / Sick / No Plant
        ↓
Field Grid + Heatmap
        ↓
User Interface + Results Export
```

The preprocessing stage includes:

- Resize to 800×600
- Gaussian Blur
- LAB color-space conversion
- CLAHE contrast enhancement
- HSV conversion
- Brightness clipping using the 99th percentile

All image processing is performed locally on the ground computer after image acquisition.

---

## Classification

The system provides four possible outputs:

- `Healthy`
- `Suspicious`
- `Sick`
- `No Plant`

For binary performance evaluation:

```text
Healthy = Negative
Suspicious + Sick = Positive
```

`Suspicious` and `Sick` are both considered positive detections because both indicate that the area requires additional attention.

---

# Field Experiments

Three field experiments were used to evaluate the final version of the system.

## Experiment 1 — Potato

| Metric | Result |
|---|---:|
| Evaluated Images | 37 |
| TP | 7 |
| TN | 24 |
| FP | 6 |
| FN | 0 |
| Accuracy | **83.8%** |
| Precision | **53.8%** |
| Recall | **100.0%** |
| F1 | **70.0%** |

All diseased samples were detected in Experiment 1, with no False Negatives.

---

## Experiment 2 — Potato

| Metric | Result |
|---|---:|
| Evaluated Images | 37 |
| TP | 9 |
| TN | 27 |
| FP | 1 |
| FN | 0 |
| Accuracy | **97.3%** |
| Precision | **90.0%** |
| Recall | **100.0%** |
| F1 | **94.7%** |

Experiment 2 achieved the highest classification accuracy and no diseased samples were missed.

---

## Experiment 3 — Sugar Beet

| Metric | Result |
|---|---:|
| Evaluated Images | 29 |
| TP | 14 |
| TN | 7 |
| FP | 7 |
| FN | 1 |
| Accuracy | **72.4%** |
| Precision | **66.7%** |
| Recall | **93.3%** |
| F1 | **77.8%** |

The Sugar Beet experiment achieved high Recall, while most classification errors were False Positives.

---

# Overall Classification Performance

Across the three final field experiments:

```text
Total evaluated images = 103

TP = 30
TN = 58
FP = 14
FN = 1
```

| Metric | Overall Result |
|---|---:|
| Accuracy | **85.4%** |
| Precision | **68.2%** |
| Recall | **96.8%** |
| F1 | **80.0%** |

The project design target was:

```text
Accuracy ≥ 85%
```

The final system therefore met the overall classification-accuracy target.

Only **1 diseased image was missed**, while **14 of the 15 classification errors were False Positives**.

This indicates that the final classifier favors high disease-detection sensitivity, even at the cost of additional false alarms.

---

# Illumination Robustness Validation

A separate engineering validation experiment was performed to evaluate the effect of illumination changes on the algorithm.

The experiment used:

- 13 representative field images
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
| Mean Underexposed Pixels | 1.77% | 2.06% |
| Mean Classifier Processing Time | 0.080 s | 0.095 s |

The preprocessing pipeline reduced the brightness range by approximately:

```text
65.55%
```

and reduced the mean Disease Score deviation by approximately:

```text
19.70%
```

Classification stability increased from:

```text
82.69% → 84.62%
```

The strongest improvement in classification stability was observed for Sugar Beet.

---

# User Interface

The system includes a Streamlit-based graphical user interface.

Main capabilities:

- Crop selection
- Image upload
- Single-image analysis
- Batch image analysis
- Image Quality Check
- Plant segmentation visualization
- Disease Score and classification
- Relative score-based confidence
- Field Grid visualization
- Disease Heatmap
- Segmentation-quality assessment
- Ground Truth validation
- CSV and ZIP export
- Illumination Robustness Test
- Configurable Grid resolution

---

# Installation

Install the required Python packages:

```bash
pip install streamlit opencv-python-headless matplotlib numpy pandas
```

---

# Run the Application

```bash
python -m streamlit run Crop_Health_Vision_FINAL.py
```

The Streamlit interface will open in the browser.

---

# Main Technologies

- Python
- OpenCV
- NumPy
- Pandas
- Matplotlib
- Streamlit

---

# System Characteristics

- RGB imaging only
- Offline processing
- No cloud dependency
- No trained ML/DL model
- Rule-based Disease Score classification
- Crop-specific processing paths
- Explainable classification logic
- Spatial visualization using Grid and Heatmap

---

# Project Team

**Daniel Haim**  
**Ofri Azar**  
**Ido Dayan**

Afeka College of Engineering  
B.Sc. Electrical Engineering

---

## Disclaimer

Crop Health Vision is an engineering computer-vision prototype intended for crop screening and field monitoring.

A `Suspicious` or `Sick` result indicates an area requiring further inspection and does not constitute a definitive agronomic diagnosis or treatment recommendation.
