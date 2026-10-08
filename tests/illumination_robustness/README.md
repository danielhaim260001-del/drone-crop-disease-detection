# Crop Health Vision — Illumination Robustness Validation

## Overview

This repository contains the **Illumination Robustness Validation Experiment** developed as part of the **Crop Health Vision** final engineering project.

The goal of this experiment is to evaluate the robustness of the crop disease-detection algorithm under controlled illumination changes, and to quantify the contribution of the image preprocessing stage.

The system analyzes RGB field images of two crops:

- **Potato**
- **Sugar Beet**

The same crop-specific classifier is evaluated using two parallel processing branches:

1. **Without Photometric Preprocessing**
2. **With Full Image Preprocessing**

> This experiment evaluates **robustness to illumination changes**.  
> It is not a disease-classification accuracy experiment against manual Ground Truth.

---

# 1. Experiment Configuration

The final validation experiment included:

| Parameter | Value |
|---|---:|
| Total unique images | **13** |
| Potato images | **7** |
| Sugar Beet images | **6** |
| Illumination levels | **5** |
| Processing branches | **2** |
| Total classifier runs | **130** |

The total number of classifier executions was:

\[
13 \times 5 \times 2 = 130
\]

---

# 2. Synthetic Illumination Changes

Each original image was transformed into five illumination levels:

| Illumination Change | Factor α |
|---:|---:|
| −30% | 0.70 |
| −20% | 0.80 |
| Original | 1.00 |
| +20% | 1.20 |
| +30% | 1.30 |

The synthetic illumination transformation is defined as:

\[
I_{\alpha}(x,y)
=
clip(\alpha \cdot I(x,y),0,255)
\]

where:

- \(I(x,y)\) is the original pixel intensity.
- \(\alpha\) is the illumination factor.
- `clip()` limits the pixel value to the valid RGB range `[0,255]`.

---

# 3. Validation Method

Each image at every illumination level is processed twice.

---

## 3.1 Branch A — Without Photometric Normalization

This branch serves as the experimental baseline.

The image is resized to the standard system resolution but does not undergo illumination normalization.

```text
Input Image
      ↓
Synthetic Illumination Change
      ↓
Resize to 800 × 600
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
```

The resize operation is kept in both branches so that the comparison isolates the effect of the photometric preprocessing rather than changes in image geometry.

---

## 3.2 Branch B — Full Image Preprocessing

The second branch applies the complete preprocessing pipeline used by the final Crop Health Vision system.

```text
Input Image
      ↓
Synthetic Illumination Change
      ↓
Resize to 800 × 600
      ↓
Gaussian Blur
Kernel = 3 × 3
σ = 0.8
      ↓
BGR → LAB
      ↓
CLAHE on L Channel
clipLimit = 2.0
tileGridSize = 8 × 8
      ↓
LAB → BGR
      ↓
BGR → HSV
      ↓
99th Percentile Clipping
on V Channel
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
```

The purpose of the preprocessing stage is to reduce sensitivity to:

- Illumination variation
- Shadows
- Strong sunlight
- Local contrast differences
- Specular highlights
- Overexposed areas

---

# 4. Measured Parameters

For every image and illumination level, the experiment records:

- Mean Brightness
- Brightness Standard Deviation
- Overexposed Pixels [%]
- Underexposed Pixels [%]
- Disease Score
- Final Classification
- Relative Classification Confidence
- Vegetation Coverage [%]
- Classifier Processing Time [s]
- Image Quality Validity

---

# 5. Classification Stability

Classification Stability measures whether the classification remains unchanged when illumination changes.

The classification obtained at the original illumination level is used as the reference.

For each image:

\[
Stable_{\alpha}
=
\begin{cases}
1, & C_{\alpha}=C_{0}\\
0, & C_{\alpha}\neq C_{0}
\end{cases}
\]

where:

- \(C_{\alpha}\) is the classification at illumination level \(\alpha\).
- \(C_{0}\) is the classification of the original image.

Overall Classification Stability is calculated as:

\[
Classification\ Stability[\%]
=
\frac{\sum Stable_{\alpha}}{N}
\times100
\]

The original `0%` illumination level is excluded from the overall robustness calculation because it is the reference itself.

---

# 6. Disease Score Deviation

The experiment also measures how much the numerical Disease Score changes relative to the original image.

\[
\Delta Score_{\alpha}
=
|Score_{\alpha}-Score_{0}|
\]

A lower value indicates that the classifier is less sensitive to illumination changes.

---

# 7. Final Validation Results

## 7.1 Overall Results

| Metric | Without Photometric Normalization | Full Preprocessing | Change |
|---|---:|---:|---:|
| **Classification Stability [%]** | **82.69%** | **84.62%** | **+1.92 pp** |
| **Mean Score Deviation** | **1.27** | **1.02** | **19.70% reduction** |
| **Brightness Range** | **68.68** | **23.66** | **65.55% reduction** |
| **Mean Overexposed Pixels [%]** | **5.04%** | **3.34%** | **33.70% reduction** |
| **Mean Underexposed Pixels [%]** | **1.77%** | **2.06%** | **16.07% increase** |
| **Mean Classifier Processing Time [s]** | **0.080 s** | **0.095 s** | **18.70% increase** |

---

# 8. Brightness Normalization

Before preprocessing, the range of mean brightness values across the five illumination levels was:

\[
\Delta B_{before}
=
155.93-87.25
=
68.68
\]

After full preprocessing:

\[
\Delta B_{after}
=
138.18-114.52
=
23.66
\]

Therefore, the reduction in brightness range was:

\[
Reduction
=
\frac{68.68-23.66}{68.68}
\times100
=
65.55\%
\]

The preprocessing stage therefore reduced the dependence of the image mean brightness on illumination variation by approximately:

\[
\boxed{65.6\%}
\]

---

## 8.1 Variation Between Mean Brightness Levels

The standard deviation between the five mean-brightness values also decreased.

Before preprocessing:

\[
STD_{before}=26.44
\]

After preprocessing:

\[
STD_{after}=8.68
\]

Therefore:

\[
Reduction
=
\frac{26.44-8.68}{26.44}
\times100
\approx67.17\%
\]

This result provides additional evidence that the preprocessing pipeline stabilizes the photometric characteristics of the images.

---

# 9. Results by Illumination Level

| Illumination Change | Stability Without PP | Stability With PP | Score Deviation Without PP | Score Deviation With PP | Mean Brightness Before | Mean Brightness After |
|---:|---:|---:|---:|---:|---:|---:|
| **−30%** | 84.62% | 92.31% | 1.08 | 0.85 | 87.25 | 114.52 |
| **−20%** | 84.62% | 84.62% | 1.00 | 0.85 | 99.84 | 118.37 |
| **0%** | 100.00% | 100.00% | 0.00 | 0.00 | 125.29 | 122.45 |
| **+20%** | 76.92% | 92.31% | 1.23 | 1.00 | 147.33 | 131.55 |
| **+30%** | 84.62% | 69.23% | 1.77 | 1.38 | 155.93 | 138.18 |

The preprocessing stage reduced Disease Score deviation at all tested non-reference illumination levels.

The largest improvement in classification stability occurred at:

- **−30% illumination**
- **+20% illumination**

The `+30%` condition remained the most challenging illumination case.

---

# 10. Results by Crop

## 10.1 Potato

| Metric | Without PP | With PP | Change |
|---|---:|---:|---:|
| Classification Stability | **100.00%** | **89.29%** | **−10.71 pp** |
| Mean Score Deviation | **0.214** | **0.536** | Increase |
| Brightness Range | **76.30** | **23.04** | **69.80% reduction** |

For the Potato subset, the baseline classifier was already highly stable under the tested illumination changes.

The preprocessing pipeline substantially reduced brightness variation, although it did not improve classification stability for this specific image subset.

This demonstrates that improved photometric normalization does not necessarily guarantee improved classification stability for every crop or dataset.

---

## 10.2 Sugar Beet

| Metric | Without PP | With PP | Change |
|---|---:|---:|---:|
| Classification Stability | **62.50%** | **79.17%** | **+16.67 pp** |
| Mean Score Deviation | **2.50** | **1.58** | **36.67% reduction** |
| Brightness Range | **59.80** | **24.40** | **59.20% reduction** |

For Sugar Beet, preprocessing produced a clear improvement in both:

- Classification Stability
- Disease Score robustness

The Classification Stability increased from:

\[
62.50\%
\rightarrow
79.17\%
\]

while Mean Disease Score Deviation decreased from:

\[
2.50
\rightarrow
1.58
\]

This corresponds to an approximately:

\[
36.7\%
\]

reduction in Disease Score sensitivity.

---

# 11. Classification Stability Graph

The experiment generates the following graph:

```text
robustness_classification_stability.png
```

It compares:

- Classification Stability without preprocessing
- Classification Stability with preprocessing

as a function of illumination change.

If the graph is stored inside a `results` folder, it can be displayed in GitHub using:

```markdown
![Classification Stability](results/robustness_classification_stability.png)
```

Example:

![Classification Stability](results/robustness_classification_stability.png)

---

# 12. Disease Score Sensitivity Graph

The second graph is:

```text
robustness_score_deviation.png
```

It compares:

\[
Mean|Disease\ Score_{\alpha}-Disease\ Score_{original}|
\]

with and without preprocessing.

GitHub syntax:

```markdown
![Disease Score Deviation](results/robustness_score_deviation.png)
```

Example:

![Disease Score Deviation](results/robustness_score_deviation.png)

---

# 13. Engineering Interpretation

The validation results demonstrate that the preprocessing stage provides a clear improvement in the **photometric robustness** of the Crop Health Vision system.

The strongest result was observed in the mean-brightness range.

Without preprocessing:

\[
\Delta B = 68.68
\]

With preprocessing:

\[
\Delta B = 23.66
\]

corresponding to a reduction of approximately:

\[
65.55\%
\]

The Mean Disease Score Deviation also decreased:

\[
1.27
\rightarrow
1.02
\]

representing a reduction of approximately:

\[
19.70\%
\]

Overall Classification Stability increased from:

\[
82.69\%
\rightarrow
84.62\%
\]

The improvement was particularly significant for Sugar Beet, where Classification Stability increased by:

\[
+16.67\ percentage\ points
\]

and Mean Score Deviation decreased by approximately:

\[
36.67\%
\]

The preprocessing stage also reduced the average proportion of overexposed pixels from:

\[
5.04\%
\rightarrow
3.34\%
\]

However, the percentage of underexposed pixels increased slightly from:

\[
1.77\%
\rightarrow
2.06\%
\]

This behavior is reasonable because CLAHE increases local contrast and may expand some darker regions while improving the overall illumination normalization.

The results therefore indicate that the preprocessing pipeline improves the robustness of the system to illumination variation, particularly in terms of brightness normalization and Disease Score stability.

---

# 14. Important Interpretation Notes

## 14.1 Robustness Is Not Accuracy

This experiment does **not** evaluate disease-detection accuracy against manual Ground Truth.

Instead, it evaluates whether the algorithm produces similar outputs when the illumination of the same image changes.

Therefore, the experiment measures:

- Robustness
- Stability
- Illumination sensitivity
- Disease Score sensitivity

Metrics such as:

- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix

must be evaluated separately using labeled Ground Truth images.

---

## 14.2 Synthetic Illumination Model

The experiment applies a global illumination multiplier:

\[
I_{\alpha}=\alpha I
\]

This provides a controlled engineering experiment but does not represent every possible lighting condition encountered in a real agricultural field.

Real-world illumination may also include:

- Directional shadows
- Cloud transitions
- Partial shadowing
- Strong localized sun glare
- Camera automatic exposure changes
- Motion blur
- Reflections from wet leaves

Therefore, the experiment should be interpreted as a controlled illumination robustness validation rather than a complete model of all outdoor lighting conditions.

---

## 14.3 Dataset Size

The validation experiment contained:

- 7 Potato images
- 6 Sugar Beet images
- 13 unique images in total

The results provide an engineering validation of the preprocessing approach but should not be interpreted as a large-scale statistical study.

---

# 15. Reproducing the Experiment

## Step 1 — Install Dependencies

```bash
pip install streamlit opencv-python-headless matplotlib numpy pandas
```

---

## Step 2 — Run Crop Health Vision

```bash
python -m streamlit run Crop_Health_Vision_FINAL.py
```

---

## Step 3 — Open Validation Mode

From the sidebar:

```text
PROJECT VALIDATION
        ↓
Illumination robustness test
```

---

## Step 4 — Upload Images

Upload the images into their correct crop groups:

```text
Potato Images
→ Potato uploader

Sugar Beet Images
→ Sugar Beet uploader
```

The same image should not be uploaded into both crop groups.

---

## Step 5 — Run the Experiment

Press:

```text
Run Robustness Test
```

The system automatically:

1. Generates the five illumination variants.
2. Runs the baseline branch.
3. Runs the complete preprocessing branch.
4. Performs crop-specific disease classification.
5. Calculates Classification Stability.
6. Calculates Disease Score Deviation.
7. Generates summary tables.
8. Generates validation graphs.
9. Exports the experiment results.

---

# 16. Experiment Output Files

The system exports:

```text
robustness_detailed_results.csv
robustness_summary.csv
robustness_by_crop.csv
robustness_classification_stability.png
robustness_score_deviation.png
```

---

## 16.1 Detailed Results

`robustness_detailed_results.csv`

Contains one row for each:

```text
Image × Illumination Level
```

Since the experiment contains 13 images and 5 illumination levels:

\[
13\times5=65
\]

rows are generated.

Each row contains results from both processing branches.

---

## 16.2 Summary Results

`robustness_summary.csv`

Contains aggregated results for each illumination level.

---

## 16.3 Results by Crop

`robustness_by_crop.csv`

Contains aggregated results separated by:

```text
Crop × Illumination Level
```

This file enables separate analysis of Potato and Sugar Beet robustness.

---

# 17. Recommended GitHub Repository Structure

```text
illumination-robustness-validation/
│
├── README.md
├── Crop_Health_Vision_FINAL.py
│
├── results/
│   ├── robustness_detailed_results.csv
│   ├── robustness_summary.csv
│   ├── robustness_by_crop.csv
│   ├── robustness_classification_stability.png
│   └── robustness_score_deviation.png
│
└── samples/
    └── README.md
```

---

# 18. Final Conclusion

The Illumination Robustness Validation demonstrated that the preprocessing stage of Crop Health Vision substantially improves the photometric consistency of field images under controlled illumination variation.

The preprocessing pipeline achieved:

- **65.55% reduction in brightness range**
- **67.17% reduction in variation between mean brightness levels**
- **19.70% reduction in overall Disease Score deviation**
- **33.70% reduction in overexposed pixels**
- **+1.92 percentage-point improvement in overall Classification Stability**

The strongest classification improvement was observed for Sugar Beet:

- Stability increased from **62.50% to 79.17%**
- Disease Score deviation decreased from **2.50 to 1.58**

The experiment therefore supports the use of:

```text
Gaussian Blur
      +
CLAHE
      +
99th-percentile brightness clipping
```

as a preprocessing stage for improving the robustness of the Crop Health Vision system under variable outdoor illumination.

---

# Crop Health Vision

**Drone-Based Smart Vision System for Detection of Plant Diseases in Agricultural Fields**

The project uses RGB drone imagery and classical computer-vision techniques for:

- Image preprocessing
- Vegetation segmentation
- Disease-feature extraction
- Rule-based disease classification
- Field-grid analysis
- Heat-map visualization
- Batch image analysis
- Engineering validation

Supported crops:

```text
Potato
Sugar Beet
```
