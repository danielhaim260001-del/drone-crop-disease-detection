# Drone-Based Crop Disease Detection

Final Year Project – B.Sc. Electrical Engineering  
Afeka Academic College of Engineering

## Overview

This project presents a drone-based RGB image-processing system for detecting visually suspicious areas in agricultural crops.

The system combines aerial image acquisition using a commercial drone with local image processing on a ground computer. The project focuses on two crops:

- Potato
- Beet

The system is designed as a decision-support tool for farmers. It identifies visual abnormalities in crop imagery and classifies regions into three levels:

- 🟢 Healthy
- 🟡 Suspicious
- 🔴 High Suspicion

The system does not provide a definitive agronomic diagnosis.

---

## Project Team

- Daniel Haim
- Ofri Azer
- Ido Dayan

Project Advisor:

Yirmiyahu Hauptman

---

## Target Crops and Classes

### Potato

The potato image-processing pipeline is evaluated using:

- Healthy
- Early Blight
- Late Blight

### Beet

The beet image-processing pipeline is evaluated using:

- Healthy / Unaffected
- Cercospora Leaf Spot

The potato and beet pipelines are processed separately because the crops have different visual characteristics, color distributions, leaf structures, and disease symptoms.

---

## System Pipeline

The main processing pipeline is:

1. RGB image acquisition
2. Image quality assessment
3. Grayscale conversion
4. Image preprocessing
5. Noise reduction
6. RGB to HSV conversion
7. Vegetation segmentation
8. Morphological operations
9. Vegetation masking / clipping
10. Color feature extraction
11. Texture feature extraction
12. RGB and HSV histogram analysis
13. Feature-vector generation
14. Crop-specific classification
15. Visual overlay generation
16. Spatial field mapping
17. User-interface presentation
18. CSV and report export

Intermediate outputs are preserved so that every processing stage can be inspected and evaluated.

---

## Main Output

The main user-facing output is a visual agricultural field map.

Regions are represented using:

- Green – Healthy
- Yellow – Suspicious
- Red – High Suspicion

Additional technical views are available for image-processing analysis, including masks, grayscale images, HSV channels, histograms, extracted features and classification results.

---

## Datasets

### Potato Dataset

PlantVillage dataset.

The project uses potato images from the following classes:

- Healthy
- Early Blight
- Late Blight

### Beet Dataset

Plant image classification dataset – Zenodo.

The beet portion contains:

- Unaffected
- Cercospora

Zenodo record: 15873360.

### Field Images

Real agricultural field images were collected by the project team during three field-photography sessions.

- Experiment 1 – Potato
- Experiment 2 – Potato
- Experiment 3 – Potato
- Experiment 3 – Beet

The public datasets are primarily used for algorithm development and calibration, while the field images are used to evaluate the system under realistic agricultural conditions.

Full external datasets are not redistributed in this repository. Representative examples and dataset documentation are included.

---

## Extracted Features

The image-processing pipeline extracts features including:

- Mean RGB values
- RGB standard deviation
- Mean HSV values
- HSV standard deviation
- Vegetation percentage
- Green-pixel percentage
- Yellow-pixel percentage
- Brown-pixel percentage
- Dark-region percentage
- Color histograms
- Texture contrast
- Texture homogeneity
- Texture energy
- Texture correlation
- Entropy

---

## Image Quality Analysis

Each input image can also be evaluated using:

- Sharpness
- Mean brightness
- Dark-pixel percentage
- Overexposed-pixel percentage

Images with insufficient quality can be flagged before classification.

---

## Repository Structure

```text
data/         Dataset samples, field images and metadata
docs/         Technical documentation and system diagrams
outputs/      Example algorithm outputs
screenshots/  GUI and processing-stage screenshots
tests/        Algorithm tests
