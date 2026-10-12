# Crop Health Vision – Software

This folder contains the software implementation of the **Crop Health Vision** final engineering project.

The system analyzes RGB drone and field images of **Potato** and **Sugar Beet** crops using classical computer-vision methods.

## Files

### `Crop_Health_Vision_FINAL.py`
Main Streamlit application.

Includes:
- User interface
- Image upload
- Crop selection
- Image preprocessing
- Plant segmentation
- Disease classification
- Single-image analysis
- Batch analysis
- Field Grid
- Heatmap visualization
- Results export
- Validation tools

### `potato_disease_detector.py`
Potato image-processing and classification logic.

Main stages:
- Plant segmentation
- Color-feature extraction
- Disease Score calculation
- Healthy / Suspicious / Sick classification

### `beet_disease_detector.py`
Sugar Beet image-processing and classification logic.

Includes:
- Leaf segmentation
- Yellow / bleached area detection
- Dark lesion detection
- Cercospora-related visual feature detection
- Disease Score calculation
- Classification

### `illumination_robustness_validation.py`
Engineering validation experiment for testing algorithm robustness under illumination changes.

The experiment compares:
- Processing without photometric normalization
- Full preprocessing pipeline

Illumination levels:

```text
-30%
-20%
Original
+20%
+30%
```

### `requirements.txt`
Python packages required to run the software.

---

## Processing Pipeline

```text
RGB Image
   ↓
Image Quality Check
   ↓
Preprocessing
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
Grid / Heatmap / Export
```

## Installation

From the project directory:

```bash
pip install -r software/requirements.txt
```

## Run

```bash
python -m streamlit run software/Crop_Health_Vision_FINAL.py
```

## Technology

- Python
- OpenCV
- NumPy
- Pandas
- Matplotlib
- Streamlit

## Notes

The system uses classical computer vision and a rule-based classifier.

It does **not** use a trained machine-learning or deep-learning model.

The output is intended as a crop-screening and decision-support tool and not as a definitive agronomic diagnosis.
