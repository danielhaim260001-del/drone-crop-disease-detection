# Illumination Robustness Test

This folder contains the images and results used for the illumination robustness test of the crop disease detection system.

## Test Objective

The purpose of this test is to evaluate the influence of illumination changes on the image-processing and disease-classification algorithm, and to examine the contribution of the preprocessing stage to classification stability.

## Method

Representative field images of potato and sugar beet crops are tested under five illumination conditions:

- -30% brightness
- -20% brightness
- Original image
- +20% brightness
- +30% brightness

Each image is analyzed twice:

1. Without the full preprocessing stage
2. With the full preprocessing stage

The preprocessing pipeline includes Gaussian filtering, CLAHE-based local contrast enhancement, and Value-channel normalization.

## Evaluation Metrics

The test records:

- Mean Brightness
- Brightness Standard Deviation
- Overexposed Pixels [%]
- Underexposed Pixels [%]
- Disease Score
- Classification
- Classification Stability

The purpose is to determine whether preprocessing reduces the sensitivity of the system to illumination variations encountered under field conditions.

## Folder Structure

- `beet_images/` – Sugar beet field images used in the test
- `potato_images/` – Potato field images used in the test
- `results/` – CSV files, plots, and summarized experimental results
