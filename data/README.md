# Project Data

This directory contains representative image samples, field images and dataset metadata used in the project.

## External datasets

### Potato

Source: PlantVillage

Classes used:

- Healthy
- Early Blight
- Late Blight

### Beet

Source: Plant image classification dataset, Zenodo record 15873360.

Classes used:

- Unaffected
- Cercospora

The complete public datasets are not redistributed in this repository.

Only representative examples required to demonstrate and test the image-processing pipeline are included.

## Field Data

Field images were collected by the project team during three photography sessions:

- Experiment 1 – Potato
- Experiment 2 – Potato
- Experiment 3 – Potato and Beet

Field images differ from the public datasets because they contain natural background, soil, shadows, overlapping vegetation and varying illumination.

## Ground Truth

Dataset labels should always be stored separately from the algorithm prediction.

The algorithm output must not overwrite the original ground-truth label.
