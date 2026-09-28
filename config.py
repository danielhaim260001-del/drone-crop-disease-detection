"""
Central project configuration
=============================
Field Disease Detector — Final Project

This file centralizes project-wide names and baseline settings.
Disease-classification thresholds are still being calibrated and should
not be presented as final validated values until the evaluation stage.
"""

APP_NAME = "Field Disease Detector"
APP_VERSION = "0.1.0-baseline"

CROPS = {
    "Potato": {
        "display_name": "Potato",
        "dataset": "PlantVillage",
        "classes": ["Healthy", "Early Blight", "Late Blight"],
    },
    "Sugar Beet": {
        "display_name": "Sugar Beet",
        "dataset": "Zenodo",
        "classes": ["Healthy / Unaffected", "Cercospora Leaf Spot"],
    },
}

OUTPUT_CLASSES = {
    "Healthy": {
        "display": "Healthy",
        "description": "No strong visual anomaly detected by the current pipeline.",
    },
    "Suspicious": {
        "display": "Suspicious",
        "description": "Visual characteristics require additional inspection.",
    },
    "Sick": {
        "display": "High Suspicion",
        "description": "Strong visual anomaly detected; agronomic inspection is recommended.",
    },
    "No Plant": {
        "display": "No Plant",
        "description": "Insufficient vegetation detected in the analyzed region.",
    },
}

# Recommended default for field-grid visualization.
DEFAULT_GRID_ROWS = 10
DEFAULT_GRID_COLS = 10

# Important project statement used in documentation/UI.
DISCLAIMER = (
    "This system is an engineering decision-support prototype. "
    "Its output is not a definitive agronomic diagnosis."
)
