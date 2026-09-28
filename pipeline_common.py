"""
Shared image-processing utilities
=================================
Reusable functions for potato and beet pipelines.

These utilities are intentionally crop-independent. Crop-specific
segmentation and classification remain in potato_detector.py and
beet_detector.py.
"""

from __future__ import annotations

import cv2
import numpy as np


def ensure_bgr(image: np.ndarray) -> np.ndarray:
    """Return a valid 3-channel BGR image."""
    if image is None or image.size == 0:
        raise ValueError("Empty image received.")
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if image.ndim == 3 and image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    if image.ndim == 3 and image.shape[2] == 3:
        return image
    raise ValueError(f"Unsupported image shape: {image.shape}")


def quality_metrics(image_bgr: np.ndarray) -> dict:
    """
    Calculate reproducible image-quality indicators.

    Returns grayscale brightness, contrast, Laplacian sharpness,
    and percentages of very dark / very bright pixels.
    """
    image_bgr = ensure_bgr(image_bgr)
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

    mean_gray = float(np.mean(gray))
    std_gray = float(np.std(gray))
    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    dark_pct = float(np.mean(gray < 25) * 100.0)
    bright_pct = float(np.mean(gray > 245) * 100.0)

    hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).ravel()
    prob = hist / max(hist.sum(), 1.0)
    nz = prob[prob > 0]
    entropy = float(-(nz * np.log2(nz)).sum())

    return {
        "mean_gray": round(mean_gray, 3),
        "std_gray": round(std_gray, 3),
        "sharpness_laplacian": round(sharpness, 3),
        "dark_pixels_pct": round(dark_pct, 3),
        "bright_pixels_pct": round(bright_pct, 3),
        "entropy_bits": round(entropy, 4),
    }


def preprocess_image(image_bgr: np.ndarray) -> dict:
    """
    Produce standard intermediate outputs for documentation and debugging.

    Stages:
      original -> grayscale -> denoised -> CLAHE -> HSV
    """
    image_bgr = ensure_bgr(image_bgr)
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    denoised = cv2.GaussianBlur(image_bgr, (5, 5), 0)

    lab = cv2.cvtColor(denoised, cv2.COLOR_BGR2LAB)
    l_chan, a_chan, b_chan = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_eq = clahe.apply(l_chan)
    enhanced = cv2.cvtColor(cv2.merge((l_eq, a_chan, b_chan)), cv2.COLOR_LAB2BGR)

    hsv = cv2.cvtColor(enhanced, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    return {
        "original": image_bgr,
        "grayscale": gray,
        "denoised": denoised,
        "enhanced": enhanced,
        "hsv": hsv,
        "h_channel": h,
        "s_channel": s,
        "v_channel": v,
    }


def clean_binary_mask(mask: np.ndarray,
                      open_kernel: int = 3,
                      close_kernel: int = 7,
                      open_iterations: int = 1,
                      close_iterations: int = 2) -> np.ndarray:
    """Apply opening and closing to a binary mask."""
    mask = (mask > 0).astype(np.uint8) * 255

    k_open = np.ones((open_kernel, open_kernel), np.uint8)
    k_close = np.ones((close_kernel, close_kernel), np.uint8)

    cleaned = cv2.morphologyEx(
        mask, cv2.MORPH_OPEN, k_open, iterations=open_iterations
    )
    cleaned = cv2.morphologyEx(
        cleaned, cv2.MORPH_CLOSE, k_close, iterations=close_iterations
    )
    return cleaned


def apply_mask(image_bgr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Clip an image to a binary vegetation/leaf mask."""
    image_bgr = ensure_bgr(image_bgr)
    mask = (mask > 0).astype(np.uint8) * 255
    return cv2.bitwise_and(image_bgr, image_bgr, mask=mask)


def rgb_statistics(image_bgr: np.ndarray, mask: np.ndarray | None = None) -> dict:
    """Mean and standard deviation of RGB channels, optionally inside a mask."""
    image_bgr = ensure_bgr(image_bgr)
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    if mask is None:
        pixels = rgb.reshape(-1, 3)
    else:
        pixels = rgb[mask > 0]

    if pixels.size == 0:
        return {k: 0.0 for k in
                ("mean_r", "mean_g", "mean_b", "std_r", "std_g", "std_b")}

    means = pixels.mean(axis=0)
    stds = pixels.std(axis=0)
    return {
        "mean_r": round(float(means[0]), 3),
        "mean_g": round(float(means[1]), 3),
        "mean_b": round(float(means[2]), 3),
        "std_r": round(float(stds[0]), 3),
        "std_g": round(float(stds[1]), 3),
        "std_b": round(float(stds[2]), 3),
    }


def hsv_statistics(image_bgr: np.ndarray, mask: np.ndarray | None = None) -> dict:
    """Mean and standard deviation of OpenCV HSV channels."""
    image_bgr = ensure_bgr(image_bgr)
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)

    if mask is None:
        pixels = hsv.reshape(-1, 3)
    else:
        pixels = hsv[mask > 0]

    if pixels.size == 0:
        return {k: 0.0 for k in
                ("mean_h", "mean_s", "mean_v", "std_h", "std_s", "std_v")}

    means = pixels.mean(axis=0)
    stds = pixels.std(axis=0)
    return {
        "mean_h": round(float(means[0]), 3),
        "mean_s": round(float(means[1]), 3),
        "mean_v": round(float(means[2]), 3),
        "std_h": round(float(stds[0]), 3),
        "std_s": round(float(stds[1]), 3),
        "std_v": round(float(stds[2]), 3),
    }


def channel_histograms(image_bgr: np.ndarray,
                       mask: np.ndarray | None = None,
                       bins: int = 256) -> dict:
    """
    Return normalized RGB and HSV histograms as arrays.
    These can be plotted by the GUI or exported for analysis.
    """
    image_bgr = ensure_bgr(image_bgr)
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)

    cv_mask = None if mask is None else ((mask > 0).astype(np.uint8) * 255)

    result = {}
    for idx, name in enumerate(("R", "G", "B")):
        hist = cv2.calcHist([rgb], [idx], cv_mask, [bins], [0, 256]).ravel()
        result[f"rgb_{name.lower()}"] = hist / max(hist.sum(), 1.0)

    # H uses OpenCV's [0,180) range.
    h_hist = cv2.calcHist([hsv], [0], cv_mask, [180], [0, 180]).ravel()
    s_hist = cv2.calcHist([hsv], [1], cv_mask, [256], [0, 256]).ravel()
    v_hist = cv2.calcHist([hsv], [2], cv_mask, [256], [0, 256]).ravel()

    result["hsv_h"] = h_hist / max(h_hist.sum(), 1.0)
    result["hsv_s"] = s_hist / max(s_hist.sum(), 1.0)
    result["hsv_v"] = v_hist / max(v_hist.sum(), 1.0)
    return result
