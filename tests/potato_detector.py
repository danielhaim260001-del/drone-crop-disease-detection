"""
=============================================================
  Potato Disease Detection — 3-Way Classification
  Pipeline per academic spec:
    9.1.3.5  Pre-processing      — image quality improvement
    9.1.3.6  Plant mask          — isolate leaf from background
    9.1.3.7  Feature extraction  — histogram & texture features
    9.1.3.8  Classification      — Green / Yellow / Red
    9.1.3.9  Geographic output   — P = (Latitude, Longitude, Class)

  Classifies potato leaf images as:
    Healthy (Green) | Suspicious (Yellow) | Sick (Red)

  Usage:
    python potato_disease_detector_v2.py <image.jpg>
    python potato_disease_detector_v2.py <folder/>
=============================================================
"""

import cv2
import numpy as np
import glob
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from skimage.feature import graycomatrix, graycoprops
from pathlib import Path


# ══════════════════════════════════════════════════════════
# STEP 9.1.3.5 — PRE-PROCESSING
# Improve image quality before anomaly identification.
# Adjusts for lighting conditions, noise, and contrast issues
# that arise from different shooting distances, angles, and
# weather conditions (sun glare, wind blur).
# Output: processed image ready for mask + feature extraction
# ══════════════════════════════════════════════════════════

def preprocess(img_bgr: np.ndarray) -> np.ndarray:
    """
    Image quality improvement pipeline:

    1. Resize to standard resolution (800×600) — consistent processing
       regardless of camera distance or zoom level.

    2. Gaussian blur — reduce high-frequency noise from wind motion
       or low-quality cameras. Kernel 3×3 preserves edge detail while
       smoothing sensor noise.

    3. CLAHE (Contrast-Limited Adaptive Histogram Equalization) —
       improve contrast in shadowed or overexposed patches.
       Applied on L-channel in LAB space so hue is not distorted.
       clipLimit=2.0 prevents noise amplification; tileGridSize 8×8.

    4. Value-channel clipping — cap the top 1% of brightness values
       to suppress specular highlights (sun glints on wet leaf surface).

    Returns:
        processed BGR image (uint8, 800×600)
    """
    # 1. Resize to standard resolution
    img = cv2.resize(img_bgr, (800, 600), interpolation=cv2.INTER_AREA)

    # 2. Gaussian blur — noise reduction
    img = cv2.GaussianBlur(img, (3, 3), sigmaX=0.8)

    # 3. CLAHE on L-channel in LAB space
    lab   = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_eq  = clahe.apply(l)
    lab_eq = cv2.merge([l_eq, a, b])
    img   = cv2.cvtColor(lab_eq, cv2.COLOR_LAB2BGR)

    # 4. Brightness clipping — suppress top-1% specular highlights
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h_ch, s_ch, v_ch = cv2.split(hsv)
    v_clip = np.clip(v_ch, 0, int(np.percentile(v_ch, 99))).astype(np.uint8)
    hsv_clipped = cv2.merge([h_ch, s_ch, v_clip])
    img = cv2.cvtColor(hsv_clipped, cv2.COLOR_HSV2BGR)

    return img


# ══════════════════════════════════════════════════════════
# STEP 9.1.3.6 — PLANT MASK
# Separate the leaf from the background.
# Output: binary mask where 255 = leaf pixel
# ══════════════════════════════════════════════════════════

def build_plant_mask(img_bgr: np.ndarray) -> dict:
    """
    Isolate the plant/leaf from the background.

    Strategy:
      1. Build a broad vegetation mask covering green (healthy),
         brown (Early Blight spots), and yellow (chlorosis) hues
      2. Morphological cleanup to fill holes and remove noise
      3. Keep only the largest connected blob (main leaf body)
      4. Return the clean leaf mask + masked image

    Returns:
        dict with:
          'leaf_mask'    — binary mask (uint8), 255 = leaf pixel
          'leaf_img'     — BGR image with background zeroed
          'leaf_area'    — number of leaf pixels
          'coverage_pct' — leaf area as % of total image
          'found'        — False if no significant plant found
    """
    h, w = img_bgr.shape[:2]
    total = h * w
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

    # Broad mask: green (healthy) + brown (Early Blight) + yellow (chlorosis)
    # Deliberately wide so diseased tissue at edges is not missed
    green_mask  = cv2.inRange(hsv, np.array([15, 20, 20]),  np.array([100, 255, 255]))
    brown_mask  = cv2.inRange(hsv, np.array([0,  40, 30]),  np.array([22,  255, 200]))
    yellow_mask = cv2.inRange(hsv, np.array([20, 60, 80]),  np.array([35,  255, 255]))
    dark_mask   = cv2.inRange(hsv, np.array([0,   0,  0]),  np.array([180,  80, 100]))

    combined = cv2.bitwise_or(
        cv2.bitwise_or(green_mask, brown_mask),
        cv2.bitwise_or(yellow_mask, dark_mask)
    )

    # Morphological cleanup
    kernel   = np.ones((7, 7), np.uint8)
    combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel, iterations=4)
    combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN,  kernel, iterations=2)

    # Keep only the largest contour (main leaf body)
    contours, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    leaf_mask = np.zeros((h, w), dtype=np.uint8)
    if contours:
        largest = max(contours, key=cv2.contourArea)
        cv2.drawContours(leaf_mask, [largest], -1, 255, -1)

    leaf_area    = int(leaf_mask.sum() // 255)
    coverage_pct = leaf_area / total * 100

    if coverage_pct < 5:
        return {
            "leaf_mask":    leaf_mask,
            "leaf_img":     img_bgr.copy(),
            "leaf_area":    leaf_area,
            "coverage_pct": coverage_pct,
            "found":        False,
        }

    leaf_img = cv2.bitwise_and(img_bgr, img_bgr, mask=leaf_mask)

    return {
        "leaf_mask":    leaf_mask,
        "leaf_img":     leaf_img,
        "leaf_area":    leaf_area,
        "coverage_pct": round(coverage_pct, 1),
        "found":        True,
    }


# ══════════════════════════════════════════════════════════
# STEP 9.1.3.7 — FEATURE EXTRACTION
# Measure features from plant area only.
# Output: feature vector X = [f1, f2, ..., fn]
# ══════════════════════════════════════════════════════════

def extract_features(img_bgr: np.ndarray, mask: dict) -> dict:
    """
    Extract histogram-based color and texture features from the
    segmented leaf area only. Background pixels excluded from all
    calculations.

    Feature vector X = [
      # Color ratio features
      f1  brown_ratio   — % of leaf pixels that are brown (Early Blight spots)
      f2  dark_ratio    — % of leaf pixels that are dark/necrotic (Late Blight)
      f3  yellow_ratio  — % of leaf pixels that are yellow (chlorosis)
      f4  green_ratio   — % of leaf pixels that are healthy green

      # Histogram-based saturation features
      f5  mean_sat      — mean HSV saturation within leaf
      f6  std_sat       — saturation standard deviation (disease spreads values)
      f7  low_sat_ratio — % of leaf pixels with low saturation (bleached/necrotic)

      # Value (brightness) histogram stats
      f8  mean_val      — mean brightness within leaf
      f9  std_val       — brightness std (spots create dark patches in bright leaf)

      # Hue histogram entropy (color diversity)
      f10 hue_entropy   — entropy of hue distribution (healthy=low, sick=high)

      # Hue histogram bins (20 bins, leaf pixels only) — f11..f30
      hue_hist_b0..b19  — normalized frequency per hue band

      # GLCM texture features (spot pattern detection)
      f31 homogeneity   — texture smoothness (lower = more textured = more spots)
      f32 contrast      — local texture contrast

      # Coverage
      f33 coverage_pct  — leaf area / total image area
    ]
    """
    if not mask["found"]:
        zero = {k: 0.0 for k in
                ("brown_ratio", "dark_ratio", "yellow_ratio", "green_ratio",
                 "mean_sat", "std_sat", "low_sat_ratio",
                 "mean_val", "std_val", "hue_entropy",
                 "homogeneity", "contrast", "coverage_pct")}
        for i in range(20):
            zero[f"hue_hist_b{i}"] = 0.0
        return zero

    leaf_mask = mask["leaf_mask"]
    leaf_area = max(mask["leaf_area"], 1)
    hsv  = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    hh, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]

    def _in_leaf(mask_bin):
        return cv2.bitwise_and(mask_bin, leaf_mask)

    # ── f1: Brown spots (Early Blight) ────────────────────────────
    b1 = cv2.inRange(hsv, np.array([10, 70, 50]), np.array([22, 255, 200]))
    b2 = cv2.inRange(hsv, np.array([0,  70, 50]), np.array([10, 255, 200]))
    brown_px    = _in_leaf(cv2.bitwise_or(b1, b2))
    brown_ratio = np.sum(brown_px > 0) / leaf_area * 100

    # ── f2: Dark spots (Late Blight) ──────────────────────────────
    dark_raw = cv2.inRange(hsv, np.array([0, 0, 0]), np.array([180, 60, 80]))
    dark_px  = cv2.morphologyEx(_in_leaf(dark_raw), cv2.MORPH_OPEN, np.ones((7, 7), np.uint8))
    dark_ratio = np.sum(dark_px > 0) / leaf_area * 100

    # ── f3: Yellow chlorosis ──────────────────────────────────────
    yellow_px    = _in_leaf(cv2.inRange(hsv, np.array([20, 60, 100]), np.array([35, 255, 255])))
    yellow_ratio = np.sum(yellow_px > 0) / leaf_area * 100

    # ── f4: Healthy green ─────────────────────────────────────────
    green_px    = _in_leaf(cv2.inRange(hsv, np.array([25, 40, 40]), np.array([90, 255, 255])))
    green_ratio = np.sum(green_px > 0) / leaf_area * 100

    # ── f5, f6: Saturation histogram stats (leaf pixels) ──────────
    s_leaf  = s[leaf_mask > 0].astype(float)
    mean_sat = float(np.mean(s_leaf)) if len(s_leaf) > 0 else 0.0
    std_sat  = float(np.std(s_leaf))  if len(s_leaf) > 0 else 0.0

    # ── f7: Low saturation fraction (bleached/necrotic tissue) ────
    low_sat_mask = _in_leaf((s < 50).astype(np.uint8) * 255)
    low_sat_ratio = np.sum(low_sat_mask > 0) / leaf_area * 100

    # ── f8, f9: Value (brightness) histogram stats (leaf pixels) ──
    v_leaf   = v[leaf_mask > 0].astype(float)
    mean_val = float(np.mean(v_leaf)) if len(v_leaf) > 0 else 0.0
    std_val  = float(np.std(v_leaf))  if len(v_leaf) > 0 else 0.0

    # ── f10: Hue entropy & hue histogram (leaf pixels) ────────────
    hue_leaf  = hh[leaf_mask > 0]
    hist, _   = np.histogram(hue_leaf, bins=20, range=(0, 180))
    hist_n    = hist / (hist.sum() + 1e-9)
    hue_entropy = float(-np.sum(hist_n * np.log2(hist_n + 1e-9)))

    # ── f31, f32: GLCM texture features ───────────────────────────
    # Grey-Level Co-occurrence Matrix captures spot pattern texture.
    # Lower homogeneity → more texture variation → more disease spots.
    gray_leaf   = cv2.bitwise_and(gray, gray, mask=leaf_mask)
    glcm        = graycomatrix(gray_leaf, distances=[1, 3],
                               angles=[0, np.pi / 4, np.pi / 2],
                               levels=256, symmetric=True, normed=True)
    homogeneity = float(graycoprops(glcm, 'homogeneity').mean())
    contrast    = float(graycoprops(glcm, 'contrast').mean())

    # Assemble feature dict
    features = {
        "brown_ratio":   round(brown_ratio,  2),
        "dark_ratio":    round(dark_ratio,   2),
        "yellow_ratio":  round(yellow_ratio, 2),
        "green_ratio":   round(green_ratio,  2),
        "mean_sat":      round(mean_sat,     1),
        "std_sat":       round(std_sat,      1),
        "low_sat_ratio": round(low_sat_ratio, 2),
        "mean_val":      round(mean_val,     1),
        "std_val":       round(std_val,      1),
        "hue_entropy":   round(hue_entropy,  4),
        "homogeneity":   round(homogeneity,  4),
        "contrast":      round(contrast,     2),
        "coverage_pct":  mask["coverage_pct"],
    }
    for i, val in enumerate(hist_n):
        features[f"hue_hist_b{i}"] = round(float(val), 4)

    # Internal masks for visualization
    features["_brown_mask"] = brown_px
    features["_dark_mask"]  = dark_px

    return features


# ══════════════════════════════════════════════════════════
# STEP 9.1.3.8 — CLASSIFICATION
# Use feature vector X to assign health class:
#   Green  = Healthy    (no anomaly detected)
#   Yellow = Suspicious (anomaly detected, insufficient certainty)
#   Red    = Sick       (anomaly detected, high confidence)
# ══════════════════════════════════════════════════════════

def classify_leaf(features: dict) -> dict:
    """
    Score-based 3-class classification using the feature vector.

    Score 0-1  → Green  = Healthy
    Score 2-4  → Yellow = Suspicious
    Score 5+   → Red    = Sick
    """
    score = 0

    # Brown spots (Early Blight) — strongest disease signal (f1)
    b = features["brown_ratio"]
    if   b > 15:  score += 3
    elif b > 5:   score += 2
    elif b > 1.5: score += 1

    # Yellow chlorosis (f3)
    y = features["yellow_ratio"]
    if   y > 15: score += 2
    elif y > 5:  score += 1

    # Low saturation mean — desaturated disease tissue (f5)
    s_mean = features["mean_sat"]
    if   s_mean < 75: score += 2
    elif s_mean < 90: score += 1

    # Low green ratio — diseased tissue displaces healthy green (f4)
    g = features["green_ratio"]
    if   g < 50: score += 2
    elif g < 65: score += 1

    # Dark necrotic area (f2)
    if features["dark_ratio"] > 3:      score += 1

    # GLCM texture — spot texture lowers homogeneity (f31)
    if features["homogeneity"] < 0.33:  score += 1

    # High saturation std — mixed healthy/diseased tissue (f6)
    if features["std_sat"] > 40:        score += 1

    # High hue entropy — many hue tones indicates mixed tissue (f10)
    if features["hue_entropy"] > 2.5:   score += 1

    # Traffic-light labels (per academic spec 9.1.3.8)
    if score >= 5:
        label, color_bgr = "Sick",       (0,   0, 220)   # Red
    elif score >= 2:
        label, color_bgr = "Suspicious", (0, 140, 255)   # Yellow
    else:
        label, color_bgr = "Healthy",    (30, 180,  30)  # Green

    return {
        "label":        label,
        "score":        score,
        "color_bgr":    color_bgr,
        "brown_ratio":  features["brown_ratio"],
        "dark_ratio":   features["dark_ratio"],
        "yellow_ratio": features["yellow_ratio"],
        "green_ratio":  features["green_ratio"],
        "mean_sat":     features["mean_sat"],
    }


# ══════════════════════════════════════════════════════════
# STEP 9.1.3.9 — GEOGRAPHIC OUTPUT
# Link result to field location.
# P = (Latitude, Longitude, Class)
# ══════════════════════════════════════════════════════════

def build_geographic_point(img_path: str, label: str) -> dict:
    """
    Extract GPS coordinates from image EXIF and attach the health class.
    Returns P = (latitude, longitude, class) per spec 9.1.3.9.

    If no GPS EXIF is present, lat/lon are None.
    """
    lat, lon = None, None

    try:
        import piexif
        exif = piexif.load(img_path)
        gps  = exif.get("GPS", {})
        if gps:
            def _dms(tag):
                dms = gps.get(tag)
                if not dms: return None
                d = dms[0][0] / dms[0][1]
                m = dms[1][0] / dms[1][1]
                s_val = dms[2][0] / dms[2][1]
                return d + m / 60.0 + s_val / 3600.0
            lat = _dms(piexif.GPSIFD.GPSLatitude)
            lon = _dms(piexif.GPSIFD.GPSLongitude)
            if lat and gps.get(piexif.GPSIFD.GPSLatitudeRef)  == b"S": lat = -lat
            if lon and gps.get(piexif.GPSIFD.GPSLongitudeRef) == b"W": lon = -lon
    except Exception:
        pass

    if lat is None:
        try:
            from PIL import Image
            from PIL.ExifTags import TAGS, GPSTAGS
            with Image.open(img_path) as pil_img:
                exif_data = pil_img._getexif() or {}
                for tag_id, value in exif_data.items():
                    if TAGS.get(tag_id) == "GPSInfo":
                        gps_info = {GPSTAGS.get(k, k): v for k, v in value.items()}
                        def _dms2(key):
                            dms = gps_info.get(key)
                            if not dms: return None
                            return dms[0] + dms[1] / 60.0 + dms[2] / 3600.0
                        lat = _dms2("GPSLatitude")
                        lon = _dms2("GPSLongitude")
                        if lat and gps_info.get("GPSLatitudeRef")  == "S": lat = -lat
                        if lon and gps_info.get("GPSLongitudeRef") == "W": lon = -lon
                        break
        except Exception:
            pass

    return {
        "latitude":  lat,
        "longitude": lon,
        "class":     label,      # P = (lat, lon, class)
        "path":      img_path,
    }


# ══════════════════════════════════════════════════════════
# VISUALIZATION
# ══════════════════════════════════════════════════════════

def annotate_image(img_bgr: np.ndarray, mask: dict, features: dict, result: dict) -> np.ndarray:
    """Draw disease overlays and classification label on the image."""
    out = img_bgr.copy()

    overlay = np.zeros_like(out)
    overlay[features["_brown_mask"] > 0] = [0, 80, 200]
    cv2.addWeighted(out, 1.0, overlay, 0.5, 0, out)

    overlay2 = np.zeros_like(out)
    overlay2[features["_dark_mask"] > 0] = [160, 0, 160]
    cv2.addWeighted(out, 1.0, overlay2, 0.5, 0, out)

    cnts, _ = cv2.findContours(mask["leaf_mask"], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(out, cnts, -1, (0, 230, 60), 2)

    cv2.rectangle(out, (0, 0), (out.shape[1], 42), (20, 20, 20), -1)
    cv2.putText(out,
                f"{result['label']} | score: {result['score']}  "
                f"(brown:{result['brown_ratio']:.0f}%  green:{result['green_ratio']:.0f}%)",
                (5, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                result["color_bgr"], 2)
    return out


# ══════════════════════════════════════════════════════════
# PUBLIC API
# ══════════════════════════════════════════════════════════

def analyze_single(img_path: str, save_output: bool = True) -> dict:
    """
    Full 4-step pipeline for one image (per spec 9.1.3.5 → 9.1.3.9):
      1. preprocess()        — image quality improvement
      2. build_plant_mask()  — isolate leaf
      3. extract_features()  — build feature vector X
      4. classify_leaf()     — assign health class
      5. geographic point    — P = (lat, lon, class)
    """
    img_raw = cv2.imread(img_path)
    if img_raw is None:
        raise FileNotFoundError(f"Cannot open image: {img_path}")

    # 9.1.3.5 Pre-processing
    img = preprocess(img_raw)

    # 9.1.3.6 Plant mask
    mask = build_plant_mask(img)

    # 9.1.3.7 Feature extraction
    features = extract_features(img, mask)

    # 9.1.3.8 Classification
    result = classify_leaf(features)

    # 9.1.3.9 Geographic output
    geo = build_geographic_point(img_path, result["label"])

    geo_str = ""
    if geo["latitude"] is not None:
        geo_str = f"  GPS:({geo['latitude']:.4f},{geo['longitude']:.4f})"
    print(f"  {Path(img_path).name:<30} → {result['label']:<12}  score={result['score']}"
          f"  (leaf={mask['coverage_pct']}%){geo_str}")

    if save_output:
        annotated = annotate_image(img, mask, features, result)
        out_path  = Path(img_path).stem + "_classified.jpg"
        cv2.imwrite(str(out_path), annotated)
        print(f"    Saved: {out_path}")

    result["filename"]  = Path(img_path).name
    result["_mask"]     = mask
    result["_features"] = features
    result["geo"]       = geo
    return result


def batch_analyze(folder_path: str, output_report: str = "report.png") -> list:
    """Analyze all images in a folder and generate a visual report."""
    folder = Path(folder_path)
    images = (sorted(folder.glob("*.JPG"))  + sorted(folder.glob("*.jpg")) +
              sorted(folder.glob("*.png"))  + sorted(folder.glob("*.jpeg")))

    print(f"\nFound {len(images)} images in: {folder_path}")
    results = []
    for img_path in images:
        try:
            results.append(analyze_single(str(img_path), save_output=False))
        except Exception as e:
            print(f"  ERROR {img_path.name}: {e}")

    if not results:
        print("No valid images found.")
        return results

    counts = {"Healthy": 0, "Suspicious": 0, "Sick": 0}
    for r in results:
        counts[r["label"]] += 1
    total = len(results)

    print(f"\n{'='*50}")
    print(f"  Summary ({total} images):")
    print(f"  Healthy:    {counts['Healthy']}  ({counts['Healthy'] / total * 100:.0f}%)")
    print(f"  Suspicious: {counts['Suspicious']}  ({counts['Suspicious'] / total * 100:.0f}%)")
    print(f"  Sick:       {counts['Sick']}  ({counts['Sick'] / total * 100:.0f}%)")
    print(f"{'='*50}")

    _create_report(results, counts, output_report)
    print(f"\nReport saved: {output_report}")
    return results


def _create_report(results, counts, output_path):
    """Visual report: sample images per class + pie chart + score histogram."""
    fig = plt.figure(figsize=(18, 10))
    fig.patch.set_facecolor('#1a1a2e')

    ax_pie = fig.add_axes([0.0, 0.55, 0.22, 0.40])
    ax_pie.set_facecolor('#16213e')
    labels = ["Healthy", "Suspicious", "Sick"]
    colors = ["#27ae60", "#f39c12", "#e74c3c"]
    sizes  = [counts[l] for l in labels]
    ax_pie.pie(sizes, labels=labels, colors=colors, autopct='%1.0f%%',
               textprops={'color': 'white', 'fontsize': 9}, startangle=90)
    ax_pie.set_title("Classification\nDistribution", color='white',
                     fontsize=10, fontweight='bold')

    ax_hist = fig.add_axes([0.0, 0.07, 0.22, 0.40])
    ax_hist.set_facecolor('#16213e')
    scores = [r["score"] for r in results]
    n, bins, patches = ax_hist.hist(scores, bins=range(0, 12),
                                    color='#3498db', edgecolor='#16213e')
    for patch, left in zip(patches, bins):
        if left >= 5:   patch.set_facecolor('#e74c3c')
        elif left >= 2: patch.set_facecolor('#f39c12')
        else:           patch.set_facecolor('#27ae60')
    ax_hist.set_xlabel("Disease Score", color='white', fontsize=9)
    ax_hist.set_ylabel("Count",         color='white', fontsize=9)
    ax_hist.set_title("Score Distribution", color='white', fontsize=10, fontweight='bold')
    ax_hist.tick_params(colors='white')
    ax_hist.axvline(2, color='#f39c12', linestyle='--', linewidth=1.2, label='Suspicious')
    ax_hist.axvline(5, color='#e74c3c', linestyle='--', linewidth=1.2, label='Sick')
    ax_hist.legend(fontsize=7, labelcolor='white', framealpha=0.3)

    class_samples = {"Healthy": [], "Suspicious": [], "Sick": []}
    for r in results:
        if len(class_samples[r["label"]]) < 3:
            class_samples[r["label"]].append(r)

    row_labels = ["Healthy", "Suspicious", "Sick"]
    row_colors = ["#27ae60", "#f39c12", "#e74c3c"]

    for row, (cls, color_hex) in enumerate(zip(row_labels, row_colors)):
        samples = class_samples[cls]
        for col, r in enumerate(samples[:3]):
            left = 0.25 + col * 0.245
            bot  = 0.65 - row * 0.31
            ax   = fig.add_axes([left, bot, 0.22, 0.28])
            ax.set_facecolor('#16213e')
            img = cv2.imread(r.get("filename", ""))
            if img is None and "_mask" in r:
                img = r["_mask"].get("leaf_img")
            if img is not None:
                annotated = annotate_image(img, r["_mask"], r["_features"], r)
                ax.imshow(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB))
            ax.set_title(
                f"Score: {r['score']} | brown: {r['brown_ratio']:.0f}%  "
                f"green: {r['green_ratio']:.0f}%",
                color=color_hex, fontsize=7.5, pad=2
            )
            ax.axis('off')
            for sp in ax.spines.values():
                sp.set_edgecolor(color_hex)
                sp.set_linewidth(2)
        if samples:
            fig.text(0.235, 0.65 - row * 0.31 + 0.14, cls,
                     color=color_hex, fontsize=11, fontweight='bold',
                     va='center', rotation=90)

    fig.suptitle("Potato Disease Detection — Classification Report",
                 color='white', fontsize=14, fontweight='bold', y=1.0)
    plt.savefig(output_path, dpi=140, bbox_inches='tight', facecolor='#1a1a2e')
    plt.close()


# ══════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        path = sys.argv[1]
        if Path(path).is_dir():
            batch_analyze(path, output_report="disease_report.png")
        else:
            r = analyze_single(path)
            print(f"\nFeature vector X:")
            for k, vv in r["_features"].items():
                if not k.startswith("_") and not k.startswith("hue_hist"):
                    print(f"  {k:<20} {vv:.3f}")
            print(f"  hue_hist_b0..b19   [20-bin hue histogram, leaf pixels]")
            geo = r["geo"]
            if geo["latitude"] is not None:
                print(f"\nGeographic point P = ({geo['latitude']:.6f}, "
                      f"{geo['longitude']:.6f}, {geo['class']})")
    else:
        print("Usage:")
        print("  python potato_disease_detector_v2.py <image.jpg>")
        print("  python potato_disease_detector_v2.py <folder/>")

