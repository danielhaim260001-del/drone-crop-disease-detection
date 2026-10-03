"""
Beet Disease Detector — Cercospora Leaf Spot
============================================
Pipeline per academic spec:
  9.1.3.5  Pre-processing      — image quality improvement
  9.1.3.6  Plant mask          — separate plant from background
  9.1.3.7  Feature extraction  — histogram & color features from plant pixels
  9.1.3.8  Classification      — Green / Yellow / Red (Healthy / Suspicious / Sick)
  9.1.3.9  Geographic output   — P = (Latitude, Longitude, Class)

Usage:
    python beet_disease_detector.py <image_path>           # Single image
    python beet_disease_detector.py <folder_path> --batch  # Batch folder

Author: Generated for ofri.azar@gmail.com
"""

import cv2
import numpy as np
import os
import sys
import glob
import argparse
import warnings
warnings.filterwarnings("ignore")

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.gridspec import GridSpec
    HAS_MPL = True
except ImportError:
    HAS_MPL = False


# ══════════════════════════════════════════════════════════
# STEP 9.1.3.5 — PRE-PROCESSING
# Improve image quality before anomaly identification.
# Adjusts for lighting variations, noise, and contrast issues
# that arise from different shooting distances, angles, and
# weather conditions (sun, wind blur).
# Output: processed image (same size, improved quality)
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
       improve contrast in shadowed regions or overexposed patches.
       Applied per-channel in LAB color space so hue is not distorted.
       clipLimit=2.0 prevents noise amplification; tileGridSize 8×8.

    4. Value-channel clipping — cap the top 1% of brightness values
       to reduce specular highlights (sun glints on wet leaves).

    Returns:
        processed BGR image (uint8, same WxH as resized input)
    """
    # 1. Resize to standard resolution
    img = cv2.resize(img_bgr, (800, 600), interpolation=cv2.INTER_AREA)

    # 2. Gaussian blur — noise reduction
    img = cv2.GaussianBlur(img, (3, 3), sigmaX=0.8)

    # 3. CLAHE on L-channel in LAB space — adaptive contrast enhancement
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
# Separate plant area from background (soil, sand, shadow).
# Output: binary mask where 255 = plant pixel
# ══════════════════════════════════════════════════════════

def build_plant_mask(img_bgr: np.ndarray) -> dict:
    """
    Segment plant material from background.

    Strategy:
      1. Build a vegetation seed: green pixels with sufficient saturation
      2. Expand the seed via dilation — captures yellow disease halos
         near green tissue while leaving distant bare soil excluded
      3. Explicitly remove soil/sand pixels (hue 8-32, low saturation)
      4. Morphological cleanup to fill holes and remove small noise blobs
      5. Return the clean plant mask + masked image

    Returns:
        dict with:
          'plant_mask'    — binary mask (uint8), 255 = plant pixel
          'plant_img'     — BGR image with background zeroed
          'plant_area'    — number of plant pixels
          'coverage_pct'  — plant area as % of total image
          'found'         — False if no significant plant was found
          '_green_mask'   — strict green-only mask (for feature stats)
    """
    h, w  = img_bgr.shape[:2]
    total = h * w
    hsv   = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    hh, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]

    # Green seed — healthy beet leaf tissue
    green_seed = ((hh >= 35) & (hh <= 85) & (s > 45) & (v > 40)).astype(np.uint8) * 255

    # Pale / bleached / dried leaf tissue (wilted beet leaves near green)
    pale_leaf = ((s < 60) & (v > 90) & (v < 240)).astype(np.uint8) * 255

    # Dark red / purple beet tissue (stems, veins, stressed leaves)
    dark_leaf = (((hh >= 150) | (hh <= 15)) & (s > 40) & (v > 30) & (v < 180)).astype(np.uint8) * 255

    # Yellowing diseased tissue (Cercospora halos)
    yellow_leaf = ((hh >= 18) & (hh <= 45) & (s > 50) & (v > 60)).astype(np.uint8) * 255

    # Explicit soil exclusion: hue ~8-32, low saturation, medium brightness
    soil = ((hh >= 8) & (hh <= 32) & (s < 65) & (v > 80)).astype(np.uint8) * 255

    # Build mask: include pale/dark/yellow only within expanded green zone
    expanded_green  = cv2.dilate(green_seed, np.ones((15, 15), np.uint8), iterations=3)
    leaf_candidates = cv2.bitwise_or(cv2.bitwise_or(pale_leaf, dark_leaf), yellow_leaf)
    leaf_near_green = cv2.bitwise_and(leaf_candidates, expanded_green)
    combined        = cv2.bitwise_or(green_seed, leaf_near_green)
    plant_mask      = cv2.bitwise_and(combined, cv2.bitwise_not(soil))

    # Morphological cleanup
    plant_mask = cv2.morphologyEx(plant_mask, cv2.MORPH_CLOSE,
                                   np.ones((9, 9), np.uint8), iterations=2)
    plant_mask = cv2.morphologyEx(plant_mask, cv2.MORPH_OPEN,
                                   np.ones((3, 3), np.uint8), iterations=1)

    plant_area   = int(plant_mask.sum() // 255)
    coverage_pct = plant_area / total * 100

    if coverage_pct < 5:
        return {
            "plant_mask":   plant_mask,
            "plant_img":    img_bgr.copy(),
            "plant_area":   plant_area,
            "coverage_pct": round(coverage_pct, 1),
            "found":        False,
            "_green_mask":  green_seed,
            "_soil_mask":   soil,
        }

    plant_img = cv2.bitwise_and(img_bgr, img_bgr, mask=plant_mask)

    return {
        "plant_mask":   plant_mask,
        "plant_img":    plant_img,
        "plant_area":   plant_area,
        "coverage_pct": round(coverage_pct, 1),
        "found":        True,
        "_green_mask":  green_seed,
        "_soil_mask":   soil,
    }


# ══════════════════════════════════════════════════════════
# STEP 9.1.3.7 — FEATURE EXTRACTION
# Measure features from plant area only.
# Output: feature vector X = [f1, f2, ..., fn]
# ══════════════════════════════════════════════════════════

def extract_features(img_bgr: np.ndarray, mask: dict) -> dict:
    """
    Extract histogram-based and color features from plant pixels only.
    Background and soil pixels are completely excluded from every calculation.

    Feature vector X = [
      # Color ratios (Cercospora disease signals)
      f1  yellow_ratio    — % of plant pixels that are yellow (Cercospora halos)
      f2  brown_ratio     — % of plant pixels with brown ring-border color
      f3  dark_ratio      — % of plant pixels that are dark (necrotic centers)
      f4  pale_ratio      — % of green pixels that are pale/bleached
      f5  green_pct       — green pixels / total image (canopy coverage)

      # Histogram-based saturation features
      f6  sat_mean        — mean saturation within green vegetation
      f7  sat_std         — saturation standard deviation (spread indicator)
      f8  high_sat_ratio  — % of green pixels with very high saturation (stress)

      # Hue histogram entropy
      f9  hue_entropy     — color diversity of green pixels
                            (healthy=low entropy, disease=higher spread)

      # Per-channel mean and std on plant pixels (full histogram summary)
      f10 hue_mean        — mean hue within plant area
      f11 val_mean        — mean brightness (Value) within plant area
      f12 val_std         — brightness variability (spots create dark patches)

      # Hue histogram bins (20 bins, plant pixels only) — f13..f32
      hue_hist_b0..b19    — normalized frequency per hue band
    ]
    """
    if not mask["found"]:
        zero = {k: 0.0 for k in
                ("yellow_ratio", "brown_ratio", "dark_ratio", "pale_ratio",
                 "green_pct", "sat_mean", "sat_std", "high_sat_ratio",
                 "hue_entropy", "hue_mean", "val_mean", "val_std")}
        for i in range(20):
            zero[f"hue_hist_b{i}"] = 0.0
        return zero

    plant_mask = mask["plant_mask"]
    green_mask = mask["_green_mask"]
    plant_area = max(mask["plant_area"], 1)
    green_area = max(int(green_mask.sum() // 255), 1)
    total      = img_bgr.shape[0] * img_bgr.shape[1]

    hsv      = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    hh, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]

    def _in_plant(bool_mask):
        return bool_mask & (plant_mask > 0)

    def _in_green(bool_mask):
        return bool_mask & (green_mask > 0)

    # ── f1: Yellow halos (Cercospora primary signal) ──────────────
    yellow_bool  = _in_plant((hh >= 18) & (hh <= 38) & (s > 70) & (v > 80))
    yellow_ratio = float(yellow_bool.sum()) / plant_area * 100

    # ── f2: Brown ring borders (spot edges) ───────────────────────
    brown_bool  = _in_plant((hh >= 5) & (hh <= 20) & (s > 80) & (v > 60))
    brown_ratio = float(brown_bool.sum()) / plant_area * 100

    # ── f3: Dark necrotic centers ─────────────────────────────────
    dark_bool  = _in_plant(v < 60)
    dark_ratio = float(dark_bool.sum()) / plant_area * 100

    # ── f4: Pale/bleached areas within green ──────────────────────
    pale_bool  = _in_green((s < 80) & (v > 120))
    pale_ratio = float(pale_bool.sum()) / green_area * 100

    # ── f5: Green canopy coverage ─────────────────────────────────
    green_pct = float(green_area) / total * 100

    # ── f6, f7: Saturation histogram stats (green pixels) ─────────
    if green_area > 500:
        s_veg    = s[green_mask > 0].astype(float)
        sat_mean = float(np.mean(s_veg))
        sat_std  = float(np.std(s_veg))
    else:
        sat_mean = sat_std = 0.0

    # ── f8: High-saturation stress fraction ───────────────────────
    high_sat_bool  = _in_green(s > 160)
    high_sat_ratio = float(high_sat_bool.sum()) / green_area * 100

    # ── f9: Hue entropy histogram (20-bin, green pixels) ─────────
    if green_area > 500:
        hue_veg     = hh[green_mask > 0]
        hist, _     = np.histogram(hue_veg, bins=20, range=(0, 180))
        hist_n      = hist / (hist.sum() + 1e-9)
        hue_entropy = float(-np.sum(hist_n * np.log2(hist_n + 1e-9)))
    else:
        hue_entropy = 0.0
        hist_n      = np.zeros(20)

    # ── f10: Hue mean within plant area ───────────────────────────
    hue_plant = hh[plant_mask > 0].astype(float)
    hue_mean  = float(np.mean(hue_plant)) if len(hue_plant) > 0 else 0.0

    # ── f11, f12: Value (brightness) stats within plant area ──────
    val_plant = v[plant_mask > 0].astype(float)
    val_mean  = float(np.mean(val_plant)) if len(val_plant) > 0 else 0.0
    val_std   = float(np.std(val_plant))  if len(val_plant) > 0 else 0.0

    # ── f13..f32: Full hue histogram bins (plant pixels) ──────────
    hue_all    = hh[plant_mask > 0]
    hist_all, _= np.histogram(hue_all, bins=20, range=(0, 180))
    hist_all_n = hist_all / (hist_all.sum() + 1e-9)

    # Assemble feature dict
    features = {
        "yellow_ratio":   round(yellow_ratio,   2),
        "brown_ratio":    round(brown_ratio,     2),
        "dark_ratio":     round(dark_ratio,      2),
        "pale_ratio":     round(pale_ratio,      2),
        "green_pct":      round(green_pct,       2),
        "sat_mean":       round(sat_mean,         1),
        "sat_std":        round(sat_std,           1),
        "high_sat_ratio": round(high_sat_ratio,  2),
        "hue_entropy":    round(hue_entropy,      4),
        "hue_mean":       round(hue_mean,          1),
        "val_mean":       round(val_mean,           1),
        "val_std":        round(val_std,            1),
    }
    for i, val in enumerate(hist_all_n):
        features[f"hue_hist_b{i}"] = round(float(val), 4)

    # Internal masks for visualization (not part of feature vector)
    features["_yellow_mask"] = yellow_bool.astype(np.uint8) * 255
    features["_dark_mask"]   = dark_bool.astype(np.uint8)   * 255

    return features


# ══════════════════════════════════════════════════════════
# STEP 9.1.3.8 — CLASSIFICATION
# Use feature vector X to assign health class:
#   Green  = Healthy    (no anomaly detected)
#   Yellow = Suspicious (anomaly detected, insufficient certainty)
#   Red    = Sick       (anomaly detected, high confidence)
# ══════════════════════════════════════════════════════════

def classify(features: dict) -> tuple:
    """
    Score-based Cercospora classification.

    Each signal from the feature vector contributes points.
    Final score maps to traffic-light health status:

      Score 0-2  → Green  = Healthy
      Score 3-5  → Yellow = Suspicious
      Score 6+   → Red    = Sick

    Returns: (label, score, reasons)
    """
    score   = 0
    reasons = []

    # Yellow halos — primary Cercospora signal (f1)
    y = features["yellow_ratio"]
    if   y > 12: score += 4; reasons.append(f"heavy yellowing ({y:.1f}%)")
    elif y > 5:  score += 2; reasons.append(f"moderate yellowing ({y:.1f}%)")
    elif y > 2:  score += 1; reasons.append(f"mild yellowing ({y:.1f}%)")

    # Saturation mean — stressed tissue is hypersaturated (f6)
    sm = features["sat_mean"]
    if   sm > 175: score += 3; reasons.append(f"high sat mean ({sm:.0f})")
    elif sm > 150: score += 2; reasons.append(f"elevated sat mean ({sm:.0f})")
    elif sm > 135: score += 1

    # High-saturation fraction within canopy (f8)
    hs = features["high_sat_ratio"]
    if   hs > 55: score += 3; reasons.append(f"high-sat in canopy ({hs:.1f}%)")
    elif hs > 35: score += 2; reasons.append(f"elevated high-sat ({hs:.1f}%)")
    elif hs > 25: score += 1

    # Hue entropy — disease spreads hue distribution (f9)
    he = features["hue_entropy"]
    if   he > 2.2: score += 2; reasons.append(f"high hue entropy ({he:.2f})")
    elif he > 1.9: score += 1

    # Green coverage — low canopy coverage signals heavy infection (f5)
    g = features["green_pct"]
    if   g < 60: score += 2; reasons.append(f"low green coverage ({g:.1f}%)")
    elif g < 75: score += 1

    # Dark necrotic areas (f3), only significant with high saturation
    d = features["dark_ratio"]
    if   d > 15 and sm > 130: score += 2; reasons.append(f"dark necrotic areas ({d:.1f}%)")
    elif d > 8  and sm > 130: score += 1

    # Brown ring spots — Cercospora spot borders (f2)
    b = features["brown_ratio"]
    if b > 2: score += 1; reasons.append(f"brown ring spots ({b:.1f}%)")

    # Brightness variability — spots create dark patches in bright canopy (f12)
    vs = features["val_std"]
    if vs > 55 and sm > 120: score += 1; reasons.append(f"high brightness variability ({vs:.0f})")

    # Traffic-light labels (per academic spec 9.1.3.8)
    if   score >= 6: label = "Sick"        # Red
    elif score >= 3: label = "Suspicious"  # Yellow
    else:            label = "Healthy"     # Green

    return label, score, reasons


# ══════════════════════════════════════════════════════════
# STEP 9.1.3.9 — GEOGRAPHIC OUTPUT
# Link result to field location.
# P = (Latitude, Longitude, Class)
# GPS metadata extracted from EXIF if available.
# ══════════════════════════════════════════════════════════

def build_geographic_point(img_path: str, label: str) -> dict:
    """
    Extract GPS coordinates from image EXIF and attach the health class.
    Returns P = (latitude, longitude, class) per spec 9.1.3.9.

    If no EXIF GPS data is present, lat/lon are None.
    The result can be used to plot field health maps.
    """
    lat, lon = None, None

    try:
        # Try piexif first (lightweight)
        import piexif
        exif = piexif.load(img_path)
        gps  = exif.get("GPS", {})
        if gps:
            def _dms(tag):
                dms = gps.get(tag)
                if not dms:
                    return None
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
            # Fallback: PIL / Pillow EXIF
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
                            return dms[0] + dms[1]/60.0 + dms[2]/3600.0
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
        "class":     label,          # P = (lat, lon, class)
        "path":      img_path,
    }


# ══════════════════════════════════════════════════════════
# VISUALIZATION
# ══════════════════════════════════════════════════════════

def _annotate(img_bgr: np.ndarray, mask: dict, features: dict,
              label: str, score: int, reasons: list) -> np.ndarray:
    """Overlay disease masks and classification label on the original image."""
    if not HAS_MPL:
        ann = img_bgr.copy()
        color_map = {"Healthy": (39, 174, 96), "Suspicious": (243, 156, 18), "Sick": (231, 76, 60)}
        c = color_map.get(label, (255, 255, 255))
        cv2.rectangle(ann, (0, 0), (450, 50), c, -1)
        cv2.putText(ann, f"{label} | score:{score}", (10, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        return ann

    H, W = img_bgr.shape[:2]
    rgb  = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    fig, ax = plt.subplots(figsize=(8, 7))
    fig.patch.set_facecolor("#0d1117")
    ax.imshow(rgb)

    if "_yellow_mask" in features and features["_yellow_mask"].sum() > 0:
        ym = features["_yellow_mask"]
        overlay = np.zeros((H, W, 4), dtype=np.float32)
        overlay[ym > 0] = [1.0, 0.55, 0.0, 0.55]
        ax.imshow(overlay)

    if "_dark_mask" in features and features["_dark_mask"].sum() > 0:
        dm = features["_dark_mask"]
        overlay2 = np.zeros((H, W, 4), dtype=np.float32)
        overlay2[dm > 0] = [0.7, 0.0, 0.7, 0.45]
        ax.imshow(overlay2)

    cnts, _ = cv2.findContours(mask["plant_mask"], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in cnts:
        if cv2.contourArea(cnt) > 200:
            pts = cnt[:, 0, :]
            ax.plot(np.append(pts[:, 0], pts[0, 0]),
                    np.append(pts[:, 1], pts[0, 1]),
                    color="#00e676", linewidth=1.5, alpha=0.75)

    colors = {"Healthy": "#27ae60", "Suspicious": "#f39c12", "Sick": "#e74c3c"}
    icons  = {"Healthy": "✓", "Suspicious": "?", "Sick": "✗"}
    ax.text(0.02, 0.98,
            f"{icons.get(label, '')} {label}  (score: {score})",
            transform=ax.transAxes, color="white",
            fontsize=13, fontweight="bold", va="top",
            bbox=dict(boxstyle="round,pad=0.4",
                      facecolor=colors.get(label, "#333"),
                      alpha=0.88, edgecolor="white"))

    if reasons:
        ax.text(0.02, 0.02,
                "\n".join(f"• {r}" for r in reasons[:4]),
                transform=ax.transAxes, color="white", fontsize=9, va="bottom",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="black", alpha=0.65))

    ax.axis("off")
    plt.tight_layout(pad=0.3)

    import io
    buf_io = io.BytesIO()
    fig.savefig(buf_io, format="png", dpi=130, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    buf_io.seek(0)
    data = np.frombuffer(buf_io.read(), dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


# ══════════════════════════════════════════════════════════
# PUBLIC API
# ══════════════════════════════════════════════════════════

def analyze_image(img_path: str, save_annotated: str = None) -> dict:
    """
    Full 4-step pipeline for one image (per spec 9.1.3.5 → 9.1.3.9):
      1. preprocess()       — quality improvement
      2. build_plant_mask() — isolate plant pixels
      3. extract_features() — build feature vector X
      4. classify()         — assign health class
      5. geographic point   — P = (lat, lon, class)
    """
    img_raw = cv2.imread(img_path)
    if img_raw is None:
        raise ValueError(f"Cannot read image: {img_path}")

    # 9.1.3.5 Pre-processing
    img = preprocess(img_raw)

    # 9.1.3.6 Plant mask
    mask = build_plant_mask(img)

    # 9.1.3.7 Feature extraction
    features = extract_features(img, mask)

    # 9.1.3.8 Classification
    label, score, reasons = classify(features)

    # 9.1.3.9 Geographic output
    geo = build_geographic_point(img_path, label)

    result = {
        "path":      img_path,
        "label":     label,
        "score":     score,
        "reasons":   reasons,
        "features":  features,
        "geo":       geo,           # P = (lat, lon, class)
    }

    if save_annotated:
        ann = _annotate(img, mask, features, label, score, reasons)
        cv2.imwrite(save_annotated, ann)

    return result


def batch_analyze(folder_path: str, extensions=("jpg", "jpeg", "png"),
                  output_dir: str = None, report_path: str = None) -> list:
    """Analyze all images in a folder. Returns list of result dicts."""
    files = []
    for ext in extensions:
        files += glob.glob(os.path.join(folder_path, f"*.{ext}"))
        files += glob.glob(os.path.join(folder_path, f"*.{ext.upper()}"))
    files = sorted(set(files))

    if not files:
        print(f"No images found in {folder_path}")
        return []

    results = []
    print(f"\nAnalyzing {len(files)} images...")
    for i, f in enumerate(files, 1):
        try:
            r = analyze_image(f)
            results.append(r)
            geo_str = ""
            if r["geo"]["latitude"] is not None:
                geo_str = f"  GPS:({r['geo']['latitude']:.4f},{r['geo']['longitude']:.4f})"
            print(f"  [{i:3d}/{len(files)}] {os.path.basename(f):<30} "
                  f"→ {r['label']:<12}  score={r['score']}{geo_str}")
        except Exception as e:
            print(f"  [ERR] {os.path.basename(f)}: {e}")

    counts = {"Healthy": 0, "Suspicious": 0, "Sick": 0}
    for r in results:
        counts[r["label"]] += 1
    total = len(results)
    print(f"\n{'─'*45}")
    print(f"  Total: {total} images")
    print(f"  Healthy:    {counts['Healthy']:3d}  ({100*counts['Healthy']//max(total,1)}%)")
    print(f"  Suspicious: {counts['Suspicious']:3d}  ({100*counts['Suspicious']//max(total,1)}%)")
    print(f"  Sick:       {counts['Sick']:3d}  ({100*counts['Sick']//max(total,1)}%)")

    if report_path and HAS_MPL:
        _generate_report(results, report_path)

    return results


def _generate_report(results: list, out_path: str, max_samples: int = 8):
    """Generate a multi-panel visual report."""
    plt.style.use("dark_background")
    n    = min(len(results), max_samples)
    cols = 4
    rows = (n + cols - 1) // cols + 2

    fig = plt.figure(figsize=(20, rows * 4 + 2), facecolor="#0f0f0f")
    gs  = GridSpec(rows, cols, figure=fig, hspace=0.45, wspace=0.25)
    fig.suptitle("Sugar Beet Disease Report — Cercospora Leaf Spot",
                 fontsize=22, fontweight="bold", color="white", y=0.99)

    color_map = {"Healthy": "#27ae60", "Suspicious": "#f39c12", "Sick": "#e74c3c"}

    for i, res in enumerate(results[:n]):
        row_i = i // cols
        col_i = i % cols
        ax    = fig.add_subplot(gs[row_i, col_i])
        img   = cv2.imread(res["path"])
        if img is not None:
            ax.imshow(cv2.cvtColor(cv2.resize(img, (400, 300)), cv2.COLOR_BGR2RGB))
        ax.axis("off")
        c = color_map[res["label"]]
        ax.set_title(f"{res['label']}\nscore={res['score']}",
                     fontsize=10, color=c, fontweight="bold")
        for spine in ax.spines.values():
            spine.set_edgecolor(c); spine.set_linewidth(3)

    ax_pie = fig.add_subplot(gs[-2, :2])
    counts = {"Healthy": 0, "Suspicious": 0, "Sick": 0}
    for r in results:
        counts[r["label"]] += 1
    labels_k = [k for k, vv in counts.items() if vv > 0]
    sizes    = [counts[k] for k in labels_k]
    colors_k = [color_map[k] for k in labels_k]
    ax_pie.pie(sizes, labels=labels_k, colors=colors_k, autopct="%1.0f%%",
               startangle=90, textprops={"color": "white", "fontsize": 11})
    ax_pie.set_title("Classification Distribution", color="white", fontsize=13)

    ax_hist = fig.add_subplot(gs[-2, 2:])
    scores  = [r["score"] for r in results]
    for label_k, color_k in color_map.items():
        sc = [r["score"] for r in results if r["label"] == label_k]
        if sc:
            ax_hist.hist(sc, bins=range(0, max(scores + [10]) + 2),
                         color=color_k, alpha=0.7, label=label_k)
    ax_hist.axvline(3, color="#f39c12", linestyle="--", linewidth=1.5, alpha=0.8)
    ax_hist.axvline(6, color="#e74c3c", linestyle="--", linewidth=1.5, alpha=0.8)
    ax_hist.set_xlabel("Disease Score", color="white")
    ax_hist.set_ylabel("Count", color="white")
    ax_hist.set_title("Score Distribution", color="white", fontsize=13)
    ax_hist.legend(fontsize=9)
    ax_hist.tick_params(colors="white")
    ax_hist.set_facecolor("#1a1a1a")

    ax_feat = fig.add_subplot(gs[-1, :])
    feat_names  = ["yellow_ratio", "sat_mean", "high_sat_ratio",
                   "dark_ratio", "hue_entropy", "green_pct"]
    feat_labels = ["Yellow %", "Sat Mean", "High-Sat %",
                   "Dark %", "Hue Entropy", "Green %"]
    def avg(lst, key):
        vals = [r["features"].get(key, 0) for r in lst]
        return float(np.mean(vals)) if vals else 0.0
    x     = np.arange(len(feat_names))
    width = 0.25
    for offset, label_k, color_k in zip([-width, 0, width],
                                         ["Healthy", "Suspicious", "Sick"],
                                         ["#27ae60", "#f39c12", "#e74c3c"]):
        avgs = [avg([r for r in results if r["label"] == label_k], k) for k in feat_names]
        if any(vv > 0 for vv in avgs):
            ax_feat.bar(x + offset, avgs, width, label=label_k, color=color_k, alpha=0.85)
    ax_feat.set_xticks(x)
    ax_feat.set_xticklabels(feat_labels, color="white", fontsize=11)
    ax_feat.set_title("Average Feature Values by Class", color="white", fontsize=13)
    ax_feat.legend(fontsize=10)
    ax_feat.tick_params(colors="white")
    ax_feat.set_facecolor("#1a1a1a")

    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="#0f0f0f")
    plt.close()
    print(f"\nReport saved to: {out_path}")


# ══════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Sugar Beet Cercospora Disease Detector")
    parser.add_argument("input",  nargs="?", help="Image path or folder path")
    parser.add_argument("--batch",  action="store_true", help="Batch mode (analyze folder)")
    parser.add_argument("--output", "-o", help="Output annotated image path")
    parser.add_argument("--report", "-r", help="Output visual report path (batch mode)")
    args = parser.parse_args()

    if not args.input:
        parser.print_help()
        return

    if args.batch or os.path.isdir(args.input):
        batch_analyze(args.input, report_path=args.report)
    else:
        r = analyze_image(args.input, save_annotated=args.output)
        print(f"\nResult: {r['label']}")
        print(f"Score:  {r['score']}")
        print(f"Reasons: {', '.join(r['reasons']) if r['reasons'] else 'none'}")
        geo = r["geo"]
        if geo["latitude"] is not None:
            print(f"GPS:    ({geo['latitude']:.6f}, {geo['longitude']:.6f})")
        print("\nFeature vector X:")
        for k, vv in r["features"].items():
            if not k.startswith("_") and not k.startswith("hue_hist"):
                print(f"  {k:<20} {vv:.3f}")
        print(f"  hue_hist_b0..b19   [20-bin hue histogram, plant pixels]")


if __name__ == "__main__":
    main()

