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


def compute_segmentation_quality(mask_u8: np.ndarray, min_coverage: float = 3.0, max_coverage: float = 92.0) -> dict:
    """Heuristic confidence for leaf isolation; independent of disease classification."""
    m = (mask_u8 > 0).astype(np.uint8)
    h, w = m.shape[:2]; total = max(h*w, 1); area = int(m.sum()); coverage = 100.0*area/total
    n, labels, stats, _ = cv2.connectedComponentsWithStats(m, 8)
    comps = [int(stats[i, cv2.CC_STAT_AREA]) for i in range(1,n) if stats[i,cv2.CC_STAT_AREA] >= max(20,int(total*0.0001))]
    largest_share = (max(comps)/area) if comps and area else 0.0
    edge = np.zeros_like(m); edge[[0,-1],:] = 1; edge[:,[0,-1]] = 1
    edge_touch = float((m & edge).sum()) / max(int(edge.sum()),1)
    score = 100.0
    if coverage < min_coverage: score -= min(45.0, (min_coverage-coverage)*10)
    if coverage > max_coverage: score -= min(35.0, (coverage-max_coverage)*4)
    if len(comps) > 18: score -= min(25.0, (len(comps)-18)*1.5)
    if largest_share < 0.25 and area: score -= 20.0
    if edge_touch > 0.75: score -= 12.0
    score = float(np.clip(score,0,100))
    level = "High" if score >= 80 else "Medium" if score >= 60 else "Low"
    return {"segmentation_quality": level, "segmentation_quality_score": round(score,1),
            "coverage_pct": round(coverage,1), "components": len(comps),
            "largest_component_share": round(100*largest_share,1)}

def build_plant_mask(img_bgr: np.ndarray) -> dict:
    """Two-stage beet segmentation: conservative green seed followed by constrained inclusion of symptomatic tissue."""
    h,w=img_bgr.shape[:2]; total=max(h*w,1); hsv=cv2.cvtColor(img_bgr,cv2.COLOR_BGR2HSV)
    hh,ss,vv=cv2.split(hsv); b,g,r=cv2.split(img_bgr); exg=2*g.astype(np.int16)-r.astype(np.int16)-b.astype(np.int16)
    seed=(((hh>=28)&(hh<=95)&(ss>=25)&(vv>=30)&(exg>5)).astype(np.uint8)*255)
    seed=cv2.morphologyEx(seed,cv2.MORPH_OPEN,np.ones((3,3),np.uint8),iterations=1)
    near=cv2.dilate(seed,np.ones((11,11),np.uint8),iterations=3)
    pale=(((ss<70)&(vv>95)&(hh>=15)&(hh<=100)).astype(np.uint8)*255)
    yellow=(((hh>=15)&(hh<=38)&(ss>=35)&(vv>=65)).astype(np.uint8)*255)
    brown=((((hh<=24)|(hh>=165))&(ss>=35)&(vv>=30)&(vv<=220)).astype(np.uint8)*255)
    dark=(((vv>=20)&(vv<=95)&(ss>=15)).astype(np.uint8)*255)
    symptomatic=cv2.bitwise_and(cv2.bitwise_or(cv2.bitwise_or(pale,yellow),cv2.bitwise_or(brown,dark)),near)
    m=cv2.bitwise_or(seed,symptomatic)
    m=cv2.morphologyEx(m,cv2.MORPH_CLOSE,np.ones((7,7),np.uint8),iterations=2)
    m=cv2.morphologyEx(m,cv2.MORPH_OPEN,np.ones((3,3),np.uint8),iterations=1)
    n,labels,stats,_=cv2.connectedComponentsWithStats(m,8); clean=np.zeros_like(m); min_area=max(30,int(total*0.0002))
    for i in range(1,n):
        if stats[i,cv2.CC_STAT_AREA]>=min_area: clean[labels==i]=255
    area=int(np.count_nonzero(clean)); cov=100.0*area/total; q=compute_segmentation_quality(clean)
    return {"plant_mask":clean,"plant_img":cv2.bitwise_and(img_bgr,img_bgr,mask=clean),"plant_area":area,
            "coverage_pct":round(cov,1),"found":bool(cov>=3.0),"segmentation_quality":q["segmentation_quality"],
            "segmentation_quality_score":q["segmentation_quality_score"],"_green_mask":seed,"_symptomatic_candidates":symptomatic,
            "_soil_mask":cv2.bitwise_not(clean)}

def detect_cercospora_spots(img_bgr: np.ndarray, plant_mask: np.ndarray):
    """Strict pale-centre/dark-ring Cercospora detector inside eroded leaf mask."""
    hsv=cv2.cvtColor(img_bgr,cv2.COLOR_BGR2HSV); gray=cv2.cvtColor(img_bgr,cv2.COLOR_BGR2GRAY)
    hh,ss,vv=hsv[:,:,0],hsv[:,:,1],hsv[:,:,2]; h,w=gray.shape
    k=5 if min(h,w)<150 else 7; interior=cv2.erode(plant_mask,np.ones((k,k),np.uint8),iterations=1)
    cand=((ss<50)&(vv>150)&(interior>0)).astype(np.uint8)*255
    cand=cv2.morphologyEx(cand,cv2.MORPH_OPEN,np.ones((2,2),np.uint8),iterations=1)
    contours,_=cv2.findContours(cand,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE); accepted=np.zeros_like(plant_mask); count=0
    for c in contours:
        a=cv2.contourArea(c); p=cv2.arcLength(c,True)
        if not (8<a<160) or p<=0 or 4*np.pi*a/(p*p)<0.45: continue
        M=cv2.moments(c)
        if M['m00']<=0: continue
        cx,cy=int(M['m10']/M['m00']),int(M['m01']/M['m00']); ri=max(2,int(np.sqrt(a/np.pi))); ro=ri+6
        inner=np.zeros_like(gray,np.uint8); outer=np.zeros_like(gray,np.uint8)
        cv2.circle(inner,(cx,cy),ri,255,-1); cv2.circle(outer,(cx,cy),ro,255,-1); cv2.circle(outer,(cx,cy),ri+1,0,-1)
        inner=cv2.bitwise_and(inner,plant_mask); outer=cv2.bitwise_and(outer,plant_mask)
        iv,ov=gray[inner>0],gray[outer>0]
        if iv.size<3 or ov.size<5 or float(iv.mean())-float(ov.mean())<15: continue
        oh,os,ovv=hh[outer>0],ss[outer>0],vv[outer>0]
        ring=(((oh<25)|(oh>140))&(os>45)&(ovv<180))
        if ring.size==0 or float(np.mean(ring))<0.10: continue
        cv2.drawContours(accepted,[c],-1,255,-1); count+=1
    area=max(int(np.count_nonzero(plant_mask)),1)
    return accepted,count,count/area*10000.0

def detect_dark_lesions(img_bgr: np.ndarray, plant_mask: np.ndarray):
    """Compact local dark-minimum detector restricted to the interior of beet leaves."""
    gray=cv2.cvtColor(img_bgr,cv2.COLOR_BGR2GRAY); hsv=cv2.cvtColor(img_bgr,cv2.COLOR_BGR2HSV); hh=hsv[:,:,0]
    h,w=gray.shape; k=5 if min(h,w)<150 else 9
    interior=cv2.erode(plant_mask,np.ones((k,k),np.uint8),iterations=1)
    mean=cv2.GaussianBlur(gray.astype(np.float32),(15,15),0); diff=gray.astype(np.float32)-mean
    cand=((diff<-40)&(interior>0)).astype(np.uint8)*255
    cand=cv2.morphologyEx(cand,cv2.MORPH_OPEN,np.ones((2,2),np.uint8),iterations=1)
    contours,_=cv2.findContours(cand,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE); out=np.zeros_like(plant_mask); count=0
    for c in contours:
        a=cv2.contourArea(c); p=cv2.arcLength(c,True)
        if not (15<=a<=120) or p<=0 or 4*np.pi*a/(p*p)<0.50: continue
        M=cv2.moments(c)
        if M['m00']<=0: continue
        cx,cy=int(M['m10']/M['m00']),int(M['m01']/M['m00']); ri=max(2,int(np.sqrt(a/np.pi))); ro=ri+5
        inner=np.zeros_like(gray,np.uint8); outer=np.zeros_like(gray,np.uint8)
        cv2.circle(inner,(cx,cy),ri,255,-1); cv2.circle(outer,(cx,cy),ro,255,-1); cv2.circle(outer,(cx,cy),ri+1,0,-1)
        inner=cv2.bitwise_and(inner,plant_mask); outer=cv2.bitwise_and(outer,plant_mask)
        iv,ov=gray[inner>0],gray[outer>0]
        if iv.size<3 or ov.size<5 or float(ov.mean())-float(iv.mean())<22: continue
        rh=hh[outer>0]
        if rh.size==0 or float(np.mean((rh>=25)&(rh<=100)))<0.45: continue
        cv2.drawContours(out,[c],-1,255,-1); count+=1
    area=max(int(np.count_nonzero(plant_mask)),1)
    return out,count,count/area*10000.0

# ══════════════════════════════════════════════════════════
# STEP 9.1.3.7 — FEATURE EXTRACTION
# HARD INVARIANT: segment leaf first; measure EVERY diagnostic feature from leaf pixels only.
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

    # Additional field-robust Cercospora features used by the final application.
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    pm = plant_mask > 0
    true_yellow = (hh >= 20) & (hh <= 30) & (s > 80) & (v > 100) & pm
    ty_pct = float(np.sum(true_yellow)) / plant_area * 100
    bleach_cerc = (hh >= 15) & (hh <= 28) & (s >= 20) & (s < 60) & (v > 130) & pm
    bleach_pct = float(np.sum(bleach_cerc)) / plant_area * 100
    # HARD LEAF-FIRST RULE: local contrast is computed on a leaf-only image.
    # Pixels outside the leaf are filled with the leaf median before Gaussian filtering,
    # preventing dark soil/background from affecting lesion contrast near leaf borders.
    leaf_gray_vals = gray[pm]
    leaf_fill = float(np.median(leaf_gray_vals)) if leaf_gray_vals.size else 0.0
    gray_leaf_only = gray.astype(np.float32).copy()
    gray_leaf_only[~pm] = leaf_fill
    blur_local = cv2.GaussianBlur(gray_leaf_only, (21, 21), 0)
    diff_local = gray_leaf_only - blur_local
    interior_pm = cv2.erode(plant_mask, np.ones((3,3),np.uint8), iterations=1) > 0
    dark_sp = float(np.sum((diff_local < -15) & interior_pm)) / plant_area * 100
    light_sp = float(np.sum((diff_local > 20) & interior_pm)) / plant_area * 100
    green_inside = pm & (hh >= 35) & (hh <= 85) & (s > 40)
    green_leaf_pct = float(np.sum(green_inside)) / plant_area * 100

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
        "ty_pct":          round(ty_pct, 3),
        "bleach_pct":      round(bleach_pct, 2),
        "dark_sp":         round(dark_sp, 1),
        "light_sp":        round(light_sp, 1),
        "green_leaf_pct":  round(green_leaf_pct, 1),
    }
    for i, val in enumerate(hist_all_n):
        features[f"hue_hist_b{i}"] = round(float(val), 4)

    # Internal masks for visualization (not part of feature vector)
    features["_yellow_mask"] = yellow_bool.astype(np.uint8) * 255
    features["_dark_mask"]   = dark_bool.astype(np.uint8)   * 255

    # Strict lesion evidence inside the segmented leaf only.
    cerc_mask, cerc_count, cerc_density = detect_cercospora_spots(img_bgr, plant_mask)
    dark_lesion_mask, dark_count, dark_density = detect_dark_lesions(img_bgr, plant_mask)
    lesion_evidence = cv2.bitwise_or(cerc_mask, dark_lesion_mask)
    lesion_stats = compute_lesion_morphology(lesion_evidence, plant_mask, min_area=8)
    features.update(lesion_stats)
    features.update({
        "cercospora_count": int(cerc_count), "cercospora_density": round(float(cerc_density),2),
        "dark_lesion_count": int(dark_count), "dark_lesion_density": round(float(dark_density),2),
        "_cercospora_mask": cerc_mask, "_dark_lesion_mask": dark_lesion_mask,
    })

    return features



# ══════════════════════════════════════════════════════════
# ENGINEERING VALIDATION — SEGMENTATION + LESION MORPHOLOGY
# These measurements do not change the classifier thresholds.
# ══════════════════════════════════════════════════════════

def compute_segmentation_metrics(pred_mask: np.ndarray, gt_mask: np.ndarray) -> dict:
    """Return IoU and Dice for a binary predicted mask against manual ground truth."""
    if pred_mask is None or gt_mask is None:
        return {"iou": 0.0, "dice": 0.0}
    if gt_mask.ndim == 3:
        gt_mask = cv2.cvtColor(gt_mask, cv2.COLOR_BGR2GRAY)
    gt = cv2.resize(gt_mask, (pred_mask.shape[1], pred_mask.shape[0]), interpolation=cv2.INTER_NEAREST) > 127
    pr = pred_mask > 0
    inter = int(np.logical_and(pr, gt).sum())
    union = int(np.logical_or(pr, gt).sum())
    denom = int(pr.sum() + gt.sum())
    return {
        "iou": round(inter / union, 4) if union else 1.0,
        "dice": round((2.0 * inter) / denom, 4) if denom else 1.0,
    }

def compute_lesion_morphology(lesion_mask: np.ndarray, plant_mask: np.ndarray, min_area: int = 8) -> dict:
    """Connected-component lesion statistics restricted to segmented leaf tissue."""
    if lesion_mask is None or plant_mask is None:
        return {"lesion_count": 0, "lesion_area_pct": 0.0, "lesion_density": 0.0,
                "mean_lesion_area": 0.0, "median_lesion_area": 0.0, "mean_circularity": 0.0}
    binary = (((lesion_mask > 0) & (plant_mask > 0)).astype(np.uint8) * 255)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    areas, circs = [], []
    clean = np.zeros_like(binary)
    for c in contours:
        a = float(cv2.contourArea(c))
        if a < min_area:
            continue
        p = float(cv2.arcLength(c, True))
        circ = 4.0 * np.pi * a / (p * p) if p > 0 else 0.0
        areas.append(a); circs.append(circ)
        cv2.drawContours(clean, [c], -1, 255, -1)
    plant_area = max(int(np.count_nonzero(plant_mask)), 1)
    lesion_pixels = int(np.count_nonzero(clean))
    return {
        "lesion_count": len(areas),
        "lesion_area_pct": round(100.0 * lesion_pixels / plant_area, 2),
        "lesion_density": round(len(areas) / plant_area * 10000.0, 2),
        "mean_lesion_area": round(float(np.mean(areas)) if areas else 0.0, 1),
        "median_lesion_area": round(float(np.median(areas)) if areas else 0.0, 1),
        "mean_circularity": round(float(np.mean(circs)) if circs else 0.0, 3),
        "_lesion_mask": clean,
    }

# ══════════════════════════════════════════════════════════
# STEP 9.1.3.8 — CLASSIFICATION
# Use feature vector X to assign health class:
#   Green  = Healthy    (no anomaly detected)
#   Yellow = Suspicious (anomaly detected, insufficient certainty)
#   Red    = Sick       (anomaly detected, high confidence)
# ══════════════════════════════════════════════════════════

def classify(features: dict) -> tuple:
    """Final beet score logic, synchronized with the integrated application."""
    score=0; reasons=[]
    ty=float(features.get("ty_pct",0)); sm=float(features.get("sat_mean",0)); bleach=float(features.get("bleach_pct",0))
    dark_sp=float(features.get("dark_sp",0)); light_sp=float(features.get("light_sp",0)); green=float(features.get("green_leaf_pct",0))
    cerc=int(features.get("cercospora_count",0)); dark_n=int(features.get("dark_lesion_count",0))
    if ty>0.30: score+=3; reasons.append(f"true yellow tissue ({ty:.2f}%)")
    elif ty>0.15: score+=2; reasons.append(f"mild true yellow tissue ({ty:.2f}%)")
    if sm<80 and bleach>1.5: score+=2; reasons.append(f"bleached tissue ({bleach:.2f}%)")
    if sm>175 and ty>0.10: score+=2; reasons.append("high saturation with yellow evidence")
    healthy_green=(green>=70.0 and ty<0.30 and bleach<1.50)
    if not healthy_green:
        if dark_sp>4.0 and light_sp>3.0: score+=1; reasons.append("local spot texture")
        if cerc>0:
            weight=max(0.3,1.0-(sm-60)/70.0); weighted=cerc*weight
            if weighted>=25: score+=4
            elif weighted>=10: score+=3
            elif weighted>=4: score+=2
            elif weighted>=1: score+=1
            reasons.append(f"Cercospora ring candidates ({cerc})")
    if dark_n>=12: score+=3; reasons.append(f"small dark lesions ({dark_n})")
    elif dark_n>=7 and green<75.0: score+=2; reasons.append(f"small dark lesions ({dark_n})")
    elif dark_n>=4 and green<65.0: score+=1; reasons.append(f"small dark lesions ({dark_n})")
    label="Sick" if score>=5 else "Suspicious" if score>=3 else "Healthy"
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
