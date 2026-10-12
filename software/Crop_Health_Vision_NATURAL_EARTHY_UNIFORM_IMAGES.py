"""
Field Disease Detection — Streamlit App
========================================
Analyzes field images for plant disease (Potato or Sugar Beet).
Outputs: classification per plant zone + heatmap + summary stats.

Run:
    pip install streamlit opencv-python-headless matplotlib numpy pandas
    streamlit run app.py
"""

import streamlit as st
import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
from io import BytesIO
import os
import sys
import csv
import time
import zipfile
import hashlib
import base64
import pandas as pd

# ── Make sure the classifiers are importable from the same folder ──
sys.path.insert(0, os.path.dirname(__file__))

# ─────────────────────────────────────────────────────────
#  Page config
# ─────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Crop Health Vision",
    page_icon="🌿",
    layout="wide",
)

# ─────────────────────────────────────────────────────────
#  Custom CSS
# ─────────────────────────────────────────────────────────
st.markdown("""
<style>
/* =========================================================
   HIGH-TECH UI LAYER — presentation only.
   No image-processing/classification logic is changed here.
   ========================================================= */
:root {
    --bg: #071018;
    --panel: rgba(13, 25, 38, 0.82);
    --panel-2: rgba(17, 34, 50, 0.88);
    --line: rgba(111, 224, 255, 0.16);
    --line-strong: rgba(66, 211, 255, 0.34);
    --text: #edf7fb;
    --muted: #8fa8b8;
    --cyan: #36d7ff;
    --cyan-2: #0ea5e9;
    --green: #32d583;
    --amber: #fbbf24;
    --red: #fb7185;
}

html, body, [class*="css"] {
    font-family: Inter, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
}

[data-testid="stAppViewContainer"] {
    background:
        radial-gradient(circle at 82% 0%, rgba(14,165,233,.12), transparent 28rem),
        radial-gradient(circle at 15% 12%, rgba(50,213,131,.07), transparent 24rem),
        linear-gradient(180deg, #071018 0%, #08121c 45%, #060d14 100%);
    color: var(--text);
}

[data-testid="stHeader"] {
    background: rgba(7,16,24,.70);
    backdrop-filter: blur(14px);
    border-bottom: 1px solid rgba(111,224,255,.08);
}

[data-testid="stMainBlockContainer"], .block-container {
    max-width: none !important;
    width: 100% !important;
    padding-top: 1.05rem !important;
    padding-left: clamp(1.1rem, 2vw, 2.2rem) !important;
    padding-right: clamp(1.1rem, 2vw, 2.2rem) !important;
    padding-bottom: 3rem !important;
}

#MainMenu {visibility: hidden;}
footer {visibility: hidden;}

/* ---------- Sidebar / control console ---------- */
[data-testid="stSidebar"] {
    background:
        linear-gradient(180deg, rgba(11,23,34,.98), rgba(7,16,24,.99));
    border-right: 1px solid rgba(54,215,255,.16);
}
[data-testid="stSidebar"] > div:first-child { padding-top: .8rem; }
[data-testid="stSidebar"] hr { border-color: rgba(143,168,184,.12); margin: 1rem 0; }

.sidebar-brand {
    position: relative;
    overflow: hidden;
    padding: 16px 16px 15px;
    margin: 0 0 16px;
    border: 1px solid var(--line-strong);
    border-radius: 14px;
    background: linear-gradient(145deg, rgba(20,43,60,.94), rgba(10,24,36,.92));
    box-shadow: 0 14px 34px rgba(0,0,0,.22), inset 0 1px 0 rgba(255,255,255,.04);
}
.sidebar-brand:before {
    content: "";
    position: absolute;
    width: 120px; height: 120px; right: -45px; top: -55px;
    border: 1px solid rgba(54,215,255,.18);
    border-radius: 50%;
    box-shadow: 0 0 45px rgba(54,215,255,.08);
}
.sidebar-kicker { color: var(--cyan); font-size: .68rem; font-weight: 800; letter-spacing: .16em; text-transform: uppercase; }
.sidebar-title { color: #f5fbff; font-size: 1.12rem; font-weight: 800; margin-top: 5px; }
.sidebar-sub { color: var(--muted); font-size: .76rem; margin-top: 4px; line-height: 1.45; }
.sidebar-section {
    color: #b9d3df; font-size: .72rem; font-weight: 800; letter-spacing: .11em;
    text-transform: uppercase; margin: 18px 0 7px;
}

/* ---------- Hero ---------- */
.tech-hero {
    position: relative;
    overflow: hidden;
    border: 1px solid var(--line-strong);
    border-radius: 20px;
    padding: 72px 48px 64px;
    margin: 6px auto 28px;
    text-align: center;
    min-height: 430px;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    background:
        linear-gradient(120deg, rgba(17,38,54,.96), rgba(8,22,34,.94) 58%, rgba(7,31,39,.90));
    box-shadow: 0 22px 55px rgba(0,0,0,.28), inset 0 1px 0 rgba(255,255,255,.045);
}
.tech-hero:after {
    content: "";
    position: absolute; right: -70px; top: -120px;
    width: 360px; height: 360px; border-radius: 50%;
    background: radial-gradient(circle, rgba(54,215,255,.13), transparent 65%);
    pointer-events: none;
}
.tech-kicker { color: var(--cyan); font-size: 1rem; font-weight: 850; letter-spacing: .18em; text-transform: uppercase; }
.tech-title { color: #f5fbff; font-size: clamp(3.6rem, 6vw, 6.4rem); line-height: 1.05; font-weight: 820; letter-spacing: -.035em; margin: 5px 0 5px; }
.tech-subtitle { color: #b7ced9; font-size: clamp(1.2rem, 1.7vw, 1.65rem); max-width: 980px; line-height: 1.5; margin: 8px auto 0; }
.tech-chips { display: flex; gap: 10px; flex-wrap: wrap; margin-top: 18px; justify-content: center; }
.tech-chip {
    display: inline-flex; align-items: center; gap: 7px;
    border: 1px solid rgba(111,224,255,.17);
    background: rgba(8,22,34,.62);
    color: #b9dce8;
    border-radius: 9px;
    padding: 5px 9px;
    font-size: .78rem; font-weight: 700;
}
.tech-dot { width: 7px; height: 7px; border-radius: 50%; background: var(--green); box-shadow: 0 0 12px rgba(50,213,131,.7); }

/* ---------- Generic section headers ---------- */
.section-eyebrow { color: var(--cyan); font-size: .68rem; font-weight: 800; letter-spacing: .14em; text-transform: uppercase; margin-bottom: 3px; }
.section-title { color: #eef8fc; font-size: 1.24rem; font-weight: 800; letter-spacing: -.01em; margin-bottom: 2px; }
.section-sub { color: var(--muted); font-size: .82rem; margin-bottom: 12px; }

/* ---------- Buttons ---------- */
.stButton > button {
    min-height: 3.15rem;
    width: 100%;
    border: 1px solid rgba(73,221,255,.48) !important;
    border-radius: 12px !important;
    background: linear-gradient(135deg, #0b8fc8 0%, #0878b8 50%, #0a9a91 115%) !important;
    color: white !important;
    font-size: .94rem !important;
    font-weight: 800 !important;
    letter-spacing: .025em;
    box-shadow: 0 9px 24px rgba(4,136,188,.20), inset 0 1px 0 rgba(255,255,255,.16);
    transition: transform .15s ease, box-shadow .15s ease, border-color .15s ease;
}
.stButton > button:hover {
    transform: translateY(-1px);
    border-color: rgba(116,231,255,.78) !important;
    box-shadow: 0 12px 30px rgba(4,153,204,.30), 0 0 0 1px rgba(54,215,255,.12);
}
.stButton > button:active { transform: translateY(0); }
.stButton > button:disabled {
    background: linear-gradient(135deg, #20313d, #182733) !important;
    border-color: rgba(143,168,184,.16) !important;
    color: #66808f !important;
    box-shadow: none;
}

/* Main Analyze CTA: intentionally much larger than ordinary controls */
.stButton > button[kind="primary"],
.stButton > button[data-testid="stBaseButton-primary"] {
    min-height: 4.8rem !important;
    border-radius: 16px !important;
    font-size: 1.18rem !important;
    font-weight: 900 !important;
    letter-spacing: .035em !important;
    background: linear-gradient(135deg, #08a7df 0%, #0b91c7 48%, #16a394 110%) !important;
    border: 1px solid rgba(112,235,255,.82) !important;
    box-shadow: 0 14px 36px rgba(0,168,222,.30), 0 0 0 1px rgba(54,215,255,.10), inset 0 1px 0 rgba(255,255,255,.20) !important;
}
.stButton > button[kind="primary"]:hover,
.stButton > button[data-testid="stBaseButton-primary"]:hover {
    transform: translateY(-2px);
    box-shadow: 0 18px 42px rgba(0,184,235,.38), 0 0 0 2px rgba(54,215,255,.14) !important;
}

.stDownloadButton > button {
    min-height: 2.85rem;
    width: 100%;
    border: 1px solid rgba(54,215,255,.22) !important;
    border-radius: 11px !important;
    background: linear-gradient(180deg, rgba(19,39,54,.92), rgba(12,28,41,.92)) !important;
    color: #d9f4fb !important;
    font-weight: 750 !important;
    box-shadow: inset 0 1px 0 rgba(255,255,255,.035);
}
.stDownloadButton > button:hover {
    border-color: rgba(54,215,255,.48) !important;
    background: linear-gradient(180deg, rgba(24,51,68,.98), rgba(14,34,48,.98)) !important;
}

/* ---------- Inputs ---------- */
[data-testid="stFileUploaderDropzone"] {
    min-height: 118px;
    border: 1px dashed rgba(54,215,255,.42) !important;
    border-radius: 13px !important;
    background: linear-gradient(145deg, rgba(16,36,50,.72), rgba(10,24,36,.74)) !important;
}
[data-testid="stFileUploaderDropzone"]:hover {
    border-color: rgba(54,215,255,.72) !important;
    background: rgba(17,42,57,.84) !important;
}
[data-baseweb="select"] > div {
    background: rgba(13,29,42,.90) !important;
    border-color: rgba(111,224,255,.18) !important;
    border-radius: 10px !important;
}
[data-testid="stRadio"] div[role="radiogroup"] { gap: .38rem; }
[data-testid="stRadio"] label {
    color: #d7e8ef !important;
    border: 1px solid rgba(111,224,255,.12);
    border-radius: 10px;
    padding: .46rem .62rem;
    background: rgba(13,29,42,.56);
    transition: .15s ease;
}
[data-testid="stRadio"] label:hover {
    border-color: rgba(54,215,255,.42);
    background: rgba(17,42,57,.72);
}
[data-testid="stRadio"] label:has(input:checked) {
    border-color: rgba(54,215,255,.58);
    background: linear-gradient(135deg, rgba(14,165,233,.18), rgba(45,212,191,.10));
    box-shadow: inset 0 0 0 1px rgba(54,215,255,.08);
}
[data-testid="stSlider"] { padding-top: .15rem; }

/* ---------- Tabs as large navigation controls ---------- */
div[data-baseweb="tab-list"] {
    gap: 7px;
    padding: 6px;
    border: 1px solid rgba(111,224,255,.13);
    border-radius: 14px;
    background: rgba(10,24,36,.72);
    box-shadow: inset 0 1px 0 rgba(255,255,255,.025);
}
button[data-baseweb="tab"] {
    min-height: 46px;
    border-radius: 10px;
    padding: 0 18px;
    color: #8fa8b8;
    font-weight: 750;
    border: 1px solid transparent;
}
button[data-baseweb="tab"]:hover { color: #dff7ff; background: rgba(54,215,255,.055); }
button[data-baseweb="tab"][aria-selected="true"] {
    color: #effbff !important;
    background: linear-gradient(135deg, rgba(14,165,233,.23), rgba(45,212,191,.13));
    border-color: rgba(54,215,255,.26);
    box-shadow: 0 7px 20px rgba(0,0,0,.14), inset 0 1px 0 rgba(255,255,255,.045);
}
button[data-baseweb="tab"] div[data-testid="stMarkdownContainer"] p { font-weight: 750; }

/* ---------- Cards / metrics / alerts ---------- */
.metric-box {
    background: linear-gradient(145deg, rgba(18,37,52,.92), rgba(11,27,40,.92));
    border: 1px solid rgba(111,224,255,.13);
    border-radius: 14px;
    padding: 18px 14px;
    text-align: center;
    margin: 3px;
    min-height: 104px;
    box-shadow: 0 12px 28px rgba(0,0,0,.16), inset 0 1px 0 rgba(255,255,255,.03);
}
.metric-label { font-size: .75rem; color: #8fa8b8; margin-bottom: 5px; text-transform: uppercase; letter-spacing: .055em; }
.metric-value { font-size: 1.95rem; font-weight: 820; letter-spacing: -.03em; }
.healthy-color { color: #32d583; }
.suspicious-color { color: #fbbf24; }
.sick-color { color: #fb7185; }

[data-testid="stMetric"] {
    background: linear-gradient(145deg, rgba(18,37,52,.78), rgba(11,27,40,.78));
    border: 1px solid rgba(111,224,255,.11);
    border-radius: 13px;
    padding: 12px 14px;
    box-shadow: inset 0 1px 0 rgba(255,255,255,.025);
}
[data-testid="stMetricLabel"] { color: #91aab8; }
[data-testid="stMetricValue"] { color: #effbff; font-weight: 800; }

.alert-high, .alert-medium, .alert-low {
    border-radius: 13px;
    padding: 14px 16px;
    box-shadow: inset 0 1px 0 rgba(255,255,255,.03);
}
.alert-high { background: rgba(93,28,43,.38); border: 1px solid rgba(251,113,133,.46); }
.alert-medium { background: rgba(92,67,15,.32); border: 1px solid rgba(251,191,36,.42); }
.alert-low { background: rgba(15,75,54,.34); border: 1px solid rgba(50,213,131,.42); }

.quality-badge {
    border-radius: 9px; padding: 8px 12px; display: inline-block;
    font-weight: 800; font-size: .82rem; letter-spacing: .02em;
}
.conf-bar { height: 11px; border-radius: 999px; margin-bottom: 4px; box-shadow: 0 0 14px rgba(255,255,255,.035); }

/* ---------- Expanders / info / data ---------- */
[data-testid="stExpander"] {
    border: 1px solid rgba(111,224,255,.12) !important;
    border-radius: 13px !important;
    background: rgba(10,24,36,.52);
    overflow: hidden;
}
[data-testid="stExpander"] summary:hover { background: rgba(54,215,255,.035); }
[data-testid="stDataFrame"] { border: 1px solid rgba(111,224,255,.12); border-radius: 12px; overflow: hidden; }
[data-testid="stAlert"] { border-radius: 12px; border: 1px solid rgba(111,224,255,.12); }
hr { border-color: rgba(143,168,184,.11) !important; }

/* ---------- Images ---------- */
[data-testid="stImage"] img {
    border-radius: 13px;
    border: 1px solid rgba(111,224,255,.11);
    box-shadow: 0 10px 24px rgba(0,0,0,.17);
}
[data-testid="stImage"] small { color: #819aa8 !important; }

/* ---------- Typography ---------- */
h1, h2, h3 { color: #eef8fc !important; letter-spacing: -.02em; }
h1 { font-weight: 820 !important; }
h2, h3 { font-weight: 780 !important; }
p, li { color: #c8d9e1; }
.stCaption, [data-testid="stCaptionContainer"] { color: #819aa8 !important; }

/* ---------- Preview metadata card ---------- */
.meta-card {
    border: 1px solid rgba(111,224,255,.14);
    border-radius: 14px;
    padding: 15px 16px;
    background: linear-gradient(145deg, rgba(17,36,51,.82), rgba(10,25,37,.82));
    box-shadow: 0 12px 28px rgba(0,0,0,.14);
}
.meta-card-title { color: #eef8fc; font-weight: 800; font-size: .9rem; margin-bottom: 9px; }
.meta-row { display:flex; justify-content:space-between; gap:14px; padding:7px 0; border-bottom:1px solid rgba(143,168,184,.08); font-size:.79rem; }
.meta-row:last-child { border-bottom:0; }
.meta-key { color:#829baa; }
.meta-val { color:#dff6ff; font-weight:750; text-align:right; }

/* Slightly cleaner scrollbar */
::-webkit-scrollbar { width: 9px; height: 9px; }
::-webkit-scrollbar-track { background: #071018; }
::-webkit-scrollbar-thumb { background: #24404f; border-radius: 999px; border: 2px solid #071018; }
::-webkit-scrollbar-thumb:hover { background: #31576a; }

@media (max-width: 900px) {
    .tech-title { font-size: 1.65rem; }
    .tech-hero { padding: 34px 20px; min-height: 330px; }
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────
#  Step 2 — Image Quality Check
# ─────────────────────────────────────────────────────────

def check_image_quality(img_bgr: np.ndarray) -> dict:
    """
    Laplacian sharpness, brightness distribution, exposure check.
    Returns: sharpness, brightness_mean, brightness_std, valid (bool), issues (list).
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # Sharpness via Laplacian variance
    lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())

    # Brightness stats
    brightness_mean = float(np.mean(gray))
    brightness_std  = float(np.std(gray))

    # Overexposed: more than 20% pixels near 255
    overexposed_pct = float(np.sum(gray > 240) / gray.size * 100)
    # Underexposed: more than 20% pixels near 0
    underexposed_pct = float(np.sum(gray < 15) / gray.size * 100)

    issues = []
    valid  = True

    if lap_var < 80:
        issues.append(f"Blurry (sharpness={lap_var:.0f} < 80)")
        valid = False
    if overexposed_pct > 20:
        issues.append(f"Overexposed ({overexposed_pct:.1f}% white pixels)")
        valid = False
    if underexposed_pct > 20:
        issues.append(f"Underexposed ({underexposed_pct:.1f}% black pixels)")
        valid = False
    if brightness_mean < 30:
        issues.append(f"Too dark (mean brightness={brightness_mean:.0f})")
        valid = False
    if brightness_mean > 230:
        issues.append(f"Too bright (mean brightness={brightness_mean:.0f})")
        valid = False

    return {
        "sharpness":         round(lap_var, 1),
        "brightness_mean":   round(brightness_mean, 1),
        "brightness_std":    round(brightness_std, 1),
        "overexposed_pct":   round(overexposed_pct, 1),
        "underexposed_pct":  round(underexposed_pct, 1),
        "valid":             valid,
        "issues":            issues,
    }


# ─────────────────────────────────────────────────────────
#  Statistical metrics per image
# ─────────────────────────────────────────────────────────

def compute_image_stats(img_bgr: np.ndarray) -> dict:
    """
    Compute statistical metrics for a single image:
    mean, std, entropy, yellow_pct, green_pct, dark_pct, brown_pct, total_pixels.
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    total_pixels = int(img_bgr.shape[0] * img_bgr.shape[1])
    mean_val = float(np.mean(gray))
    std_val  = float(np.std(gray))

    # Shannon entropy from grayscale histogram
    hist, _  = np.histogram(gray.flatten(), bins=256, range=(0, 255))
    hist     = hist.astype(float)
    hist_nz  = hist[hist > 0]
    prob     = hist_nz / hist_nz.sum()
    entropy_val = float(-np.sum(prob * np.log2(prob)))

    hsv   = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
    h_ch  = hsv[:, :, 0]
    s_ch  = hsv[:, :, 1]
    v_ch  = hsv[:, :, 2]

    yellow_pct = float(np.sum((h_ch >= 15) & (h_ch <= 35) & (s_ch > 60) & (v_ch > 80)) / total_pixels * 100)
    green_pct  = float(np.sum((h_ch >= 35) & (h_ch <= 85) & (s_ch > 40) & (v_ch > 50)) / total_pixels * 100)
    dark_pct   = float(np.sum(v_ch < 50) / total_pixels * 100)
    brown_pct  = float(np.sum((h_ch >= 5) & (h_ch <= 20) & (s_ch > 40) & (v_ch > 40)) / total_pixels * 100)

    return {
        "stat_mean":       round(mean_val, 2),
        "stat_std":        round(std_val, 2),
        "stat_entropy":    round(entropy_val, 4),
        "stat_yellow_pct": round(yellow_pct, 2),
        "stat_green_pct":  round(green_pct, 2),
        "stat_dark_pct":   round(dark_pct, 2),
        "stat_brown_pct":  round(brown_pct, 2),
        "total_pixels":    total_pixels,
    }


# ─────────────────────────────────────────────────────────
#  Step 11 — Histogram plots
# ─────────────────────────────────────────────────────────

def render_rgb_histogram(img_bgr: np.ndarray) -> bytes:
    """Plot R, G, B channel histograms."""
    fig, axes = plt.subplots(1, 3, figsize=(12, 3))
    fig.patch.set_facecolor("#0d1117")

    channel_info = [
        (2, "Red",   "#e74c3c"),
        (1, "Green", "#27ae60"),
        (0, "Blue",  "#3498db"),
    ]
    for ax, (ch, name, color) in zip(axes, channel_info):
        ax.set_facecolor("#0d1117")
        hist = cv2.calcHist([img_bgr], [ch], None, [256], [0, 256])
        ax.fill_between(range(256), hist.ravel(), color=color, alpha=0.7)
        ax.plot(hist.ravel(), color=color, linewidth=1)
        ax.set_title(f"{name} Channel", color="white", fontsize=10, pad=4)
        ax.set_xlabel("Intensity", color="#aaa", fontsize=8)
        ax.set_ylabel("Pixel count", color="#aaa", fontsize=8)
        ax.tick_params(colors="#aaa", labelsize=7)
        for spine in ax.spines.values():
            spine.set_color("#333")

    fig.suptitle("RGB Histogram", color="white", fontsize=12, y=1.01)
    plt.tight_layout(pad=0.8)
    return _fig_to_img(fig)


def render_hsv_histogram(img_bgr: np.ndarray) -> bytes:
    """Plot H, S, V channel histograms."""
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    fig, axes = plt.subplots(1, 3, figsize=(12, 3))
    fig.patch.set_facecolor("#0d1117")

    channel_info = [
        (0, "Hue",        "#9b59b6", [0, 180]),
        (1, "Saturation", "#e67e22", [0, 256]),
        (2, "Value",      "#f1c40f", [0, 256]),
    ]
    for ax, (ch, name, color, rng) in zip(axes, channel_info):
        ax.set_facecolor("#0d1117")
        bins = 180 if ch == 0 else 256
        hist = cv2.calcHist([hsv], [ch], None, [bins], rng)
        ax.fill_between(range(bins), hist.ravel(), color=color, alpha=0.7)
        ax.plot(hist.ravel(), color=color, linewidth=1)
        ax.set_title(f"{name} Channel", color="white", fontsize=10, pad=4)
        ax.set_xlabel("Value", color="#aaa", fontsize=8)
        ax.set_ylabel("Pixel count", color="#aaa", fontsize=8)
        ax.tick_params(colors="#aaa", labelsize=7)
        for spine in ax.spines.values():
            spine.set_color("#333")

    fig.suptitle("HSV Histogram", color="white", fontsize=12, y=1.01)
    plt.tight_layout(pad=0.8)
    return _fig_to_img(fig)


def render_hue_detail(img_bgr: np.ndarray, plant_mask: np.ndarray = None) -> bytes:
    """Hue histogram (vegetation pixels only if mask provided)."""
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    hue_ch = hsv[:, :, 0]

    fig, ax = plt.subplots(figsize=(9, 3))
    fig.patch.set_facecolor("#0d1117")
    ax.set_facecolor("#0d1117")

    if plant_mask is not None and plant_mask.sum() > 0:
        hue_vals = hue_ch[plant_mask > 0]
        title = "Hue Histogram (vegetation pixels)"
    else:
        hue_vals = hue_ch.ravel()
        title = "Hue Histogram (full image)"

    hist, edges = np.histogram(hue_vals, bins=36, range=(0, 180))
    centers = (edges[:-1] + edges[1:]) / 2

    # Color each bar by approximate hue color
    colors_hue = []
    for h in centers:
        rgb = cv2.cvtColor(np.uint8([[[int(h), 255, 200]]]), cv2.COLOR_HSV2RGB)[0][0]
        colors_hue.append(tuple(c / 255 for c in rgb))

    ax.bar(centers, hist, width=5, color=colors_hue, alpha=0.85, edgecolor="none")
    ax.set_title(title, color="white", fontsize=11, pad=6)
    ax.set_xlabel("Hue (0–180)", color="#aaa", fontsize=9)
    ax.set_ylabel("Pixel count", color="#aaa", fontsize=9)
    ax.tick_params(colors="#aaa", labelsize=8)
    for spine in ax.spines.values():
        spine.set_color("#333")

    plt.tight_layout(pad=0.5)
    return _fig_to_img(fig)


# ─────────────────────────────────────────────────────────
#  Step 15 — Confidence score
# ─────────────────────────────────────────────────────────

def compute_confidence(score: int, crop: str) -> dict:
    """
    Convert the rule-based Disease Score into a RELATIVE class-confidence indicator.
    This is not a calibrated probability and is not produced by a trained ML model.
    """
    if crop == "Potato":
        # thresholds: sick>=5, suspicious>=2
        centers = {"Healthy": 0.5, "Suspicious": 3.0, "Sick": 6.0}
    else:
        # thresholds: sick>=5, suspicious>=3 (v8 algorithm)
        centers = {"Healthy": 1.0, "Suspicious": 4.0, "Sick": 7.0}

    # Relative affinity to the three score centers.
    raw = {cls: -((score - c) ** 2) for cls, c in centers.items()}
    exp_vals = {k: np.exp(v / 3.0) for k, v in raw.items()}
    total = sum(exp_vals.values())
    return {k: round(v / total, 3) for k, v in exp_vals.items()}


# ─────────────────────────────────────────────────────────
#  Step 20 — CSV row builder
# ─────────────────────────────────────────────────────────

def build_csv_row(
    image_name: str,
    crop: str,
    result: dict,
    quality: dict,
    confidence: dict,
    processing_time: float,
    experiment: str = "",
    ground_truth: str = "",
    img_stats: dict = None,
) -> dict:
    """Build one CSV row per the Step 20 spec."""
    label    = result.get("label", "")
    correct  = ""
    if ground_truth and label:
        correct = "1" if ground_truth.strip().lower() == label.strip().lower() else "0"

    gps_str = ""
    if result.get("latitude") and result.get("longitude"):
        gps_str = f"{result['latitude']:.6f},{result['longitude']:.6f}"

    stats = img_stats or {}
    return {
        "Experiment":      experiment,
        "Crop":            crop,
        "Image_ID":        image_name,
        "Ground_Truth":    ground_truth,
        "Mean_H":          result.get("mean_h", ""),
        "Mean_S":          result.get("sat_mean", result.get("mean_sat", "")),
        "Mean_V":          result.get("val_mean", ""),
        "Green_Percent":   result.get("green_ratio", result.get("green_pct", "")),
        "Yellow_Percent":  result.get("yellow_ratio", ""),
        "Brown_Percent":   result.get("brown_ratio", ""),
        "Dark_Percent":    result.get("dark_ratio", ""),
        "Vegetation_Pct":  result.get("veg_ratio", ""),
        "Prediction":      label,
        "Confidence":      confidence.get(label, ""),
        "Conf_Healthy":    confidence.get("Healthy", ""),
        "Conf_Suspicious": confidence.get("Suspicious", ""),
        "Conf_Sick":       confidence.get("Sick", ""),
        "Processing_Time": round(processing_time, 3),
        "GPS":             gps_str,
        "Sharpness":       quality.get("sharpness", ""),
        "Brightness":      quality.get("brightness_mean", ""),
        "Valid":           int(quality.get("valid", True)),
        "Correct":         correct,
        # ── Statistical metrics ──
        "Stat_Mean":       stats.get("stat_mean", ""),
        "Stat_Std":        stats.get("stat_std", ""),
        "Stat_Entropy":    stats.get("stat_entropy", ""),
        "Stat_Yellow_Pct": stats.get("stat_yellow_pct", ""),
        "Stat_Green_Pct":  stats.get("stat_green_pct", ""),
        "Stat_Dark_Pct":   stats.get("stat_dark_pct", ""),
        "Stat_Brown_Pct":  stats.get("stat_brown_pct", ""),
        "Total_Pixels":    stats.get("total_pixels", ""),
    }


def export_csv(rows: list) -> bytes:
    """Convert list of dicts to CSV bytes."""
    if not rows:
        return b""
    import io
    sio = io.StringIO()
    writer = csv.DictWriter(sio, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)
    return sio.getvalue().encode("utf-8")


def compute_performance_metrics(rows: list) -> dict:
    """
    Compute Accuracy, Precision, Recall, F1 and Confusion Matrix
    for rows that have both Ground_Truth and Prediction filled.
    Classes: Healthy, Suspicious, Sick.
    Returns dict with metrics and confusion matrix, or None if not enough labeled rows.
    """
    classes = ["Healthy", "Suspicious", "Sick"]
    # Metrics describe the 3-class disease classifier only. Rows classified as
    # No Plant (or carrying an unsupported label) are excluded rather than being
    # silently counted in the denominator without appearing in the confusion matrix.
    labeled = [
        r for r in rows
        if str(r.get("Ground_Truth", "")).strip() in classes
        and str(r.get("Prediction", "")).strip() in classes
    ]
    if len(labeled) < 2:
        return None
    # Confusion matrix: cm[true][pred]
    cm = {c: {p: 0 for p in classes} for c in classes}
    for r in labeled:
        gt  = r["Ground_Truth"].strip()
        pred = r["Prediction"].strip()
        if gt in cm and pred in classes:
            cm[gt][pred] += 1

    n = len(labeled)
    correct = sum(cm[c][c] for c in classes)
    accuracy = correct / n if n > 0 else 0.0

    precision, recall, f1 = {}, {}, {}
    for c in classes:
        tp = cm[c][c]
        fp = sum(cm[r][c] for r in classes) - tp
        fn = sum(cm[c][p] for p in classes) - tp
        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r_ = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f = 2 * p * r_ / (p + r_) if (p + r_) > 0 else 0.0
        precision[c] = round(p, 3)
        recall[c]    = round(r_, 3)
        f1[c]        = round(f, 3)

    macro_precision = round(sum(precision.values()) / len(classes), 3)
    macro_recall    = round(sum(recall.values()) / len(classes), 3)
    macro_f1        = round(sum(f1.values()) / len(classes), 3)

    return {
        "n_labeled": n,
        "accuracy":  round(accuracy, 3),
        "precision": precision,
        "recall":    recall,
        "f1":        f1,
        "macro_precision": macro_precision,
        "macro_recall":    macro_recall,
        "macro_f1":        macro_f1,
        "confusion_matrix": cm,
        "classes": classes,
    }


def export_csv_with_metrics(rows: list) -> bytes:
    """
    Export all rows as CSV, then append blank line + performance metrics section
    (Accuracy, Precision, Recall, F1, Confusion Matrix) if Ground_Truth is available.
    """
    if not rows:
        return b""
    import io
    sio = io.StringIO()
    writer = csv.DictWriter(sio, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)

    metrics = compute_performance_metrics(rows)
    if metrics:
        classes = metrics["classes"]
        sio.write("\n")
        sio.write("# ── Performance Metrics ──\n")
        sio.write(f"# N labeled,{metrics['n_labeled']}\n")
        sio.write(f"# Accuracy,{metrics['accuracy']}\n")
        sio.write("\n")
        sio.write("# Per-class Precision\n")
        sio.write("# Class," + ",".join(classes) + ",Macro\n")
        sio.write("# Precision," + ",".join(str(metrics['precision'][c]) for c in classes)
                  + f",{metrics['macro_precision']}\n")
        sio.write("# Recall,"    + ",".join(str(metrics['recall'][c])    for c in classes)
                  + f",{metrics['macro_recall']}\n")
        sio.write("# F1,"        + ",".join(str(metrics['f1'][c])        for c in classes)
                  + f",{metrics['macro_f1']}\n")
        sio.write("\n")
        sio.write("# Confusion Matrix (rows=True, cols=Predicted)\n")
        sio.write("# True\\Pred," + ",".join(classes) + "\n")
        for c in classes:
            sio.write(f"# {c}," + ",".join(str(metrics['confusion_matrix'][c][p]) for p in classes) + "\n")

    return sio.getvalue().encode("utf-8")


# ─────────────────────────────────────────────────────────
#  Step 21 — ZIP folder structure builder
# ─────────────────────────────────────────────────────────

def build_output_zip(
    image_name: str,
    img_bgr:    np.ndarray,
    img_proc:   np.ndarray,
    result:     dict,
    quality:    dict,
    confidence: dict,
    annotated:  bytes,
    plant_mask: np.ndarray,
    segmented_plant: np.ndarray,
    rgb_hist:   bytes,
    hsv_hist:   bytes,
    hue_hist:   bytes,
    csv_bytes:  bytes,
) -> bytes:
    """
    Build a ZIP with the organized folder structure (Step 21):
    <image_name>/
      original/   gray/   preprocessed/
      h_channel/  s_channel/  v_channel/
      clean_mask/ segmented/
      histograms/ overlay/    features/   result/
    """
    stem = os.path.splitext(image_name)[0]
    buf  = BytesIO()

    def _encode_bgr(img: np.ndarray) -> bytes:
        ok, enc = cv2.imencode(".png", img)
        return enc.tobytes() if ok else b""

    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        base = f"{stem}/"

        # original
        zf.writestr(base + "original/original.png", _encode_bgr(img_bgr))

        # gray
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        ok, enc = cv2.imencode(".png", gray)
        zf.writestr(base + "gray/gray.png", enc.tobytes() if ok else b"")

        # preprocessed
        zf.writestr(base + "preprocessed/preprocessed.png", _encode_bgr(img_proc))

        # HSV channels
        hsv = cv2.cvtColor(img_proc, cv2.COLOR_BGR2HSV)
        h_ch, s_ch, v_ch = cv2.split(hsv)
        ok_h, enc_h = cv2.imencode(".png", h_ch)
        ok_s, enc_s = cv2.imencode(".png", s_ch)
        ok_v, enc_v = cv2.imencode(".png", v_ch)
        zf.writestr(base + "h_channel/hue.png",        enc_h.tobytes() if ok_h else b"")
        zf.writestr(base + "s_channel/saturation.png", enc_s.tobytes() if ok_s else b"")
        zf.writestr(base + "v_channel/value.png",      enc_v.tobytes() if ok_v else b"")

        # Exact classifier plant mask + transparent leaf-only PNG
        ok_m, enc_m = cv2.imencode(".png", plant_mask.astype(np.uint8))
        zf.writestr(base + "clean_mask/plant_mask.png", enc_m.tobytes() if ok_m else b"")
        zf.writestr(base + "segmented/segmented_plant.png", _encode_bgr(segmented_plant))

        # histograms
        zf.writestr(base + "histograms/rgb_histogram.png", rgb_hist)
        zf.writestr(base + "histograms/hsv_histogram.png", hsv_hist)
        zf.writestr(base + "histograms/hue_detail.png",    hue_hist)

        # overlay (annotated image)
        zf.writestr(base + "overlay/annotated.png", annotated)

        # features (JSON-like text)
        feat_lines = [f"{k}: {v}" for k, v in result.items()]
        feat_lines += [f"confidence_{k}: {v}" for k, v in confidence.items()]
        feat_lines += [f"sharpness: {quality.get('sharpness', '')}"]
        feat_lines += [f"brightness_mean: {quality.get('brightness_mean', '')}"]
        zf.writestr(base + "features/features.txt", "\n".join(feat_lines).encode())

        # result
        label = result.get("label", "unknown")
        result_txt = (
            f"Classification: {label}\n"
            f"Score: {result.get('score', '')}\n"
            f"Confidence:\n"
            + "\n".join(f"  {k}: {v}" for k, v in confidence.items())
        )
        zf.writestr(base + "result/result.txt", result_txt.encode())

        # CSV
        if csv_bytes:
            zf.writestr(base + "result/data.csv", csv_bytes)

    buf.seek(0)
    return buf.read()


# ─────────────────────────────────────────────────────────
#  Shared background-removal helpers
# ─────────────────────────────────────────────────────────

def _remove_bg_potato(hsv: np.ndarray, total: int) -> tuple:
    """
    Build a plant mask for potato that excludes bare soil / row gaps.
    Designed for aerial/overhead field shots where pinkish sandy soil
    sits between rows of green plants.

    Returns (plant_mask_u8, plant_area, veg_ratio, soil_ratio).
    """
    hh = hsv[:, :, 0]
    s  = hsv[:, :, 1]
    v  = hsv[:, :, 2]

    # ── Broad soil exclusion: orange-red-tan hue, LOW-MEDIUM saturation ──
    soil_mask = (
        ((hh >= 0)  & (hh <= 30)) &
        (s  >= 10)  & (s  <= 110) &
        (v  >= 40)
    ).astype(np.uint8) * 255

    # Also catch pale pinkish/beige sand (very low saturation, bright)
    soil_sand = (
        ((hh >= 0) & (hh <= 25)) &
        (s <= 80) & (v >= 100)
    ).astype(np.uint8) * 255
    soil_mask = cv2.bitwise_or(soil_mask, soil_sand)

    # Dilate soil slightly to suppress soil/leaf border pixels
    soil_mask = cv2.dilate(soil_mask, np.ones((3, 3), np.uint8), iterations=1)

    # ── Green plant tissue anchor — stricter saturation >= 50 ──
    green_seed = (
        (hh >= 20) & (hh <= 100) &
        (s  >= 50) &
        (v  >= 30)
    ).astype(np.uint8) * 255
    green_seed = cv2.bitwise_and(green_seed, cv2.bitwise_not(soil_mask))

    # Expand green zone to capture adjacent disease pixels
    expanded_green = cv2.dilate(green_seed, np.ones((7, 7), np.uint8), iterations=3)
    expanded_green = cv2.bitwise_and(expanded_green, cv2.bitwise_not(soil_mask))

    # Disease candidates ONLY inside expanded green zone
    # Higher saturation threshold for brown (>= 70) to avoid catching dry earth
    brown1 = cv2.inRange(hsv, np.array([5,  70, 40]), np.array([22, 255, 220]))
    brown2 = cv2.inRange(hsv, np.array([0,  70, 40]), np.array([5,  255, 220]))
    brown_in_leaf = cv2.bitwise_and(cv2.bitwise_or(brown1, brown2), expanded_green)

    yellow_in_leaf = cv2.bitwise_and(
        cv2.inRange(hsv, np.array([15, 60, 80]), np.array([38, 255, 255])),
        expanded_green
    )
    dark_raw = ((v >= 20) & (v <= 80) & (s >= 40)).astype(np.uint8) * 255
    dark_in_leaf = cv2.bitwise_and(
        cv2.bitwise_and(dark_raw, cv2.bitwise_not(soil_mask)),
        expanded_green
    )

    plant_mask = cv2.bitwise_or(green_seed,
                  cv2.bitwise_or(brown_in_leaf,
                  cv2.bitwise_or(yellow_in_leaf, dark_in_leaf)))

    plant_mask = cv2.morphologyEx(plant_mask, cv2.MORPH_CLOSE,
                                   np.ones((7, 7), np.uint8), iterations=2)
    plant_mask = cv2.morphologyEx(plant_mask, cv2.MORPH_OPEN,
                                   np.ones((3, 3), np.uint8), iterations=1)

    soil_area  = int(np.sum(soil_mask > 0))
    plant_area = max(int(np.sum(plant_mask > 0)), 1)
    veg_ratio  = plant_area / total * 100
    soil_ratio = soil_area  / total * 100
    return plant_mask, plant_area, veg_ratio, soil_ratio




# ─────────────────────────────────────────────────────────
#  Potato classifier
# ─────────────────────────────────────────────────────────

def _analyze_cell_potato(cell_bgr: np.ndarray, forced_plant_mask: np.ndarray = None) -> dict:
    """Classify one grid cell: background removal → feature extraction → score."""
    if cell_bgr is None or cell_bgr.size == 0:
        return {"label": "Healthy", "score": 0, "veg_ratio": 0}
    h, w = cell_bgr.shape[:2]
    if h < 10 or w < 10:
        return {"label": "Healthy", "score": 0, "veg_ratio": 0}

    hsv   = cv2.cvtColor(cell_bgr, cv2.COLOR_BGR2HSV)
    gray  = cv2.cvtColor(cell_bgr, cv2.COLOR_BGR2GRAY)
    total = h * w

    # ── Step 1: background removal ──
    if forced_plant_mask is None:
        plant_mask, plant_area, veg_ratio, soil_ratio = _remove_bg_potato(hsv, total)
    else:
        plant_mask = ((forced_plant_mask > 0).astype(np.uint8) * 255)
        plant_area = int(np.sum(plant_mask > 0))
        veg_ratio = plant_area / max(total, 1) * 100.0
        soil_ratio = 0.0
    # A cell is "No Plant" only if very little vegetation detected (<10%)
    # Do NOT discard based on soil ratio alone — field images naturally have lots of soil
    if veg_ratio < 10:
        return {"label": "No Plant", "score": -1, "veg_ratio": round(veg_ratio, 1)}

    # ── Step 2: features on plant pixels only ──
    def _on_plant(mask):
        return cv2.bitwise_and(mask, plant_mask)

    b1 = cv2.inRange(hsv, np.array([10, 70, 50]), np.array([22, 255, 200]))
    b2 = cv2.inRange(hsv, np.array([0,  70, 50]), np.array([10, 255, 200]))
    brown_ratio  = np.sum(_on_plant(cv2.bitwise_or(b1, b2)) > 0) / plant_area * 100

    dark_raw  = cv2.inRange(hsv, np.array([0, 0, 0]), np.array([180, 60, 80]))
    dark_px   = cv2.morphologyEx(_on_plant(dark_raw), cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    dark_ratio = np.sum(dark_px > 0) / plant_area * 100

    yellow_px    = _on_plant(cv2.inRange(hsv, np.array([20, 60, 100]), np.array([35, 255, 255])))
    yellow_ratio = np.sum(yellow_px > 0) / plant_area * 100

    green_px    = _on_plant(cv2.inRange(hsv, np.array([25, 40, 40]), np.array([90, 255, 255])))
    green_ratio = np.sum(green_px > 0) / plant_area * 100

    sat_vals = hsv[:, :, 1][plant_mask > 0]
    mean_sat = float(np.mean(sat_vals)) if len(sat_vals) > 0 else 0

    # Mean H, V on plant pixels
    hue_vals = hsv[:, :, 0][plant_mask > 0]
    val_vals = hsv[:, :, 2][plant_mask > 0]
    mean_h   = float(np.mean(hue_vals)) if len(hue_vals) > 0 else 0
    val_mean = float(np.mean(val_vals)) if len(val_vals) > 0 else 0

    # ── Step 3: score (on plant pixels only — soil already excluded from mask) ──
    score = 0
    # Brown lesions on leaf tissue — thresholds raised slightly since mask is cleaner
    if   brown_ratio > 20:  score += 3
    elif brown_ratio > 8:   score += 2
    elif brown_ratio > 3:   score += 1
    # Yellowing
    if   yellow_ratio > 15: score += 2
    elif yellow_ratio > 5:  score += 1
    # Low saturation across the leaf (stress / wilt signal)
    if   mean_sat < 70:     score += 2
    elif mean_sat < 85:     score += 1
    # Low green coverage relative to total plant area (leaf loss / necrosis)
    if   green_ratio < 45:  score += 2
    elif green_ratio < 60:  score += 1
    # Dark necrotic patches (only penalise if substantial)
    if dark_ratio > 5:      score += 1

    label = "Sick" if score >= 5 else "Suspicious" if score >= 2 else "Healthy"
    return {
        "label": label, "score": score, "veg_ratio": round(veg_ratio, 1),
        "brown_ratio": round(brown_ratio, 1), "yellow_ratio": round(yellow_ratio, 1),
        "green_ratio": round(green_ratio, 1), "dark_ratio": round(dark_ratio, 1),
        "mean_sat": round(mean_sat, 1), "mean_h": round(mean_h, 1),
        "val_mean": round(val_mean, 1),
    }


# ─────────────────────────────────────────────────────────
#  Sugar Beet leaf segmentation
# ─────────────────────────────────────────────────────────


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

def _remove_bg_beet_leaves(cell_bgr: np.ndarray):
    """
    Segment beet LEAF tissue from soil/background.

    Uses a conservative vegetation seed in HSV plus Excess Green (ExG), then
    morphology and connected-component cleanup. The goal is to keep leaf pixels
    for feature extraction while rejecting grey/brown soil that previously
    entered the beet mask and created false Cercospora texture detections.

    Returns: plant_mask (uint8 0/255), plant_area, veg_ratio.
    """
    if cell_bgr is None or cell_bgr.size == 0:
        return np.zeros((1, 1), dtype=np.uint8), 0, 0.0

    h, w = cell_bgr.shape[:2]
    total = h * w
    hsv = cv2.cvtColor(cell_bgr, cv2.COLOR_BGR2HSV)
    hh = hsv[:, :, 0].astype(np.float32)
    ss = hsv[:, :, 1].astype(np.float32)
    vv = hsv[:, :, 2].astype(np.float32)

    # Excess Green: positive values indicate green-dominant vegetation.
    b, g, r = cv2.split(cell_bgr)
    exg = (2 * g.astype(np.int16) - r.astype(np.int16) - b.astype(np.int16))

    # Beet-leaf seed. H starts above the typical brown/grey soil range.
    leaf_seed = (
        (hh >= 28) & (hh <= 95) &
        (ss >= 25) & (vv >= 30) &
        (exg > 5)
    ).astype(np.uint8) * 255

    # Remove isolated soil noise and reconnect small gaps inside leaves.
    leaf_mask = cv2.morphologyEx(
        leaf_seed, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1
    )
    leaf_mask = cv2.morphologyEx(
        leaf_mask, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8), iterations=2
    )

    # Remove tiny disconnected fragments while keeping all meaningful leaf regions.
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(leaf_mask, connectivity=8)
    clean = np.zeros_like(leaf_mask)
    min_area = max(30, int(total * 0.0002))
    for idx in range(1, n_labels):
        if stats[idx, cv2.CC_STAT_AREA] >= min_area:
            clean[labels == idx] = 255

    plant_area = int(np.sum(clean > 0))
    veg_ratio = (plant_area / total * 100.0) if total else 0.0
    return clean, plant_area, veg_ratio


# ─────────────────────────────────────────────────────────
#  Beet Cercospora ring-lesion detector
# ─────────────────────────────────────────────────────────

def _detect_beet_cercospora_spots(cell_bgr: np.ndarray, plant_mask_u8: np.ndarray):
    """
    Detect classic Cercospora lesions: compact pale/grey centres surrounded by
    a darker brown/purple ring.  All candidates must lie inside the eroded leaf
    mask, which sharply reduces sun-glare and soil false positives.
    """
    if cell_bgr is None or cell_bgr.size == 0:
        shape = plant_mask_u8.shape if plant_mask_u8 is not None else (1, 1)
        return np.zeros(shape, dtype=np.uint8), 0, 0.0

    hsv = cv2.cvtColor(cell_bgr, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(cell_bgr, cv2.COLOR_BGR2GRAY)
    hh, ss, vv = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]

    h, w = gray.shape
    erode_k = 5 if min(h, w) < 150 else 7
    interior = cv2.erode(
        plant_mask_u8, np.ones((erode_k, erode_k), np.uint8), iterations=1
    )

    # Desaturated bright lesion centres.
    centre_mask = ((ss < 50) & (vv > 150) & (interior > 0)).astype(np.uint8) * 255
    centre_mask = cv2.morphologyEx(
        centre_mask, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8), iterations=1
    )

    contours, _ = cv2.findContours(centre_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    accepted = np.zeros_like(plant_mask_u8)
    count = 0

    for c in contours:
        area_c = cv2.contourArea(c)
        if not (8 < area_c < 160):
            continue
        perim = cv2.arcLength(c, True)
        if perim <= 0:
            continue
        circularity = 4.0 * np.pi * area_c / (perim * perim)
        if circularity < 0.45:
            continue

        M = cv2.moments(c)
        if M["m00"] <= 0:
            continue
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        r_in = max(2, int(np.sqrt(area_c / np.pi)))
        r_out = r_in + 6

        inner = np.zeros_like(gray, dtype=np.uint8)
        outer = np.zeros_like(gray, dtype=np.uint8)
        cv2.circle(inner, (cx, cy), r_in, 255, -1)
        cv2.circle(outer, (cx, cy), r_out, 255, -1)
        cv2.circle(outer, (cx, cy), r_in + 1, 0, -1)
        inner = cv2.bitwise_and(inner, plant_mask_u8)
        outer = cv2.bitwise_and(outer, plant_mask_u8)

        iv = gray[inner > 0]
        ov = gray[outer > 0]
        if iv.size < 3 or ov.size < 5:
            continue
        if float(iv.mean()) - float(ov.mean()) < 15.0:
            continue

        # A true Cercospora ring commonly contains brown/red-purple pixels.
        oh = hh[outer > 0]
        os = ss[outer > 0]
        ovv = vv[outer > 0]
        ring_like = (((oh < 25) | (oh > 140)) & (os > 45) & (ovv < 180))
        if ring_like.size == 0 or float(np.mean(ring_like)) < 0.10:
            continue

        cv2.drawContours(accepted, [c], -1, 255, -1)
        count += 1

    plant_area = max(int(np.sum(plant_mask_u8 > 0)), 1)
    density = count / plant_area * 10000.0
    return accepted, count, density


# ─────────────────────────────────────────────────────────
#  Beet small dark-lesion detector
# ─────────────────────────────────────────────────────────

def _detect_beet_dark_lesions(cell_bgr: np.ndarray, plant_mask_u8: np.ndarray):
    """
    Detect compact dark lesions INSIDE the segmented beet leaves.

    This is deliberately conservative: it looks for small, compact local minima
    whose centre is substantially darker than the surrounding leaf ring.  The
    eroded plant mask prevents soil, leaf borders and gaps between leaves from
    being counted as disease spots.

    Returns: (lesion_mask_u8, lesion_count, lesion_density_per_10k_leaf_pixels)
    """
    if cell_bgr is None or cell_bgr.size == 0 or plant_mask_u8 is None:
        shape = cell_bgr.shape[:2] if cell_bgr is not None and cell_bgr.size else (1, 1)
        return np.zeros(shape, dtype=np.uint8), 0, 0.0

    gray = cv2.cvtColor(cell_bgr, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(cell_bgr, cv2.COLOR_BGR2HSV)
    hh = hsv[:, :, 0]

    h, w = gray.shape
    # Grid cells are small; use a slightly smaller erosion there.
    erode_k = 5 if min(h, w) < 150 else 9
    interior = cv2.erode(
        plant_mask_u8, np.ones((erode_k, erode_k), np.uint8), iterations=1
    )

    local_mean = cv2.GaussianBlur(gray.astype(np.float32), (15, 15), 0)
    local_diff = gray.astype(np.float32) - local_mean

    # Strong local dark minima only.  This rejects ordinary leaf texture.
    candidate = ((local_diff < -40) & (interior > 0)).astype(np.uint8) * 255
    candidate = cv2.morphologyEx(
        candidate, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8), iterations=1
    )

    contours, _ = cv2.findContours(candidate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    lesion_mask = np.zeros_like(plant_mask_u8)
    count = 0

    for c in contours:
        area_c = cv2.contourArea(c)
        if not (15 <= area_c <= 120):
            continue

        perim = cv2.arcLength(c, True)
        if perim <= 0:
            continue
        circularity = 4.0 * np.pi * area_c / (perim * perim)
        if circularity < 0.50:
            continue

        M = cv2.moments(c)
        if M["m00"] <= 0:
            continue
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])

        r_in = max(2, int(np.sqrt(area_c / np.pi)))
        r_out = r_in + 5
        inner = np.zeros_like(gray, dtype=np.uint8)
        outer = np.zeros_like(gray, dtype=np.uint8)
        cv2.circle(inner, (cx, cy), r_in, 255, -1)
        cv2.circle(outer, (cx, cy), r_out, 255, -1)
        cv2.circle(outer, (cx, cy), r_in + 1, 0, -1)
        inner = cv2.bitwise_and(inner, plant_mask_u8)
        outer = cv2.bitwise_and(outer, plant_mask_u8)

        iv = gray[inner > 0]
        ov = gray[outer > 0]
        if iv.size < 3 or ov.size < 5:
            continue

        # Disease spot centre must be clearly darker than its immediate leaf ring.
        if float(ov.mean()) - float(iv.mean()) < 22.0:
            continue

        # Most of the ring should still look like leaf tissue, not soil/background.
        ring_h = hh[outer > 0]
        leaf_ring_fraction = float(np.mean((ring_h >= 25) & (ring_h <= 100))) if ring_h.size else 0.0
        if leaf_ring_fraction < 0.45:
            continue

        cv2.drawContours(lesion_mask, [c], -1, 255, -1)
        count += 1

    plant_area = max(int(np.sum(plant_mask_u8 > 0)), 1)
    density = count / plant_area * 10000.0
    return lesion_mask, count, density


# ─────────────────────────────────────────────────────────
#  Beet classifier
# ─────────────────────────────────────────────────────────

def _analyze_cell_beet(cell_bgr: np.ndarray, forced_plant_mask: np.ndarray = None) -> dict:
    """
    Classify one beet cell — v8 algorithm.
    Key fix: bleach_cerc restricted to H=15-28 only (not H>28 = green range).
    true_yellow (H=20-30, S>80, V>100) is the real disease signal.
    Thresholds: Sick ≥5, Suspicious ≥3, Healthy <3.
    """
    if cell_bgr is None or cell_bgr.size == 0:
        return {"label": "Healthy", "score": 0, "veg_ratio": 0}
    h, w = cell_bgr.shape[:2]
    if h < 10 or w < 10:
        return {"label": "Healthy", "score": 0, "veg_ratio": 0}

    hsv   = cv2.cvtColor(cell_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
    gray  = cv2.cvtColor(cell_bgr, cv2.COLOR_BGR2GRAY)
    total = h * w
    hh, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]

    # Leaf-only segmentation: soil/background is excluded BEFORE feature extraction.
    if forced_plant_mask is None:
        plant_mask_u8, plant_area, veg_ratio = _remove_bg_beet_leaves(cell_bgr)
    else:
        plant_mask_u8 = ((forced_plant_mask > 0).astype(np.uint8) * 255)
        plant_area = int(np.sum(plant_mask_u8 > 0))
        veg_ratio = plant_area / max(total, 1) * 100.0
    plant_mask = plant_mask_u8 > 0
    if plant_area == 0:
        return {"label": "No Plant", "score": -1, "veg_ratio": 0.0}
    if veg_ratio < 10:
        return {"label": "No Plant", "score": -1, "veg_ratio": round(veg_ratio, 1)}

    true_yellow = (hh >= 20) & (hh <= 30) & (s > 80) & (v > 100) & plant_mask
    ty_pct      = float(np.sum(true_yellow)) / plant_area * 100

    s_mean = float(np.mean(s[plant_mask]))

    # CRITICAL FIX: H ≤ 28 only — H>28 is green range, catches sun-lit healthy leaves
    bleach_cerc = (hh >= 15) & (hh <= 28) & (s >= 20) & (s < 60) & (v > 130) & plant_mask
    bleach_pct  = float(np.sum(bleach_cerc)) / plant_area * 100

    # HARD LEAF-FIRST RULE: local texture is calculated only after segmentation.
    # Outside-leaf pixels are replaced by the leaf median before local filtering,
    # so dark soil cannot create false lesion contrast along the leaf boundary.
    leaf_gray_vals = gray[plant_mask]
    leaf_fill = float(np.median(leaf_gray_vals)) if leaf_gray_vals.size else 0.0
    gray_leaf_only = gray.astype(np.float32).copy()
    gray_leaf_only[~plant_mask] = leaf_fill
    blur = cv2.GaussianBlur(gray_leaf_only, (21, 21), 0)
    diff = gray_leaf_only - blur
    interior_mask = cv2.erode(plant_mask_u8, np.ones((3,3), np.uint8), iterations=1) > 0
    dark_sp  = float(np.sum((diff < -15) & interior_mask)) / plant_area * 100
    light_sp = float(np.sum((diff > 20) & interior_mask)) / plant_area * 100

    green_px  = plant_mask & (hh >= 35) & (hh <= 85) & (s > 40)
    # Healthy-green coverage is measured INSIDE the segmented leaf area, not the full image.
    green_pct = float(np.sum(green_px)) / plant_area * 100

    yellow_ratio = float(np.sum(plant_mask & (hh >= 18) & (hh <= 38) & (s > 70) & (v > 80))) / plant_area * 100
    dark_ratio   = float(np.sum(plant_mask & (v < 60))) / plant_area * 100
    mean_h   = float(np.mean(hh[plant_mask]))
    val_mean = float(np.mean(v[plant_mask]))

    # Compact black/dark lesions: a complementary Cercospora signal.
    dark_lesion_mask, dark_lesion_count, dark_lesion_density = _detect_beet_dark_lesions(
        cell_bgr, plant_mask_u8
    )

    # ── Cercospora ring lesions ──
    cercospora_mask, cercospora_count, cercospora_density = _detect_beet_cercospora_spots(
        cell_bgr, plant_mask_u8
    )

    score = 0
    if   ty_pct > 0.30: score += 3
    elif ty_pct > 0.15: score += 2
    if s_mean < 80 and bleach_pct > 1.5:
        score += 2
    if s_mean > 175 and ty_pct > 0.10:
        score += 2
    # Strong healthy-foliage evidence: when most segmented leaf pixels are green and
    # yellow/bleached evidence is low, local sun/shadow texture must not dominate the score.
    healthy_green_dominant = (
        green_pct >= 70.0 and ty_pct < 0.30 and bleach_pct < 1.50
    )

    if not healthy_green_dominant:
        if dark_sp > 4.0 and light_sp > 3.0:
            score += 1

        # Cercospora spots — weighted by inverse of s_mean.
        if cercospora_count > 0:
            spot_weight = max(0.3, 1.0 - (s_mean - 60) / 70.0)
            weighted = cercospora_count * spot_weight
            if   weighted >= 25: score += 4
            elif weighted >= 10: score += 3
            elif weighted >= 4:  score += 2
            elif weighted >= 1:  score += 1

    # Small black lesions are common in field Cercospora images and may appear
    # while most of the leaf is still green.  Add a conservative independent
    # contribution so they are not hidden by the healthy-green safeguard.
    if dark_lesion_count >= 12:
        score += 3
    elif dark_lesion_count >= 7 and green_pct < 75.0:
        score += 2
    elif dark_lesion_count >= 4 and green_pct < 65.0:
        score += 1

    label = "Sick" if score >= 5 else "Suspicious" if score >= 3 else "Healthy"
    return {
        "label": label, "score": score, "veg_ratio": round(veg_ratio, 1),
        "ty_pct": round(ty_pct, 3), "s_mean": round(s_mean, 1),
        "bleach_pct": round(bleach_pct, 2), "dark_sp": round(dark_sp, 1),
        "light_sp": round(light_sp, 1),
        "yellow_ratio": round(yellow_ratio, 1), "sat_mean": round(s_mean, 1),
        "green_pct": round(green_pct, 1), "dark_ratio": round(dark_ratio, 1),
        "mean_h": round(mean_h, 1), "val_mean": round(val_mean, 1),
        "hue_entropy": 0.0,
        "cercospora_count": cercospora_count,
        "cercospora_density": round(cercospora_density, 2),
        "dark_lesion_count": int(dark_lesion_count),
        "dark_lesion_density": round(float(dark_lesion_density), 2),
        "purple_ratio": 0.0,
        "healthy_green_dominant": bool(healthy_green_dominant),
    }




# ─────────────────────────────────────────────────────────
#  Grid analysis engine
# ─────────────────────────────────────────────────────────

def analyze_field(img_bgr: np.ndarray, crop: str, grid_rows: int, grid_cols: int):
    """
    LEAF-FIRST field analysis.

    The complete image is segmented ONCE before it is divided into grid cells.
    Every cell receives the corresponding crop of that global leaf mask. In
    addition, a global lesion-evidence mask is projected into each cell so
    small diseased spots are not lost simply because the rest of the cell is
    mostly healthy green tissue.
    """
    H, W = img_bgr.shape[:2]
    cell_h = H // grid_rows
    cell_w = W // grid_cols
    analyze = _analyze_cell_potato if crop == "Potato" else _analyze_cell_beet

    # Isolate leaves on the WHOLE image first.
    global_leaf_mask = get_classifier_plant_mask(img_bgr, crop)
    global_leaf_mask = ((global_leaf_mask > 0).astype(np.uint8) * 255)
    global_lesion_mask = build_lesion_evidence_mask(img_bgr, crop, global_leaf_mask)

    results_matrix = []
    score_matrix = np.zeros((grid_rows, grid_cols))
    label_matrix = np.full((grid_rows, grid_cols), "No Plant", dtype=object)

    progress = st.progress(0, text="Isolating leaves and analyzing field...")
    for row in range(grid_rows):
        row_results = []
        for col in range(grid_cols):
            y1 = row * cell_h; y2 = H if row == grid_rows-1 else (row+1)*cell_h
            x1 = col * cell_w; x2 = W if col == grid_cols-1 else (col+1)*cell_w
            cell = img_bgr[y1:y2, x1:x2]
            cell_leaf_mask = global_leaf_mask[y1:y2, x1:x2]
            cell_lesion_mask = global_lesion_mask[y1:y2, x1:x2]
            leaf_px = int(np.count_nonzero(cell_leaf_mask))
            leaf_pct = 100.0 * leaf_px / max(cell_leaf_mask.size, 1)

            # Background if almost no leaf pixels are present.
            if leaf_pct < 6.0:
                res = {"label": "No Plant", "score": -1, "veg_ratio": round(leaf_pct, 1)}
                lesion_ratio = 0.0
                lesion_px = 0
            else:
                res = analyze(cell, forced_plant_mask=cell_leaf_mask)
                lesion_px = int(np.count_nonzero((cell_lesion_mask > 0) & (cell_leaf_mask > 0)))
                lesion_ratio = 100.0 * lesion_px / max(leaf_px, 1)

                # Spatial refinement: visible lesion evidence inside this cell can
                # upgrade the local map classification even when the cell-average
                # colour statistics are diluted by surrounding healthy green tissue.
                if crop == "Potato":
                    if lesion_ratio >= 5.0 or lesion_px >= max(18, int(0.012 * leaf_px)):
                        res["label"] = "Sick"
                        res["score"] = max(int(res.get("score", 0)), 5)
                    elif lesion_ratio >= 1.0 or lesion_px >= max(8, int(0.004 * leaf_px)):
                        if res.get("label") == "Healthy":
                            res["label"] = "Suspicious"
                        res["score"] = max(int(res.get("score", 0)), 2)
                else:
                    if lesion_ratio >= 3.0 or lesion_px >= max(16, int(0.009 * leaf_px)):
                        res["label"] = "Sick"
                        res["score"] = max(int(res.get("score", 0)), 5)
                    elif lesion_ratio >= 0.8 or lesion_px >= max(7, int(0.003 * leaf_px)):
                        if res.get("label") == "Healthy":
                            res["label"] = "Suspicious"
                        res["score"] = max(int(res.get("score", 0)), 2)

            res["row"] = row; res["col"] = col
            res["lesion_ratio_cell"] = round(float(lesion_ratio), 2)
            res["lesion_px_cell"] = int(lesion_px)
            row_results.append(res)
            score_matrix[row, col] = max(res.get("score", -1), 0)
            label_matrix[row, col] = res["label"]
        results_matrix.append(row_results)
        progress.progress((row+1)/grid_rows, text=f"Analyzing row {row+1}/{grid_rows}...")
    progress.empty()

    counts = {"Healthy":0, "Suspicious":0, "Sick":0, "No Plant":0}
    for rr in results_matrix:
        for r in rr:
            counts[r["label"]] = counts.get(r["label"], 0) + 1
    plant = counts["Healthy"] + counts["Suspicious"] + counts["Sick"]
    summary = {
        "healthy":counts["Healthy"], "suspicious":counts["Suspicious"],
        "sick":counts["Sick"], "no_plant":counts["No Plant"], "plant_cells":plant,
        "sick_pct":round(counts["Sick"]/max(plant,1)*100,1),
        "suspicious_pct":round(counts["Suspicious"]/max(plant,1)*100,1),
        "alert_level": "HIGH" if counts["Sick"] > plant*0.30 else ("MEDIUM" if counts["Sick"]+counts["Suspicious"] > plant*0.30 else "LOW"),
        "global_leaf_mask": global_leaf_mask,
        "global_lesion_mask": global_lesion_mask,
    }
    return results_matrix, score_matrix, label_matrix, summary


# ─────────────────────────────────────────────────────────
#  Map rendering
# ─────────────────────────────────────────────────────────

def render_overlay(img_bgr, results_matrix, grid_rows, grid_cols):
    """Original image with colored cell overlays.

    The final row/column explicitly consume any remainder pixels so every supported
    grid size covers the complete image without leaving an unclassified strip.
    """
    H, W = img_bgr.shape[:2]
    cell_h = H // grid_rows
    cell_w = W // grid_cols

    fig, ax = plt.subplots(figsize=(8, 6))
    fig.patch.set_facecolor("#0d1117")
    ax.set_facecolor("#0d1117")
    ax.imshow(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))

    rgba = {
        "Healthy":    (0.15, 0.75, 0.25, 0.20),
        "Suspicious": (1.00, 0.65, 0.00, 0.32),
        "Sick":       (0.85, 0.15, 0.15, 0.42),
        "No Plant":   (0.00, 0.00, 0.00, 0.00),
    }
    for row in range(grid_rows):
        for col in range(grid_cols):
            r = results_matrix[row][col]
            c = rgba.get(r["label"], (0, 0, 0, 0))
            y1 = row * cell_h
            y2 = H if row == grid_rows - 1 else (row + 1) * cell_h
            x1 = col * cell_w
            x2 = W if col == grid_cols - 1 else (col + 1) * cell_w
            ax.add_patch(plt.Rectangle(
                (x1, y1), x2 - x1, y2 - y1,
                linewidth=0.6, edgecolor="white",
                facecolor=c[:3], alpha=c[3]
            ))
            if r["label"] in ("Sick", "Suspicious"):
                ax.text((x1 + x2) / 2, (y1 + y2) / 2,
                        str(r["score"]),
                        color="white", fontsize=max(5, min(9, 70 // grid_rows)),
                        ha="center", va="center", fontweight="bold")

    ax.set_xlim(0, W); ax.set_ylim(H, 0); ax.axis("off")
    ax.set_title("Field — Classification Overlay", color="white", fontsize=11, pad=6)
    plt.tight_layout(pad=0.3)
    return _fig_to_img(fig)


def render_heatmap(score_matrix, grid_rows, grid_cols, label_matrix=None):
    """Traffic-light heat map calibrated to the final zone labels.

    Healthy cells stay clearly green, suspicious cells become yellow, sick cells red,
    and background/no-plant cells are muted dark gray. This prevents contradictions
    between the grid classification and the heat-map view.
    """
    display_scores = np.full((grid_rows, grid_cols), np.nan, dtype=float)
    text_scores = np.zeros((grid_rows, grid_cols), dtype=float)

    for rr in range(grid_rows):
        for cc in range(grid_cols):
            label = label_matrix[rr, cc] if label_matrix is not None else None
            score = float(score_matrix[rr, cc])
            text_scores[rr, cc] = score
            if label == "No Plant":
                display_scores[rr, cc] = np.nan
            elif label == "Healthy":
                display_scores[rr, cc] = 0.14
            elif label == "Suspicious":
                display_scores[rr, cc] = min(0.74, 0.54 + 0.20 * min(score / 8.0, 1.0))
            elif label == "Sick":
                display_scores[rr, cc] = min(1.0, 0.82 + 0.18 * min(score / 8.0, 1.0))
            else:
                display_scores[rr, cc] = 0.14 if score <= 0 else min(score / 8.0, 1.0)

    fig, ax = plt.subplots(figsize=(6.8, 4.6))
    fig.patch.set_facecolor("#0d1117")
    ax.set_facecolor("#0d1117")

    cmap = LinearSegmentedColormap.from_list(
        "disease_calibrated", [(0.0, "#1a8f3a"), (0.55, "#f0c040"), (1.0, "#c0392b")], N=256
    )
    cmap.set_bad("#1f2937")
    im = ax.imshow(display_scores, cmap=cmap, vmin=0, vmax=1, interpolation="nearest", aspect="auto")

    for x in range(grid_cols + 1):
        ax.axvline(x - 0.5, color="white", linewidth=0.3, alpha=0.35)
    for y in range(grid_rows + 1):
        ax.axhline(y - 0.5, color="white", linewidth=0.3, alpha=0.35)

    for row in range(grid_rows):
        for col in range(grid_cols):
            label = label_matrix[row, col] if label_matrix is not None else None
            val = display_scores[row, col]
            if np.isnan(val):
                continue
            if label in ("Suspicious", "Sick"):
                ax.text(col, row, f"{int(text_scores[row, col])}",
                        ha="center", va="center",
                        color="white", fontsize=max(5, min(9, 70 // grid_rows)),
                        fontweight="bold")

    cbar = plt.colorbar(im, ax=ax, fraction=0.035, pad=0.03)
    cbar.set_ticks([0.14, 0.60, 0.90])
    cbar.set_ticklabels(["Healthy", "Suspicious", "Sick"])
    cbar.set_label("Zone status", color="white", fontsize=9)
    cbar.ax.yaxis.set_tick_params(color="white", labelcolor="white")
    ax.set_xlabel("Column", color="#aaa", fontsize=9)
    ax.set_ylabel("Row",    color="#aaa", fontsize=9)
    ax.tick_params(colors="#aaa", labelsize=8)
    ax.set_title("Traffic-light disease heat map", color="white", fontsize=11, pad=6)
    plt.tight_layout(pad=0.3)
    return _fig_to_img(fig)


def render_categorical(label_matrix, grid_rows, grid_cols, summary):
    """Categorical ✓ / ? / ✗ classification grid."""
    fig, ax = plt.subplots(figsize=(7, 6))
    fig.patch.set_facecolor("#0d1117")
    ax.set_facecolor("#0d1117")

    lti  = {"Healthy": 0, "Suspicious": 1, "Sick": 2, "No Plant": -1}
    mat  = np.array([[lti.get(label_matrix[r, c], 0)
                      for c in range(grid_cols)]
                     for r in range(grid_rows)])
    cmap = LinearSegmentedColormap.from_list(
        "cls", ["#666666", "#1a7a1a", "#d4a017", "#c0392b"], N=4
    )
    ax.imshow(mat, cmap=cmap, vmin=-1, vmax=2,
              interpolation="nearest", aspect="auto")

    sym = {"Healthy": "✓", "Suspicious": "?", "Sick": "✗", "No Plant": "·"}
    for r in range(grid_rows):
        for c in range(grid_cols):
            ax.text(c, r, sym.get(label_matrix[r, c], ""),
                    ha="center", va="center", color="white",
                    fontsize=max(6, min(11, 70 // grid_rows)), fontweight="bold")

    for x in range(grid_cols + 1):
        ax.axvline(x - 0.5, color="white", linewidth=0.4, alpha=0.45)
    for y in range(grid_rows + 1):
        ax.axhline(y - 0.5, color="white", linewidth=0.4, alpha=0.45)

    legend_patches = [
        mpatches.Patch(color="#1a7a1a", label=f"✓ Healthy    ({summary['healthy']})"),
        mpatches.Patch(color="#d4a017", label=f"? Suspicious ({summary['suspicious']})"),
        mpatches.Patch(color="#c0392b", label=f"✗ Sick       ({summary['sick']})"),
        mpatches.Patch(color="#666666", label=f"· No Plant   ({summary['no_plant']})"),
    ]
    ax.legend(handles=legend_patches, loc="lower right",
              facecolor="#1a1a2e", edgecolor="#444",
              labelcolor="white", fontsize=8, framealpha=0.85)

    ax.set_xlabel("Column", color="#aaa", fontsize=9)
    ax.set_ylabel("Row",    color="#aaa", fontsize=9)
    ax.tick_params(colors="#aaa", labelsize=8)
    ax.set_title("Classification Map", color="white", fontsize=11, pad=6)
    plt.tight_layout(pad=0.3)
    return _fig_to_img(fig)


def _fig_to_img(fig) -> bytes:
    """Convert matplotlib figure to PNG bytes."""
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    buf.seek(0)
    return buf.read()


# ─────────────────────────────────────────────────────────
#  Single-plant / single-leaf mode
# ─────────────────────────────────────────────────────────

def detect_image_type(img_bgr: np.ndarray) -> str:
    """
    Decide whether an image shows a single close-up plant/leaf or a field/multi-plant scene.
    Returns 'single_plant' or 'field'.

    Primary signal: single-leaf lab images have a large uniform neutral background
    (gray/white), while field images do not.
    Secondary signals: soil presence, dark shadows, stem colour.
    """
    H, W = img_bgr.shape[:2]
    scale = min(800 / max(H, W), 1.0)
    img = cv2.resize(img_bgr, (int(W * scale), int(H * scale)), interpolation=cv2.INTER_AREA)
    h, w = img.shape[:2]
    total = h * w

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hh, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]

    # ── PRIMARY: uniform neutral background (lab / studio shot) ──────────────
    # Neutral = low saturation + not very dark (not shadow)
    lab_bg = ((s < 25) & (v > 130)).astype(np.uint8) * 255
    lab_closed = cv2.morphologyEx(lab_bg, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
    cnts, _ = cv2.findContours(lab_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    biggest_blob_pct = max((cv2.contourArea(c) for c in cnts), default=0) / total * 100
    # A single connected neutral blob >12% of the image = plain lab background
    if biggest_blob_pct > 12:
        return "single_plant"

    # ── SECONDARY scoring: field-specific signals ─────────────────────────────
    field_score = 0

    # Soil (sandy/brown) between plant rows
    soil = (((hh >= 5) & (hh <= 35)) & (s < 90) & (v > 50)).astype(np.uint8) * 255
    soil_pct = float(np.sum(soil > 0)) / total * 100
    if soil_pct > 8:
        field_score += 3
    elif soil_pct > 4:
        field_score += 2
    elif soil_pct > 2:
        field_score += 1

    # Shadow pixels between small leaves in dense field
    dark_pct = float(np.sum(v < 50)) / total * 100
    if dark_pct > 15:
        field_score += 2
    elif dark_pct > 8:
        field_score += 1

    # Potato/beet stem colour (reddish-brown stems visible from above)
    stem = (((hh >= 140) | (hh <= 10)) & (s > 60) & (v > 30) & (v < 180)).astype(np.uint8) * 255
    stem = cv2.morphologyEx(stem, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8), iterations=2)
    stem_pct = float(np.sum(stem > 0)) / total * 100
    if stem_pct > 8:
        field_score += 2
    elif stem_pct > 4:
        field_score += 1

    # High green coverage fraction (field = dense canopy)
    green = ((hh >= 25) & (hh <= 90) & (s >= 40) & (v >= 35)).astype(np.uint8) * 255
    green_pct = float(np.sum(green > 0)) / total * 100
    if green_pct > 55:
        field_score += 1

    return "field" if field_score >= 3 else "single_plant"


def _segment_potato_closeup_foreground(img_bgr: np.ndarray) -> np.ndarray:
    """Extract the dominant foreground potato leaf on a CLOSE-UP photograph.

    Unlike a green HSV threshold, foreground/background modelling can separate
    bright symptomatic leaf tissue from other green leaves in the background.
    GrabCut uses a spatial prior (foreground at the photographed centre),
    vegetation-colour seeds and corner background seeds; holes caused by brown
    or black lesions are filled after segmentation. This is NOT a replacement
    for the field-canopy vegetation mask.

    Returns an 8-bit binary mask (255 leaf, 0 background) at input resolution.
    """
    if img_bgr is None or img_bgr.size == 0:
        return np.zeros((1, 1), dtype=np.uint8)
    h0, w0 = img_bgr.shape[:2]
    if h0 < 30 or w0 < 30:
        return np.zeros((h0, w0), dtype=np.uint8)

    # GrabCut is expensive at large image sizes. Segment at up to 480 px on
    # the long side and upscale the BINARY result with nearest-neighbour.
    scale = min(1.0, 480.0 / max(h0, w0))
    w = max(30, int(round(w0 * scale)))
    h = max(30, int(round(h0 * scale)))
    work = cv2.resize(img_bgr, (w, h), interpolation=cv2.INTER_AREA) if scale < 1 else img_bgr.copy()
    work = cv2.GaussianBlur(work, (3, 3), 0)
    hsv = cv2.cvtColor(work, cv2.COLOR_BGR2HSV)
    sat, val = hsv[:, :, 1], hsv[:, :, 2]
    yy, xx = np.ogrid[:h, :w]

    # The central subject is estimated as the dominant foreground leaf.
    # The photograph may contain OTHER GREEN LEAVES, so colour alone is not
    # a valid foreground/background criterion.
    centre_x, centre_y = 0.53 * w, 0.51 * h
    ellipse = ((xx - centre_x) / (0.47 * w)) ** 2 + ((yy - centre_y) / (0.49 * h)) ** 2
    central_seed = ((xx - centre_x) / (0.21 * w)) ** 2 + ((yy - centre_y) / (0.26 * h)) ** 2

    gc_mask = np.full((h, w), cv2.GC_PR_BGD, dtype=np.uint8)
    gc_mask[ellipse < 1.0] = cv2.GC_PR_FGD
    border = ((xx < .03 * w) | (xx > .97 * w) |
              (yy < .03 * h) | (yy > .97 * h))
    # Keep the top middle row unconstrained: the close-up leaf can reach the
    # image edge. Corner wedges, not the full border, are definite background.
    safe_bg = border & ((xx < .22*w) | (xx > .93*w)) & (ellipse >= 1.0)
    gc_mask[safe_bg] = cv2.GC_BGD
    # Only unambiguously plant-coloured central pixels are HARD foreground.
    core = (central_seed < 1.0) & (sat > 30) & (val > 70)
    # A second seed on the upper blade prevents a yellowed/diseased top half
    # from being discarded when the central lower blade is brighter.
    upper_seed = (((xx - .56*w)/(.17*w))**2 + ((yy - .30*h)/(.15*h))**2 < 1)
    upper_core = upper_seed & (sat > 30) & (val > 95)
    gc_mask[core | upper_core] = cv2.GC_FGD

    # Both foreground and background must have initial seeds.
    if int(np.count_nonzero(core)) < 30 or int(np.count_nonzero(gc_mask == cv2.GC_BGD)) < 30:
        return np.zeros((h0, w0), dtype=np.uint8)

    background_model = np.zeros((1, 65), np.float64)
    foreground_model = np.zeros((1, 65), np.float64)
    try:
        # OpenCV GrabCut uses random initial colour clusters. Fix the seed so
        # repeated analyses of the same image give the same binary mask.
        cv2.setRNGSeed(1729)
        cv2.grabCut(work, gc_mask, None, background_model,
                    foreground_model, 3, cv2.GC_INIT_WITH_MASK)
    except cv2.error:
        return np.zeros((h0, w0), dtype=np.uint8)

    region = np.where(
        (gc_mask == cv2.GC_FGD) | (gc_mask == cv2.GC_PR_FGD),
        255, 0
    ).astype(np.uint8)
    region = cv2.morphologyEx(region, cv2.MORPH_CLOSE,
                              np.ones((9, 9), np.uint8), iterations=1)

    # Retain the dominant foreground connected region; suppress isolated
    # distant foliage and fill internal holes (necrotic lesions are leaf).
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(region, 8)
    if count <= 1:
        return np.zeros((h0, w0), dtype=np.uint8)
    center_pt = np.array([centre_x, centre_y], dtype=np.float32)
    best = 0
    best_score = -1.0
    for i in range(1, count):
        area = float(stats[i, cv2.CC_STAT_AREA])
        if area < 0.02 * w * h:
            continue
        dist = float(np.linalg.norm((centroids[i] - center_pt) / np.array([w, h])))
        score = area / (w * h) - .08 * dist
        if score > best_score:
            best_score, best = score, i
    if best == 0:
        return np.zeros((h0, w0), dtype=np.uint8)

    region = (labels == best).astype(np.uint8) * 255
    contours, _ = cv2.findContours(region, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled = np.zeros_like(region)
    if contours:
        cv2.drawContours(filled, [max(contours, key=cv2.contourArea)], -1, 255, -1)
    # Remove jagged pixel artefacts but retain the outline and lesions.
    filled = cv2.morphologyEx(filled, cv2.MORPH_CLOSE,
                              np.ones((5, 5), np.uint8), iterations=1)

    return cv2.resize(filled, (w0, h0), interpolation=cv2.INTER_NEAREST)



def _remove_bg_potato_leaf(hsv: np.ndarray, total: int) -> tuple:
    """Single close-up potato leaf segmentation used by the classifier AND GUI.

    Returns (plant_mask, plant_area, coverage_percent, symptomatic_fraction).
    A binary mask contains all leaf tissue, including necrotic regions.
    It does not consist of spots alone. All outputs derive from one mask.
    """
    img_bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    plant_mask = _segment_potato_closeup_foreground(img_bgr)
    plant_area = int(np.count_nonzero(plant_mask))
    coverage = 100.0 * plant_area / max(int(total), 1)

    hh, ss, vv = cv2.split(hsv)
    brown = ((hh <= 24) & (ss >= 35) & (vv >= 25) & (vv <= 225))
    yellow = ((hh >= 14) & (hh <= 40) & (ss >= 30) & (vv >= 55))
    dark = ((vv >= 18) & (vv <= 90) & (ss >= 18))
    symptomatic = (brown | yellow | dark) & (plant_mask > 0)
    symptomatic_fraction = 100.0 * int(np.count_nonzero(symptomatic)) / max(plant_area, 1)
    return plant_mask, max(plant_area, 1), coverage, symptomatic_fraction


def _analyze_single_leaf_potato(cell_bgr: np.ndarray, supplied_mask: np.ndarray = None) -> dict:
    """
    Classify a single isolated potato leaf (lab / close-up image with neutral background).
    Uses more sensitive thresholds than the field-grid version.
    """
    if cell_bgr is None or cell_bgr.size == 0:
        return {"label": "Healthy", "score": 0, "veg_ratio": 0}
    h, w = cell_bgr.shape[:2]
    if h < 20 or w < 20:
        return {"label": "Healthy", "score": 0, "veg_ratio": 0}

    hsv   = cv2.cvtColor(cell_bgr, cv2.COLOR_BGR2HSV)
    total = h * w

    if supplied_mask is None:
        plant_mask, plant_area, veg_ratio, disease_ratio = _remove_bg_potato_leaf(hsv, total)
    else:
        plant_mask = ((supplied_mask > 0).astype(np.uint8) * 255)
        plant_area = max(int(np.count_nonzero(plant_mask)), 1)
        veg_ratio = 100.0 * plant_area / max(total, 1)
        hh, ss, vv = cv2.split(hsv)
        symptomatic = ((hh <= 24) & (ss >= 35) & (vv >= 25) & (vv <= 225)) | ((hh >= 14) & (hh <= 40) & (ss >= 30) & (vv >= 55)) | ((vv >= 18) & (vv <= 90) & (ss >= 18))
        disease_ratio = 100.0 * int(np.count_nonzero(symptomatic & (plant_mask > 0))) / plant_area

    if veg_ratio < 5:
        return {"label": "No Plant", "score": -1, "veg_ratio": round(veg_ratio, 1)}

    hh = hsv[:, :, 0]
    s  = hsv[:, :, 1]
    v  = hsv[:, :, 2]

    def _on_plant(mask):
        return cv2.bitwise_and(mask, plant_mask)

    # Brown lesion ratio (sensitive thresholds for close-up leaves)
    b1 = cv2.inRange(hsv, np.array([5,  35, 35]), np.array([22, 255, 220]))
    b2 = cv2.inRange(hsv, np.array([0,  35, 35]), np.array([5,  255, 220]))
    brown_ratio = np.sum(_on_plant(cv2.bitwise_or(b1, b2)) > 0) / plant_area * 100

    # Yellow/chlorotic tissue ratio
    yellow_px    = _on_plant(cv2.inRange(hsv, np.array([15, 35, 60]), np.array([38, 255, 255])))
    yellow_ratio = np.sum(yellow_px > 0) / plant_area * 100

    # Dark necrotic patches
    dark_raw   = (v >= 20) & (v <= 80) & (s >= 20)
    dark_in    = _on_plant((dark_raw.astype(np.uint8)) * 255)
    dark_ratio = np.sum(dark_in > 0) / plant_area * 100

    # Green ratio (healthy tissue)
    green_px    = _on_plant(cv2.inRange(hsv, np.array([22, 35, 35]), np.array([90, 255, 255])))
    green_ratio = np.sum(green_px > 0) / plant_area * 100

    # Mean saturation on leaf pixels
    sat_vals = s[plant_mask > 0]
    mean_sat = float(np.mean(sat_vals)) if len(sat_vals) > 0 else 0.0

    hue_vals = hh[plant_mask > 0]
    val_vals = v[plant_mask > 0]
    mean_h   = float(np.mean(hue_vals)) if len(hue_vals) > 0 else 0.0
    val_mean = float(np.mean(val_vals)) if len(val_vals) > 0 else 0.0

    # ── Scoring — sensitive thresholds for isolated leaf images ──
    score = 0

    # Brown lesions — even 2% is suspicious on an isolated leaf
    if   brown_ratio > 15: score += 3
    elif brown_ratio > 5:  score += 2
    elif brown_ratio > 1.5: score += 1

    # Yellowing (chlorosis / mosaic)
    if   yellow_ratio > 20: score += 2
    elif yellow_ratio > 8:  score += 1

    # Dark necrosis
    if   dark_ratio > 10:  score += 2
    elif dark_ratio > 3:   score += 1

    # Low green coverage = significant diseased area
    if   green_ratio < 40:  score += 2
    elif green_ratio < 60:  score += 1

    # Low mean saturation = pale / stressed leaf
    if   mean_sat < 65:  score += 2
    elif mean_sat < 80:  score += 1

    # Precomputed overall disease pixel ratio (from mask step)
    if   disease_ratio > 25: score += 2
    elif disease_ratio > 10: score += 1

    label = "Sick" if score >= 5 else "Suspicious" if score >= 2 else "Healthy"
    return {
        "label": label, "score": score, "veg_ratio": round(veg_ratio, 1),
        "brown_ratio": round(brown_ratio, 1), "yellow_ratio": round(yellow_ratio, 1),
        "green_ratio": round(green_ratio, 1), "dark_ratio": round(dark_ratio, 1),
        "disease_ratio": round(disease_ratio, 1),
        "mean_sat": round(mean_sat, 1), "mean_h": round(mean_h, 1),
        "val_mean": round(val_mean, 1),
    }


def analyze_single_plant(img_bgr: np.ndarray, crop: str) -> dict:
    """Classify a single-plant / single-leaf image as one unit.
    Auto-detects field vs. single-leaf and uses the appropriate classifier."""
    if crop == "Potato":
        img_type = detect_image_type(img_bgr)
        if img_type == "field":
            return _analyze_cell_potato(img_bgr)
        else:
            return _analyze_single_leaf_potato(img_bgr)
    else:
        return _analyze_cell_beet(img_bgr)


def get_classifier_plant_mask(img_bgr: np.ndarray, crop: str) -> np.ndarray:
    """
    Return the exact plant mask used by the active single-image classifier.
    This helper is visualization-only: it does not change classification logic.
    """
    if img_bgr is None or img_bgr.size == 0:
        return np.zeros((1, 1), dtype=np.uint8)

    h, w = img_bgr.shape[:2]
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    total = h * w

    if crop == "Potato":
        img_type = detect_image_type(img_bgr)
        if img_type == "field":
            plant_mask, _, _, _ = _remove_bg_potato(hsv, total)
        else:
            plant_mask, _, _, _ = _remove_bg_potato_leaf(hsv, total)
        return plant_mask.astype(np.uint8)

    # Sugar Beet: EXACT leaf-only mask used inside _analyze_cell_beet().
    plant_mask, _, _ = _remove_bg_beet_leaves(img_bgr)
    return plant_mask


def make_segmented_plant(img_bgr: np.ndarray, plant_mask: np.ndarray) -> np.ndarray:
    """Return BGRA plant image with a truly transparent background."""
    if plant_mask.shape[:2] != img_bgr.shape[:2]:
        plant_mask = cv2.resize(
            plant_mask, (img_bgr.shape[1], img_bgr.shape[0]),
            interpolation=cv2.INTER_NEAREST
        )
    alpha = np.where(plant_mask > 0, 255, 0).astype(np.uint8)
    bgra = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2BGRA)
    bgra[:, :, 3] = alpha
    return bgra


def _clean_binary_mask(mask_u8: np.ndarray, min_area_px: int = 25, open_ks: int = 3, close_ks: int = 5) -> np.ndarray:
    """Morphologically clean a binary lesion mask and remove tiny speckles."""
    binary = ((mask_u8 > 0).astype(np.uint8) * 255)
    if binary.size == 0:
        return binary

    if open_ks and open_ks > 1:
        k_open = np.ones((open_ks, open_ks), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, k_open, iterations=1)
    if close_ks and close_ks > 1:
        k_close = np.ones((close_ks, close_ks), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, k_close, iterations=1)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    cleaned = np.zeros_like(binary)
    for idx in range(1, num_labels):
        area = int(stats[idx, cv2.CC_STAT_AREA])
        if area >= int(min_area_px):
            cleaned[labels == idx] = 255
    return cleaned


def _traffic_light_thresholds(crop: str):
    """Cell-level lesion ratio thresholds (%) used to paint traffic-light outputs."""
    if crop == "Sugar Beet":
        return 0.7, 2.2
    return 0.9, 3.0


def _build_spatial_status_maps(leaf_mask_u8: np.ndarray, lesion_mask_u8: np.ndarray, overall_status: str, grid_size: int, crop: str):
    """Create grid-aligned traffic-light labels shared by field-grid and heat-map views."""
    h, w = leaf_mask_u8.shape[:2]
    rows = cols = max(2, int(grid_size))
    cell_h = h / rows
    cell_w = w / cols

    label_matrix = np.full((rows, cols), "No Plant", dtype=object)
    ratio_matrix = np.zeros((rows, cols), dtype=float)
    intensity_matrix = np.full((rows, cols), np.nan, dtype=float)
    susp_thr, sick_thr = _traffic_light_thresholds(crop)

    for r in range(rows):
        for c in range(cols):
            y1 = int(round(r * cell_h)); y2 = int(round((r + 1) * cell_h)) if r < rows - 1 else h
            x1 = int(round(c * cell_w)); x2 = int(round((c + 1) * cell_w)) if c < cols - 1 else w
            lm = leaf_mask_u8[y1:y2, x1:x2] > 0
            area = max(1, (y2 - y1) * (x2 - x1))
            leaf_px = int(lm.sum())

            if leaf_px < max(20, int(0.03 * area)):
                continue

            lesion_px = int(((lesion_mask_u8[y1:y2, x1:x2] > 0) & lm).sum())
            ratio = 100.0 * lesion_px / max(leaf_px, 1)
            ratio_matrix[r, c] = ratio

            if overall_status == "Healthy":
                label = "Healthy"
            elif overall_status == "Suspicious":
                label = "Suspicious" if ratio >= susp_thr else "Healthy"
            else:
                if ratio >= sick_thr:
                    label = "Sick"
                elif ratio >= susp_thr:
                    label = "Suspicious"
                else:
                    label = "Healthy"

            label_matrix[r, c] = label
            if label == "Healthy":
                intensity_matrix[r, c] = 0.14
            elif label == "Suspicious":
                intensity_matrix[r, c] = min(0.74, 0.54 + 0.20 * min(ratio / max(sick_thr, 1e-6), 1.0))
            elif label == "Sick":
                intensity_matrix[r, c] = min(1.0, 0.82 + 0.18 * min((ratio - sick_thr) / max(sick_thr, 1e-6), 1.0))

    return label_matrix, ratio_matrix, intensity_matrix


def build_lesion_evidence_mask(img_bgr: np.ndarray, crop: str, plant_mask_u8: np.ndarray) -> np.ndarray:
    """Build a cleaner lesion-evidence mask inside the segmented plant only.

    The classifier itself is unchanged. This mask is used for visualization and lesion
    morphology, so it is aggressively cleaned to keep the heat map aligned with the
    field-grid result and to avoid noisy red/yellow artefacts.
    """
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    hh, ss, vv = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    plant = plant_mask_u8 > 0
    plant_area = max(int(np.count_nonzero(plant_mask_u8)), 1)

    if crop == "Sugar Beet":
        yellow = ((hh >= 18) & (hh <= 38) & (ss > 85) & (vv > 90) & plant).astype(np.uint8) * 255
        cerc, _, _ = _detect_beet_cercospora_spots(img_bgr, plant_mask_u8)
        dark, _, _ = _detect_beet_dark_lesions(img_bgr, plant_mask_u8)
        raw_mask = cv2.bitwise_or(yellow, cv2.bitwise_or(cerc, dark))
        cleaned = _clean_binary_mask(raw_mask, min_area_px=max(18, int(plant_area * 0.00008)), open_ks=3, close_ks=5)
    else:
        green = ((hh >= 32) & (hh <= 98) & (ss >= 30) & (vv >= 30))
        brown = (((hh >= 4) & (hh <= 26) & (ss >= 45) & (vv >= 28) & (vv <= 225)) |
                 ((hh <= 5) & (ss >= 45) & (vv >= 28) & (vv <= 225)))
        yellow = ((hh >= 14) & (hh <= 40) & (ss >= 40) & (vv >= 65))
        dark_necrotic = (vv >= 18) & (vv <= 112) & (ss >= 18) & (~green)
        raw_mask = ((brown | yellow | dark_necrotic) & plant).astype(np.uint8) * 255
        cleaned = _clean_binary_mask(raw_mask, min_area_px=max(16, int(plant_area * 0.00008)), open_ks=3, close_ks=5)
        cleaned = cv2.dilate(cleaned, np.ones((3, 3), np.uint8), iterations=1)
        cleaned = cv2.bitwise_and(cleaned, plant_mask_u8)

    return cv2.bitwise_and(cleaned, plant_mask_u8)


def compute_lesion_morphology(lesion_mask_u8: np.ndarray, plant_mask_u8: np.ndarray) -> dict:
    """Connected-component morphology of disease evidence inside leaf area."""
    plant_area = int(np.count_nonzero(plant_mask_u8))
    if plant_area <= 0:
        return {"lesion_count": 0, "lesion_area_pct": 0.0, "lesion_density": 0.0,
                "mean_lesion_area": 0.0, "median_lesion_area": 0.0, "mean_circularity": 0.0}
    binary = (lesion_mask_u8 > 0).astype(np.uint8) * 255
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    areas, circs = [], []
    min_area = max(4.0, plant_area * 0.00001)
    for c in contours:
        a = float(cv2.contourArea(c))
        if a < min_area:
            continue
        p = float(cv2.arcLength(c, True))
        if p <= 0:
            continue
        areas.append(a)
        circs.append(float(4.0 * np.pi * a / (p * p)))
    lesion_pixels = int(np.count_nonzero(binary))
    return {
        "lesion_count": len(areas),
        "lesion_area_pct": round(100.0 * lesion_pixels / plant_area, 2),
        "lesion_density": round(len(areas) / plant_area * 10000.0, 2),
        "mean_lesion_area": round(float(np.mean(areas)) if areas else 0.0, 1),
        "median_lesion_area": round(float(np.median(areas)) if areas else 0.0, 1),
        "mean_circularity": round(float(np.mean(circs)) if circs else 0.0, 3),
    }

def run_core_pipeline(img_bgr: np.ndarray, crop: str, preprocessed: np.ndarray = None) -> dict:
    """Run the exact shared disease-analysis chain used by single and batch modes.

    Classification logic is identical in both modes:
    preprocess -> plant segmentation -> crop-specific classification -> lesion evidence.
    Candidate lesion evidence is only reported to the user when the final class is
    Suspicious or Sick, preventing contradictory Healthy + red-lesion displays.
    """
    img_proc = preprocessed if preprocessed is not None else preprocess_img(img_bgr)
    # Single close-up potato leaves require object-level foreground separation.
    # Reuse the IDENTICAL binary mask in classification, morphology, overlays
    # and the segmentation stage; do not recompute it independently.
    if crop == "Potato" and detect_image_type(img_proc) == "single_plant":
        hsv_proc = cv2.cvtColor(img_proc, cv2.COLOR_BGR2HSV)
        plant_mask, _, coverage_pct, _ = _remove_bg_potato_leaf(
            hsv_proc, img_proc.shape[0] * img_proc.shape[1]
        )
        result = _analyze_single_leaf_potato(img_proc, supplied_mask=plant_mask)
    else:
        result = analyze_single_plant(img_proc, crop)
        plant_mask = get_classifier_plant_mask(img_proc, crop)
    plant_mask = ((plant_mask > 0).astype(np.uint8) * 255)

    raw_lesion_mask = build_lesion_evidence_mask(img_proc, crop, plant_mask)
    raw_lesion_mask = cv2.bitwise_and(raw_lesion_mask, plant_mask)

    if result.get("label") in ("Suspicious", "Sick"):
        reported_lesion_mask = raw_lesion_mask
    else:
        reported_lesion_mask = np.zeros_like(raw_lesion_mask)

    lesion_metrics = compute_lesion_morphology(reported_lesion_mask, plant_mask)
    result = dict(result)
    result.update(lesion_metrics)

    return {
        "processed": img_proc,
        "result": result,
        "plant_mask": plant_mask,
        "raw_lesion_mask": raw_lesion_mask,
        "lesion_mask": reported_lesion_mask,
        "lesion_metrics": lesion_metrics,
    }


def _uploaded_files_signature(uploaded_files, crop: str, grid_size: int):
    """Small deterministic signature used to keep results alive across Streamlit reruns.

    Uses filename, byte length and hashes of the first/last 64 KiB. This is fast enough
    for field batches while still detecting a replaced file with the same name.
    """
    if not uploaded_files:
        return None
    items = []
    for f in uploaded_files:
        data = f.getvalue()
        sample = data if len(data) <= 131072 else data[:65536] + data[-65536:]
        digest = hashlib.sha1(sample).hexdigest()
        items.append((f.name, len(data), digest))
    return (crop, int(grid_size), tuple(items))



def evaluate_segmentation_mask(pred_mask_u8: np.ndarray, gt_bytes) -> dict | None:
    """Compute IoU/Dice against an optional manually annotated binary GT mask."""
    if not gt_bytes:
        return None
    arr = np.frombuffer(gt_bytes, np.uint8)
    gt = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)
    if gt is None:
        return None
    if gt.ndim == 3:
        if gt.shape[2] == 4:
            alpha = gt[:, :, 3]
            rgb = cv2.cvtColor(gt[:, :, :3], cv2.COLOR_BGR2GRAY)
            gt = np.maximum(alpha, rgb)
        else:
            gt = cv2.cvtColor(gt, cv2.COLOR_BGR2GRAY)
    gt = cv2.resize(gt, (pred_mask_u8.shape[1], pred_mask_u8.shape[0]), interpolation=cv2.INTER_NEAREST)
    gt_b = gt > 127
    pr_b = pred_mask_u8 > 0
    inter = int(np.count_nonzero(gt_b & pr_b))
    union = int(np.count_nonzero(gt_b | pr_b))
    denom = int(np.count_nonzero(gt_b)) + int(np.count_nonzero(pr_b))
    iou = inter / union if union else 1.0
    dice = (2.0 * inter / denom) if denom else 1.0
    return {"iou": iou, "dice": dice, "gt_pixels": int(np.count_nonzero(gt_b)),
            "pred_pixels": int(np.count_nonzero(pr_b))}




# ─────────────────────────────────────────────────────────
#  UI
# ─────────────────────────────────────────────────────────

def preprocess_img(img_bgr: np.ndarray) -> np.ndarray:
    """
    Pre-processing (Step 9.1.3.5):
    1. Resize to 800×600
    2. Gaussian blur (noise reduction)
    3. CLAHE on L-channel (adaptive contrast)
    4. Value-channel clipping (highlight suppression)
    """
    img = cv2.resize(img_bgr, (800, 600), interpolation=cv2.INTER_AREA)
    img = cv2.GaussianBlur(img, (3, 3), sigmaX=0.8)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img = cv2.cvtColor(cv2.merge([clahe.apply(l), a, b]), cv2.COLOR_LAB2BGR)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h_ch, s_ch, v_ch = cv2.split(hsv)
    v_clip = np.clip(v_ch, 0, int(np.percentile(v_ch, 99))).astype(np.uint8)
    return cv2.cvtColor(cv2.merge([h_ch, s_ch, v_clip]), cv2.COLOR_HSV2BGR)


# ─────────────────────────────────────────────────────────
#  Robustness experiment — illumination / preprocessing
# ─────────────────────────────────────────────────────────

ROBUSTNESS_LEVELS = [
    (-30, 0.70),
    (-20, 0.80),
    (0,   1.00),
    (20,  1.20),
    (30,  1.30),
]


def adjust_illumination(img_bgr: np.ndarray, factor: float) -> np.ndarray:
    """Synthetic illumination change: I_alpha = clip(alpha * I, 0, 255)."""
    arr = img_bgr.astype(np.float32) * float(factor)
    return np.clip(arr, 0, 255).astype(np.uint8)


def resize_baseline(img_bgr: np.ndarray) -> np.ndarray:
    """
    Baseline used in the robustness experiment.
    Resize is kept in BOTH branches so the comparison isolates the photometric
    pre-processing (Gaussian + CLAHE + P99 clipping) instead of image geometry.
    """
    return cv2.resize(img_bgr, (800, 600), interpolation=cv2.INTER_AREA)


def analyze_robustness_image(img_bgr: np.ndarray, crop: str) -> dict:
    """Use the field classifier consistently for the robustness experiment."""
    return _analyze_cell_potato(img_bgr) if crop == "Potato" else _analyze_cell_beet(img_bgr)


def _safe_confidence(result: dict, crop: str) -> dict:
    """Confidence only for actual plant classifications; No Plant gets blanks."""
    if result.get("label") == "No Plant" or result.get("score", -1) < 0:
        return {"Healthy": np.nan, "Suspicious": np.nan, "Sick": np.nan}
    return compute_confidence(int(result.get("score", 0)), crop)


def run_robustness_experiment(image_records: list) -> tuple:
    """
    Run each image at -30%, -20%, original, +20%, +30% illumination.

    Each field-image variant is analyzed twice with the same crop-specific field classifier:
      A) Baseline: resize to 800x600 only.
      B) Full preprocessing: resize + Gaussian + CLAHE + P99 V clipping.

    Returns:
      detailed_df: one row per image × illumination level.
      summary_df: aggregate results by illumination level.
      crop_summary_df: aggregate results by crop and illumination level.
    """
    rows = []
    total_runs = max(len(image_records) * len(ROBUSTNESS_LEVELS), 1)
    progress = st.progress(0, text="Running robustness experiment...")
    done = 0

    for rec in image_records:
        image_name = rec["image_name"]
        crop = rec["crop"]
        img_bgr = rec["img_bgr"]

        for delta_pct, factor in ROBUSTNESS_LEVELS:
            variant = adjust_illumination(img_bgr, factor)

            # A — baseline: same geometry, no Gaussian/CLAHE/P99 normalization
            baseline = resize_baseline(variant)
            q_before = check_image_quality(baseline)
            t0 = time.time()
            res_before = analyze_robustness_image(baseline, crop)
            t_before = time.time() - t0
            conf_before = _safe_confidence(res_before, crop)

            # B — full preprocessing
            processed = preprocess_img(variant)
            q_after = check_image_quality(processed)
            t0 = time.time()
            res_after = analyze_robustness_image(processed, crop)
            t_after = time.time() - t0
            conf_after = _safe_confidence(res_after, crop)

            rows.append({
                "Image_ID": image_name,
                "Crop": crop,
                "Illumination_Change_%": delta_pct,
                "Alpha": factor,
                "Brightness_Before": q_before["brightness_mean"],
                "Brightness_After": q_after["brightness_mean"],
                "Brightness_STD_Before": q_before["brightness_std"],
                "Brightness_STD_After": q_after["brightness_std"],
                "Overexposed_%_Before": q_before["overexposed_pct"],
                "Overexposed_%_After": q_after["overexposed_pct"],
                "Underexposed_%_Before": q_before["underexposed_pct"],
                "Underexposed_%_After": q_after["underexposed_pct"],
                "Quality_Valid_Before": int(q_before["valid"]),
                "Quality_Valid_After": int(q_after["valid"]),
                "Score_Without_PP": res_before.get("score", np.nan),
                "Score_With_PP": res_after.get("score", np.nan),
                "Classification_Without_PP": res_before.get("label", ""),
                "Classification_With_PP": res_after.get("label", ""),
                "Confidence_Without_PP": conf_before.get(res_before.get("label", ""), np.nan),
                "Confidence_With_PP": conf_after.get(res_after.get("label", ""), np.nan),
                "Vegetation_%_Without_PP": res_before.get("veg_ratio", np.nan),
                "Vegetation_%_With_PP": res_after.get("veg_ratio", np.nan),
                "Processing_Time_Without_PP_s": round(t_before, 4),
                "Processing_Time_With_PP_s": round(t_after, 4),
            })

            done += 1
            progress.progress(done / total_runs,
                              text=f"{done}/{total_runs}: {image_name} ({delta_pct:+d}%)")

    progress.empty()
    detailed_df = pd.DataFrame(rows)
    if detailed_df.empty:
        return detailed_df, pd.DataFrame(), pd.DataFrame()

    # Original (0%) result is the reference for stability, separately for each branch.
    refs = detailed_df[detailed_df["Illumination_Change_%"] == 0][[
        "Image_ID", "Crop", "Score_Without_PP", "Score_With_PP",
        "Classification_Without_PP", "Classification_With_PP"
    ]].rename(columns={
        "Score_Without_PP": "Reference_Score_Without_PP",
        "Score_With_PP": "Reference_Score_With_PP",
        "Classification_Without_PP": "Reference_Class_Without_PP",
        "Classification_With_PP": "Reference_Class_With_PP",
    })

    detailed_df = detailed_df.merge(refs, on=["Image_ID", "Crop"], how="left")
    detailed_df["Stable_Without_PP"] = (
        detailed_df["Classification_Without_PP"] == detailed_df["Reference_Class_Without_PP"]
    ).astype(int)
    detailed_df["Stable_With_PP"] = (
        detailed_df["Classification_With_PP"] == detailed_df["Reference_Class_With_PP"]
    ).astype(int)
    detailed_df["Score_Deviation_Without_PP"] = (
        detailed_df["Score_Without_PP"] - detailed_df["Reference_Score_Without_PP"]
    ).abs()
    detailed_df["Score_Deviation_With_PP"] = (
        detailed_df["Score_With_PP"] - detailed_df["Reference_Score_With_PP"]
    ).abs()

    def _summarize(df: pd.DataFrame, group_cols: list) -> pd.DataFrame:
        out = df.groupby(group_cols, as_index=False).agg(
            Images=("Image_ID", "count"),
            Mean_Brightness_Before=("Brightness_Before", "mean"),
            Mean_Brightness_After=("Brightness_After", "mean"),
            Mean_Brightness_STD_Before=("Brightness_STD_Before", "mean"),
            Mean_Brightness_STD_After=("Brightness_STD_After", "mean"),
            Mean_Overexposed_Before=("Overexposed_%_Before", "mean"),
            Mean_Overexposed_After=("Overexposed_%_After", "mean"),
            Mean_Underexposed_Before=("Underexposed_%_Before", "mean"),
            Mean_Underexposed_After=("Underexposed_%_After", "mean"),
            Mean_Score_Without_PP=("Score_Without_PP", "mean"),
            Mean_Score_With_PP=("Score_With_PP", "mean"),
            Mean_Score_Deviation_Without_PP=("Score_Deviation_Without_PP", "mean"),
            Mean_Score_Deviation_With_PP=("Score_Deviation_With_PP", "mean"),
            Stability_Without_PP=("Stable_Without_PP", "mean"),
            Stability_With_PP=("Stable_With_PP", "mean"),
        )
        out["Classification_Stability_Without_PP_%"] = out.pop("Stability_Without_PP") * 100
        out["Classification_Stability_With_PP_%"] = out.pop("Stability_With_PP") * 100
        numeric_cols = out.select_dtypes(include=[np.number]).columns
        out[numeric_cols] = out[numeric_cols].round(2)
        return out

    summary_df = _summarize(detailed_df, ["Illumination_Change_%"])
    crop_summary_df = _summarize(detailed_df, ["Crop", "Illumination_Change_%"])
    return detailed_df, summary_df, crop_summary_df


def render_robustness_mode(potato_files, beet_files):
    """Dedicated UI for the illumination robustness experiment."""
    st.markdown("## 🧪 Illumination Robustness Test")
    st.markdown(
        "This experiment compares the classifier under **−30%, −20%, original, +20%, +30%** "
        "illumination, with and without the photometric preprocessing stage. "
        "Both branches are resized to 800×600 so image geometry is identical."
    )

    image_records = []
    skipped_files = []
    for crop, files in [("Potato", potato_files or []), ("Sugar Beet", beet_files or [])]:
        for f in files:
            raw = f.getvalue()
            data = np.frombuffer(raw, np.uint8)
            img = cv2.imdecode(data, cv2.IMREAD_COLOR)
            if img is None:
                skipped_files.append(f.name)
                continue
            # Same safety cap used by the normal application.
            h, w = img.shape[:2]
            if max(h, w) > 2000:
                scale = 2000 / max(h, w)
                img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
            sample = raw if len(raw) <= 131072 else raw[:65536] + raw[-65536:]
            fingerprint = hashlib.sha1(sample).hexdigest()
            image_records.append({"image_name": f.name, "crop": crop, "img_bgr": img, "fingerprint": fingerprint})

    if skipped_files:
        st.warning("Skipped unreadable image file(s): " + ", ".join(skipped_files))

    n_potato = sum(1 for r in image_records if r["crop"] == "Potato")
    n_beet = sum(1 for r in image_records if r["crop"] == "Sugar Beet")
    st.info(
        f"Loaded **{len(image_records)} images**: {n_potato} Potato + {n_beet} Sugar Beet. "
        f"The experiment will perform **{len(image_records) * 5 * 2} classifier runs**."
    )

    if not image_records:
        st.caption("Upload representative field images in the sidebar, then run the experiment.")
        return

    if len(image_records) < 10:
        st.warning("For the project book, use about 10–15 representative field images if possible.")

    dataset_signature = tuple(
        sorted((r["crop"], r["image_name"], r.get("fingerprint", "")) for r in image_records)
    )
    run_clicked = st.button("▶ Run Robustness Test", key="run_robustness_test")

    if run_clicked:
        with st.spinner("Generating illumination variants and running both analysis branches..."):
            detailed_df, summary_df, crop_summary_df = run_robustness_experiment(image_records)

        st.session_state["robustness_detailed"] = detailed_df
        st.session_state["robustness_summary"] = summary_df
        st.session_state["robustness_crop_summary"] = crop_summary_df
        st.session_state["robustness_dataset_signature"] = dataset_signature
        st.success("Robustness experiment completed.")
    elif st.session_state.get("robustness_dataset_signature") == dataset_signature:
        detailed_df = st.session_state.get("robustness_detailed", pd.DataFrame())
        summary_df = st.session_state.get("robustness_summary", pd.DataFrame())
        crop_summary_df = st.session_state.get("robustness_crop_summary", pd.DataFrame())
        if detailed_df.empty:
            return
    else:
        st.caption("Click **Run Robustness Test** to generate the comparison tables and graphs.")
        return

    # Overall key result: exclude the original 0% row because it is trivially stable by definition.
    changed = detailed_df[detailed_df["Illumination_Change_%"] != 0]
    stability_no = changed["Stable_Without_PP"].mean() * 100 if len(changed) else np.nan
    stability_yes = changed["Stable_With_PP"].mean() * 100 if len(changed) else np.nan
    score_dev_no = changed["Score_Deviation_Without_PP"].mean() if len(changed) else np.nan
    score_dev_yes = changed["Score_Deviation_With_PP"].mean() if len(changed) else np.nan

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Stability without PP", f"{stability_no:.1f}%")
    m2.metric("Stability with PP", f"{stability_yes:.1f}%",
              delta=f"{stability_yes - stability_no:+.1f} pp")
    m3.metric("Mean score deviation without PP", f"{score_dev_no:.2f}")
    m4.metric("Mean score deviation with PP", f"{score_dev_yes:.2f}",
              delta=f"{score_dev_yes - score_dev_no:+.2f}", delta_color="inverse")

    st.markdown("### Summary table — all images")
    st.dataframe(summary_df, hide_index=True, use_container_width=True)

    with st.expander("Summary by crop", expanded=False):
        st.dataframe(crop_summary_df, hide_index=True, use_container_width=True)

    with st.expander("Detailed results — every image and illumination level", expanded=False):
        st.dataframe(detailed_df, hide_index=True, use_container_width=True)

    # Illumination robustness graphs — balanced, side-by-side presentation.
    plot_df = summary_df[summary_df["Illumination_Change_%"] != 0]
    graph_col1, graph_col2 = st.columns(2, gap="large")

    with graph_col1:
        st.markdown("#### Classification stability")
        fig, ax = plt.subplots(figsize=(8.8, 5.2))
        ax.plot(plot_df["Illumination_Change_%"],
                plot_df["Classification_Stability_Without_PP_%"], marker="o", label="Without preprocessing")
        ax.plot(plot_df["Illumination_Change_%"],
                plot_df["Classification_Stability_With_PP_%"], marker="o", label="With preprocessing")
        ax.set_xlabel("Illumination change [%]")
        ax.set_ylabel("Classification stability [%]")
        ax.set_ylim(0, 105)
        ax.set_xticks([-30, -20, 20, 30])
        ax.grid(True, alpha=0.25)
        ax.legend()
        ax.set_title("Classification Stability vs. Illumination Change")
        plt.tight_layout()
        stability_png = _fig_to_img(fig)
        st.image(stability_png, caption="Graph 1 — Stability of the final classification", use_container_width=True)

    with graph_col2:
        st.markdown("#### Disease-score sensitivity")
        fig, ax = plt.subplots(figsize=(8.8, 5.2))
        ax.plot(plot_df["Illumination_Change_%"],
                plot_df["Mean_Score_Deviation_Without_PP"], marker="o", label="Without preprocessing")
        ax.plot(plot_df["Illumination_Change_%"],
                plot_df["Mean_Score_Deviation_With_PP"], marker="o", label="With preprocessing")
        ax.set_xlabel("Illumination change [%]")
        ax.set_ylabel("Mean |Disease Score − Original Score|")
        ax.set_xticks([-30, -20, 20, 30])
        ax.grid(True, alpha=0.25)
        ax.legend()
        ax.set_title("Disease-Score Sensitivity to Illumination")
        plt.tight_layout()
        score_png = _fig_to_img(fig)
        st.image(score_png, caption="Graph 2 — Deviation from the original disease score", use_container_width=True)

    # Visual example: first image, -30/original/+30 before and after preprocessing.
    first = image_records[0]
    st.markdown("### Visual comparison — effect of illumination normalization")
    st.caption("The same image is shown before and after preprocessing at three illumination levels.")
    cols = st.columns(3, gap="large")
    for idx, (delta, factor) in enumerate([(-30, 0.70), (0, 1.00), (30, 1.30)]):
        variant = adjust_illumination(first["img_bgr"], factor)
        before = resize_baseline(variant)
        after = preprocess_img(variant)
        with cols[idx]:
            st.markdown(f"#### {delta:+d}% illumination")
            st.image(cv2.cvtColor(before, cv2.COLOR_BGR2RGB), caption="Before preprocessing", use_container_width=True)
            st.image(cv2.cvtColor(after, cv2.COLOR_BGR2RGB), caption="After preprocessing", use_container_width=True)

    # Downloads for the project book / analysis.
    st.markdown("### Export robustness results")
    d1, d2, d3, d4, d5 = st.columns(5)
    d1.download_button(
        "⬇ Detailed CSV",
        detailed_df.to_csv(index=False).encode("utf-8-sig"),
        file_name="robustness_detailed_results.csv",
        mime="text/csv",
    )
    d2.download_button(
        "⬇ Summary CSV",
        summary_df.to_csv(index=False).encode("utf-8-sig"),
        file_name="robustness_summary.csv",
        mime="text/csv",
    )
    d3.download_button(
        "⬇ By-crop CSV",
        crop_summary_df.to_csv(index=False).encode("utf-8-sig"),
        file_name="robustness_by_crop.csv",
        mime="text/csv",
    )
    d4.download_button(
        "⬇ Stability graph",
        stability_png,
        file_name="robustness_classification_stability.png",
        mime="image/png",
    )
    d5.download_button(
        "⬇ Score graph",
        score_png,
        file_name="robustness_score_deviation.png",
        mime="image/png",
    )


def _ui_fit_canvas(image, target_w: int = 420, target_h: int = 300) -> np.ndarray:
    """Fit an image into a fixed UI canvas without cropping. Display-only helper."""
    if isinstance(image, (bytes, bytearray)):
        decoded = cv2.imdecode(np.frombuffer(image, np.uint8), cv2.IMREAD_UNCHANGED)
        if decoded is None:
            return np.zeros((target_h, target_w, 3), dtype=np.uint8)
        if decoded.ndim == 3 and decoded.shape[2] == 4:
            image = cv2.cvtColor(decoded, cv2.COLOR_BGRA2RGBA)
        elif decoded.ndim == 3:
            image = cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB)
        else:
            image = decoded

    if image is None or not hasattr(image, "size") or image.size == 0:
        return np.zeros((target_h, target_w, 3), dtype=np.uint8)

    if image.ndim == 2:
        src = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        has_alpha = False
    elif image.shape[2] == 4:
        src = image.copy()
        has_alpha = True
    else:
        src = image.copy()
        has_alpha = False

    h, w = src.shape[:2]
    scale = min(target_w / max(w, 1), target_h / max(h, 1))
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
    resized = cv2.resize(src, (new_w, new_h), interpolation=interp)

    if has_alpha:
        canvas = np.zeros((target_h, target_w, 4), dtype=np.uint8)
        # Fully transparent padding preserves the "leaves only" presentation.
        canvas[:, :, :3] = np.array([7, 18, 27], dtype=np.uint8)
        canvas[:, :, 3] = 0
    else:
        canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
        canvas[:] = np.array([7, 18, 27], dtype=np.uint8)

    x0 = (target_w - new_w) // 2
    y0 = (target_h - new_h) // 2
    canvas[y0:y0 + new_h, x0:x0 + new_w] = resized
    return canvas


def render_leaf_first_user_overlay(original_bgr: np.ndarray, target_shape, plant_mask: np.ndarray, lesion_mask: np.ndarray, label: str) -> bytes:
    """User-facing overlay on the ORIGINAL image.

    HARD INVARIANT: every colored disease pixel is first intersected with the final
    leaf mask. Background/soil can never be highlighted as disease.
    """
    H, W = target_shape[:2]
    base = cv2.resize(original_bgr, (W, H), interpolation=cv2.INTER_AREA)
    pm = (plant_mask > 0).astype(np.uint8) * 255
    lm = cv2.bitwise_and((lesion_mask > 0).astype(np.uint8) * 255, pm)
    out = base.copy()

    # Thin green boundary shows exactly what the algorithm accepted as crop tissue.
    cnts, _ = cv2.findContours(pm, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(out, [c for c in cnts if cv2.contourArea(c) > 80], -1, (70, 220, 80), 2)

    # Disease/suspicion highlighting is strictly leaf-only. Healthy images receive no red/orange fill.
    if label in ("Sick", "Suspicious") and np.any(lm > 0):
        overlay = out.copy()
        color = (30, 45, 235) if label == "Sick" else (0, 165, 255)
        overlay[lm > 0] = color
        out = cv2.addWeighted(overlay, 0.42, out, 0.58, 0)
        lc, _ = cv2.findContours(lm, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, lc, -1, color, 2)

    ok, enc = cv2.imencode(".png", out)
    return enc.tobytes() if ok else b""


def _uniform_processing_preview(image_data, canvas_size=(900, 600), mask=False):
    """Presentation only: letterbox an image without distorting or cropping it.

    Every processing stage has an identical canvas size on screen, regardless
    of the photograph orientation or the algorithm's working resolution. The
    original image arrays, alpha mask, classifier and exported files are untouched.
    """
    from PIL import Image

    if isinstance(image_data, (bytes, bytearray)):
        with Image.open(BytesIO(image_data)) as raw:
            image = raw.copy()
    elif isinstance(image_data, Image.Image):
        image = image_data.copy()
    else:
        array = np.asarray(image_data)
        if array.ndim == 2:
            image = Image.fromarray(array.astype(np.uint8), mode="L")
        elif array.ndim == 3 and array.shape[-1] in (3, 4):
            image = Image.fromarray(array.astype(np.uint8), mode="RGB" if array.shape[-1] == 3 else "RGBA")
        else:
            raise ValueError("Unsupported image format in processing preview")

    if image.width < 1 or image.height < 1:
        raise ValueError("Empty processing preview")

    resampling = getattr(Image, "Resampling", Image).LANCZOS
    # LANCZOS is for photographs; nearest neighbour keeps binary masks binary.
    if mask:
        image = image.convert("L").point(lambda x: 255 if x > 127 else 0)
        background = Image.new("RGB", canvas_size, (19, 26, 22))
        resampling = getattr(Image, "Resampling", Image).NEAREST
        image.thumbnail(canvas_size, resampling)
        image = image.convert("RGB")
    else:
        background = Image.new("RGB", canvas_size, (250, 248, 242))
        if image.mode == "RGBA":
            # Composite ONLY the preview: true alpha remains in segmented_plant.
            transparent_base = Image.new("RGBA", image.size, (250, 248, 242, 255))
            image = Image.alpha_composite(transparent_base, image).convert("RGB")
        else:
            image = image.convert("RGB")
        image.thumbnail(canvas_size, resampling)

    x = (canvas_size[0] - image.width) // 2
    y = (canvas_size[1] - image.height) // 2
    background.paste(image, (x, y))
    return background


def render_single_analysis_tab(img_bgr: np.ndarray, crop: str, image_name: str = "image", segmentation_gt_bytes=None):
    """Full pipeline output for the 'Single Image Analysis' tab."""

    t_start = time.time()

    # ── Step 2: Image Quality Check ──
    quality = check_image_quality(img_bgr)

    # ── Step 3: Grayscale ──
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # ── Steps 4–9: shared core pipeline ──
    with st.spinner("Pre-processing, segmenting & classifying..."):
        core = run_core_pipeline(img_bgr, crop)
        img_proc = core["processed"]
        result = core["result"]
        plant_mask = core["plant_mask"]
        lesion_mask = core["lesion_mask"]
        segmented_plant = make_segmented_plant(img_proc, plant_mask)
        annotated = render_leaf_first_user_overlay(
            img_bgr, img_proc.shape, plant_mask, lesion_mask, result.get("label", "Healthy")
        )
        seg_metrics = evaluate_segmentation_mask(plant_mask, segmentation_gt_bytes)
        mask_coverage_pct = 100.0 * np.count_nonzero(plant_mask) / max(plant_mask.size, 1)

    t_end = time.time()
    proc_time = t_end - t_start

    # ── Statistical metrics ──
    img_stats = compute_image_stats(img_bgr)

    # ── Step 11: Histograms ──
    rgb_hist = render_rgb_histogram(img_proc)
    hsv_hist = render_hsv_histogram(img_proc)
    hue_hist = render_hue_detail(img_proc)

    # ── Step 15: Confidence ──
    confidence = {} if result.get("label") == "No Plant" else compute_confidence(result.get("score", 0), crop)

    # ── Step 20: CSV row ──
    # Ground Truth is persisted per crop+image. The sidebar value is used only
    # as the initial value; an applied result label survives later Streamlit reruns.
    _gt_store_key = f"{crop}|{image_name}"
    ground_truth = st.session_state.get("ground_truth_labels", {}).get(
        _gt_store_key, st.session_state.get("ground_truth_input", "")
    )
    csv_row = build_csv_row(image_name, crop, result, quality, confidence, proc_time,
                            ground_truth=ground_truth, img_stats=img_stats)
    # Store result dict in session so Ground Truth can be updated later without re-analysis
    st.session_state["last_result"] = {
        "image_name": image_name,
        "crop": crop,
        "result": result,
        "quality": quality,
        "confidence": confidence,
        "proc_time": proc_time,
        "img_stats": img_stats,
    }
    csv_bytes = export_csv([csv_row])
    # Accumulate rows — replace if same image was analyzed before
    st.session_state.csv_rows = [
        r for r in st.session_state.csv_rows
        if not (r.get("Image_ID") == image_name and r.get("Crop") == crop)
    ]
    st.session_state.csv_rows.append(csv_row)

    label = result["label"]
    score = result.get("score", 0)
    icons  = {"Healthy": "✓", "Suspicious": "?", "Sick": "✗", "No Plant": "·"}
    alert_cls = {"Healthy": "alert-low", "Suspicious": "alert-medium",
                 "Sick": "alert-high", "No Plant": "alert-low"}
    color_map  = {"Healthy": "#27ae60", "Suspicious": "#f39c12", "Sick": "#e74c3c"}
    badge_color = color_map.get(label, "#888")

    # ═══════════════════════════════════════════════════
    # ── Step 2 display: Quality banner ──
    # ═══════════════════════════════════════════════════
    with st.expander("Image Quality", expanded=False):
        q_color = "#27ae60" if quality["valid"] else "#e74c3c"
        q_label = "✅ Valid — ready for analysis" if quality["valid"] else "⚠️ Quality Issues Detected"
        st.markdown(
            f"<div class='quality-badge' style='background:{q_color}22;border:1px solid {q_color};color:{q_color}'>"
            f"{q_label}</div>",
            unsafe_allow_html=True
        )
        qc1, qc2, qc3 = st.columns(3)
        qc1.metric("Sharpness (Laplacian)", f"{quality['sharpness']}", delta="≥80 OK")
        qc2.metric("Brightness mean",       f"{quality['brightness_mean']}",
                   delta=f"std: {quality['brightness_std']}")
        qc3.metric("Overexposed pixels",    f"{quality['overexposed_pct']}%",
                   delta=f"underexposed: {quality['underexposed_pct']}%")
        if quality["issues"]:
            for iss in quality["issues"]:
                st.warning(f"⚠️ {iss}")

    st.divider()

    if crop == "Potato" and (mask_coverage_pct < 4.0 or mask_coverage_pct > 94.0):
        st.warning(
            "Potato leaf segmentation is uncertain for this image. "
            "Review Plant Mask before using the result as quantitative evidence. "
            "For academic validation, compare the mask with a manually labelled Ground Truth."
        )

    # ═══════════════════════════════════════════════════
    # ── Visual processing chain ──
    # ═══════════════════════════════════════════════════
    st.markdown("""<div class="section-eyebrow">VISION PIPELINE</div><div class="section-title">Image Processing</div>""", unsafe_allow_html=True)

    # Presentation only: show full-resolution source/processing outputs, rather
    # than resampling them to 420 × 300 and stretching them on screen again.
    # The classification pipeline above, its masks and its scoring stay unchanged.
    chain_items = [
        ("01 · Original", cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)),
        ("02 · Grayscale", gray),
        ("03 · Preprocessed", cv2.cvtColor(img_proc, cv2.COLOR_BGR2RGB)),
        ("04 · Plant Mask", plant_mask),
        ("05 · Leaf Segmentation", cv2.cvtColor(segmented_plant, cv2.COLOR_BGRA2RGBA)),
        ("06 · Disease Overlay", annotated),
    ]
    # All six stages: identical frame geometry, centered with preserved aspect.
    # Show 3 evenly sized previews per row for BOTH Potato and Sugar Beet.
    # Presentation-only; no resampling enters core segmentation/classification.
    for row_start in (0, 3):
        chain_cols = st.columns(3, gap="medium")
        for col, (title, image_data) in zip(chain_cols, chain_items[row_start:row_start + 3]):
            with col:
                st.markdown(
                    f"<div class='chv-stage-label'>{title}</div>",
                    unsafe_allow_html=True,
                )
                preview = _uniform_processing_preview(image_data, mask=(title == "04 · Plant Mask"))
                st.image(preview, use_container_width=True)

    st.caption("Leaf Segmentation uses a true transparent alpha background — soil/background pixels are removed.")
    st.markdown("")
    st.download_button("⬇ Download annotated image", data=annotated,
                       file_name=f"analysis_{label.lower()}.png", mime="image/png")

    # Quantitative lesion morphology — connected components inside the leaf mask.
    st.markdown("### Lesion Morphology")
    lm1, lm2, lm3, lm4 = st.columns(4)
    lm1.metric("Detected lesions", result.get("lesion_count", 0))
    lm2.metric("Disease area / leaf", f"{result.get('lesion_area_pct', 0):.2f}%")
    lm3.metric("Lesion density", f"{result.get('lesion_density', 0):.2f} / 10k px")
    lm4.metric("Mean circularity", f"{result.get('mean_circularity', 0):.3f}")
    st.caption("Connected-components measurements are calculated only inside the segmented leaf area and currently do not alter the classification score.")

    if seg_metrics is not None:
        st.markdown("### Segmentation Validation")
        sv1, sv2 = st.columns(2)
        sv1.metric("IoU", f"{seg_metrics['iou']:.3f}")
        sv2.metric("Dice", f"{seg_metrics['dice']:.3f}")
        st.caption("Compared with the optional manually annotated Ground-Truth leaf mask uploaded in Validation.")

    st.divider()

    # ═══════════════════════════════════════════════════
    # ── Classification + Confidence ──
    # ═══════════════════════════════════════════════════
    st.markdown(f"""
<div class="{alert_cls.get(label, 'alert-low')}">
  <b>{icons.get(label, '')} Classification: {label}</b>
  &nbsp;|&nbsp; Disease score: <b>{score}</b>
  &nbsp;|&nbsp; Processing time: <b>{proc_time:.2f}s</b>
</div>""", unsafe_allow_html=True)
    st.markdown("")

    cl1, cl2 = st.columns([1, 2])
    with cl1:
        st.markdown("**Diagnosis**")
        st.markdown(f"""
<div style='background:{badge_color};border-radius:8px;padding:14px 16px;text-align:center'>
  <span style='color:white;font-size:1.6rem;font-weight:700'>{icons.get(label, '')} {label}</span><br>
  <span style='color:rgba(255,255,255,0.8);font-size:0.9rem'>Score: {score}</span>
</div>""", unsafe_allow_html=True)

    with cl2:
        st.markdown("**Relative Classification Confidence**")
        st.caption("Score-based indicator · not field-area percentage")
        conf_colors = {"Healthy": "#27ae60", "Suspicious": "#f39c12", "Sick": "#e74c3c"}
        if not confidence:
            st.info("No class-confidence indicator is reported because no plant was detected.")
        else:
            for cls_name, prob in confidence.items():
                pct = int(prob * 100)
                c   = conf_colors.get(cls_name, "#888")
                st.markdown(
                    f"<div style='margin-bottom:6px'>"
                    f"<span style='color:#ccc;font-size:0.85rem;width:110px;display:inline-block'>{cls_name}</span>"
                    f"<div class='conf-bar' style='width:{pct}%;background:{c};display:inline-block;vertical-align:middle;margin:0 8px'></div>"
                    f"<span style='color:{c};font-weight:700'>{prob:.2%}</span>"
                    f"</div>",
                    unsafe_allow_html=True
                )

    st.divider()

    # ═══════════════════════════════════════════════════
    # ── Feature vector ──
    # ═══════════════════════════════════════════════════
    with st.expander("Feature Details", expanded=False):
        if crop == "Sugar Beet":
            feats = [
                ("True Yellow %",    result.get("ty_pct", 0),        "%",  "#f1c40f"),
                ("Sat mean",         result.get("s_mean", result.get("sat_mean", 0)), "", "#3498db"),
                ("Bleach %",         result.get("bleach_pct", 0),    "%",  "#e67e22"),
                ("Cercospora spots", result.get("cercospora_count", 0), "", "#e74c3c"),
                ("Small dark lesions", result.get("dark_lesion_count", 0), "", "#ff5f87"),
                ("Dark texture %",   result.get("dark_sp", 0),       "%",  "#888"),
                ("Green leaf %",     result.get("green_pct", 0),     "%",  "#27ae60"),
                ("Mean Hue",         result.get("mean_h", 0),        "",   "#9b59b6"),
            ]
        else:
            feats = [
                ("Brown ratio",   result.get("brown_ratio", 0),   "%",  "#e67e22"),
                ("Yellow ratio",  result.get("yellow_ratio", 0),  "%",  "#f1c40f"),
                ("Green ratio",   result.get("green_ratio", 0),   "%",  "#27ae60"),
                ("Dark ratio",    result.get("dark_ratio", 0),    "%",  "#888"),
                ("Avg saturation",result.get("mean_sat", 0),      "",   "#3498db"),
                ("Mean Hue",      result.get("mean_h", 0),        "",   "#9b59b6"),
                ("Mean Value",    result.get("val_mean", 0),      "",   "#f1c40f"),
            ]
        feats.append(("Vegetation %", result.get("veg_ratio", 0), "%", "#8e44ad"))

        col_fa, col_fb = st.columns(2)
        for i, (name, val, unit, color) in enumerate(feats):
            col = col_fa if i % 2 == 0 else col_fb
            with col:
                ca, cb = st.columns([3, 2])
                ca.markdown(f"<span style='color:#aaa;font-size:0.88rem'>{name}</span>",
                            unsafe_allow_html=True)
                cb.markdown(f"<span style='color:{color};font-weight:700'>{val}{unit}</span>",
                            unsafe_allow_html=True)

    # ═══════════════════════════════════════════════════
    # ── Statistical Metrics ──
    # ═══════════════════════════════════════════════════
    with st.expander("📐 Statistical Metrics", expanded=False):
        sm1, sm2, sm3, sm4 = st.columns(4)
        sm1.metric("Mean",         f"{img_stats['stat_mean']:.1f}")
        sm2.metric("Std Dev",      f"{img_stats['stat_std']:.1f}")
        sm3.metric("Entropy",      f"{img_stats['stat_entropy']:.3f}")
        sm4.metric("Total Pixels", f"{img_stats['total_pixels']:,}")
        sm5, sm6, sm7, sm8 = st.columns(4)
        sm5.metric("Yellow %",     f"{img_stats['stat_yellow_pct']:.1f}%")
        sm6.metric("Green %",      f"{img_stats['stat_green_pct']:.1f}%")
        sm7.metric("Dark %",       f"{img_stats['stat_dark_pct']:.1f}%")
        sm8.metric("Brown %",      f"{img_stats['stat_brown_pct']:.1f}%")
        st.caption(f"⏱ Processing time: {proc_time:.3f}s")

    # ═══════════════════════════════════════════════════
    # ── Histograms (Step 11) ──
    # ═══════════════════════════════════════════════════
    with st.expander("Histograms", expanded=False):
        ht1, ht2, ht3 = st.tabs(["RGB", "HSV", "Hue Detail"])
        with ht1:
            st.image(rgb_hist, use_container_width=True)
        with ht2:
            st.image(hsv_hist, use_container_width=True)
        with ht3:
            st.image(hue_hist, use_container_width=True)

    # ═══════════════════════════════════════════════════
    # ── Pre-processing steps checklist ──
    # ═══════════════════════════════════════════════════
    with st.expander("Preprocessing", expanded=False):
        st.markdown("""
- ✅ **Resize** → 800 × 600
- ✅ **Gaussian blur** (σ=0.8, kernel 3×3) — noise reduction
- ✅ **CLAHE** on LAB L-channel (clipLimit=2.0, tile 8×8) — adaptive contrast
- ✅ **Brightness clipping** at 99th percentile — highlight suppression
        """)

    st.divider()

    # ═══════════════════════════════════════════════════
    # ── Export: CSV + ZIP (Steps 20–21) ──
    # ═══════════════════════════════════════════════════
    st.markdown("""<div class="section-eyebrow">OUTPUT PACKAGE</div><div class="section-title">Export Results</div>""", unsafe_allow_html=True)
    exp_c1, exp_c2 = st.columns(2)

    with exp_c1:
        # Allow updating Ground Truth after seeing results, without re-running analysis
        st.markdown("**Ground Truth for this image**")
        _gt_widget_id = hashlib.sha1(f"{crop}|{image_name}".encode("utf-8")).hexdigest()[:10]
        gt_live = st.selectbox(
            "True label",
            options=["", "Healthy", "Suspicious", "Sick"],
            index=["", "Healthy", "Suspicious", "Sick"].index(ground_truth) if ground_truth in ["", "Healthy", "Suspicious", "Sick"] else 0,
            key=f"gt_live_select_{_gt_widget_id}",
            label_visibility="collapsed",
        )
        if st.button("✅ Apply & save Ground Truth", key=f"apply_gt_{_gt_widget_id}"):
            last = st.session_state.get("last_result", {})
            if last:
                st.session_state.ground_truth_labels[_gt_store_key] = gt_live
                updated_row = build_csv_row(
                    last["image_name"], last["crop"], last["result"],
                    last["quality"], last["confidence"], last["proc_time"],
                    ground_truth=gt_live, img_stats=last.get("img_stats"),
                )
                st.session_state.csv_rows = [
                    r for r in st.session_state.csv_rows
                    if not (r.get("Image_ID") == last["image_name"] and r.get("Crop") == last["crop"])
                ]
                st.session_state.csv_rows.append(updated_row)
                csv_row = updated_row
                csv_bytes = export_csv([updated_row])
                st.success(f"Ground Truth set to: **{gt_live or '(empty)'}**")

        st.download_button(
            "Download CSV",
            data=export_csv([csv_row]),
            file_name=f"results_{image_name}.csv",
            mime="text/csv",
            help="One row with all features, classification, confidence and metadata"
        )

    # Multi-image accumulated CSV with performance metrics
    all_rows = st.session_state.get("csv_rows", [])
    if len(all_rows) > 0:
        _metric_classes = {"Healthy", "Suspicious", "Sick"}
        n_labeled = sum(
            1 for r in all_rows
            if str(r.get("Ground_Truth", "")).strip() in _metric_classes
            and str(r.get("Prediction", "")).strip() in _metric_classes
        )
        metrics_available = n_labeled >= 2
        metrics = compute_performance_metrics(all_rows) if metrics_available else None

        with st.expander(
            f"📊 Session CSV — {len(all_rows)} image(s)"
            + (f" · {n_labeled} labeled" if n_labeled > 0 else ""),
            expanded=False
        ):
            if metrics:
                classes = metrics["classes"]
                mc1, mc2, mc3, mc4 = st.columns(4)
                mc1.metric("Accuracy",  f"{metrics['accuracy']:.1%}")
                mc2.metric("Precision", f"{metrics['macro_precision']:.1%}", help="Macro average")
                mc3.metric("Recall",    f"{metrics['macro_recall']:.1%}",    help="Macro average")
                mc4.metric("F1",        f"{metrics['macro_f1']:.1%}",        help="Macro average")

                st.markdown("**Per-class metrics**")
                tbl_rows = []
                for c in classes:
                    tbl_rows.append({
                        "Class":     c,
                        "Precision": f"{metrics['precision'][c]:.3f}",
                        "Recall":    f"{metrics['recall'][c]:.3f}",
                        "F1":        f"{metrics['f1'][c]:.3f}",
                    })
                st.dataframe(pd.DataFrame(tbl_rows), hide_index=True, use_container_width=True)

                st.markdown("**Confusion Matrix** (rows = True label, cols = Predicted)")
                cm_data = []
                for c in classes:
                    row_d = {"True \\ Predicted": c}
                    row_d.update({p: metrics["confusion_matrix"][c][p] for p in classes})
                    cm_data.append(row_d)
                st.dataframe(pd.DataFrame(cm_data), hide_index=True, use_container_width=True)
                st.caption("Metrics include only rows with both Ground Truth and prediction in Healthy / Suspicious / Sick; No Plant rows are excluded from the 3-class confusion matrix.")
            elif n_labeled > 0:
                st.info(f"Need at least 2 labeled images for metrics. Currently: {n_labeled}.")
            else:
                st.info("Add Ground Truth labels in the sidebar to enable Accuracy / Precision / Recall / F1.")

            # Download full CSV with metrics appended
            full_csv = export_csv_with_metrics(all_rows)
            st.download_button(
                "⬇ Download full session CSV (with metrics)",
                data=full_csv,
                file_name="session_results_with_metrics.csv",
                mime="text/csv",
            )
            if st.button("🗑 Clear session history", key="clear_csv"):
                st.session_state.csv_rows = []
                st.rerun()

    with exp_c2:
        # Build ZIP on button click to avoid building it every render
        if st.button("Prepare output ZIP", key="zip_btn"):
            with st.spinner("Building ZIP..."):
                zip_bytes = build_output_zip(
                    image_name=image_name,
                    img_bgr=img_bgr,
                    img_proc=img_proc,
                    result=result,
                    quality=quality,
                    confidence=confidence,
                    annotated=annotated,
                    plant_mask=plant_mask,
                    segmented_plant=segmented_plant,
                    rgb_hist=rgb_hist,
                    hsv_hist=hsv_hist,
                    hue_hist=hue_hist,
                    csv_bytes=csv_bytes,
                )
            st.download_button(
                "⬇ Download ZIP folder",
                data=zip_bytes,
                file_name=f"{os.path.splitext(image_name)[0]}_outputs.zip",
                mime="application/zip",
            )

    # ── Step 19: Summary Report ──
    with st.expander("Summary Report", expanded=False):
        st.markdown(f"""
**Experiment Summary**

| Field              | Value |
|--------------------|-------|
| Image              | `{image_name}` |
| Crop               | {crop} |
| Quality            | {'✅ Valid' if quality['valid'] else '⚠️ Invalid'} |
| Sharpness          | {quality['sharpness']} |
| Brightness mean    | {quality['brightness_mean']} |
| Classification     | **{label}** |
| Disease score      | {score} |
| Relative score-based confidence ({label}) | {confidence.get(label, 0):.2%} |
| Vegetation %       | {result.get('veg_ratio', '')}% |
| Processing time    | {proc_time:.3f}s |
        """)
        if quality["issues"]:
            st.markdown("**Quality issues:** " + "; ".join(quality["issues"]))








def _prepare_batch_map_layers(img_bgr, plant_mask, lesion_mask, status, grid_size, crop):
    """Shared leaf-only spatial categories for batch grid and heat map.

    No classifier thresholds or disease scores are changed. Both displays use
    the same status matrix and the same leaf segmentation. Background/soil is
    kept uncoloured. All colours are specified in OpenCV BGR order.
    """
    h, w = img_bgr.shape[:2]
    leaf = cv2.resize(
        ((plant_mask > 0).astype(np.uint8) * 255),
        (w, h), interpolation=cv2.INTER_NEAREST,
    )
    lesion = cv2.resize(
        ((lesion_mask > 0).astype(np.uint8) * 255),
        (w, h), interpolation=cv2.INTER_NEAREST,
    )
    lesion = cv2.bitwise_and(lesion, leaf)
    labels, ratios, intensities = _build_spatial_status_maps(
        leaf, lesion, status, grid_size, crop,
    )
    # BGR: green / amber-yellow / red.  Never colour No Plant as healthy.
    colors = {
        "Healthy": (45, 171, 42),
        "Suspicious": (20, 203, 245),
        "Sick": (43, 53, 235),
    }
    rows = cols = max(2, int(grid_size))
    cell_colors = np.zeros((h, w, 3), dtype=np.uint8)
    valid = np.zeros((h, w), dtype=np.uint8)
    borders = []
    for r in range(rows):
        y1 = int(round(r * h / rows))
        y2 = int(round((r + 1) * h / rows)) if r < rows - 1 else h
        for c in range(cols):
            x1 = int(round(c * w / cols))
            x2 = int(round((c + 1) * w / cols)) if c < cols - 1 else w
            label = labels[r, c]
            if label not in colors:
                borders.append((x1, y1, x2, y2, None))
                continue
            color = colors[label]
            cell_colors[y1:y2, x1:x2] = color
            valid[y1:y2, x1:x2] = 255
            borders.append((x1, y1, x2, y2, label))
    # The per-cell label must correspond to actual plant pixels, not bare soil.
    vegetation = (leaf > 0) & (valid > 0)
    return cell_colors, vegetation, borders, labels, ratios, intensities


def render_batch_heatmap(img_bgr, proc_shape, plant_mask, lesion_mask, status, grid_size=10, crop="Potato"):
    """Strong, visible leaf-only traffic-light heat map for batch analysis.

    The older version used only 18% healthy-green alpha and looked identical to
    the original field photo. This map uses a readable 68% vegetation overlay.
    It never adds a disease colour to an image classified Healthy.
    """
    colors, vegetation, _, _, _, _ = _prepare_batch_map_layers(
        img_bgr, plant_mask, lesion_mask, status, grid_size, crop,
    )
    out = img_bgr.astype(np.float32).copy()
    # Smoothly blend coloured zones across cell edges: a heat map should not
    # look like a wireframe grid. Unsegmented soil remains the original image.
    valid = np.any(colors > 0, axis=2).astype(np.float32)
    smooth_weight = cv2.GaussianBlur(valid, (0, 0), sigmaX=8, sigmaY=8)
    smooth = cv2.GaussianBlur(colors.astype(np.float32), (0, 0), sigmaX=8, sigmaY=8)
    # Normalise at valid-zone borders so a No Plant cell never creates dark edges.
    smooth /= np.maximum(smooth_weight[..., None], 1e-6)
    # Keep bare soil pixel-exact; smoothing affects colour only within leaves.
    alpha = vegetation.astype(np.float32) * 0.68
    out = out * (1.0 - alpha[..., None]) + smooth * alpha[..., None]
    return np.clip(out, 0, 255).astype(np.uint8)


def render_batch_result_grid(img_bgr, proc_shape, plant_mask, lesion_mask, status, grid_size, crop="Potato"):
    """Clearly coloured field grid using exactly the same cell labels as heat map.

    Plant tissue is visibly tinted (not nearly transparent). Unsegmented soil
    is not interpreted as healthy or diseased. Cell borders show grid layout.
    """
    colors, vegetation, borders, _, _, _ = _prepare_batch_map_layers(
        img_bgr, plant_mask, lesion_mask, status, grid_size, crop,
    )
    out = img_bgr.astype(np.float32).copy()
    # Strong vegetation tint (54%). The subtle 8% full-cell tint improves
    # recognition of classified zones without presenting soil as diseased.
    active = np.any(colors > 0, axis=2)
    alpha = np.where(vegetation, 0.54, np.where(active, 0.08, 0.0)).astype(np.float32)
    out = out * (1.0 - alpha[..., None]) + colors.astype(np.float32) * alpha[..., None]
    out = np.clip(out, 0, 255).astype(np.uint8)
    border_colors = {
        "Healthy": (31, 119, 29),
        "Suspicious": (0, 144, 191),
        "Sick": (23, 29, 185),
    }
    for x1, y1, x2, y2, label in borders:
        color = border_colors.get(label, (148, 148, 148))
        thickness = 2 if label in border_colors else 1
        cv2.rectangle(out, (x1, y1), (max(x1, x2 - 1), max(y1, y2 - 1)), color, thickness)
    return out


def render_batch_analysis(uploaded_files, crop, grid_size, gps_df=None):
    """Batch mode uses the exact whole-image pipeline used by Single Image Analysis."""
    st.markdown('<div class="section-eyebrow">BATCH SCREENING</div><div class="section-title">Field Image Triage</div><div class="section-sub">Each image is segmented first as a complete image. Analysis and highlighting are restricted to the final leaf mask.</div>', unsafe_allow_html=True)
    rows, flagged = [], []
    progress = st.progress(0, text="Preparing batch analysis...")
    for i, uf in enumerate(uploaded_files):
        arr = np.frombuffer(uf.getvalue(), np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            rows.append({"Image": uf.name, "Status": "Unreadable", "Disease score": None, "Lesions": 0, "Disease area %": 0.0, "Segmentation quality": "-", "Segmentation score": 0})
            progress.progress((i + 1) / len(uploaded_files), text=f"Skipping unreadable file {i+1}/{len(uploaded_files)} - {uf.name}")
            continue
        h, w = img.shape[:2]
        if max(h, w) > 2000:
            scale = 2000 / max(h, w)
            img = cv2.resize(img, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_AREA)
        # EXACT SAME classification/segmentation chain as single-image mode.
        core = run_core_pipeline(img, crop)
        proc = core["processed"]
        plant_mask = core["plant_mask"]
        result = core["result"]
        status = result.get("label", "Healthy")
        lesion_mask = core["lesion_mask"]
        lm = core["lesion_metrics"]
        sq = compute_segmentation_quality(plant_mask)
        rows.append({"Image": uf.name, "Status": status, "Disease score": result.get("score", 0),
                     "Lesions": lm.get("lesion_count", 0), "Disease area %": lm.get("lesion_area_pct", 0.0),
                     "Lesion density": lm.get("lesion_density", 0.0), "Segmentation quality": sq["segmentation_quality"],
                     "Segmentation score": sq["segmentation_quality_score"]})
        # Always keep a farmer-facing image. Diagnosis is still whole-image; grid is display/spatial localization only.
        grid_overlay = render_batch_result_grid(img, proc.shape, plant_mask, lesion_mask, status, grid_size, crop)
        heatmap_overlay = render_batch_heatmap(img, proc.shape, plant_mask, lesion_mask, status, grid_size, crop)
        flagged.append((uf.name, status, result, lm, grid_overlay, heatmap_overlay, sq))
        progress.progress((i+1)/len(uploaded_files), text=f"Analyzing {i+1}/{len(uploaded_files)} - {uf.name}")
    progress.empty()

    flagged.sort(key=lambda x: ({"Sick":0,"Suspicious":1,"Healthy":2}.get(x[1],3), -float(x[2].get("score", 0))))
    healthy_n = sum(r.get("Status") == "Healthy" for r in rows)
    susp_n = sum(r.get("Status") == "Suspicious" for r in rows)
    sick_n = sum(r.get("Status") == "Sick" for r in rows)
    attention_n = susp_n + sick_n
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Images", len(rows)); c2.metric("Healthy", healthy_n); c3.metric("Suspicious", susp_n); c4.metric("Sick", sick_n)
    if attention_n == 0:
        st.success("No suspicious or sick images were detected in this batch.")
    else:
        st.warning(f"{attention_n} image(s) require attention. They are shown first below.")
    st.markdown("### Analyzed field images")
    st.caption("Every uploaded image is shown below. 🟢 Green = healthy vegetation, 🟡 yellow = suspicious area, 🔴 red = sick area; uncoloured/gray = soil or insufficient plant. The field grid shows separate zones; the heat map shows the same findings without grid lines. Both views colour only segmented plant tissue. Image classification is performed before spatial visualisation.")
    for start_i in range(0, len(flagged), 3):
        cols = st.columns(3, gap="medium")
        for col, item in zip(cols, flagged[start_i:start_i+3]):
            name, status, result, lm, overlay, heatmap_overlay, sq = item
            with col:
                badge = "🔴" if status == "Sick" else ("🟠" if status == "Suspicious" else "🟢")
                st.markdown(f"**{badge} {name} — {status}**")
                tab_grid, tab_heat = st.tabs(["Field grid", "Heat map"])
                with tab_grid:
                    st.image(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB), use_container_width=True)
                with tab_heat:
                    st.image(cv2.cvtColor(heatmap_overlay, cv2.COLOR_BGR2RGB), use_container_width=True)
                st.caption(f"Score {result.get('score',0)} · Lesions {lm.get('lesion_count',0)} · Disease area {lm.get('lesion_area_pct',0.0):.2f}% · Leaf isolation {sq['segmentation_quality']} ({sq['segmentation_quality_score']:.0f}/100)")
    df = pd.DataFrame(rows)
    if gps_df is not None and not gps_df.empty:
        gps = gps_df.copy(); gps.columns = [str(c).strip().lower() for c in gps.columns]
        aliases = {"image":"filename", "file":"filename", "image name":"filename", "lat":"latitude", "lon":"longitude", "lng":"longitude", "long":"longitude"}
        gps = gps.rename(columns={c: aliases.get(c, c) for c in gps.columns})
        if {"filename","latitude","longitude"}.issubset(gps.columns):
            gps["filename"] = gps["filename"].astype(str).map(os.path.basename)
            gps["latitude"] = pd.to_numeric(gps["latitude"], errors="coerce"); gps["longitude"] = pd.to_numeric(gps["longitude"], errors="coerce")
            df["filename_key"] = df["Image"].astype(str).map(os.path.basename)
            df = df.merge(gps[["filename","latitude","longitude"]], left_on="filename_key", right_on="filename", how="left").drop(columns=["filename_key","filename"], errors="ignore")
            mapped = df.dropna(subset=["latitude","longitude"]).copy()
            if not mapped.empty:
                st.markdown("### Crop Health Map")
                color_map = {"Healthy":[34,197,94,190], "Suspicious":[245,158,11,210], "Sick":[239,68,68,220], "Unreadable":[107,114,128,170]}
                mapped["map_color"] = mapped["Status"].map(color_map).apply(lambda x: x if isinstance(x,list) else [107,114,128,170])
                try:
                    import pydeck as pdk
                    layer = pdk.Layer("ScatterplotLayer", data=mapped, get_position="[longitude, latitude]", get_fill_color="map_color", radius_min_pixels=7, radius_max_pixels=18, pickable=True, stroked=True, get_line_color=[255,255,255,220])
                    view = pdk.ViewState(latitude=float(mapped["latitude"].mean()), longitude=float(mapped["longitude"].mean()), zoom=16)
                    st.pydeck_chart(pdk.Deck(layers=[layer], initial_view_state=view, tooltip={"html":"<b>{Image}</b><br>Status: {Status}<br>Score: {Disease score}<br>Lesions: {Lesions}"}), use_container_width=True)
                except Exception:
                    st.map(mapped.rename(columns={"latitude":"lat","longitude":"lon"})[["lat","lon"]], use_container_width=True)
            else: st.warning("GPS file loaded, but no filenames matched the uploaded images.")
        else: st.warning("GPS CSV must contain filename, latitude and longitude columns.")
    with st.expander("Batch statistics and full results", expanded=True): st.dataframe(df, use_container_width=True, hide_index=True)
    st.download_button("⬇ Download batch CSV", df.to_csv(index=False).encode("utf-8-sig"), file_name="batch_analysis.csv", mime="text/csv")


def _read_embedded_asset_b64(filename):
    """Read an unchanged image asset bundled beside this Python file."""
    assets_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Crop_Health_Vision_assets")
    asset_path = os.path.join(assets_dir, filename)
    try:
        with open(asset_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("ascii")
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"Missing bundled image: {asset_path}. Extract the entire ZIP "
            "with its Crop_Health_Vision_assets folder."
        ) from exc


@st.cache_data(show_spinner=False)
def _load_best_home_hero_b64():
    """Return the embedded high-resolution Crop Health Vision landing artwork (modern dark-green GUI).

    The original PNG is kept unchanged in the bundled image assets folder.
    This loader is retained for compatibility with the original app.
    """
    return _read_embedded_asset_b64("legacy_home.png")


EARTHY_HOME = '<main class="chv-home" lang="en">\n  <section class="chv-intro" aria-label="Crop Health Vision introduction">\n    <div class="chv-kicker">PRECISION AGRICULTURE <span>·</span> COMPUTER VISION</div>\n    <h1>Crop Health <em>Vision</em></h1>\n    <h2>Image-Based Plant Disease Detection</h2>\n    <p>Analyze agricultural field imagery using computer vision and image processing to identify signs of plant disease and explore crop health through clear, visual results.</p>\n    <a href="#chv-upload-target" class="chv-start" aria-label="Go to the image upload controls">Start Analysis <span aria-hidden="true">&#8594;</span></a>\n  </section>\n  <section class="chv-benefits" aria-label="What this application offers">\n    <article class="chv-benefit">\n      <div class="chv-benefit-icon"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.75" stroke-linecap="round" stroke-linejoin="round"><rect x="3" y="4" width="18" height="16" rx="2"/><circle cx="8.5" cy="9" r="1.6"/><path d="m3 17 5-5 4 4 3-3 6 5"/></svg></div>\n      <div><h3>Upload Field Images</h3><p>Use RGB photos from your drone or camera, individually or in batches.</p></div>\n    </article>\n    <article class="chv-benefit">\n      <div class="chv-benefit-icon"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.75" stroke-linecap="round" stroke-linejoin="round"><path d="M20 4c-8 0-15 4-15 11a5 5 0 0 0 5 5c7 0 11-8 10-16Z"/><path d="M5 20c3-5 7-9 12-12"/></svg></div>\n      <div><h3>Detect Plant Diseases</h3><p>Vegetation segmentation and crop-specific disease classification.</p></div>\n    </article>\n    <article class="chv-benefit">\n      <div class="chv-benefit-icon"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.75" stroke-linecap="round" stroke-linejoin="round"><path d="M3 20h18"/><rect x="5" y="12" width="3" height="8" rx=".3"/><rect x="11" y="8" width="3" height="12" rx=".3"/><rect x="17" y="4" width="3" height="16" rx=".3"/></svg></div>\n      <div><h3>Visualize Results</h3><p>Review classifications, heatmaps, field grids, graphs and exports.</p></div>\n    </article>\n  </section>\n  <section class="chv-bottom" aria-label="Crops and workflow">\n    <div class="chv-panel chv-crops">\n      <h2>Our Crops</h2><p class="chv-panel-lead">Currently supporting potato and sugar beet crops.</p>\n      <div class="chv-crop-cards">\n        <article class="chv-crop-card"><div class="chv-crop-photo chv-potato-photo" role="img" aria-label="Green potato plants"></div><div class="chv-crop-label"><span class="chv-crop-dot">●</span><div><strong>Potato</strong><small>Solanum tuberosum</small></div></div></article>\n        <article class="chv-crop-card"><div class="chv-crop-photo chv-beet-photo" role="img" aria-label="Sugar beet plants"></div><div class="chv-crop-label"><span class="chv-crop-dot">✳</span><div><strong>Sugar Beet</strong><small>Beta vulgaris</small></div></div></article>\n      </div>\n    </div>\n    <div class="chv-panel chv-how">\n      <h2>How It Works</h2><p class="chv-panel-lead">From aerial photographs to clear, actionable results.</p>\n      <div class="chv-how-steps">\n        <div class="chv-how-step"><div class="chv-how-top"><span class="chv-step-index">1</span><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7"><rect x="3" y="4" width="18" height="16" rx="2"/><circle cx="8" cy="9" r="1.5"/><path d="m3 17 5-5 4 4 3-3 6 5"/></svg></div><h3>Upload Images</h3><p>Drone or field photographs</p></div>\n        <span class="chv-step-arrow" aria-hidden="true">&#8594;</span>\n        <div class="chv-how-step"><div class="chv-how-top"><span class="chv-step-index">2</span><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7"><path d="M20 4c-8 0-15 4-15 11a5 5 0 0 0 5 5c7 0 11-8 10-16Z"/><path d="M5 20c3-5 7-9 12-12"/></svg></div><h3>Run Analysis</h3><p>Segment and classify plant health</p></div>\n        <span class="chv-step-arrow" aria-hidden="true">&#8594;</span>\n        <div class="chv-how-step"><div class="chv-how-top"><span class="chv-step-index">3</span><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7"><path d="M3 20h18"/><rect x="5" y="12" width="3" height="8" rx=".3"/><rect x="11" y="8" width="3" height="12" rx=".3"/><rect x="17" y="4" width="3" height="16" rx=".3"/></svg></div><h3>Explore Results</h3><p>Heatmaps, grids and export options</p></div>\n      </div>\n    </div>\n  </section>\n</main>'
EARTHY_STYLE = (
    '\n<style>\n/* NATURAL & EARTHY — UI-only override; processing functions unchanged */\n:root{--bg:#f5f1e'
    '8;--panel:#fffdf8;--panel-2:#f1eddf;--line:#ded9ca;--line-strong:#c9c8ae;--text:#243325;--muted:#6f7'
    '768;--cyan:#547247;--cyan-2:#3d613d;--green:#527b45;--amber:#a8813b;--red:#a04f3f}\nhtml,body,[class*'
    '="css"],.stApp{font-family:\'Segoe UI\',Arial,sans-serif!important;color:#243325!important}\n[data-test'
    'id="stAppViewContainer"],[data-testid="stAppViewContainer"]>.main{background:#f5f1e8!important;color'
    ':#243325!important}\n[data-testid="stHeader"]{background:#f5f1e8!important;border:0!important;box-sha'
    'dow:none!important}\n[data-testid="stSidebar"]{background:#f9f6ee!important;border-right:1px solid #d'
    'dd9cb!important;box-shadow:none!important}\n[data-testid="stSidebar"] *{text-shadow:none!important}\n['
    'data-testid="stSidebar"] .sb-kicker,.section-eyebrow{color:#778a62!important;letter-spacing:.13em!im'
    'portant;font-weight:700!important}\n[data-testid="stSidebar"] .sb-title,.section-title{color:#263e2b!'
    "important;font-family:Georgia,'Times New Roman',serif!important;letter-spacing:-.025em!important}\n[d"
    'ata-testid="stSidebar"] .sb-sub,.section-sub{color:#697464!important}\n[data-testid="stSidebar"] labe'
    'l,[data-testid="stSidebar"] p,[data-testid="stSidebar"] h3,[data-testid="stSidebar"] span{color:#304'
    '12f!important}\n[data-testid="stSidebar"] hr{border-color:#ded9cb!important}\n.stButton>button,.stDown'
    'loadButton>button,[data-testid="stSidebar"] .stButton>button{background:#fdfbf5!important;color:#385'
    '53a!important;border:1px solid #c8cdb9!important;border-radius:9px!important;box-shadow:none!importa'
    'nt;font-weight:650!important;transition:background .16s,border-color .16s!important}\n.stButton>butto'
    'n:hover,.stDownloadButton>button:hover{background:#e9ecdd!important;border-color:#7b9872!important;c'
    'olor:#294b31!important;transform:none!important}\n.stButton>button[kind="primary"],.st-key-sidebar_an'
    'alyze button,.st-key-open_validation button{background:#405f3b!important;color:#fffdf8!important;bor'
    'der:1px solid #405f3b!important;box-shadow:none!important}\n.stButton>button[kind="primary"]:hover,.s'
    't-key-sidebar_analyze button:hover,.st-key-open_validation button:hover{background:#314d31!important'
    ';color:white!important;border-color:#314d31!important}\n.st-key-reset_uploaded_images button{backgrou'
    'nd:#fbf4f0!important;color:#8c493b!important;border:1px solid #dfc2b6!important;box-shadow:none!impo'
    'rtant}\n[data-testid="stFileUploader"] section{background:#fbfaf5!important;border:1px dashed #aebaa3'
    '!important;border-radius:12px!important}\n[data-testid="stFileUploader"] section *{color:#40553c!impo'
    'rtant}\n[data-testid="stFileUploader"] button{background:#e8eadf!important;color:#3b5839!important;bo'
    'rder:1px solid #ccd3c2!important}\n[data-baseweb="select"]>div,[data-baseweb="input"]>div,[data-basew'
    'eb="textarea"]>div{background:#fffdf8!important;color:#293c2d!important;border:1px solid #d7d9ca!imp'
    'ortant;border-radius:9px!important;box-shadow:none!important}\n[data-baseweb="select"] *{color:#293c2'
    'd!important}\n[data-baseweb="tab-list"]{background:#eeeadd!important;border:1px solid #e0ddcf!importa'
    'nt;border-radius:10px!important;padding:5px!important;gap:4px!important}\nbutton[data-baseweb="tab"]{'
    'background:transparent!important;color:#687563!important;border-radius:7px!important;font-weight:600'
    '!important}\nbutton[data-baseweb="tab"][aria-selected="true"]{background:#fffdf9!important;color:#385'
    'b3b!important;box-shadow:0 1px 4px #b7b7a52c!important}\n[data-testid="stExpander"],.metric-box,.meta'
    '-card,[class*="card"],div[data-testid="stMetric"]{background:#fffdf8!important;border:1px solid #e0d'
    'dcf!important;border-radius:12px!important;box-shadow:0 5px 18px #4b52320c!important;color:#263e2b!i'
    'mportant}\n.metric-box *, .meta-card *,[class*="card"] *{text-shadow:none!important}\n.metric-box .met'
    'ric-value,.meta-val{color:#345b39!important}\n.metric-box .metric-label,.meta-key{color:#6c7668!impor'
    'tant}\n.stMarkdown h1,.stMarkdown h2,.stMarkdown h3,.stMarkdown h4{color:#263b2b!important}\n.stMarkdo'
    "wn h1,.stMarkdown h2{font-family:Georgia,'Times New Roman',serif!important}\n.stMarkdown p,.stMarkdow"
    'n li,.stMarkdown strong,.stMarkdown span{color:inherit}\n[data-testid="stDataFrame"],.stTable{border:'
    '1px solid #e1ddcf!important;border-radius:9px!important;overflow:hidden}\n[data-testid="stImage"] img'
    '{border-radius:10px!important}\n[data-testid="stAlert"]{background:#edf0e5!important;color:#3a563b!im'
    'portant;border:1px solid #d4dccb!important;border-radius:10px!important}\n[data-testid="stProgress"] '
    '[role="progressbar"]>div{background:#56784b!important}\nhr{border-color:#e2ddcf!important}\n.home-hero'
    '{position:relative!important;min-height:calc(100vh - 3px)!important;background-position:center!impor'
    'tant;background-size:cover!important;background-color:#e7e5d6!important;isolation:isolate!important}'
    '\n.home-hero:before{content:"";position:absolute;inset:0;background:linear-gradient(90deg,rgba(249,24'
    '6,236,.98) 0%,rgba(249,246,236,.93) 30%,rgba(249,246,236,.64) 56%,rgba(25,47,29,.15) 100%);z-index:0'
    '}\n.earthy-home{position:relative;z-index:2;padding:clamp(38px,7vw,112px) clamp(28px,6vw,110px);max-w'
    "idth:820px;color:#253c2b}\n.earthy-kicker{font:700 12px 'Segoe UI',sans-serif;letter-spacing:.22em;co"
    "lor:#6a805b;margin-bottom:24px}\n.earthy-title{font:normal clamp(50px,6.1vw,94px)/1.04 Georgia,'Times"
    " New Roman',serif;letter-spacing:-.055em;color:#263b2a;margin:0 0 14px}\n.earthy-title em{color:#5274"
    '48;font-style:normal}\n.earthy-subtitle{font:normal clamp(19px,2.1vw,29px)/1.3 Georgia,serif;color:#3'
    "b563a;margin:0 0 24px}\n.earthy-copy{font:400 16px/1.75 'Segoe UI',sans-serif;color:#5d6a59;max-width"
    ':480px;margin:0 0 32px}\n.earthy-note{display:inline-block;background:#405f3b;color:#fff!important;pa'
    'dding:15px 24px;border-radius:8px;font-weight:600;font-size:14px;box-shadow:0 8px 20px #253f271c}\n.e'
    'arthy-steps{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:12px;margin-top:52px;max-'
    'width:740px}\n.earthy-step{background:#fffdf8e8;border:1px solid #e4dfd1;border-radius:11px;padding:1'
    '8px 15px;box-shadow:0 8px 24px #43492e13}\n.earthy-step b{display:block;color:#35573b;font-size:14px;'
    'margin-bottom:7px}\n.earthy-step span{display:block;color:#727b6e;font-size:12px;line-height:1.5}\n@me'
    'dia(max-width:850px){.earthy-home{padding:52px 28px}.earthy-steps{grid-template-columns:1fr}.home-he'
    'ro:before{background:rgba(249,246,236,.90)}}\n</style>\n'
)

def main():
    # ── Session state ──
    if "csv_rows" not in st.session_state:
        st.session_state.csv_rows = []
    if "engineering_validation" not in st.session_state:
        st.session_state.engineering_validation = False
    if "upload_reset_nonce" not in st.session_state:
        st.session_state.upload_reset_nonce = 0
    if "analysis_active" not in st.session_state:
        st.session_state.analysis_active = False
    if "analysis_signature" not in st.session_state:
        st.session_state.analysis_signature = None
    if "ground_truth_labels" not in st.session_state:
        st.session_state.ground_truth_labels = {}

    # Global full-width application shell. The home artwork is embedded in this
    # Python file so it can never fail because a PNG was moved or renamed.
    st.markdown("""
    <style>
    [data-testid="stAppViewContainer"] {background:#06131d;}
    [data-testid="stAppViewContainer"] > .main {background:#06131d;}
    .block-container {max-width:none !important; width:100% !important; padding:0 !important;}
    section.main > div {max-width:none !important; width:100% !important;}
    header[data-testid="stHeader"] {background:transparent !important; height:0 !important;}
    [data-testid="stSidebar"] {background:#061923 !important; border-right:1px solid rgba(59,211,238,.18);}
    [data-testid="stSidebar"] > div:first-child {padding-top:.55rem;}
    .home-hero {width:100%; min-height:calc(100vh - 4px); overflow:hidden; background-color:#06131d; background-size:cover; background-position:center center; background-repeat:no-repeat; position:relative; display:block; margin:0; padding:0; border-radius:0; image-rendering:auto;}
    .sb-kicker {color:#28d9f5;font-weight:900;font-size:.72rem;letter-spacing:.12em;text-transform:uppercase;margin:.1rem 0 .15rem;}
    .sb-title {color:#f3fbff;font-weight:900;font-size:1.05rem;margin:0 0 .2rem;}
    .sb-sub {color:#8fa8b7;font-size:.76rem;line-height:1.45;margin:0 0 .75rem;}
    [data-testid="stSidebar"] .stButton > button {width:100%;border-radius:10px;min-height:2.7rem;font-weight:800;}
    /* Primary workflow buttons: blue. Reset remains red below. */
    .st-key-sidebar_analyze button, .st-key-open_validation button {
       background:linear-gradient(90deg,#0891d1,#13b8df) !important; color:white !important;
       border:1px solid #42d8f5 !important; box-shadow:0 0 16px rgba(14,165,233,.22) !important;
    }
    .st-key-sidebar_analyze button:hover, .st-key-open_validation button:hover {
       background:linear-gradient(90deg,#0782bd,#0ea5c6) !important; border-color:#7be8ff !important;
    }
    [data-testid="stSidebar"] div[data-testid="stFileUploader"] section {padding:.65rem !important;}
    [data-testid="stSidebar"] hr {border-color:rgba(120,170,190,.18);margin:.8rem 0;}
    /* Reset is intentionally the final button in the sidebar. */
    .st-key-reset_uploaded_images button {
       background:linear-gradient(90deg,#b91c1c,#ef4444) !important;color:white !important;
       border:1px solid #ff6b6b !important;box-shadow:0 0 18px rgba(239,68,68,.24) !important;
    }
    </style>
    """, unsafe_allow_html=True)

    st.markdown(EARTHY_STYLE, unsafe_allow_html=True)
    st.markdown('<style>\n/* Premium Natural & Earthy — LANDING PAGE ONLY. All analysis styles and logic remain unchanged. */\n.home-hero {\n  background-image:linear-gradient(90deg,rgba(250,248,239,.98) 0%,rgba(250,248,239,.94) 24%,rgba(250,248,239,.72) 42%,rgba(250,248,239,.16) 76%,rgba(250,248,239,.02) 100%),\n    linear-gradient(0deg,rgba(250,248,239,.80) 0%,rgba(250,248,239,0) 38%),\n    url("data:image/jpeg;base64,__CROP_HOME_PHOTO__") !important;\n  background-position:center center,center center,center 42% !important;\n  background-size:cover!important;\n  background-repeat:no-repeat!important;\n  background-color:#f7f3e9!important;\n  min-height:100vh!important;\n  overflow:visible!important;\n  isolation:isolate!important;\n  border:0!important;\n  border-radius:0!important;\n  box-shadow:none!important;\n}\n.home-hero::before,.home-hero::after{display:none!important;}\n.chv-home{font-family:"Segoe UI",Arial,sans-serif;color:#20372a;box-sizing:border-box;\n  max-width:1700px;margin:0 auto;padding:clamp(44px,4vw,78px) clamp(28px,4vw,74px) 34px;}\n.chv-home *{box-sizing:border-box;}\n.chv-intro{max-width:710px;}\n.chv-kicker{font-size:12px;font-weight:750;letter-spacing:.19em;color:#496b4f;margin:0 0 36px;}\n.chv-kicker span{padding:0 6px;color:#9a9f83;}\n.stMarkdown .chv-intro h1{font-family:Georgia,"Times New Roman",serif!important;\n  font-weight:600!important;color:#132f26!important;letter-spacing:-.055em;\n  font-size:clamp(51px,5.0vw,87px);line-height:1.05;margin:0 0 12px;white-space:nowrap;}\n.stMarkdown .chv-intro h1 em{font-family:inherit!important;font-style:normal;color:#4d7851;}\n.stMarkdown .chv-intro h2{font-family:Georgia,"Times New Roman",serif!important;font-size:clamp(24px,2.25vw,36px)!important;\n  line-height:1.25;font-weight:400!important;color:#365a3e!important;margin:0 0 20px;letter-spacing:-.025em;}\n.chv-intro p{font-size:clamp(15px,1.09vw,18px);line-height:1.65;color:#283e30;\n  margin:0 0 23px;max-width:640px;text-wrap:pretty;}\n.chv-start{display:inline-flex;align-items:center;justify-content:space-between;gap:54px;\n  border-radius:9px;padding:16px 25px;min-width:235px;min-height:54px;text-decoration:none!important;\n  background:#315f3b;color:#fff!important;font-size:16px;font-weight:700;\n  box-shadow:0 9px 24px #29452a26;transition:background .15s,transform .15s,box-shadow .15s;}\n.chv-start span{color:#fff!important;font-size:22px;line-height:1;}\n.chv-start:hover,.chv-start:focus-visible{background:#234b2d;color:#fff!important;\n  transform:translateY(-1px);box-shadow:0 12px 28px #29452a42;}\n.chv-benefits{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:18px;margin-top:43px;}\n.chv-benefit{display:flex;align-items:center;gap:17px;padding:19px 22px;min-height:118px;\n  border:1px solid rgba(210,213,195,.92);border-radius:13px;background:rgba(255,253,247,.96);\n  box-shadow:0 12px 30px rgba(54,61,35,.11);backdrop-filter:blur(3px);}\n.chv-benefit-icon{display:flex;align-items:center;justify-content:center;width:65px;height:65px;\n  flex:0 0 65px;border-radius:50%;background:#e8ecdf;color:#486d4e;}\n.chv-benefit-icon svg{width:30px;height:30px;}\n.stMarkdown .chv-benefit h3{font-family:Georgia,"Times New Roman",serif!important;color:#21352b!important;\n  font-weight:650;font-size:clamp(16px,1.15vw,21px);line-height:1.15;margin:0 0 7px;}\n.chv-benefit p{font-size:13px;color:#526154;margin:0;line-height:1.45;}\n.chv-bottom{display:grid;grid-template-columns:minmax(0,.94fr) minmax(0,1.15fr);gap:19px;margin-top:20px;}\n.chv-panel{background:rgba(255,253,248,.96);border:1px solid rgba(217,218,206,.9);border-radius:14px;\n  padding:24px 27px 25px;box-shadow:0 14px 35px rgba(45,55,37,.08);min-height:258px;}\n.stMarkdown .chv-panel h2{font-family:Georgia,"Times New Roman",serif!important;font-size:clamp(23px,1.8vw,29px)!important;\n  font-weight:650;color:#1a3026!important;line-height:1.18;margin:0 0 7px;}\n.chv-panel-lead{color:#536052;font-size:14px;line-height:1.45;margin:0 0 19px;}\n.chv-crop-cards{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:15px;}\n.chv-crop-card{border-radius:11px;background:#fcfbf6;overflow:hidden;border:1px solid #e6e8da;\n  box-shadow:0 4px 13px rgba(51,70,45,.07);}\n.chv-crop-photo{height:103px;background-size:cover;background-position:center center;background-repeat:no-repeat;}\n.chv-potato-photo{background-image:url("data:image/jpeg;base64,__CROP_POTATO__");}\n.chv-beet-photo{background-image:url("data:image/jpeg;base64,__CROP_BEET__");}\n.chv-crop-label{display:flex;align-items:center;gap:12px;padding:10px 12px 12px;}\n.chv-crop-dot{display:flex;align-items:center;justify-content:center;flex:0 0 34px;\n  background:#dfe7d4;color:#486e43;width:34px;height:34px;border-radius:50%;font-size:18px;}\n.chv-crop-card:nth-child(2) .chv-crop-dot{background:#f2ded1;color:#b75c4f;}\n.stMarkdown .chv-crop-label strong{display:block;font-family:Georgia,"Times New Roman",serif!important;\n  font-size:17px;color:#172e26;font-weight:700;}\n.chv-crop-label small{display:block;color:#667064;font-size:12px;font-style:italic;margin-top:2px;}\n.chv-how-steps{display:flex;align-items:flex-start;justify-content:space-between;gap:12px;margin:21px 0 0;}\n.chv-how-step{flex:1 1 0;text-align:center;min-width:0;}\n.chv-how-top{display:flex;align-items:center;justify-content:center;gap:15px;height:56px;color:#466c4e;}\n.chv-step-index{display:flex;align-items:center;justify-content:center;width:39px;height:39px;border-radius:50%;\n  background:#e4e9dc;color:#193b2b;font-size:18px;font-family:Georgia,serif;font-weight:700;}\n.chv-how-top svg{width:33px;height:33px;flex:0 0 33px;}\n.chv-how-step h3{font-size:14px;font-weight:750;color:#23382c;margin:10px 0 8px;}\n.chv-how-step p{font-size:12.5px;color:#697267;margin:0;line-height:1.5;}\n.chv-step-arrow{flex:0 0 18px;align-self:center;font-size:28px;color:#a8b8a0;margin:8px -5px 0;}\n#chv-upload-target{scroll-margin-top:25px;}\n@media(max-width:1180px){.chv-home{padding:43px 29px 40px;}\n .chv-benefits{gap:10px;}.chv-benefit{padding:15px 13px;gap:10px;}.chv-benefit-icon{width:50px;height:50px;flex-basis:50px;}\n .chv-benefit-icon svg{width:25px;height:25px;}.chv-bottom{gap:12px;}.chv-panel{padding:20px 17px;}}\n@media(max-width:900px){.home-hero{background-image:linear-gradient(90deg,rgba(250,248,239,.97),rgba(250,248,239,.75)),url("data:image/jpeg;base64,__CROP_HOME_PHOTO__")!important;\n background-position:center center,right top!important;}.chv-intro h1{white-space:normal;}\n .chv-benefits{grid-template-columns:1fr;}.chv-bottom{grid-template-columns:1fr;}.chv-benefit{min-height:94px;}}\n@media(max-width:520px){.chv-home{padding:30px 19px;}.chv-kicker{margin-bottom:28px;font-size:10px;}\n .chv-intro h1{font-size:48px;}.chv-intro h2{font-size:24px;}\n .chv-crop-cards{gap:8px;}.chv-benefit{padding:13px;}.chv-panel{padding:19px 14px;}\n .chv-start{width:100%;}.chv-how-top{gap:7px;}.chv-how-top svg{width:25px;height:25px;flex-basis:25px;}\n .chv-step-arrow{font-size:20px;}.chv-how-step p{font-size:11px;}}\n</style>'.replace("__CROP_HOME_PHOTO__", _read_embedded_asset_b64("drone_field_clean.jpg")).replace("__CROP_POTATO__", _read_embedded_asset_b64("potato_crop.jpg")).replace("__CROP_BEET__", _read_embedded_asset_b64("beet_crop.jpg")), unsafe_allow_html=True)

    # Final accessibility/contrast layer: white lettering on dark forest-green controls.
    # CSS only; analysis, graphs, widgets, and exports are untouched.
    st.markdown("""<style>
    :root { --forest-action:#405f3b; --forest-action-hover:#314d31; --on-forest:#ffffff; }
    .stButton > button[kind="primary"],
    .stDownloadButton > button[kind="primary"],
    .st-key-sidebar_analyze button,
    .st-key-open_validation button,
    .earthy-note {
        background:var(--forest-action)!important;
        border-color:var(--forest-action)!important;
        color:var(--on-forest)!important;
    }
    .stButton > button[kind="primary"] *,
    .stDownloadButton > button[kind="primary"] *,
    .st-key-sidebar_analyze button *,
    .st-key-open_validation button *,
    .earthy-note * {
        color:var(--on-forest)!important;
        fill:var(--on-forest)!important;
    }
    .stButton > button[kind="primary"]:hover,
    .stDownloadButton > button[kind="primary"]:hover,
    .st-key-sidebar_analyze button:hover,
    .st-key-open_validation button:hover {
        background:var(--forest-action-hover)!important;
        border-color:var(--forest-action-hover)!important;
        color:var(--on-forest)!important;
    }
    .stButton > button[kind="primary"]:disabled,
    .st-key-sidebar_analyze button:disabled {
        color:#ffffff!important;
        opacity:.65!important;
    }
    .stButton > button[kind="primary"]:disabled *,
    .st-key-sidebar_analyze button:disabled * {color:#ffffff!important;}
    </style>""", unsafe_allow_html=True)


    # Natural & Earthy accessibility pass: presentation only.
    # Applied LAST to override colors inherited from the historical dark theme.
    # This does not modify a widget key, callback, algorithm or export.
    st.markdown("""<style>
    /* ================================================
       CROP HEALTH VISION — CONSISTENT READABLE INTERFACE
       Forest-green actions / white lettering / dark content
       ================================================ */
    :root {
      --chv-ink: #172c21;
      --chv-copy: #293b2c;
      --chv-secondary: #405444;
      --chv-forest: #355b3c;
      --chv-forest-hover: #28472e;
      --chv-forest-selected: #203c2a;
      --chv-red: #aa3030;
      --chv-red-hover: #882424;
      --chv-paper: #f5f1e8;
      --chv-white: #fffdf8;
      --chv-font: "Segoe UI", Arial, Helvetica, sans-serif;
    }
    html, body, .stApp, [data-testid="stAppViewContainer"], [data-testid="stSidebar"],
    [data-testid="stAppViewContainer"] button,
    [data-testid="stAppViewContainer"] input,
    [data-testid="stAppViewContainer"] textarea,
    [data-testid="stAppViewContainer"] select {
      font-family: var(--chv-font) !important;
    }
    /* One font family across the interface, including display titles. */
    .stMarkdown h1, .stMarkdown h2, .stMarkdown h3, .stMarkdown h4,
    .stMarkdown h5, .stMarkdown h6, .stMarkdown p, .stMarkdown li,
    .sb-kicker, .sb-title, .sb-sub, .section-eyebrow, .section-title, .section-sub,
    .sidebar-title, .sidebar-sub, .sidebar-section, .sidebar-kicker,
    .landing-heading h1, .landing-step-grid h3, .landing-number,
    .landing-lead, .landing-action, .landing-footer, .earthy-title, .earthy-subtitle,
    .metric-box, .metric-box *, .meta-card, .meta-card * {
      font-family: var(--chv-font) !important;
    }
    .stMarkdown h1, .stMarkdown h2, .stMarkdown h3, .stMarkdown h4,
    .section-title, .sb-title {
      color: var(--chv-ink) !important;
      font-weight: 750 !important;
    }
    .stMarkdown h2 {font-size: clamp(1.65rem, 2.2vw, 2.05rem) !important; line-height: 1.3 !important;}
    .stMarkdown h3 {font-size: 1.35rem !important; line-height: 1.35 !important;}
    [data-testid="stAppViewContainer"] .stMarkdown > div > p,
    [data-testid="stAppViewContainer"] .stMarkdown li {
      color: var(--chv-copy) !important;
      font-size: 1rem !important;
      line-height: 1.65 !important;
    }
    .section-sub, .meta-key { color: var(--chv-secondary) !important; font-size: .96rem !important; }
    .section-eyebrow { color: #426448 !important; font-size: .86rem !important; }
    /* Streamlit Metric: old dark-theme styling made values almost white. */
    [data-testid="stMetric"], .metric-box {
      background: var(--chv-white) !important;
      border: 1px solid #ddd8ca !important;
      color: var(--chv-ink) !important;
    }
    [data-testid="stMetric"] [data-testid="stMetricLabel"],
    [data-testid="stMetric"] [data-testid="stMetricLabel"] *,
    .metric-box .metric-label {
      color: var(--chv-copy) !important;
      opacity: 1 !important;
      font-size: 1rem !important;
      font-weight: 700 !important;
      line-height: 1.4 !important;
      letter-spacing: 0 !important;
    }
    [data-testid="stMetric"] [data-testid="stMetricValue"],
    [data-testid="stMetric"] [data-testid="stMetricValue"] *,
    .metric-box .metric-value {
      color: #213e29 !important;
      opacity: 1 !important;
      font-size: clamp(1.9rem, 2.5vw, 2.45rem) !important;
      font-weight: 800 !important;
      line-height: 1.25 !important;
    }
    [data-testid="stMetric"] [data-testid="stMetricDelta"],
    [data-testid="stMetric"] [data-testid="stMetricDelta"] * {
      font-size: .93rem !important;
      font-weight: 700 !important;
      opacity: 1 !important;
    }
    /* Alerts (loaded images / experiment completed / warnings): dark and legible. */
    [data-testid="stAlert"] {
      border-color: #c6d0bf !important;
      background-color: #e3eddf !important;
      color: var(--chv-ink) !important;
    }
    [data-testid="stAlert"] :is(p, span, div, strong, li, code) {
      color: var(--chv-ink) !important;
      font-size: 1rem !important;
      font-weight: 600 !important;
      line-height: 1.55 !important;
      opacity: 1 !important;
    }
    [data-testid="stAlert"] svg {color: #355b3c !important;}
    [data-testid="stCaptionContainer"], [data-testid="stCaptionContainer"] *,
    .stCaption, [data-testid="stImage"] small {
      color: var(--chv-secondary) !important;
      font-size: .88rem !important;
      opacity: 1 !important;
    }
    /* All real action and download buttons share the forest-green treatment. */
    .stButton > button,
    .stDownloadButton > button,
    [data-testid="stFileUploader"] button {
      font-family: var(--chv-font) !important;
      background: var(--chv-forest) !important;
      border: 1px solid var(--chv-forest) !important;
      border-radius: 9px !important;
      color: #ffffff !important;
      font-size: .96rem !important;
      font-weight: 700 !important;
      min-height: 2.8rem !important;
      text-shadow: none !important;
      box-shadow: none !important;
      opacity: 1 !important;
      transition: background .15s ease, border-color .15s ease !important;
    }
    .stButton > button :is(span, p, div, small),
    .stDownloadButton > button :is(span, p, div, small),
    [data-testid="stFileUploader"] button :is(span, p, div, small) {
      color: #ffffff !important;
      font-family: var(--chv-font) !important;
      opacity: 1 !important;
    }
    .stButton > button svg,
    .stDownloadButton > button svg,
    [data-testid="stFileUploader"] button svg {color: #ffffff !important;stroke: #ffffff !important;}
    .stButton > button:hover:not(:disabled),
    .stDownloadButton > button:hover:not(:disabled),
    [data-testid="stFileUploader"] button:hover:not(:disabled) {
      background: var(--chv-forest-hover) !important;
      border-color: var(--chv-forest-hover) !important;
      color: #ffffff !important;
      transform: none !important;
      box-shadow: none !important;
    }
    .stButton > button:disabled,
    .stDownloadButton > button:disabled {
      background: #70866f !important;
      border-color: #70866f !important;
      color: #ffffff !important;
      opacity: .78 !important;
      box-shadow: none !important;
    }
    .stButton > button:disabled :is(span,p,div,small),
    .stDownloadButton > button:disabled :is(span,p,div,small) {color: #ffffff !important; opacity:1 !important;}
    /* Tab navigation is a row of rubric/action buttons. */
    div[data-baseweb="tab-list"] {background:#e3e8dc !important;border:1px solid #cbd5c7 !important;}
    button[data-baseweb="tab"] {
      background: var(--chv-forest) !important;
      border: 1px solid #456e4b !important;
      border-radius: 8px !important;
      color: #ffffff !important;
      font-family: var(--chv-font) !important;
      font-size: .98rem !important;
      font-weight: 700 !important;
      opacity: 1 !important;
    }
    button[data-baseweb="tab"] :is(span,p,div) {color: #ffffff !important; font-family:var(--chv-font) !important;}
    button[data-baseweb="tab"]:hover {background: var(--chv-forest-hover) !important;color:#ffffff !important;}
    button[data-baseweb="tab"][aria-selected="true"] {background: var(--chv-forest-selected) !important;color:#ffffff !important;border-color:var(--chv-forest-selected) !important;}
    /* Collapsible rubrics: dark green header, light open content. */
    [data-testid="stExpander"] {background:var(--chv-white) !important;}
    [data-testid="stExpander"] summary {
      background: var(--chv-forest) !important;
      color: #ffffff !important;
      padding: .7rem .85rem !important;
    }
    [data-testid="stExpander"] summary:hover {background: var(--chv-forest-hover) !important;}
    [data-testid="stExpander"] summary :is(span,p,div,small),
    [data-testid="stExpander"] summary svg {color: #ffffff !important;stroke: #ffffff !important;}
    /* The sidebar uses exactly the same type system as the work area. */
    [data-testid="stSidebar"] {color: var(--chv-ink) !important;}
    [data-testid="stSidebar"] .sb-kicker,
    [data-testid="stSidebar"] .sidebar-kicker {
      color: #426044 !important;
      font-size: .83rem !important;
      font-weight: 750 !important;
      letter-spacing: .09em !important;
    }
    [data-testid="stSidebar"] .sb-title,
    [data-testid="stSidebar"] .sidebar-title {
      color: var(--chv-ink) !important;
      font-size: 1.28rem !important;
      font-weight: 750 !important;
      line-height: 1.35 !important;
    }
    [data-testid="stSidebar"] .sb-sub,
    [data-testid="stSidebar"] .sidebar-sub {
      color: var(--chv-copy) !important;
      font-size: .94rem !important;
      line-height: 1.55 !important;
    }
    [data-testid="stSidebar"] .stMarkdown h3 {
      color: var(--chv-ink) !important;
      font-size: 1.08rem !important;
      font-weight: 750 !important;
      margin-bottom: .3rem !important;
    }
    [data-testid="stSidebar"] [data-testid="stWidgetLabel"],
    [data-testid="stSidebar"] [data-testid="stWidgetLabel"] *,
    [data-testid="stSidebar"] [data-testid="stFileUploader"] label,
    [data-testid="stSidebar"] [data-testid="stSelectbox"] label,
    [data-testid="stSidebar"] .stCaption, [data-testid="stSidebar"] [data-testid="stCaptionContainer"] * {
      color: var(--chv-copy) !important;
      opacity: 1 !important;
      font-size: .94rem !important;
      font-family: var(--chv-font) !important;
    }
    [data-testid="stSidebar"] [data-testid="stRadio"] label {
      background:#f4f5ee !important;
      border:1px solid #bdcbb9 !important;
      color:var(--chv-ink) !important;
    }
    [data-testid="stSidebar"] [data-testid="stRadio"] label:has(input:checked) {
      background:#dce9d7 !important;
      border-color:#618565 !important;
    }
    [data-testid="stSidebar"] [data-testid="stRadio"] label :is(span,p,div) {color:var(--chv-ink) !important;font-size:.95rem !important;}
    [data-testid="stSidebar"] input[type="radio"] {accent-color: #355b3c !important;}
    [data-baseweb="select"] *, [data-baseweb="input"] *,
    [data-baseweb="textarea"] *, [data-testid="stFileUploaderDropzone"] small {
      color: var(--chv-copy) !important;
      opacity: 1 !important;
      font-family: var(--chv-font) !important;
    }
    /* Reset must ALWAYS be destructive red; preserve prominent white label. */
    .st-key-reset_uploaded_images button,
    [data-testid="stSidebar"] .st-key-reset_uploaded_images button {
      background: var(--chv-red) !important;
      border-color: var(--chv-red) !important;
      color: #ffffff !important;
      opacity: 1 !important;
    }
    .st-key-reset_uploaded_images button:hover,
    [data-testid="stSidebar"] .st-key-reset_uploaded_images button:hover {
      background: var(--chv-red-hover) !important;
      border-color: var(--chv-red-hover) !important;
      color: #ffffff !important;
    }
    .st-key-reset_uploaded_images button :is(span,p,div,small),
    .st-key-reset_uploaded_images button svg {color:#ffffff !important;stroke:#ffffff !important;}
    /* Keep home landing copy dark against the existing pale photo overlay. */
    .landing-heading h1,.landing-step-grid h3 {color:var(--chv-ink) !important;}
    .landing-lead,.landing-step-grid p,.landing-action p {color:var(--chv-copy) !important;}
    @media (max-width: 700px) {
      [data-testid="stMetric"] [data-testid="stMetricValue"] {font-size: 1.95rem !important;}
      [data-testid="stSidebar"] .sb-title {font-size:1.2rem !important;}
    }
    </style>""", unsafe_allow_html=True)


    # Sidebar visual QA: fix the validation CTA contrast and file-uploader layout.
    # This is a CSS-only change. Original widget keys, callbacks and analysis are untouched.
    st.markdown("""<style>
    /* Project validation button: override legacy sidebar secondary-button CSS. */
    [data-testid="stSidebar"] .st-key-open_validation button,
    [data-testid="stSidebar"] [class*="st-key-open_validation"] button {
        background: #355b3c !important;
        background-color: #355b3c !important;
        border: 1px solid #355b3c !important;
        color: #ffffff !important;
        opacity: 1 !important;
        min-height: 46px !important;
        font-size: .89rem !important;
        font-weight: 750 !important;
        line-height: 1.35 !important;
        white-space: normal !important;
        overflow-wrap: anywhere !important;
        text-align: center !important;
        text-shadow: none !important;
        box-shadow: none !important;
    }
    [data-testid="stSidebar"] .st-key-open_validation button :is(span,div,p,small),
    [data-testid="stSidebar"] [class*="st-key-open_validation"] button :is(span,div,p,small) {
        color: #ffffff !important;
        opacity: 1 !important;
        font-size: inherit !important;
        font-weight: inherit !important;
        white-space: normal !important;
    }
    [data-testid="stSidebar"] .st-key-open_validation button:hover,
    [data-testid="stSidebar"] [class*="st-key-open_validation"] button:hover {
        background: #28472e !important;
        background-color: #28472e !important;
        border-color: #28472e !important;
        color: #ffffff !important;
    }
    [data-testid="stSidebar"] .st-key-open_validation button:focus-visible {
        outline: 3px solid #95ba91 !important;
        outline-offset: 2px !important;
    }

    /* FILE UPLOADER: no overlapping native labels. CSS-only; Streamlit keeps
       its original clickable button, keyboard access and multi-file handling. */
    [data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] {
        box-sizing: border-box !important;
        display: flex !important;
        flex-direction: column !important;
        align-items: stretch !important;
        justify-content: center !important;
        gap: 0 !important;
        width: 100% !important;
        min-width: 0 !important;
        min-height: 105px !important;
        padding: 12px !important;
        background: #fffdf8 !important;
        border: 1px dashed #9eaf98 !important;
        border-radius: 11px !important;
        overflow: hidden !important;
    }
    [data-testid="stSidebar"] [data-testid="stFileUploaderDropzoneInstructions"],
    [data-testid="stSidebar"] [data-testid="stFileUploaderDropzoneInstructionsPrimaryText"],
    [data-testid="stSidebar"] [data-testid="stFileUploaderDropzoneInstructionsSecondaryText"] {
        display: none !important;
    }
    [data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] > div:not(:has(button)),
    [data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] > span:not(:has(button)) {
        flex: 0 0 auto !important;
        min-width: 0 !important;
    }
    [data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] button {
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        flex: 0 0 auto !important;
        align-self: stretch !important;
        width: 100% !important;
        max-width: 100% !important;
        min-width: 0 !important;
        min-height: 44px !important;
        margin: 0 !important;
        padding: 11px !important;
        box-sizing: border-box !important;
        background: #355b3c !important;
        border: 1px solid #355b3c !important;
        border-radius: 8px !important;
        color: #ffffff !important;
        font-size: 0 !important; /* native text cannot overlap the single rendered label */
        line-height: 0 !important;
        letter-spacing: 0 !important;
        white-space: nowrap !important;
        text-shadow: none !important;
        box-shadow: none !important;
        opacity: 1 !important;
        overflow: hidden !important;
        position: relative !important;
    }
    [data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] button > *,
    [data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] button :is(span, p, div, small, svg) {
        font-size: 0 !important;
        line-height: 0 !important;
        letter-spacing: 0 !important;
        color: transparent !important;
        opacity: 0 !important;
    }
    [data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] button::before {
        content: none !important;
    }
    /* One visible label only. Native button remains the actual working uploader. */
    [data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] button::after {
        content: "Choose files" !important;
        display: block !important;
        position: relative !important;
        flex: 0 0 auto !important;
        font-family: "Segoe UI", Arial, sans-serif !important;
        font-size: 14px !important;
        font-weight: 700 !important;
        line-height: 1.3 !important;
        letter-spacing: .01em !important;
        color: #ffffff !important;
        white-space: nowrap !important;
        opacity: 1 !important;
    }
    [data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] button:hover {
        background: #28472e !important;
        border-color: #28472e !important;
    }
    [data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] button:focus-visible {
        outline: 3px solid #aac7a2 !important;
        outline-offset: 2px !important;
    }
    [data-testid="stSidebar"] [class*="st-key-standard_image_uploader_"] [data-testid="stFileUploaderDropzone"]::after {
        content: "JPG, PNG · Up to 200 MB per file";
        display: block !important;
        width: 100% !important;
        box-sizing: border-box !important;
        margin: 8px 0 0 !important;
        text-align: center !important;
        color: #405543 !important;
        font-family: "Segoe UI", Arial, sans-serif !important;
        font-size: 11px !important;
        font-weight: 500 !important;
        line-height: 1.35 !important;
    }
    </style>""", unsafe_allow_html=True)


    # Finishing pass — display, typography and the validation back control ONLY.
    # Keeps the existing algorithms, uploader logic, result charts and exports.
    st.markdown("""<style>
    /* A single typeface for the full application, including the home hero,
       metrics, captions, sidebar, image panels, charts' HTML labels and buttons. */
    html, body, .stApp, [data-testid="stSidebar"],
    .stApp :is(h1,h2,h3,h4,h5,h6,p,li,label,button,input,textarea,select,small,strong,em,figcaption),
    .stApp [data-testid="stCaptionContainer"],
    .stApp [data-testid="stCaptionContainer"] p,
    .stApp [data-testid="stWidgetLabel"],
    .stApp .stMarkdown,
    .stApp .stMarkdown .chv-intro h1,
    .stApp .stMarkdown .chv-intro h1 em,
    .stApp .stMarkdown .chv-intro h2,
    .stApp .stMarkdown .chv-benefit h3,
    .stApp .stMarkdown .chv-panel h2,
    .stApp .stMarkdown .chv-crop-label strong,
    .stApp .chv-step-index, .stApp .chv-crop-label small,
    .stApp .earthy-title, .stApp .earthy-subtitle,
    .stApp .section-title, .stApp .section-eyebrow {
        font-family: "Segoe UI", Arial, Helvetica, sans-serif !important;
    }
    /* The six single-image results are images, not dark canvases.
       Borders / shadows are understated and consistent with Natural & Earthy. */
    [data-testid="stImage"] img {
        height: auto !important;
        object-fit: contain !important;
        background-color: #faf8f1 !important;
        border: 1px solid #dedfd3 !important;
        border-radius: 12px !important;
        box-shadow: 0 6px 20px rgba(36, 52, 39, .10) !important;
    }
    .chv-stage-label {
        color: #20382a !important;
        font-family: "Segoe UI", Arial, sans-serif !important;
        font-size: 1rem !important;
        font-weight: 750 !important;
        letter-spacing: 0 !important;
        margin: .25rem 0 .65rem !important;
        line-height: 1.35 !important;
    }
    /* The same forest-green/white button as the main analysis actions. */
    [data-testid="stSidebar"] [class*="st-key-back_from_validation"] button {
        background: #355b3c !important;
        background-color: #355b3c !important;
        border: 1px solid #355b3c !important;
        color: #ffffff !important;
        opacity: 1 !important;
        font-family: "Segoe UI", Arial, sans-serif !important;
        font-size: .98rem !important;
        font-weight: 700 !important;
        border-radius: 9px !important;
        min-height: 44px !important;
    }
    [data-testid="stSidebar"] [class*="st-key-back_from_validation"] button :is(span,p,div,small),
    [data-testid="stSidebar"] [class*="st-key-back_from_validation"] button svg {
        color: #ffffff !important;
        fill: none !important;
        stroke: #ffffff !important;
        opacity: 1 !important;
    }
    [data-testid="stSidebar"] [class*="st-key-back_from_validation"] button:hover {
        color: #ffffff !important;
        background: #28472e !important;
        background-color: #28472e !important;
        border-color: #28472e !important;
    }
    [data-testid="stSidebar"] [class*="st-key-back_from_validation"] button:focus-visible {
        outline: 3px solid #94b98d !important;
        outline-offset: 2px !important;
    }
    </style>""", unsafe_allow_html=True)

    crop = "Potato"
    grid_size = 10
    uploaded = []
    analyze_btn = False
    gps_df = None
    segmentation_gt = None
    ground_truth_input = ""
    potato_robust_files, beet_robust_files = [], []

    with st.sidebar:
        if st.session_state.engineering_validation:
            st.markdown('<div class="sb-kicker">PROJECT VALIDATION</div><div class="sb-title">Illumination robustness test</div><div class="sb-sub">Test classifier stability under different light conditions.</div>', unsafe_allow_html=True)
            if st.button("← Back to field analysis", use_container_width=True, key="back_from_validation"):
                st.session_state.engineering_validation = False
                st.rerun()

            st.markdown("### 1. Upload images")
            potato_robust_files = st.file_uploader("🥔 Potato images", type=["jpg","jpeg","png"], accept_multiple_files=True,
                key=f"robust_potato_uploader_{st.session_state.upload_reset_nonce}")
            beet_robust_files = st.file_uploader("🌿 Sugar Beet images", type=["jpg","jpeg","png"], accept_multiple_files=True,
                key=f"robust_beet_uploader_{st.session_state.upload_reset_nonce}")
            st.markdown("### 2. Brightness variations")
            st.caption("−30%   ·   −20%   ·   Original   ·   +20%   ·   +30%")
            st.caption("The full robustness test, calculations, tables, graphs and exports are unchanged.")
        else:
            st.markdown('<div class="sb-kicker">CROP HEALTH VISION</div><div class="sb-title">Field analysis</div><div class="sb-sub">Upload crop images and run the disease-detection workflow.</div>', unsafe_allow_html=True)
            st.markdown("### 1. Choose crop")
            crop = st.radio("Crop", ["Potato", "Sugar Beet"], horizontal=True, key="sidebar_crop", label_visibility="collapsed")
            st.markdown('<span id="chv-upload-target" aria-hidden="true"></span>', unsafe_allow_html=True)
            st.markdown("### 2. Upload images")
            uploaded = st.file_uploader("Upload image(s)", type=["jpg","jpeg","png"], accept_multiple_files=True,
                key=f"standard_image_uploader_{st.session_state.upload_reset_nonce}",
                help="One image = detailed inspection. Multiple images = batch screening.") or []
            st.markdown("### 3. Area detail")
            grid_size = st.selectbox(
                "Grid resolution",
                options=[5, 8, 10, 12, 15, 20],
                index=2,
                help="Choose the field-grid resolution. 10 × 10 is recommended.",
                key="sidebar_grid",
            )
            st.caption(f"Grid: {grid_size} × {grid_size} = {grid_size**2} zones")
            analyze_btn = st.button("▶ Analyze crop images", disabled=not uploaded, use_container_width=True, type="primary", key="sidebar_analyze")
            with st.expander("Optional field map & validation inputs", expanded=False):
                gps_upload = st.file_uploader("GPS coordinates CSV", type=["csv"], key=f"gps_csv_uploader_{st.session_state.upload_reset_nonce}")
                if gps_upload is not None:
                    try:
                        gps_df = pd.read_csv(gps_upload)
                        st.success(f"GPS file loaded · {len(gps_df)} rows")
                    except Exception as e:
                        st.error(f"Could not read GPS CSV: {e}")
                ground_truth_input = st.selectbox("Ground truth", ["", "Healthy", "Suspicious", "Sick"], key="ground_truth_input")
                segmentation_gt = st.file_uploader("Leaf mask GT (optional)", type=["png","jpg","jpeg"], key=f"segmentation_gt_uploader_{st.session_state.upload_reset_nonce}")

            st.markdown("---")
            st.markdown('<div class="sb-kicker">PROJECT VALIDATION</div>', unsafe_allow_html=True)
            if st.button("🧪 Illumination robustness test", use_container_width=True, key="open_validation"):
                st.session_state.engineering_validation = True
                st.rerun()

        st.markdown("---")
        if st.button("🗑 Reset uploaded images", use_container_width=True, key="reset_uploaded_images"):
            # New widget keys guarantee that Streamlit file uploaders are physically cleared.
            st.session_state.upload_reset_nonce += 1
            st.session_state.engineering_validation = False
            st.session_state.analysis_active = False
            st.session_state.analysis_signature = None
            st.session_state.csv_rows = []
            st.session_state.ground_truth_labels = {}
            for k in [
                "robustness_detailed", "robustness_summary", "robustness_crop_summary",
                "robustness_dataset_signature", "last_result"
            ]:
                st.session_state.pop(k, None)
            for k in list(st.session_state.keys()):
                if str(k).startswith("gt_live_select_") or str(k).startswith("apply_gt_"):
                    st.session_state.pop(k, None)
            # Return to a clean home page immediately.
            st.rerun()

    if st.session_state.engineering_validation:
        render_robustness_mode(potato_robust_files, beet_robust_files)
        return

    current_signature = _uploaded_files_signature(uploaded, crop, grid_size)
    if analyze_btn and current_signature is not None:
        st.session_state.analysis_active = True
        st.session_state.analysis_signature = current_signature

    analysis_ready = bool(
        uploaded
        and st.session_state.analysis_active
        and st.session_state.analysis_signature == current_signature
    )

    # ── Passive home screen ──
    # Presentation-only landing page. Start Analysis navigates to the existing
    # functional sidebar uploader; analysis itself remains unchanged.
    if not analysis_ready:
        st.markdown(f'<div class="home-hero">{EARTHY_HOME}</div>', unsafe_allow_html=True)
        return

    # Results workspace: use the whole central area instead of keeping the landing page visible.
    st.markdown("""
    <style>
      .block-container {max-width:none !important; width:100% !important; padding:1.15rem 1.6rem 2.2rem !important;}
    </style>
    """, unsafe_allow_html=True)

    # Multiple files switch automatically to batch screening.
    if len(uploaded) > 1:
        render_batch_analysis(uploaded, crop, grid_size, gps_df=gps_df)
        return

    # One file keeps the detailed single-image workflow.
    uploaded = uploaded[0]

    # ── Load image ──
    file_bytes = np.frombuffer(uploaded.getvalue(), np.uint8)
    img_bgr    = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if img_bgr is None:
        st.error("Could not read the uploaded image. Please try a different file.")
        return

    # Keep true uploaded dimensions for display only. Analysis still uses
    # its established 2000-pixel safety cap and 800×600 pre-processing.
    source_h, source_w = img_bgr.shape[:2]
    # ── Cap resolution to 2000px on the long side for fast processing ──
    _MAX_DIM = 2000
    _h, _w = img_bgr.shape[:2]
    if max(_h, _w) > _MAX_DIM:
        _scale = _MAX_DIM / max(_h, _w)
        img_bgr = cv2.resize(img_bgr, (int(_w * _scale), int(_h * _scale)),
                             interpolation=cv2.INTER_AREA)

    H, W = img_bgr.shape[:2]

    # Presentation only: display the exact uploaded file, retaining its full
    # original resolution. The separate 2000px cap above is unchanged and still
    # applies to the computation pipeline; no extra pixels are invented here.
    col_prev, col_meta = st.columns([1.7, 1], gap="large")
    with col_prev:
        st.image(uploaded.getvalue(), caption=uploaded.name, width=760)
    with col_meta:
        st.markdown(f"""
<div class="meta-card">
  <div class="meta-card-title">IMAGE INPUT</div>
  <div class="meta-row"><span class="meta-key">Crop</span><span class="meta-val">{crop}</span></div>
  <div class="meta-row"><span class="meta-key">Resolution</span><span class="meta-val">{source_w} × {source_h}</span></div>
  <div class="meta-row"><span class="meta-key">Grid</span><span class="meta-val">{grid_size} × {grid_size}</span></div>
</div>
""", unsafe_allow_html=True)

    # ── Detect image type ──
    st.markdown("""
<div class="section-eyebrow">RESULTS</div>
<div class="section-title">Crop Health Results</div>
""", unsafe_allow_html=True)

    with st.spinner("Detecting image type..."):
        img_type = detect_image_type(img_bgr)

    # ── Two tabs: single image analysis + field grid ──
    tab_single, tab_field = st.tabs(["Image diagnosis", "Field map"])

    with tab_single:
        render_single_analysis_tab(
            img_bgr, crop, image_name=uploaded.name,
            segmentation_gt_bytes=(segmentation_gt.getvalue() if segmentation_gt is not None else None)
        )

    with tab_field:
        if img_type == "single_plant":
            st.info("🔍 This appears to be a close-up plant image. A spatial map is still available if needed.")
        else:
            st.info("🌾 Field image detected. Spatial analysis is available below.")

        with st.spinner("Processing field grid..."):
            # Use the same preprocessing chain as Single Image Analysis so the
            # implementation matches the documented system pipeline.
            img_field_proc = preprocess_img(img_bgr)
            results_matrix, score_matrix, label_matrix, summary = analyze_field(
                img_field_proc, crop, grid_size, grid_size
            )

        # Summary metrics
        alert_class = f"alert-{summary['alert_level'].lower()}"
        st.markdown(f"""
<div class="{alert_class}">
  <b>Field health overview</b> &nbsp;|&nbsp; Attention level: <b>{summary['alert_level']}</b> &nbsp;|&nbsp;
  Sick zones: <b>{summary['sick_pct']}%</b> &nbsp;|&nbsp;
  Suspicious zones: <b>{summary['suspicious_pct']}%</b> &nbsp;|&nbsp;
  Analyzed plant zones: <b>{summary['plant_cells']}</b>
</div>
""", unsafe_allow_html=True)
        st.caption("Percentages show the share of vegetation zones classified by the system.")

        st.markdown("")
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.markdown(f"""<div class="metric-box">
                <div class="metric-label">Healthy zones</div>
                <div class="metric-value healthy-color">{summary['healthy']}</div>
            </div>""", unsafe_allow_html=True)
        with m2:
            st.markdown(f"""<div class="metric-box">
                <div class="metric-label">Suspicious zones</div>
                <div class="metric-value suspicious-color">{summary['suspicious']}</div>
            </div>""", unsafe_allow_html=True)
        with m3:
            st.markdown(f"""<div class="metric-box">
                <div class="metric-label">Sick zones</div>
                <div class="metric-value sick-color">{summary['sick']}</div>
            </div>""", unsafe_allow_html=True)
        with m4:
            st.markdown(f"""<div class="metric-box">
                <div class="metric-label">Background / no plant</div>
                <div class="metric-value" style="color:#8b9dc3">{summary['no_plant']}</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("")

        # Maps inside nested tabs — compact and centered for a farmer-facing view.
        mt1, mt2, mt3 = st.tabs(["Areas requiring attention", "Disease intensity", "All classified zones"])
        img_overlay = render_overlay(img_bgr, results_matrix, grid_size, grid_size)
        img_heat = render_heatmap(score_matrix, grid_size, grid_size, label_matrix)
        img_cat = render_categorical(label_matrix, grid_size, grid_size, summary)
        has_attention = (summary["sick"] + summary["suspicious"]) > 0

        with mt1:
            cL, cM, cR = st.columns([1, 3.2, 1])
            with cM:
                st.image(img_overlay, width=820)
            st.download_button("Download overlay map", data=img_overlay,
                               file_name=f"field_overlay_{crop.lower().replace(' ','_')}.png",
                               mime="image/png")
        with mt2:
            if not has_attention:
                st.success("No suspicious or sick zones were detected. A disease-intensity heat map is therefore not required for this image.")
                st.caption("Healthy cells are assigned zero disease intensity. The grid remains available under All classified zones for spatial orientation.")
            else:
                cL, cM, cR = st.columns([1, 3.2, 1])
                with cM:
                    st.image(img_heat, width=760)
                st.download_button("Download heatmap", data=img_heat,
                                   file_name=f"field_heatmap_{crop.lower().replace(' ','_')}.png",
                                   mime="image/png")
        with mt3:
            cL, cM, cR = st.columns([1, 3.2, 1])
            with cM:
                st.image(img_cat, width=760)
            st.download_button("Download classification grid", data=img_cat,
                               file_name=f"field_grid_{crop.lower().replace(' ','_')}.png",
                               mime="image/png")

        # Only surface a heatmap as a primary result when there is actual attention evidence.
        if has_attention:
            st.divider()
            st.markdown("""<div class="section-eyebrow">FIELD OVERVIEW</div><div class="section-title">Disease intensity across the field</div>""", unsafe_allow_html=True)
            cL, cM, cR = st.columns([1, 3.2, 1])
            with cM:
                st.image(img_heat, width=760)
        else:
            st.info("Field overview: all analyzed vegetation zones are healthy; no disease heat map is shown.")

        # ── Sick zone details ──
        sick_zones = [
            (r["row"] + 1, r["col"] + 1, r["score"])
            for row in results_matrix for r in row
            if r["label"] == "Sick"
        ]
        if sick_zones:
            with st.expander(f"🔴 Sick zones detail ({len(sick_zones)} zones)", expanded=False):
                st.markdown("These grid positions need immediate attention:")
                cols = st.columns(4)
                for i, (row, col, score) in enumerate(sorted(sick_zones, key=lambda x: -x[2])):
                    cols[i % 4].markdown(
                        f"<div class='metric-box'>"
                        f"<div class='metric-label'>Row {row}, Col {col}</div>"
                        f"<div class='metric-value sick-color'>{score}</div>"
                        f"<div class='metric-label'>disease score</div></div>",
                        unsafe_allow_html=True
                    )

        susp_zones = [
            (r["row"] + 1, r["col"] + 1, r["score"])
            for row in results_matrix for r in row
            if r["label"] == "Suspicious"
        ]
        if susp_zones:
            with st.expander(f"🟡 Suspicious zones ({len(susp_zones)} zones)", expanded=False):
                st.markdown("These zones show early disease signals — monitor closely:")
                cols = st.columns(4)
                for i, (row, col, score) in enumerate(sorted(susp_zones, key=lambda x: -x[2])):
                    cols[i % 4].markdown(
                        f"<div class='metric-box'>"
                        f"<div class='metric-label'>Row {row}, Col {col}</div>"
                        f"<div class='metric-value suspicious-color'>{score}</div>"
                        f"<div class='metric-label'>disease score</div></div>",
                        unsafe_allow_html=True
                    )

        # ── Step 19: Field Summary Report ──
        with st.expander("Field Summary", expanded=False):
            total_zones = grid_size * grid_size
            plant_zones = summary["plant_cells"]
            st.markdown(f"""
**Field Analysis Summary**

| Field              | Value |
|--------------------|-------|
| Crop               | {crop} |
| Total zones        | {total_zones} |
| Plant zones        | {plant_zones} ({round(plant_zones/total_zones*100,1)}%) |
| No-plant zones     | {summary['no_plant']} |
| ✅ Healthy          | {summary['healthy']} ({round(summary['healthy']/max(plant_zones,1)*100,1)}%) |
| 🟡 Suspicious      | {summary['suspicious']} ({summary['suspicious_pct']}%) |
| 🔴 Sick            | {summary['sick']} ({summary['sick_pct']}%) |
| Alert Level        | **{summary['alert_level']}** |
            """)


if __name__ == "__main__":
    main()
