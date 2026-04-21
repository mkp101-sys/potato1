"""
╔══════════════════════════════════════════════════════════════════════════╗
║         POTATO SMART IRRIGATION DASHBOARD                               ║
║  Predicts surface soil moisture using LSTM hybrid model                 ║
║  Inputs: Sowing Date, Current Date, Lat/Lon                             ║
╚══════════════════════════════════════════════════════════════════════════╝
"""

import streamlit as st
import numpy as np
import pandas as pd
import requests
import math
import gzip
import joblib
import os
from datetime import datetime, date, timedelta

# ─────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────
st.set_page_config(
    page_title="🥔 Potato Smart Irrigation",
    page_icon="🥔",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Main background */
    .stApp { background-color: #0f1117; }

    /* Metric cards */
    .metric-card {
        background: linear-gradient(135deg, #1e2130 0%, #252840 100%);
        border: 1px solid #3a3f5c;
        border-radius: 14px;
        padding: 18px 20px;
        margin-bottom: 12px;
        text-align: center;
    }
    .metric-card .label {
        color: #8892b0;
        font-size: 0.78rem;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 6px;
    }
    .metric-card .value {
        color: #e6f1ff;
        font-size: 1.9rem;
        font-weight: 700;
        line-height: 1;
    }
    .metric-card .unit {
        color: #64ffda;
        font-size: 0.85rem;
        margin-top: 4px;
    }

    /* Stage badge */
    .stage-badge {
        display: inline-block;
        background: linear-gradient(90deg, #1a6b3c, #23a05a);
        color: white;
        padding: 6px 18px;
        border-radius: 20px;
        font-size: 0.9rem;
        font-weight: 600;
        letter-spacing: 0.5px;
    }

    /* Alert boxes */
    .alert-critical {
        background: linear-gradient(135deg, #3d1515, #5a1f1f);
        border: 1px solid #e05252;
        border-left: 4px solid #ff4b4b;
        border-radius: 10px;
        padding: 14px 18px;
        color: #ffcccc;
        margin: 10px 0;
    }
    .alert-warning {
        background: linear-gradient(135deg, #3d2e0a, #5a4410);
        border: 1px solid #d4900a;
        border-left: 4px solid #ffb700;
        border-radius: 10px;
        padding: 14px 18px;
        color: #fff3cc;
        margin: 10px 0;
    }
    .alert-ok {
        background: linear-gradient(135deg, #0a2e1a, #103d24);
        border: 1px solid #2ea84e;
        border-left: 4px solid #00d26a;
        border-radius: 10px;
        padding: 14px 18px;
        color: #ccffe0;
        margin: 10px 0;
    }

    /* Section headers */
    .section-header {
        font-size: 1.05rem;
        font-weight: 700;
        color: #64ffda;
        margin: 18px 0 10px;
        padding-bottom: 6px;
        border-bottom: 1px solid #2a2f4a;
        letter-spacing: 0.5px;
    }

    /* Progress bar container */
    .sm-bar-wrap {
        background: #1a1e2e;
        border-radius: 8px;
        height: 22px;
        width: 100%;
        margin: 6px 0;
        overflow: hidden;
        border: 1px solid #2a2f4a;
    }
    .sm-bar-fill {
        height: 100%;
        border-radius: 8px;
        display: flex;
        align-items: center;
        padding-left: 10px;
        font-size: 0.75rem;
        font-weight: 600;
        color: white;
        text-shadow: 0 1px 2px rgba(0,0,0,0.5);
    }

    /* Sidebar */
    .sidebar-title {
        font-size: 1.1rem;
        font-weight: 700;
        color: #64ffda;
        margin-bottom: 12px;
    }

    div[data-testid="stSidebar"] {
        background-color: #161822;
    }

    /* Input labels */
    .stDateInput label, .stTextInput label {
        color: #a8b2d8 !important;
        font-size: 0.85rem !important;
    }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────
# PURE-NUMPY LSTM INFERENCE  (no TensorFlow needed)
# Architecture: LSTM(64, relu) → Dropout → Dense(32, relu) → Dense(1)
# ─────────────────────────────────────────────────────────
def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))

def _relu(x):
    return np.maximum(0.0, x)

def _lstm_forward(x_seq, W, U, b):
    """
    x_seq : (batch, timesteps, features)
    Returns final hidden state h of shape (batch, units).
    Gate order: i, f, c_tilde, o  (Keras default).
    LSTM activation = relu, recurrent_activation = sigmoid.
    """
    batch = x_seq.shape[0]
    units = U.shape[0]
    h = np.zeros((batch, units), dtype=np.float32)
    c = np.zeros((batch, units), dtype=np.float32)
    for t in range(x_seq.shape[1]):
        z = x_seq[:, t, :] @ W + h @ U + b   # (batch, 4*units)
        n = units
        i      = _sigmoid(z[:, 0*n:1*n])
        f      = _sigmoid(z[:, 1*n:2*n])
        c_tild = _relu   (z[:, 2*n:3*n])      # activation='relu'
        o      = _sigmoid(z[:, 3*n:4*n])
        c = f * c + i * c_tild
        h = o * _relu(c)                       # recurrent uses same activation
    return h

def numpy_predict(weights, x_scaled):
    """
    weights : dict with keys W_lstm, U_lstm, b_lstm, W_d1, b_d1, W_d2, b_d2
    x_scaled: (batch, timesteps, features) float32
    Returns scalar prediction.
    """
    h = _lstm_forward(x_scaled,
                      weights['W_lstm'], weights['U_lstm'], weights['b_lstm'])
    out = _relu(h @ weights['W_d1'] + weights['b_d1'])
    out = out @ weights['W_d2'] + weights['b_d2']
    return float(out[0, 0])


# ─────────────────────────────────────────────────────────
# MODEL & SCALER LOADING
# ─────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

@st.cache_resource(show_spinner="⚙️ Loading AI model...")
def load_resources():
    """Load scaler and model weights (no TensorFlow required)."""
    errors = []

    # -- Scaler --
    scaler = None
    scaler_path = os.path.join(BASE_DIR, "data_scaler.gz")
    try:
        with gzip.open(scaler_path, "rb") as f:
            raw = f.read()
        import tempfile
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pkl") as tmp:
            tmp.write(raw)
            tmp_path = tmp.name
        scaler = joblib.load(tmp_path)
        os.unlink(tmp_path)
    except Exception as e:
        errors.append(f"Scaler load error: {e}")

    # -- Model weights (NumPy .npz — no TensorFlow needed) --
    weights = None
    npz_path = os.path.join(BASE_DIR, "potato_weights.npz")
    if os.path.exists(npz_path):
        try:
            data = np.load(npz_path)
            weights = {k: data[k] for k in data.files}
        except Exception as e:
            errors.append(f"Weights load error: {e}")
    else:
        # Fallback: try to load from original Keras/H5 file via TensorFlow
        try:
            os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
            import tensorflow as tf
            keras_path = os.path.join(BASE_DIR, "potato_hybrid_model.keras")
            h5_path    = os.path.join(BASE_DIR, "potato_hybrid_model__1_.h5")
            if os.path.exists(keras_path):
                m = tf.keras.models.load_model(keras_path)
            elif os.path.exists(h5_path):
                m = tf.keras.models.load_model(h5_path)
            else:
                errors.append("No model file found (potato_weights.npz, .keras, or .h5).")
                return None, scaler, errors
            w = m.get_weights()
            weights = dict(W_lstm=w[0], U_lstm=w[1], b_lstm=w[2],
                           W_d1=w[3], b_d1=w[4], W_d2=w[5], b_d2=w[6])
        except Exception as e:
            errors.append(f"Model load error: {e}")

    return weights, scaler, errors


# ─────────────────────────────────────────────────────────
# CROP STAGE LOGIC
# ─────────────────────────────────────────────────────────
STAGES = [
    (0,   15,  "Pre-Emergence",    "🌱", "#607d8b",
        "Soil moist; no canopy yet.",  0.40, 0.50),
    (15,  30,  "Emergence",        "🌿", "#43a047",
        "First leaves, roots establishing.", 0.50, 0.60),
    (30,  55,  "Vegetative Growth","🍃", "#2e7d32",
        "Rapid top growth, water demand rising.", 0.60, 0.75),
    (55,  75,  "Tuber Initiation", "🔵", "#1565c0",
        "Critical! Maintain adequate moisture.", 0.75, 0.85),
    (75, 105,  "Tuber Bulking",    "💪", "#6a1b9a",
        "Peak water demand. Irrigation vital.", 0.80, 0.90),
    (105,125,  "Maturation",       "🌾", "#e65100",
        "Reduce irrigation. Skin set phase.", 0.55, 0.65),
    (125,999,  "Post-Harvest",     "✅", "#795548",
        "Crop ready. No irrigation needed.", 0.30, 0.40),
]

def get_stage(das: int):
    for s_min, s_max, name, icon, color, desc, kc_lo, kc_hi in STAGES:
        if s_min <= das < s_max:
            return dict(name=name, icon=icon, color=color, desc=desc,
                        kc=(kc_lo + kc_hi) / 2, kc_range=(kc_lo, kc_hi),
                        das_range=(s_min, s_max))
    return dict(name="Unknown", icon="❓", color="#9e9e9e", desc="",
                kc=0.6, kc_range=(0.5,0.7), das_range=(0,999))


# ─────────────────────────────────────────────────────────
# NDVI ESTIMATION
# ─────────────────────────────────────────────────────────
NDVI_CURVE = {0:0.08, 10:0.12, 20:0.22, 30:0.40, 45:0.62, 55:0.72,
              65:0.77, 80:0.80, 95:0.75, 105:0.60, 115:0.38, 125:0.18}

def estimate_ndvi(das: int) -> float:
    das = max(0, min(das, 125))
    keys = sorted(NDVI_CURVE.keys())
    for i in range(len(keys)-1):
        k0, k1 = keys[i], keys[i+1]
        if k0 <= das <= k1:
            t = (das - k0) / (k1 - k0)
            return round(NDVI_CURVE[k0] + t * (NDVI_CURVE[k1] - NDVI_CURVE[k0]), 4)
    return 0.1


# ─────────────────────────────────────────────────────────
# WEATHER API  (Open-Meteo — free, no key)
# ─────────────────────────────────────────────────────────
@st.cache_data(ttl=1800, show_spinner="🌦️ Fetching weather data...")
def fetch_weather(lat: float, lon: float, start: str, end: str) -> dict:
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat,
        "longitude": lon,
        "daily": [
            "temperature_2m_max",
            "temperature_2m_min",
            "relative_humidity_2m_max",
            "relative_humidity_2m_min",
            "wind_speed_10m_max",
            "et0_fao_evapotranspiration",
            "shortwave_radiation_sum",
            "precipitation_sum",
        ],
        "start_date": start,
        "end_date": end,
        "timezone": "auto",
    }
    r = requests.get(url, params=params, timeout=20)
    r.raise_for_status()
    return r.json()


# ─────────────────────────────────────────────────────────
# ETo  — FAO-56 Penman-Monteith
# ─────────────────────────────────────────────────────────
def eto_penman_monteith(
    tmax: float, tmin: float, rh_mean: float,
    ws_10m: float, rs_mj: float,
    lat_deg: float, doy: int, elev_m: float = 250
) -> float:
    """Returns ETo in mm/day using FAO-56 PM method."""
    T      = (tmax + tmin) / 2.0
    lat_r  = math.radians(lat_deg)

    # Atmospheric pressure & psychrometric constant
    P     = 101.3 * ((293 - 0.0065 * elev_m) / 293) ** 5.26
    gamma = 0.000665 * P

    # Saturation and actual vapour pressure
    eT_max = 0.6108 * math.exp(17.27 * tmax / (tmax + 237.3))
    eT_min = 0.6108 * math.exp(17.27 * tmin / (tmin + 237.3))
    es     = (eT_max + eT_min) / 2.0
    ea     = es * (rh_mean / 100.0)

    # Slope of saturation vapour pressure curve
    delta = 4098 * (0.6108 * math.exp(17.27 * T / (T + 237.3))) / (T + 237.3) ** 2

    # Net shortwave radiation
    Rns = (1 - 0.23) * rs_mj

    # Extraterrestrial radiation (Ra)
    dr   = 1 + 0.033 * math.cos(2 * math.pi / 365 * doy)
    decl = 0.409 * math.sin(2 * math.pi / 365 * doy - 1.39)
    ws_a = math.acos(max(-1, min(1, -math.tan(lat_r) * math.tan(decl))))
    Ra   = (24 * 60 / math.pi) * 0.0820 * dr * (
        ws_a * math.sin(lat_r) * math.sin(decl)
        + math.cos(lat_r) * math.cos(decl) * math.sin(ws_a)
    )
    Rso  = (0.75 + 2e-5 * elev_m) * Ra
    fcd  = max(0.05, min(1.0, 1.35 * rs_mj / max(Rso, 0.001) - 0.35))

    # Net longwave radiation
    sigma_TK4 = 4.903e-9 * ((tmax + 273.16) ** 4 + (tmin + 273.16) ** 4) / 2
    Rnl  = sigma_TK4 * (0.34 - 0.14 * math.sqrt(max(0, ea))) * fcd
    Rn   = Rns - Rnl

    # Wind speed at 2m
    u2 = ws_10m * (4.87 / math.log(67.8 * 10 - 5.42))

    # ETo
    num = 0.408 * delta * (Rn - 0) + gamma * (900 / (T + 273)) * u2 * (es - ea)
    den = delta + gamma * (1 + 0.34 * u2)
    return max(0.0, round(num / den, 3))


# ─────────────────────────────────────────────────────────
# GDD CALCULATION  (base temp = 7 °C for potato)
# ─────────────────────────────────────────────────────────
def calc_gdd(tmax_list, tmin_list, t_base=7.0) -> tuple:
    daily = [max(0.0, (tx + tn) / 2 - t_base) for tx, tn in zip(tmax_list, tmin_list)]
    return round(sum(daily), 1), [round(g, 2) for g in daily]


# ─────────────────────────────────────────────────────────
# SOIL MOISTURE → IRRIGATION DECISION
# ─────────────────────────────────────────────────────────
def irrigation_decision(sm_pred: float, stage_name: str) -> dict:
    """
    Thresholds (m³/m³) vary by stage.
    Returns status, deficit, and recommended application (mm).
    Root depth assumed 300 mm.
    """
    THRESHOLDS = {
        "Pre-Emergence":    (0.18, 0.26, 0.32),
        "Emergence":        (0.20, 0.28, 0.35),
        "Vegetative Growth":(0.22, 0.30, 0.38),
        "Tuber Initiation": (0.24, 0.32, 0.40),
        "Tuber Bulking":    (0.24, 0.33, 0.42),
        "Maturation":       (0.18, 0.24, 0.30),
        "Post-Harvest":     (0.10, 0.18, 0.25),
    }
    crit, opt, fc = THRESHOLDS.get(stage_name, (0.20, 0.28, 0.36))
    root_depth_mm = 300

    if sm_pred >= opt:
        status = "optimal"
        deficit_m3m3 = 0
    elif sm_pred >= crit:
        status = "low"
        deficit_m3m3 = opt - sm_pred
    else:
        status = "critical"
        deficit_m3m3 = fc - sm_pred

    apply_mm = round(deficit_m3m3 * root_depth_mm, 1)
    return dict(
        status=status, critical=crit, optimal=opt, fc=fc,
        deficit=round(deficit_m3m3, 4), apply_mm=apply_mm,
    )


# ─────────────────────────────────────────────────────────
# MODEL PREDICTION
# ─────────────────────────────────────────────────────────
def run_prediction(weights, scaler, tmax, tmin, eto, rh,
                   ndvi, soil_pot_surf, soil_pot_rz,
                   tmean, das) -> float:
    """
    9 features (matching scaler order):
    [Tmax, Tmin, ETo, RH, NDVI, SoilPot_surf(kPa), SoilPot_rz(kPa), Tmean, DAS]
    Returns predicted surface soil moisture (m³/m³).
    """
    x = np.array([[tmax, tmin, eto, rh, ndvi,
                   soil_pot_surf, soil_pot_rz,
                   tmean, das]], dtype=np.float32)

    x_scaled = scaler.transform(x)                   # shape (1, 9)
    x_lstm   = x_scaled.reshape(1, 1, 9).astype(np.float32)  # shape (1, 1, 9)
    pred     = numpy_predict(weights, x_lstm)
    return float(pred)


# ─────────────────────────────────────────────────────────
# HELPER: custom HTML metric card
# ─────────────────────────────────────────────────────────
def metric_card(label: str, value: str, unit: str = "", col=None):
    html = f"""
    <div class="metric-card">
        <div class="label">{label}</div>
        <div class="value">{value}</div>
        <div class="unit">{unit}</div>
    </div>"""
    (col or st).markdown(html, unsafe_allow_html=True)


def soil_bar(sm_val: float, crit: float, opt: float, fc: float):
    """Render a colour-coded soil moisture bar."""
    pct = min(100, sm_val / fc * 100) if fc > 0 else 50
    if sm_val >= opt:
        color, label = "#00d26a", "Optimal"
    elif sm_val >= crit:
        color, label = "#ffb700", "Low — irrigate soon"
    else:
        color, label = "#ff4b4b", "Critical — irrigate now"

    st.markdown(f"""
    <div style='margin:4px 0 2px'>
        <span style='color:#8892b0;font-size:0.78rem'>Surface Soil Moisture</span>
    </div>
    <div class='sm-bar-wrap'>
        <div class='sm-bar-fill' style='width:{pct:.0f}%; background:{color};'>
            {sm_val:.3f} m³/m³ — {label}
        </div>
    </div>
    <div style='display:flex;justify-content:space-between;font-size:0.7rem;color:#607080;margin-top:3px'>
        <span>Critical: {crit}</span><span>Optimal: {opt}</span><span>FC: {fc}</span>
    </div>
    """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════
#  SIDEBAR — INPUTS
# ═══════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown('<div class="sidebar-title">🥔 Potato Smart Irrigation</div>', unsafe_allow_html=True)
    st.caption("LSTM Hybrid Model · FAO-56 Penman-Monteith")
    st.divider()

    st.markdown("##### 📅 Crop Dates")
    sowing_date  = st.date_input("Sowing Date",  value=None, key="sowing",
                                  help="Date when crop was sown/transplanted")
    current_date = st.date_input("Current Date", value=None, key="current",
                                  help="Today's date or target forecast date")

    st.divider()
    st.markdown("##### 📍 Location")
    latlon_str = st.text_input(
        "Latitude, Longitude",
        placeholder="e.g.  28.35, 79.41",
        help="Enter decimal degrees — comma-separated"
    )

    st.divider()
    st.markdown("##### ⚙️ Advanced Options")
    elev_m    = st.number_input("Elevation (m asl)", min_value=0, max_value=5000,
                                 value=250, step=10)
    t_base    = st.number_input("GDD Base Temp (°C)", min_value=0.0, max_value=15.0,
                                 value=7.0, step=0.5,
                                 help="Potato: 7 °C is standard")
    sm_init   = st.slider("Initial Soil Moisture (m³/m³)",
                           min_value=0.10, max_value=0.45, value=0.30, step=0.01,
                           help="Soil moisture at sowing / field capacity start")

    st.divider()
    run_btn = st.button("🚀  Run Analysis", type="primary", use_container_width=True)


# ═══════════════════════════════════════════════════════════
#  MAIN PANEL
# ═══════════════════════════════════════════════════════════
st.markdown("""
<h2 style='color:#e6f1ff;margin-bottom:4px'>🥔 Potato Smart Irrigation Dashboard</h2>
<p style='color:#607080;margin-top:0'>LSTM · Penman-Monteith ETo · Open-Meteo Weather · FAO-56 Irrigation</p>
""", unsafe_allow_html=True)
st.divider()

# ── Idle state ───────────────────────────────────────────
if not run_btn:
    st.info("👈  Fill in the Sowing Date, Current Date, and Location in the sidebar, then click **Run Analysis**.")
    st.markdown("""
    <div style='background:#1a1e2e;border-radius:12px;padding:22px;border:1px solid #2a2f4a;margin-top:20px'>
    <h4 style='color:#64ffda;margin-top:0'>How it works</h4>
    <ul style='color:#a8b2d8;line-height:1.8'>
      <li>Fetches live weather data from <b>Open-Meteo API</b> (free, no key needed)</li>
      <li>Calculates <b>DAS</b>, crop growth <b>stage</b>, and expected <b>NDVI</b></li>
      <li>Computes <b>GDD</b> and <b>ETo (Penman-Monteith FAO-56)</b></li>
      <li>Feeds 9 features into your <b>LSTM hybrid model</b></li>
      <li>Outputs surface soil moisture + <b>irrigation recommendation</b></li>
    </ul>
    </div>
    """, unsafe_allow_html=True)
    st.stop()


# ── Validate inputs ──────────────────────────────────────
errors_ui = []
if not sowing_date:
    errors_ui.append("Sowing Date is required.")
if not current_date:
    errors_ui.append("Current Date is required.")
if not latlon_str.strip():
    errors_ui.append("Latitude/Longitude is required.")

lat, lon = None, None
if latlon_str.strip():
    try:
        parts = latlon_str.split(",")
        lat, lon = float(parts[0].strip()), float(parts[1].strip())
        if not (-90 <= lat <= 90 and -180 <= lon <= 180):
            errors_ui.append("Lat/Lon out of valid range.")
    except Exception:
        errors_ui.append("Invalid Lat/Lon — enter as: 28.35, 79.41")

if sowing_date and current_date and current_date < sowing_date:
    errors_ui.append("Current Date must be ≥ Sowing Date.")

if errors_ui:
    for e in errors_ui:
        st.error(f"❌  {e}")
    st.stop()


# ── Load model ────────────────────────────────────────────
weights, scaler, load_errs = load_resources()
if load_errs:
    for e in load_errs:
        st.warning(f"⚠️  {e}")
if weights is None or scaler is None:
    st.error("Model or scaler could not be loaded. Check that **potato_weights.npz** and **data_scaler.gz** are in the same folder as app.py.")
    st.stop()


# ═══════════════════════════════════════════════════════════
#  CALCULATION ENGINE
# ═══════════════════════════════════════════════════════════
with st.spinner("🔄  Calculating... fetching weather and running model..."):

    # ── DAS & Stage ──────────────────────────────────────
    das         = (current_date - sowing_date).days
    stage_info  = get_stage(das)
    stage_name  = stage_info["name"]
    ndvi_today  = estimate_ndvi(das)

    # ── Weather window: sowing → current ─────────────────
    weather_start = sowing_date.strftime("%Y-%m-%d")
    weather_end   = current_date.strftime("%Y-%m-%d")

    try:
        wx = fetch_weather(lat, lon, weather_start, weather_end)
        daily = wx.get("daily", {})

        tmax_all  = daily.get("temperature_2m_max",            [25.0])
        tmin_all  = daily.get("temperature_2m_min",            [15.0])
        rh_max_all = daily.get("relative_humidity_2m_max",     [60.0])
        rh_min_all = daily.get("relative_humidity_2m_min",     [40.0])
        ws_all    = daily.get("wind_speed_10m_max",            [2.0])
        eto_api_all = daily.get("et0_fao_evapotranspiration",  [3.0])
        rs_all    = daily.get("shortwave_radiation_sum",       [15.0])
        rain_all  = daily.get("precipitation_sum",             [0.0])
        dates_all = daily.get("time",                          [current_date.strftime("%Y-%m-%d")])

        wx_error = None
    except Exception as exc:
        wx_error = str(exc)
        # Fallback defaults
        n = max(das + 1, 1)
        tmax_all  = [30.0] * n
        tmin_all  = [18.0] * n
        rh_max_all = [65.0] * n
        rh_min_all = [45.0] * n
        ws_all    = [2.5]  * n
        eto_api_all = [4.0] * n
        rs_all    = [18.0] * n
        rain_all  = [0.0]  * n
        dates_all = [(sowing_date + timedelta(days=i)).strftime("%Y-%m-%d") for i in range(n)]

    # ── Today's values (last day in series) ──────────────
    tmax_t   = tmax_all[-1]  if tmax_all   else 30.0
    tmin_t   = tmin_all[-1]  if tmin_all   else 18.0
    rh_t     = (rh_max_all[-1] + rh_min_all[-1]) / 2 if rh_max_all else 55.0
    ws_t     = ws_all[-1]   if ws_all     else 2.5
    rs_t     = rs_all[-1]   if rs_all     else 18.0
    eto_api_t = eto_api_all[-1] if eto_api_all else 4.0
    rain_t   = rain_all[-1] if rain_all   else 0.0
    tmean_t  = (tmax_t + tmin_t) / 2

    # ── GDD (cumulative, sowing→today) ───────────────────
    gdd_total, gdd_daily = calc_gdd(tmax_all, tmin_all, t_base)

    # ── ETo Penman-Monteith (today) ───────────────────────
    doy_t = current_date.timetuple().tm_yday
    eto_pm_t = eto_penman_monteith(
        tmax_t, tmin_t, rh_t, ws_t, rs_t, lat, doy_t, elev_m
    )

    # ── Kc · ETo = ETc ───────────────────────────────────
    kc_t  = stage_info["kc"]
    etc_t = round(kc_t * eto_pm_t, 3)

    # ── Soil potential proxy (kPa) ────────────────────────
    cum_rain = sum(rain_all) if rain_all else 0.0
    cum_etc  = sum(e * kc_t for e in eto_api_all) if eto_api_all else das * etc_t
    water_bal = cum_rain - cum_etc   # positive = surplus, negative = deficit

    # Map water balance to soil potential (empirical)
    def balance_to_pot(wb, min_pot, max_pot):
        """Clamp and map water balance to soil potential range."""
        norm = 1 / (1 + math.exp(wb / 20))   # sigmoid: deficit→1, surplus→0
        return round(min_pot + norm * (max_pot - min_pot), 3)

    soil_pot_surf = balance_to_pot(water_bal, -33.0, -12.0)
    soil_pot_rz   = balance_to_pot(water_bal * 0.6, -15.8, -4.4)

    # ── Model prediction ─────────────────────────────────
    model_ok  = False
    model_err = ""
    sm_pred   = 0.0
    try:
        sm_pred = run_prediction(
            weights, scaler,
            tmax=tmax_t, tmin=tmin_t, eto=eto_pm_t, rh=rh_t,
            ndvi=ndvi_today,
            soil_pot_surf=soil_pot_surf,
            soil_pot_rz=soil_pot_rz,
            tmean=tmean_t, das=float(das)
        )
        # Clamp to physically reasonable range
        sm_pred = max(0.05, min(0.50, sm_pred))
        model_ok = True
    except Exception as exc:
        model_err = str(exc)

    # ── Irrigation decision ───────────────────────────────
    irr = irrigation_decision(sm_pred, stage_name)

    # ── Root-zone water content (estimated from surface SM) ──
    rz_sm = round(min(0.50, sm_pred * 1.08), 3)


# ═══════════════════════════════════════════════════════════
#  DISPLAY RESULTS
# ═══════════════════════════════════════════════════════════

if wx_error:
    st.warning(f"⚠️ Weather API unavailable ({wx_error}). Using fallback defaults — connect to the internet for live data.")
if not model_ok:
    st.error(f"❌ Model prediction failed: {model_err}")
    st.stop()


# ── Row 1 — Key Metrics ───────────────────────────────────
st.markdown('<div class="section-header">📊 Crop Status Overview</div>', unsafe_allow_html=True)

c1, c2, c3, c4, c5, c6 = st.columns(6)
metric_card("Days After Sowing",  str(das), "DAS", c1)
metric_card("Crop Stage", f"{stage_info['icon']} {stage_name}", "", c2)
metric_card("Cumulative GDD", f"{gdd_total}", "°C·days", c3)
metric_card("ETo (PM FAO-56)",   f"{eto_pm_t:.2f}", "mm/day", c4)
metric_card("Expected NDVI",     f"{ndvi_today:.3f}", "unitless", c5)
metric_card("Crop Coeff. (Kc)",  f"{kc_t:.2f}", "stage avg", c6)


st.divider()

# ── Row 2 — Prediction & Recommendation ──────────────────
left, right = st.columns([3, 2], gap="large")

with left:
    st.markdown('<div class="section-header">💧 Soil Moisture Prediction</div>',
                unsafe_allow_html=True)

    soil_bar(sm_pred, irr["critical"], irr["optimal"], irr["fc"])

    st.markdown(f"""
    <div style='display:flex;gap:14px;margin-top:16px'>
      <div class='metric-card' style='flex:1'>
        <div class='label'>Surface SM (Predicted)</div>
        <div class='value' style='color:#64ffda'>{sm_pred:.3f}</div>
        <div class='unit'>m³/m³</div>
      </div>
      <div class='metric-card' style='flex:1'>
        <div class='label'>Root-Zone SM (Est.)</div>
        <div class='value' style='color:#b39ddb'>{rz_sm:.3f}</div>
        <div class='unit'>m³/m³</div>
      </div>
      <div class='metric-card' style='flex:1'>
        <div class='label'>Crop Water Use (ETc)</div>
        <div class='value' style='color:#ffd740'>{etc_t:.2f}</div>
        <div class='unit'>mm/day</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # 7-day ETo trend (if data available)
    if len(eto_api_all) >= 2:
        st.markdown('<div class="section-header" style="margin-top:20px">📈 ETo Trend (Season)</div>',
                    unsafe_allow_html=True)
        chart_n = min(len(dates_all), len(eto_api_all))
        df_eto = pd.DataFrame({
            "Date": pd.to_datetime(dates_all[:chart_n]),
            "ETo_API (mm/d)": eto_api_all[:chart_n],
        }).set_index("Date")
        st.line_chart(df_eto, use_container_width=True, height=180, color=["#64ffda"])


with right:
    st.markdown('<div class="section-header">🚿 Irrigation Decision</div>',
                unsafe_allow_html=True)

    status = irr["status"]
    if status == "optimal":
        st.markdown("""
        <div class='alert-ok'>
          <b>✅ No Irrigation Needed</b><br>
          Soil moisture is in the optimal range. Monitor daily.
        </div>""", unsafe_allow_html=True)
    elif status == "low":
        st.markdown(f"""
        <div class='alert-warning'>
          <b>⚠️ Irrigation Recommended Soon</b><br>
          SM is below optimal. Schedule irrigation within 1–2 days.
          <br><br>
          Apply ≈ <b>{irr['apply_mm']} mm</b> to restore to optimal.
        </div>""", unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class='alert-critical'>
          <b>🚨 Irrigate Immediately!</b><br>
          SM is below critical threshold — crop water stress likely.
          <br><br>
          Apply ≈ <b>{irr['apply_mm']} mm</b> urgently.
        </div>""", unsafe_allow_html=True)

    # Thresholds table
    st.markdown(f"""
    <div style='background:#1a1e2e;border-radius:10px;padding:14px;
                border:1px solid #2a2f4a;margin-top:12px'>
      <div style='color:#8892b0;font-size:0.75rem;margin-bottom:8px;text-transform:uppercase;letter-spacing:1px'>
        Thresholds for {stage_name}</div>
      <table style='width:100%;color:#ccd6f6;font-size:0.82rem;border-collapse:collapse'>
        <tr style='border-bottom:1px solid #2a2f4a'>
          <td style='padding:5px 0;color:#ff4b4b'>🔴 Critical</td>
          <td style='text-align:right'><b>{irr["critical"]}</b> m³/m³</td>
        </tr>
        <tr style='border-bottom:1px solid #2a2f4a'>
          <td style='padding:5px 0;color:#ffb700'>🟡 Optimal Low</td>
          <td style='text-align:right'><b>{irr["optimal"]}</b> m³/m³</td>
        </tr>
        <tr>
          <td style='padding:5px 0;color:#00d26a'>🟢 Field Capacity</td>
          <td style='text-align:right'><b>{irr["fc"]}</b> m³/m³</td>
        </tr>
      </table>
    </div>
    """, unsafe_allow_html=True)

    # Stage progress bar
    s_min, s_max = stage_info["das_range"]
    prog = 0 if s_max == s_min else min(100, int((das - s_min) / (s_max - s_min) * 100))
    st.markdown(f"""
    <div style='margin-top:14px'>
      <div style='color:#8892b0;font-size:0.75rem;text-transform:uppercase;
                  letter-spacing:1px;margin-bottom:4px'>Stage Progress</div>
      <div style='background:#1a1e2e;border-radius:6px;height:12px;
                  border:1px solid #2a2f4a;overflow:hidden'>
        <div style='height:100%;width:{prog}%;
                    background:linear-gradient(90deg,{stage_info["color"]},{stage_info["color"]}99)'>
        </div>
      </div>
      <div style='color:#607080;font-size:0.7rem;margin-top:2px'>
        DAS {das} / {s_max}  ({prog}% through {stage_name})
      </div>
    </div>
    """, unsafe_allow_html=True)


# ── Row 3 — Weather Summary ──────────────────────────────
st.divider()
st.markdown('<div class="section-header">🌡️ Today\'s Weather Parameters</div>',
            unsafe_allow_html=True)

w1, w2, w3, w4, w5, w6, w7 = st.columns(7)
metric_card("Tmax", f"{tmax_t:.1f}", "°C", w1)
metric_card("Tmin", f"{tmin_t:.1f}", "°C", w2)
metric_card("Tmean", f"{tmean_t:.1f}", "°C", w3)
metric_card("Rel. Humidity", f"{rh_t:.0f}", "%", w4)
metric_card("Wind Speed", f"{ws_t:.1f}", "km/h", w5)
metric_card("Solar Rad.", f"{rs_t:.1f}", "MJ/m²/d", w6)
metric_card("Rainfall", f"{rain_t:.1f}", "mm", w7)


# ── Row 4 — Feature vector (expandable) ─────────────────
with st.expander("🔬 Model Feature Vector (debug view)"):
    feat_df = pd.DataFrame({
        "Feature": [
            "Tmax (°C)", "Tmin (°C)", "ETo PM (mm/d)", "RH (%)",
            "NDVI", "Soil Pot. Surf (kPa)", "Soil Pot. RZ (kPa)",
            "Tmean (°C)", "DAS"
        ],
        "Raw Value": [
            tmax_t, tmin_t, eto_pm_t, rh_t,
            ndvi_today, soil_pot_surf, soil_pot_rz,
            tmean_t, float(das)
        ],
        "Scaled [0–1]": scaler.transform([[
            tmax_t, tmin_t, eto_pm_t, rh_t,
            ndvi_today, soil_pot_surf, soil_pot_rz,
            tmean_t, float(das)
        ]])[0].round(4).tolist(),
    })
    st.dataframe(feat_df, use_container_width=True, hide_index=True)

    st.caption("""
    ℹ️ Feature mapping is inferred from scaler min/max ranges.
    Features 5 & 6 (soil water potential, kPa) are derived from a running
    water balance (cumulative rain − ETc). Verify against your training
    feature set if predictions seem off.
    Model runs via pure NumPy (no TensorFlow) — compatible with any Python version.
    """)


# ── Footer ───────────────────────────────────────────────
st.divider()
st.markdown("""
<div style='text-align:center;color:#3d4a5c;font-size:0.75rem;padding:6px 0'>
  🥔 Potato Smart Irrigation Dashboard &nbsp;·&nbsp;
  Model: LSTM Hybrid (NumPy) &nbsp;·&nbsp;
  Weather: Open-Meteo API &nbsp;·&nbsp;
  ETo: FAO-56 Penman-Monteith
</div>
""", unsafe_allow_html=True)
