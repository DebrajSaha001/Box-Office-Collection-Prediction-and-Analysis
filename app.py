# ============================================================
#   INDIAN BOX OFFICE PREDICTOR — app.py  (Streamlit UI)
#   Run with:  streamlit run app.py
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from difflib import get_close_matches

# ── Helper functions ─────────────────────────────────────────
def verdict_from_profit(p):
    if   p < -50 : return "DISASTER"
    elif p < 0   : return "FLOP"
    elif p < 50  : return "AVERAGE"
    elif p < 100 : return "HIT"
    elif p < 200 : return "SUPER HIT"
    elif p < 400 : return "BLOCKBUSTER"
    else          : return "ALL TIME BLOCKBUSTER"

# ── Page config ──────────────────────────────────────────────
st.set_page_config(
    page_title="Box Office Predictor",
    page_icon="🎬",
    layout="centered",
)

# ── Custom CSS ───────────────────────────────────────────────
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        text-align: center;
        background: linear-gradient(135deg, #7F77DD, #D85A30);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .subtitle {
        text-align: center;
        color: #888;
        font-size: 0.95rem;
        margin-bottom: 2rem;
    }
    .result-card {
        background: linear-gradient(135deg, #1a1a2e, #16213e);
        border-radius: 16px;
        padding: 2rem;
        text-align: center;
        margin-top: 1.5rem;
        border: 1px solid #333;
    }
    .result-crore {
        font-size: 3rem;
        font-weight: 800;
        color: #FFD700;
    }
    .result-label {
        font-size: 0.85rem;
        color: #aaa;
        text-transform: uppercase;
        letter-spacing: 0.1em;
    }
    .verdict-badge {
        display: inline-block;
        padding: 0.4rem 1.4rem;
        border-radius: 30px;
        font-size: 1.1rem;
        font-weight: 700;
        margin-top: 0.8rem;
    }
    .stAlert { border-radius: 10px; }
</style>
""", unsafe_allow_html=True)

# ── Load models ──────────────────────────────────────────────
@st.cache_resource
def load_models():
    models_dir = "models"
    required = [
        "regressor.pkl", "classifier.pkl", "label_encoder.pkl",
        "label_language.pkl", "label_season.pkl",
        "star_power_map.pkl", "director_power_map.pkl", "meta.pkl"
    ]
    missing = [f for f in required if not os.path.exists(os.path.join(models_dir, f))]
    if missing:
        return None, missing

    return {
        "regressor"       : joblib.load(f"{models_dir}/regressor.pkl"),
        "classifier"      : joblib.load(f"{models_dir}/classifier.pkl"),
        "label_encoder"   : joblib.load(f"{models_dir}/label_encoder.pkl"),
        "label_language"  : joblib.load(f"{models_dir}/label_language.pkl"),
        "label_season"    : joblib.load(f"{models_dir}/label_season.pkl"),
        "star_power_map"  : joblib.load(f"{models_dir}/star_power_map.pkl"),
        "director_power_map": joblib.load(f"{models_dir}/director_power_map.pkl"),
        "meta"            : joblib.load(f"{models_dir}/meta.pkl"),
    }, []

def get_season(month):
    if   month in [1, 10, 11, 12]: return "Holiday"
    elif month in [3, 4, 5]:       return "Summer"
    elif month in [6, 7]:          return "Monsoon"
    else:                           return "Normal"

def verdict_color(verdict):
    colors = {
        "ALL TIME BLOCKBUSTER" : ("#FFD700", "#1a1a00"),
        "BLOCKBUSTER"          : ("#FF6B35", "#1a0d00"),
        "SUPER HIT"            : ("#4CAF50", "#001a00"),
        "HIT"                  : ("#2196F3", "#001020"),
        "AVERAGE"              : ("#9E9E9E", "#111111"),
        "FLOP"                 : ("#FF5252", "#1a0000"),
        "DISASTER"             : ("#B71C1C", "#0d0000"),
    }
    return colors.get(verdict, ("#9E9E9E", "#111111"))

def verdict_emoji(verdict):
    emojis = {
        "ALL TIME BLOCKBUSTER" : "🏆",
        "BLOCKBUSTER"          : "🔥",
        "SUPER HIT"            : "⭐",
        "HIT"                  : "✅",
        "AVERAGE"              : "➡️",
        "FLOP"                 : "📉",
        "DISASTER"             : "💥",
    }
    return emojis.get(verdict, "🎬")

# ── Header ────────────────────────────────────────────────────
st.markdown('<div class="main-title">🎬 Box Office Predictor</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Predict lifetime worldwide collection & verdict for Indian films</div>', unsafe_allow_html=True)

# ── Load models ───────────────────────────────────────────────
models, missing = load_models()

if models is None:
    st.error(
        "⚠️ Trained models not found!\n\n"
        "Please run `python main.py` first to train and save the models.\n\n"
        f"Missing files: {missing}"
    )
    st.stop()

m          = models
meta       = m["meta"]
star_map   = m["star_power_map"]
dir_map    = m["director_power_map"]
lang_enc   = m["label_language"]
sea_enc    = m["label_season"]

# Known stars and directors for autocomplete hints
known_stars     = sorted(star_map.index.tolist())
known_directors = sorted(dir_map.index.tolist())

# ── Input form ────────────────────────────────────────────────
st.markdown("### 🎥 Movie Details")

col1, col2 = st.columns(2)

with col1:
    movie_name = st.text_input("Movie Name", placeholder="e.g. Tiger 4")
    star_name  = st.text_input(
        "Lead Star",
        placeholder="e.g. Salman Khan",
        help=f"Known stars in dataset: {', '.join(known_stars[:8])}..."
    )
    budget = st.number_input(
        "Budget (₹ Crore)", min_value=1.0, max_value=2000.0,
        value=100.0, step=5.0
    )
    opening_day = st.number_input(
        "Expected Opening Day (₹ Crore)",
        min_value=0.1, max_value=500.0, value=20.0, step=1.0,
        help="First day box office collection estimate"
    )

with col2:
    director_name = st.text_input(
        "Director",
        placeholder="e.g. Rohit Shetty",
        help=f"Known directors: {', '.join(known_directors[:8])}..."
    )
    language = st.selectbox(
        "Language",
        options=sorted(lang_enc.classes_.tolist()),
        index=list(lang_enc.classes_).index("Hindi")
            if "Hindi" in lang_enc.classes_ else 0
    )
    screens = st.number_input(
        "Worldwide Screens", min_value=100, max_value=15000,
        value=4000, step=100,
        help="Total number of screens worldwide"
    )
    release_month = st.selectbox(
        "Release Month",
        options=list(range(1, 13)),
        format_func=lambda m: [
            "", "January", "February", "March", "April",
            "May", "June", "July", "August", "September",
            "October", "November", "December"
        ][m],
        index=9
    )

col3, col4 = st.columns(2)
with col3:
    is_franchise = st.checkbox(
        "🔁 Sequel / Franchise film?",
        help="Check if this is part 2, 3, or an established franchise"
    )
with col4:
    release_year = st.selectbox(
        "Release Year",
        options=[2024, 2025, 2026],
        index=1
    )

# ── Predict button ────────────────────────────────────────────
st.markdown("---")
predict_btn = st.button("🔮 Predict Box Office", use_container_width=True, type="primary")

if predict_btn:
    if not movie_name.strip():
        st.warning("Please enter a movie name.")
        st.stop()

    # ── Resolve star power ────────────────────────────────────
    star_input = star_name.strip().title()
    star_match = get_close_matches(star_input, star_map.index.tolist(), n=1, cutoff=0.6)
    if star_match:
        star_val      = float(star_map[star_match[0]])
        star_note     = f"Matched: **{star_match[0]}**"
    else:
        star_val      = float(meta["global_star_mean"])
        star_note     = "Star not in dataset — using average star power"

    # ── Resolve director power ────────────────────────────────
    dir_input = director_name.strip().title()
    dir_match = get_close_matches(dir_input, dir_map.index.tolist(), n=1, cutoff=0.6)
    if dir_match:
        dir_val   = float(dir_map[dir_match[0]])
        dir_note  = f"Matched: **{dir_match[0]}**"
    else:
        dir_val   = float(meta["global_director_mean"])
        dir_note  = "Director not in dataset — using average director power"

    # ── Encode language & season ──────────────────────────────
    if language in lang_enc.classes_:
        lang_val = int(lang_enc.transform([language])[0])
    else:
        lang_val = 0

    season_str = get_season(release_month)
    if season_str in sea_enc.classes_:
        season_val = int(sea_enc.transform([season_str])[0])
    else:
        season_val = 0

    # ── Build feature row ─────────────────────────────────────
    future = pd.DataFrame([{
        "Budget"             : float(budget),
        "Opening_Day"        : float(opening_day),
        "Screens"            : float(screens),
        "Language_Label"     : lang_val,
        "Season_Label"       : season_val,
        "Franchise"          : int(is_franchise),
        "Opening_to_Budget"  : float(opening_day) / float(budget),
        "Screens_to_Budget"  : float(screens) / float(budget),
        "Opening_per_Screen" : float(opening_day) / float(screens) if screens > 0 else 0,
        "Release_Year"       : int(release_year),
        "Star_Power"         : star_val,
        "Director_Power"     : dir_val,
    }])

    # ── Predict ───────────────────────────────────────────────
    pred_log       = m["regressor"].predict(future)[0]
    pred_worldwide = round(float(np.expm1(pred_log)), 2)
    clf_verdict_id = m["classifier"].predict(future)[0]
    clf_verdict    = m["label_encoder"].inverse_transform([clf_verdict_id])[0]

    profit     = pred_worldwide - float(budget)
    profit_pct = (profit / float(budget)) * 100

    # Primary verdict always from calculated profit %
    pred_verdict = verdict_from_profit(profit_pct)

    # Confidence signal
    if clf_verdict == pred_verdict:
        confidence_label = "🟢 High confidence — both models agree"
        confidence_color = "#1D9E75"
    else:
        confidence_label = f"🟡 Medium confidence — classifier suggested {clf_verdict}"
        confidence_color = "#EF9F27"

    roi_label  = f"{'▲' if profit >= 0 else '▼'} ₹{abs(round(profit,1))} Cr  ({round(profit_pct,1)}%)"

    v_color, v_bg = verdict_color(pred_verdict)
    emoji = verdict_emoji(pred_verdict)

    # ── Display result ────────────────────────────────────────
    st.markdown(f"""
    <div class="result-card">
        <div class="result-label">Predicted Lifetime Worldwide Collection</div>
        <div class="result-crore">₹{pred_worldwide:,.1f} Cr</div>
        <div style="color:#ccc; margin-top:0.4rem; font-size:0.95rem;">{roi_label}</div>
        <div class="verdict-badge" style="background:{v_color}; color:{v_bg}; margin-top:1rem;">
            {emoji} {pred_verdict}
        </div>
        <div style="color:#666; font-size:0.8rem; margin-top:1rem;">
            Release season: {season_str} · Franchise: {'Yes' if is_franchise else 'No'}
        </div>
        <div style="color:{confidence_color}; font-size:0.85rem; margin-top:0.6rem;">
            {confidence_label}
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Insight cards ─────────────────────────────────────────
    st.markdown("#### 📊 Breakdown")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Budget",        f"₹{budget} Cr")
    m2.metric("Opening Day",   f"₹{opening_day} Cr")
    m3.metric("Worldwide",     f"₹{pred_worldwide} Cr",
              delta=f"₹{round(profit,1)} Cr profit" if profit >= 0 else f"₹{round(abs(profit),1)} Cr loss")
    m4.metric("ROI",           f"{round(profit_pct,1)}%")

    # ── Match notes ───────────────────────────────────────────
    with st.expander("ℹ️ How this prediction was made"):
        st.markdown(f"**Star:** {star_note}")
        st.markdown(f"**Director:** {dir_note}")
        st.markdown(f"**Season:** {season_str} (Month {release_month})")
        st.markdown(f"**Opening/Screen:** ₹{round(opening_day/screens*100000):,} per screen")
        st.markdown("""
---
This prediction uses two XGBoost models trained on 500+ Indian films (2021–2024):
- A **regressor** predicts the log-transformed worldwide collection
- A **classifier** predicts the verdict category directly

Both models consider budget, opening day, screens, star power, director track record, language, season, and franchise status.
        """)

# ── Footer ─────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<div style='text-align:center; color:#666; font-size:0.8rem;'>"
    "Trained on Indian box office data 2021–2024 · For educational purposes"
    "</div>",
    unsafe_allow_html=True
)