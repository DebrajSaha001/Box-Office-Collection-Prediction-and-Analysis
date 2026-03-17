# ============================================================
#   INDIAN BOX OFFICE PREDICTOR + TRACKER — app.py
#   Run with:  streamlit run app.py
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import json
import urllib.request
import urllib.parse
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

# ── Page config ──────────────────────────────────────────────
st.set_page_config(
    page_title="Box Office Predictor",
    page_icon="🎬",
    layout="centered",
)

# ── Global CSS ───────────────────────────────────────────────
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
        margin-bottom: 1.5rem;
    }
    .result-card {
        background: linear-gradient(135deg, #1a1a2e, #16213e);
        border-radius: 16px;
        padding: 2rem;
        text-align: center;
        margin-top: 1.5rem;
        border: 1px solid #2a2a4a;
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
    .tracker-card {
        background: linear-gradient(135deg, #1a1a2e, #16213e);
        border-radius: 16px;
        padding: 1.6rem;
        margin-top: 1.2rem;
        border: 1px solid #2a2a4a;
        display: flex;
        gap: 1.5rem;
        flex-wrap: wrap;
    }
    .tracker-poster img {
        width: 160px;
        border-radius: 10px;
        object-fit: cover;
        border: 2px solid #2a2a4a;
    }
    .tracker-poster .no-poster {
        width: 160px;
        height: 240px;
        border-radius: 10px;
        background: #1e1e3a;
        border: 2px dashed #2a2a4a;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 3rem;
    }
    .tracker-details { flex: 1; min-width: 220px; }
    .tracker-title {
        font-size: 1.4rem;
        font-weight: 700;
        background: linear-gradient(135deg, #7F77DD, #D85A30);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.6rem;
    }
    .tracker-info {
        font-size: 0.88rem;
        color: #bbb;
        margin: 5px 0;
        line-height: 1.5;
    }
    .tracker-info span { color: #eee; font-weight: 500; }
    .tracker-box {
        background: rgba(127, 119, 221, 0.10);
        border: 1px solid rgba(127, 119, 221, 0.22);
        border-radius: 10px;
        padding: 10px 14px;
        margin-top: 10px;
        font-size: 0.88rem;
        color: #ccc;
        line-height: 1.6;
    }
    .tracker-box-title {
        color: #7F77DD;
        font-weight: 600;
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-bottom: 4px;
    }
    .tracker-box-office {
        font-size: 1.3rem;
        font-weight: 700;
        color: #FFD700;
    }
    .stAlert { border-radius: 10px; }
</style>
""", unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────
st.markdown('<div class="main-title">🎬 Box Office Predictor</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">Predict future collections · Look up real box office data</div>',
    unsafe_allow_html=True
)

# ── Tabs ──────────────────────────────────────────────────────
tab1, tab2 = st.tabs(["🔮 Predict a Movie", "🔍 Look Up a Movie"])


# ════════════════════════════════════════════════════════════
#  TAB 1 — PREDICTOR
# ════════════════════════════════════════════════════════════
with tab1:

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
            "regressor"         : joblib.load(f"{models_dir}/regressor.pkl"),
            "classifier"        : joblib.load(f"{models_dir}/classifier.pkl"),
            "label_encoder"     : joblib.load(f"{models_dir}/label_encoder.pkl"),
            "label_language"    : joblib.load(f"{models_dir}/label_language.pkl"),
            "label_season"      : joblib.load(f"{models_dir}/label_season.pkl"),
            "star_power_map"    : joblib.load(f"{models_dir}/star_power_map.pkl"),
            "director_power_map": joblib.load(f"{models_dir}/director_power_map.pkl"),
            "meta"              : joblib.load(f"{models_dir}/meta.pkl"),
        }, []

    models, missing_files = load_models()

    if models is None:
        st.error(
            "⚠️ Trained models not found!\n\n"
            "Please run `python main.py` first to train and save the models.\n\n"
            f"Missing: {missing_files}"
        )
    else:
        m            = models
        meta         = m["meta"]
        star_map     = m["star_power_map"]
        dir_map      = m["director_power_map"]
        lang_enc     = m["label_language"]
        sea_enc      = m["label_season"]
        known_stars  = sorted(star_map.index.tolist())
        known_dirs   = sorted(dir_map.index.tolist())

        st.markdown("### 🎥 Enter Movie Details")
        col1, col2 = st.columns(2)

        with col1:
            movie_name  = st.text_input("Movie Name", placeholder="e.g. Tiger 4")
            star_name   = st.text_input(
                "Lead Star", placeholder="e.g. Salman Khan",
                help=f"Known stars: {', '.join(known_stars[:8])}..."
            )
            budget      = st.number_input("Budget (₹ Crore)", min_value=1.0, max_value=2000.0, value=100.0, step=5.0)
            opening_day = st.number_input(
                "Expected Opening Day (₹ Crore)",
                min_value=0.1, max_value=500.0, value=20.0, step=1.0,
                help="First day box office collection estimate"
            )

        with col2:
            director_name = st.text_input(
                "Director", placeholder="e.g. Rohit Shetty",
                help=f"Known directors: {', '.join(known_dirs[:8])}..."
            )
            language = st.selectbox(
                "Language",
                options=sorted(lang_enc.classes_.tolist()),
                index=list(lang_enc.classes_).index("Hindi") if "Hindi" in lang_enc.classes_ else 0
            )
            screens = st.number_input(
                "Worldwide Screens", min_value=100, max_value=15000,
                value=4000, step=100, help="Total screens worldwide"
            )
            release_month = st.selectbox(
                "Release Month",
                options=list(range(1, 13)),
                format_func=lambda mo: [
                    "", "January", "February", "March", "April", "May", "June",
                    "July", "August", "September", "October", "November", "December"
                ][mo],
                index=9
            )

        col3, col4 = st.columns(2)
        with col3:
            is_franchise = st.checkbox("🔁 Sequel / Franchise film?")
        with col4:
            release_year = st.selectbox("Release Year", options=[2024, 2025, 2026], index=1)

        st.markdown("---")
        predict_btn = st.button("🔮 Predict Box Office", use_container_width=True, type="primary")

        if predict_btn:
            if not movie_name.strip():
                st.warning("Please enter a movie name.")
            else:
                # Star power
                star_input = star_name.strip().title()
                star_match = get_close_matches(star_input, star_map.index.tolist(), n=1, cutoff=0.6)
                star_val   = float(star_map[star_match[0]]) if star_match else float(meta["global_star_mean"])
                star_note  = f"Matched: **{star_match[0]}**" if star_match else "Not in dataset — using average"

                # Director power
                dir_input = director_name.strip().title()
                dir_match = get_close_matches(dir_input, dir_map.index.tolist(), n=1, cutoff=0.6)
                dir_val   = float(dir_map[dir_match[0]]) if dir_match else float(meta["global_director_mean"])
                dir_note  = f"Matched: **{dir_match[0]}**" if dir_match else "Not in dataset — using average"

                # Encode
                lang_val   = int(lang_enc.transform([language])[0]) if language in lang_enc.classes_ else 0
                season_str = get_season(release_month)
                season_val = int(sea_enc.transform([season_str])[0]) if season_str in sea_enc.classes_ else 0

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

                pred_log       = m["regressor"].predict(future)[0]
                pred_worldwide = round(float(np.expm1(pred_log)), 2)
                clf_id         = m["classifier"].predict(future)[0]
                clf_verdict    = m["label_encoder"].inverse_transform([clf_id])[0]

                profit       = pred_worldwide - float(budget)
                profit_pct   = (profit / float(budget)) * 100
                pred_verdict = verdict_from_profit(profit_pct)
                roi_label    = f"{'▲' if profit >= 0 else '▼'} ₹{abs(round(profit,1))} Cr  ({round(profit_pct,1)}%)"

                conf_label = "🟢 High confidence — both models agree" if clf_verdict == pred_verdict \
                    else f"🟡 Medium confidence — classifier suggested {clf_verdict}"
                conf_color = "#1D9E75" if clf_verdict == pred_verdict else "#EF9F27"

                v_color, v_bg = verdict_color(pred_verdict)
                emoji = verdict_emoji(pred_verdict)

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
                    <div style="color:{conf_color}; font-size:0.85rem; margin-top:0.5rem;">
                        {conf_label}
                    </div>
                </div>
                """, unsafe_allow_html=True)

                st.markdown("#### 📊 Breakdown")
                mc1, mc2, mc3, mc4 = st.columns(4)
                mc1.metric("Budget",      f"₹{budget} Cr")
                mc2.metric("Opening Day", f"₹{opening_day} Cr")
                mc3.metric("Worldwide",   f"₹{pred_worldwide} Cr",
                           delta=f"₹{round(profit,1)} Cr profit" if profit >= 0 else f"₹{round(abs(profit),1)} Cr loss")
                mc4.metric("ROI",         f"{round(profit_pct,1)}%")

                with st.expander("ℹ️ How this prediction was made"):
                    st.markdown(f"**Star:** {star_note}")
                    st.markdown(f"**Director:** {dir_note}")
                    st.markdown(f"**Season:** {season_str} (Month {release_month})")
                    st.markdown(f"**Opening/Screen:** ₹{round(opening_day/screens*100000):,} per screen")
                    st.markdown("""
---
Two XGBoost models trained on 500+ Indian films (2021–2024):
- **Regressor** → predicts worldwide collection in ₹ Crore
- **Classifier** → predicts verdict category as a cross-check
                    """)


# ════════════════════════════════════════════════════════════
#  TAB 2 — OMDB MOVIE TRACKER
# ════════════════════════════════════════════════════════════
with tab2:

    st.markdown("### 🔍 Look Up Any Movie")
    st.markdown(
        "<p style='color:#888; font-size:0.88rem; margin-bottom:1rem;'>"
        "Search any film to see its real box office, cast, ratings and plot via OMDB.</p>",
        unsafe_allow_html=True
    )

    with st.expander("⚙️ OMDB API Key (optional)", expanded=False):
        api_key = st.text_input(
            "Your OMDB API key",
            type="password",
            placeholder="Get a free key at omdbapi.com",
            help="Free key at https://www.omdbapi.com/apikey.aspx"
        )
        st.caption("Free key at [omdbapi.com/apikey.aspx](https://www.omdbapi.com/apikey.aspx)")

    if not api_key:
        api_key = "trilogy"

    s_col, b_col = st.columns([5, 1])
    with s_col:
        movie_query = st.text_input(
            "search", placeholder="e.g. Pathaan, Inception, RRR...",
            label_visibility="collapsed"
        )
    with b_col:
        search_btn = st.button("Search", use_container_width=True, type="primary")

    if search_btn and movie_query.strip():
        encoded = urllib.parse.quote(movie_query.strip())
        url     = f"https://www.omdbapi.com/?t={encoded}&apikey={api_key}&plot=full"

        try:
            with urllib.request.urlopen(url, timeout=8) as resp:
                data = json.loads(resp.read().decode())

            if data.get("Response") == "False":
                st.error(f"Movie not found: **{movie_query}**. Try a different spelling or the English title.")
            else:
                title      = data.get("Title",      "N/A")
                year       = data.get("Year",        "N/A")
                rating     = data.get("imdbRating",  "N/A")
                genre      = data.get("Genre",       "N/A")
                director   = data.get("Director",    "N/A")
                actors     = data.get("Actors",      "N/A")
                plot       = data.get("Plot",        "N/A")
                box_office = data.get("BoxOffice",   "Not Available")
                runtime    = data.get("Runtime",     "N/A")
                language   = data.get("Language",    "N/A")
                country    = data.get("Country",     "N/A")
                awards     = data.get("Awards",      "N/A")
                poster     = data.get("Poster",      "N/A")

                poster_html = f'<img src="{poster}" alt="{title}"/>' \
                    if poster and poster != "N/A" \
                    else '<div class="no-poster">🎬</div>'

                try:
                    r_val   = float(rating)
                    r_color = "#4CAF50" if r_val >= 7 else "#ff9800" if r_val >= 5 else "#FF5252"
                    r_bg    = "rgba(76,175,80,0.12)" if r_val >= 7 else "rgba(255,152,0,0.12)" if r_val >= 5 else "rgba(255,82,82,0.12)"
                except:
                    r_color, r_bg = "#888", "rgba(128,128,128,0.12)"

                st.markdown(f"""
                <div class="tracker-card">
                    <div class="tracker-poster">{poster_html}</div>
                    <div class="tracker-details">
                        <div class="tracker-title">{title} ({year})</div>
                        <div class="tracker-info">⏱ <span>{runtime}</span> &nbsp;·&nbsp; 🌐 <span>{language}</span> &nbsp;·&nbsp; 🏳️ <span>{country}</span></div>
                        <div class="tracker-info">🎭 <span>{genre}</span></div>
                        <div class="tracker-info">🎬 Director: <span>{director}</span></div>
                        <div class="tracker-info">🌟 Cast: <span>{actors}</span></div>
                        <div class="tracker-info">
                            ⭐ IMDb:&nbsp;
                            <span style="display:inline-block; background:{r_bg}; border:1px solid {r_color}44;
                                         color:{r_color}; font-weight:700; padding:2px 10px;
                                         border-radius:20px; font-size:0.88rem;">
                                {rating} / 10
                            </span>
                        </div>
                        <div class="tracker-box">
                            <div class="tracker-box-title">💰 Box Office Collection</div>
                            <div class="tracker-box-office">{box_office}</div>
                        </div>
                        <div class="tracker-box">
                            <div class="tracker-box-title">🏆 Awards</div>
                            {awards}
                        </div>
                    </div>
                </div>
                <div class="tracker-box" style="margin-top:12px;">
                    <div class="tracker-box-title">📖 Plot</div>
                    {plot}
                </div>
                """, unsafe_allow_html=True)

        except Exception as e:
            st.error(f"Could not fetch data. Check your connection or API key.\n\n`{e}`")

    elif search_btn:
        st.warning("Please enter a movie name to search.")

# ── Footer ────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<div style='text-align:center; color:#555; font-size:0.8rem;'>"
    "Trained on Indian box office data 2021–2024 · Movie data via OMDB API · For educational purposes"
    "</div>",
    unsafe_allow_html=True
)