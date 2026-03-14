# ============================================================
#   INDIAN BOX OFFICE PREDICTOR — main.py
#   Trains XGBoost regressor + classifier, saves .pkl models
# ============================================================

import pandas as pd
import numpy as np
import glob
import os
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from difflib import get_close_matches

sns.set_style("whitegrid")

from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, r2_score, accuracy_score, confusion_matrix

from xgboost import XGBRegressor, XGBClassifier

# ─────────────────────────────────────────────
#  STEP 1 — LOAD & STANDARDISE ALL CSV FILES
# ─────────────────────────────────────────────

print("\n" + "="*55)
print("  INDIAN BOX OFFICE PREDICTOR")
print("="*55)
print("\n[1/7] Loading datasets...")

# Column name map — every possible variant → standard name
COLUMN_MAP = {
    # Movie name
    "movie name"        : "Movie_Name",
    "movie"             : "Movie_Name",
    "movies"            : "Movie_Name",

    # Star
    "stars_featuring"   : "Star_Featuring",
    "star_featuring"    : "Star_Featuring",
    "star_power"        : "Star_Featuring",   # 2023 mislabelled this

    # Director
    "director"          : "Director",

    # Language
    "language"          : "Language",

    # Date
    "released_date"     : "Released_Date",
    "released date"     : "Released_Date",

    # Financials
    "budget"            : "Budget",
    "worldwide collection" : "Worldwide",
    "worldwide"         : "Worldwide",
    "india gross collection" : "India_Gross",
    "india gross"       : "India_Gross",
    "india_gross"       : "India_Gross",
    "overseas collection"  : "Overseas",
    "overseas"          : "Overseas",
    "india_hindi_net"   : "India_Hindi_Net",

    # Opening / screens (already standardised)
    "opening_day"       : "Opening_Day",
    "screens"           : "Screens",

    # Verdict
    "verdict"           : "Verdict",

    # Extra cols we don't need (kept but not used)
    "profit"            : "Profit_Raw",
    "profit in percentage" : "Profit_Pct_Raw",
    "rating"            : "Rating",
}

def load_and_standardise(path):
    df = pd.read_csv(path)
    # Strip whitespace from column names, lowercase for mapping
    df.columns = df.columns.str.strip()
    rename = {}
    for col in df.columns:
        std = COLUMN_MAP.get(col.lower().strip())
        if std:
            rename[col] = std
    df = df.rename(columns=rename)
    return df

files = glob.glob("data/*.csv")
if not files:
    raise FileNotFoundError(
        "No CSV files found in data/ folder.\n"
        "Please put your updated CSVs inside a 'data/' subfolder."
    )

dataframes = []
for f in sorted(files):
    try:
        df_temp = load_and_standardise(f)
        dataframes.append(df_temp)
        print(f"  ✓ Loaded {f}  ({df_temp.shape[0]} rows)")
    except Exception as e:
        print(f"  ✗ Failed {f}: {e}")

df = pd.concat(dataframes, ignore_index=True)
print(f"\n  Total movies after merge: {df.shape[0]}")

# ─────────────────────────────────────────────
#  STEP 2 — ENSURE ALL EXPECTED COLUMNS EXIST
# ─────────────────────────────────────────────

expected = [
    "Movie_Name", "Language", "Director", "Star_Featuring",
    "Budget", "Worldwide", "India_Gross", "Overseas",
    "Opening_Day", "Screens", "Released_Date", "Verdict"
]
for col in expected:
    if col not in df.columns:
        df[col] = np.nan

# ─────────────────────────────────────────────
#  STEP 3 — CLEAN & CONVERT
# ─────────────────────────────────────────────

print("\n[2/7] Cleaning data...")

# Keep only first star name when multiple are listed
df["Star_Featuring"] = (
    df["Star_Featuring"].astype(str)
    .str.split(";").str[0]
    .str.strip()
    .replace("nan", "Unknown")
)
df["Star_Featuring"] = df["Star_Featuring"].fillna("Unknown")
df["Director"]       = df["Director"].fillna("Unknown")
df["Language"]       = df["Language"].fillna("Unknown")
df["Movie_Name"]     = df["Movie_Name"].fillna("Unknown")

# Convert all financial columns to numeric
numeric_cols = ["Budget", "Worldwide", "India_Gross", "Overseas", "Opening_Day", "Screens"]
for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# Drop rows with no Worldwide or Budget — can't train without these
df = df.dropna(subset=["Worldwide", "Budget"])
df = df[df["Budget"] > 0]
df = df[df["Worldwide"] > 0]

# Fill remaining numeric nulls with median
df["Opening_Day"] = df["Opening_Day"].fillna(df["Opening_Day"].median())
df["Screens"]     = df["Screens"].fillna(df["Screens"].median())
df["India_Gross"] = df["India_Gross"].fillna(df["India_Gross"].median())
df["Overseas"]    = df["Overseas"].fillna(df["Overseas"].median())

df.replace([np.inf, -np.inf], 0, inplace=True)
df = df.reset_index(drop=True)

print(f"  Clean rows ready: {df.shape[0]}")

# ─────────────────────────────────────────────
#  STEP 4 — STANDARDISE VERDICT LABELS
# ─────────────────────────────────────────────

VERDICT_MAP = {
    "all time blockbuster" : "ALL TIME BLOCKBUSTER",
    "blockbuster"          : "BLOCKBUSTER",
    "super hit"            : "SUPER HIT",
    "above average"        : "HIT",          # merge into HIT
    "hit"                  : "HIT",
    "average"              : "AVERAGE",
    "below average"        : "FLOP",         # merge into FLOP
    "flop"                 : "FLOP",
    "disaster"             : "DISASTER",
}

df["Verdict_Clean"] = (
    df["Verdict"].astype(str)
    .str.strip().str.lower()
    .map(VERDICT_MAP)
    .fillna("AVERAGE")
)

# ─────────────────────────────────────────────
#  STEP 5 — FEATURE ENGINEERING
# ─────────────────────────────────────────────

print("\n[3/7] Engineering features...")

# Log transform target (handles skewed crore values)
df["Log_Worldwide"] = np.log1p(df["Worldwide"])

# Profit
df["Profit"]            = df["Worldwide"] - df["Budget"]
df["Profit_Percentage"] = (df["Profit"] / df["Budget"]) * 100

# Ratio features
df["Opening_to_Budget"]  = df["Opening_Day"] / df["Budget"]
df["Screens_to_Budget"]  = df["Screens"]     / df["Budget"]
df["Opening_per_Screen"] = df["Opening_Day"] / df["Screens"].replace(0, np.nan)
df.replace([np.inf, -np.inf], 0, inplace=True)
df["Opening_per_Screen"] = df["Opening_per_Screen"].fillna(0)

# Release date features
df["Released_Date"] = pd.to_datetime(df["Released_Date"], dayfirst=True, errors="coerce")
df["Release_Month"] = df["Released_Date"].dt.month.fillna(6).astype(int)
df["Release_Year"]  = df["Released_Date"].dt.year.fillna(2022).astype(int)

def get_season(month):
    if   month in [1, 10, 11, 12]: return "Holiday"
    elif month in [3, 4, 5]:       return "Summer"
    elif month in [6, 7]:          return "Monsoon"
    else:                           return "Normal"

df["Season"] = df["Release_Month"].apply(get_season)

# Franchise flag — sequel/part keywords in title
df["Franchise"] = df["Movie_Name"].astype(str).str.contains(
    r"\b(2|3|4|II|III|IV|Part|Chapter|Return|Reloaded|Revolution|Legacy)\b",
    case=False, regex=True
).astype(int)

# Encode Language and Season
label_language = LabelEncoder()
label_season   = LabelEncoder()
df["Language_Label"] = label_language.fit_transform(df["Language"].astype(str))
df["Season_Label"]   = label_season.fit_transform(df["Season"])

print(f"  Languages found : {list(label_language.classes_)}")
print(f"  Seasons found   : {list(label_season.classes_)}")
print(f"\n  Verdict distribution:")
print(df["Verdict_Clean"].value_counts().to_string())

# ─────────────────────────────────────────────
#  STEP 6 — TRAIN / TEST SPLIT
# ─────────────────────────────────────────────

print("\n[4/7] Splitting data...")

FEATURES = [
    "Budget",
    "Opening_Day",
    "Screens",
    "Language_Label",
    "Season_Label",
    "Franchise",
    "Opening_to_Budget",
    "Screens_to_Budget",
    "Opening_per_Screen",
    "Release_Year",
]

X = df[FEATURES]
Y_reg = df["Log_Worldwide"]

X_train, X_test, Y_train_reg, Y_test_reg = train_test_split(
    X, Y_reg, test_size=0.2, random_state=42
)

# ── Star Power & Director Power computed on TRAIN only (no data leakage) ──
train_idx = X_train.index
test_idx  = X_test.index

train_df = df.loc[train_idx].copy()
test_df  = df.loc[test_idx].copy()

star_power_map     = train_df.groupby("Star_Featuring")["Worldwide"].median()
director_power_map = train_df.groupby("Director")["Worldwide"].mean()

global_star_mean     = star_power_map.mean()
global_director_mean = director_power_map.mean()

train_df["Star_Power"]     = train_df["Star_Featuring"].map(star_power_map).fillna(global_star_mean)
train_df["Director_Power"] = train_df["Director"].map(director_power_map).fillna(global_director_mean)

test_df["Star_Power"]      = test_df["Star_Featuring"].map(star_power_map).fillna(global_star_mean)
test_df["Director_Power"]  = test_df["Director"].map(director_power_map).fillna(global_director_mean)

FEATURES_FULL = FEATURES + ["Star_Power", "Director_Power"]

X_train_full = train_df[FEATURES_FULL]
X_test_full  = test_df[FEATURES_FULL]

Y_train_reg  = train_df["Log_Worldwide"]
Y_test_reg   = test_df["Log_Worldwide"]

print(f"  Train: {len(X_train_full)} rows | Test: {len(X_test_full)} rows")

# ─────────────────────────────────────────────
#  STEP 7A — XGBOOST REGRESSOR
# ─────────────────────────────────────────────

print("\n[5/7] Training XGBoost Regressor...")

regressor = XGBRegressor(
    n_estimators  = 1000,
    learning_rate = 0.02,
    max_depth     = 5,
    subsample     = 0.8,
    colsample_bytree = 0.8,
    gamma         = 1,
    min_child_weight = 3,
    random_state  = 42,
    verbosity     = 0,
)
regressor.fit(X_train_full, Y_train_reg)

Y_pred_reg = regressor.predict(X_test_full)

mae      = mean_absolute_error(Y_test_reg, Y_pred_reg)
r2       = r2_score(Y_test_reg, Y_pred_reg)
mae_cr   = mean_absolute_error(
    np.expm1(Y_test_reg), np.expm1(Y_pred_reg)
)

print(f"\n  ── Regression Results ──")
print(f"  R2 Score       : {r2:.4f}  (target > 0.75)")
print(f"  MAE (log)      : {mae:.4f}")
print(f"  MAE (Crores)   : ₹{mae_cr:.1f} Cr")

# Cross-validation for honest estimate
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# For CV we need star/director on full df
df["Star_Power_Full"]     = df["Star_Featuring"].map(star_power_map).fillna(global_star_mean)
df["Director_Power_Full"] = df["Director"].map(director_power_map).fillna(global_director_mean)
FEATURES_CV = FEATURES + ["Star_Power_Full", "Director_Power_Full"]

cv_scores = cross_val_score(
    XGBRegressor(n_estimators=500, learning_rate=0.02, max_depth=5,
                 random_state=42, verbosity=0),
    df[FEATURES_CV], df["Log_Worldwide"],
    cv=kf, scoring="r2"
)
print(f"  5-Fold CV R2   : {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

# ─────────────────────────────────────────────
#  STEP 7B — XGBOOST CLASSIFIER
# ─────────────────────────────────────────────

print("\n[6/7] Training XGBoost Classifier...")

label_encoder = LabelEncoder()
df["Verdict_Label"] = label_encoder.fit_transform(df["Verdict_Clean"])

Y_clf_train = train_df.loc[train_df.index.isin(X_train_full.index), "Verdict_Label"] \
    if "Verdict_Label" in train_df.columns \
    else df.loc[X_train_full.index, "Verdict_Label"]

Y_clf_train = df.loc[X_train_full.index, "Verdict_Label"]
Y_clf_test  = df.loc[X_test_full.index,  "Verdict_Label"]

# Class weights to handle imbalance (rare verdicts like ATB get more weight)
from collections import Counter
counts   = Counter(Y_clf_train)
n_total  = len(Y_clf_train)
n_classes = len(counts)
weights  = {cls: n_total / (n_classes * cnt) for cls, cnt in counts.items()}
sample_weights = np.array([weights[y] for y in Y_clf_train])

classifier = XGBClassifier(
    n_estimators     = 800,
    learning_rate    = 0.03,
    max_depth        = 5,
    subsample        = 0.8,
    colsample_bytree = 0.8,
    eval_metric      = "mlogloss",
    random_state     = 42,
    verbosity        = 0,
)
classifier.fit(X_train_full, Y_clf_train, sample_weight=sample_weights)

Y_pred_clf = classifier.predict(X_test_full)
acc = accuracy_score(Y_clf_test, Y_pred_clf)

print(f"\n  ── Classification Results ──")
print(f"  Accuracy : {acc:.4f}  (target > 0.55)")
print(f"\n  Classes  : {list(label_encoder.classes_)}")

# ─────────────────────────────────────────────
#  STEP 8 — SAVE MODELS
# ─────────────────────────────────────────────

os.makedirs("models", exist_ok=True)

joblib.dump(regressor,        "models/regressor.pkl")
joblib.dump(classifier,       "models/classifier.pkl")
joblib.dump(label_encoder,    "models/label_encoder.pkl")
joblib.dump(label_language,   "models/label_language.pkl")
joblib.dump(label_season,     "models/label_season.pkl")
joblib.dump(star_power_map,   "models/star_power_map.pkl")
joblib.dump(director_power_map, "models/director_power_map.pkl")
joblib.dump({
    "global_star_mean"    : global_star_mean,
    "global_director_mean": global_director_mean,
    "features_full"       : FEATURES_FULL,
}, "models/meta.pkl")

print("\n  Models saved to models/ folder ✓")

# ─────────────────────────────────────────────
#  STEP 9 — VISUALISATIONS
# ─────────────────────────────────────────────

print("\n[7/7] Generating charts...")

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle("Box Office Predictor — Model Results", fontsize=14, fontweight="bold")

# Chart 1 — Actual vs Predicted
ax1 = axes[0]
actual_cr    = np.expm1(Y_test_reg.values)
predicted_cr = np.expm1(Y_pred_reg)
ax1.scatter(actual_cr, predicted_cr, alpha=0.5, color="#7F77DD", s=30)
lim = max(actual_cr.max(), predicted_cr.max())
ax1.plot([0, lim], [0, lim], "r--", linewidth=1.5, label="Perfect Prediction")
ax1.set_xlabel("Actual Worldwide (Cr)")
ax1.set_ylabel("Predicted Worldwide (Cr)")
ax1.set_title(f"Actual vs Predicted\nR² = {r2:.3f}")
ax1.legend()

# Chart 2 — Confusion Matrix
ax2 = axes[1]
cm = confusion_matrix(Y_clf_test, Y_pred_clf)
sns.heatmap(
    cm, annot=True, fmt="d", ax=ax2,
    xticklabels=label_encoder.classes_,
    yticklabels=label_encoder.classes_,
    cmap="Purples"
)
ax2.set_title(f"Confusion Matrix\nAccuracy = {acc:.3f}")
ax2.set_xlabel("Predicted")
ax2.set_ylabel("Actual")
plt.setp(ax2.get_xticklabels(), rotation=30, ha="right", fontsize=8)
plt.setp(ax2.get_yticklabels(), rotation=0, fontsize=8)

# Chart 3 — Feature Importance
ax3 = axes[2]
importance = pd.Series(regressor.feature_importances_, index=FEATURES_FULL).sort_values()
colors = ["#7F77DD" if i >= len(importance) - 3 else "#B4B2A9" for i in range(len(importance))]
importance.plot(kind="barh", ax=ax3, color=colors)
ax3.set_title("Feature Importance\n(Top 3 highlighted)")
ax3.set_xlabel("Importance Score")

plt.tight_layout()
plt.savefig("models/results.png", dpi=150, bbox_inches="tight")
plt.show()
print("  Chart saved to models/results.png ✓")

# ─────────────────────────────────────────────
#  STEP 10 — HELPER: VERDICT FROM PROFIT %
# ─────────────────────────────────────────────

def verdict_from_profit(p):
    if   p < -50 : return "DISASTER"
    elif p < 0   : return "FLOP"
    elif p < 50  : return "AVERAGE"
    elif p < 100 : return "HIT"
    elif p < 200 : return "SUPER HIT"
    elif p < 400 : return "BLOCKBUSTER"
    else          : return "ALL TIME BLOCKBUSTER"

# ─────────────────────────────────────────────
#  STEP 11 — TERMINAL PREDICTION (OPTIONAL)
# ─────────────────────────────────────────────

print("\n" + "="*55)
print("  PREDICT A MOVIE")
print("="*55)

try:
    movie_name    = input("\nMovie Name           : ").strip()
    star_name     = input("Star Featuring       : ").strip().title()
    director_name = input("Director             : ").strip().title()
    budget        = float(input("Budget (Cr)          : "))
    opening_day   = float(input("Opening Day (Cr)     : "))
    screens       = int(input("Worldwide Screens    : "))
    language      = input("Language             : ").strip().title()
    release_month = int(input("Release Month (1-12) : "))
    is_franchise  = input("Sequel/Franchise? (y/n): ").strip().lower() == "y"

    # Star power lookup with fuzzy match
    star_match = get_close_matches(star_name, star_power_map.index.tolist(), n=1, cutoff=0.6)
    star_val   = star_power_map[star_match[0]] if star_match else global_star_mean
    if star_match:
        print(f"  → Matched star: '{star_match[0]}'")
    else:
        print(f"  → Star not found, using dataset average")

    # Director power lookup with fuzzy match
    dir_match = get_close_matches(director_name, director_power_map.index.tolist(), n=1, cutoff=0.6)
    dir_val   = director_power_map[dir_match[0]] if dir_match else global_director_mean
    if dir_match:
        print(f"  → Matched director: '{dir_match[0]}'")
    else:
        print(f"  → Director not found, using dataset average")

    # Language encoding
    if language in label_language.classes_:
        lang_val = int(label_language.transform([language])[0])
    else:
        lang_val = int(label_language.transform(["Unknown"])[0]) \
            if "Unknown" in label_language.classes_ \
            else 0

    season_str = get_season(release_month)
    season_val = int(label_season.transform([season_str])[0])

    future = pd.DataFrame([{
        "Budget"            : budget,
        "Opening_Day"       : opening_day,
        "Screens"           : screens,
        "Language_Label"    : lang_val,
        "Season_Label"      : season_val,
        "Franchise"         : int(is_franchise),
        "Opening_to_Budget" : opening_day / budget,
        "Screens_to_Budget" : screens / budget,
        "Opening_per_Screen": opening_day / screens if screens > 0 else 0,
        "Release_Year"      : 2025,
        "Star_Power"        : star_val,
        "Director_Power"    : dir_val,
    }])

    pred_log        = regressor.predict(future)[0]
    pred_worldwide  = round(float(np.expm1(pred_log)), 2)
    pred_verdict_id = classifier.predict(future)[0]
    clf_verdict     = label_encoder.inverse_transform([pred_verdict_id])[0]

    profit       = pred_worldwide - budget
    profit_pct   = (profit / budget) * 100
    # Primary verdict — always derived from predicted profit %
    # Classifier is used as a secondary cross-check only
    final_verdict = verdict_from_profit(profit_pct)

    # Confidence note: flag when classifier agrees or disagrees
    if clf_verdict == final_verdict:
        confidence = "HIGH  (both models agree)"
    else:
        confidence = f"MEDIUM (classifier suggested {clf_verdict})"

    print("\n" + "="*55)
    print("  PREDICTION RESULT")
    print("="*55)
    print(f"  Movie          : {movie_name}")
    print(f"  Predicted WW   : ₹{pred_worldwide} Cr")
    print(f"  Budget         : ₹{budget} Cr")
    print(f"  Profit         : ₹{round(profit, 2)} Cr  ({round(profit_pct, 1)}%)")
    print(f"  Verdict        : {final_verdict}")
    print(f"  Confidence     : {confidence}")
    print("="*55)

except KeyboardInterrupt:
    print("\n\n  Prediction skipped.")

print("\n✓ Program finished successfully.\n")