# ---------------------- STEP 5 (FINAL ROBUST): CAUSAL INFERENCE ----------------------
# Goals:
# 1. Define Treatment T (High vs Low Interest Rate) dynamically.
# 2. Train Propensity Model P(T|X) to control for confounding (Risk).
# 3. Estimate ATE (Average Treatment Effect) using Doubly Robust estimator.
# 4. Estimate CATE (Conditional ATE) to see who is sensitive to rates.

import os, warnings, joblib, json
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")
ARTIFACT_DIR = Path("artifacts")
IMG_DIR = ARTIFACT_DIR / "images"
IMG_DIR.mkdir(exist_ok=True)

print("STEP 5: Causal Inference (Propensity, ATE, CATE)")

# ---------- 1. Load Preprocessed Data ----------
train_path = ARTIFACT_DIR / "train_pre.parquet"
val_path   = ARTIFACT_DIR / "val_pre.parquet"
feature_meta_path = ARTIFACT_DIR / "feature_names.json"

if not train_path.exists():
    raise FileNotFoundError("Step 4 artifacts not found. Run Step 4 first.")

print("Loading training data...")
train_df = pd.read_parquet(train_path)
val_df   = pd.read_parquet(val_path)

with open(feature_meta_path, "r") as f:
    feature_names = json.load(f)

print(f"Train size: {len(train_df):,}, Val size: {len(val_df):,}")

# ---------- 2. Define Treatment (T) & Outcome (Y) ----------
print("Defining Treatment (T)...")

# Robustly find the rate column (Step 4 saves it as 'int_rate_val')
if 'int_rate_val' in train_df.columns:
    rate_col = 'int_rate_val'
elif 'int_rate' in train_df.columns:
    rate_col = 'int_rate'
else:
    # If not found in columns, checks if it was normalized into X_num
    # We look for it in the feature list
    matches = [f for f in feature_names if 'int_rate' in f]
    if matches:
        rate_col = matches[0]
    else:
        raise KeyError("Could not find 'int_rate' column for Treatment.")

# DYNAMIC THRESHOLD (Median Split)
# Since Step 4 scaled the data (RobustScaler), 0.0 is roughly the median.
# We calculate the exact median of the current training set to be safe.
train_rates = train_df[rate_col]
T_THRESHOLD = train_rates.median()

print(f"Using column '{rate_col}' for Treatment.")
print(f"Treatment Threshold (Median): {T_THRESHOLD:.4f}")

# Define T: 1 if rate > median (High Rate), 0 if rate <= median (Low Rate)
train_df['T'] = (train_rates > T_THRESHOLD).astype(int)
val_df['T']   = (val_df[rate_col] > T_THRESHOLD).astype(int)

# Verify Prevalence
prevalence = train_df['T'].mean()
print(f"Treatment Prevalence (High Rate): {prevalence:.2%}")
if prevalence < 0.01 or prevalence > 0.99:
    raise ValueError("Treatment T is skewed/constant! Causal inference impossible.")

# Define Outcome Y (Default = 1)
# Ensure target exists
if 'target' not in train_df.columns:
    raise KeyError("Target column missing from train_df.")
    
train_df['Y'] = train_df['target'].astype(int)
val_df['Y']   = val_df['target'].astype(int)

# ---------- 3. Define Covariates (X) ----------
# We must exclude the Treatment itself and its proxies to prevent leakage.
# Proxies include: 'grade', 'sub_grade', 'fico_grade_interaction'
proxies = [rate_col, 'grade_ord', 'sub_grade_te', 'fico_grade_interaction', 'int_rate']
# Filter out proxies from the feature list
propensity_features = [f for f in feature_names if f not in proxies and f in train_df.columns]

print(f"Propensity Features used: {len(propensity_features)}")

X_train = train_df[propensity_features].fillna(0)
y_train_T = train_df['T']
y_train_Y = train_df['Y']

X_val = val_df[propensity_features].fillna(0)

# ---------- 4. Train Propensity Model P(T|X) ----------
print("\nTraining Propensity Model (Random Forest)...")
# Predicts: "Given these risk features, should this person have a High Interest Rate?"
ps_model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,       # Restrict depth to prevent overfitting
    min_samples_leaf=50,
    random_state=42,
    n_jobs=-1
)
ps_model.fit(X_train, y_train_T)

# Predict Propensity Scores
ps_train = ps_model.predict_proba(X_train)[:, 1]
ps_val   = ps_model.predict_proba(X_val)[:, 1]

# Visualization: Overlap Check
plt.figure(figsize=(10, 5))
sns.kdeplot(ps_train[y_train_T==0], label='Control (Low Rate)', fill=True, clip=(0,1))
sns.kdeplot(ps_train[y_train_T==1], label='Treated (High Rate)', fill=True, clip=(0,1))
plt.title("Propensity Score Overlap (Common Support)")
plt.xlabel("Probability of Receiving High Rate")
plt.legend()
plt.tight_layout()
plt.savefig(IMG_DIR / "propensity_overlap.png")
print("Saved Propensity Overlap plot.")

train_df['ps'] = ps_train
val_df['ps']   = ps_val
joblib.dump(ps_model, ARTIFACT_DIR / "propensity_model.joblib")

# ---------- 5. Train Outcome Model E[Y|X,T] (S-Learner) ----------
print("\nTraining Outcome Model (Gradient Boosting)...")
# Predicts Default Probability using Features + Treatment
outcome_features = propensity_features + ['T']
X_out_train = train_df[propensity_features].copy()
X_out_train['T'] = train_df['T']

outcome_model = GradientBoostingClassifier(
    n_estimators=100,
    max_depth=5,
    random_state=42
)
outcome_model.fit(X_out_train, y_train_Y)
joblib.dump(outcome_model, ARTIFACT_DIR / "outcome_model.joblib")

# Calculate Potential Outcomes (Counterfactuals)
# What if everyone got High Rate? (T=1)
X_T1 = train_df[propensity_features].copy()
X_T1['T'] = 1
train_df['mu1'] = outcome_model.predict_proba(X_T1)[:, 1]

# What if everyone got Low Rate? (T=0)
X_T0 = train_df[propensity_features].copy()
X_T0['T'] = 0
train_df['mu0'] = outcome_model.predict_proba(X_T0)[:, 1]

# ---------- 6. Estimate ATE (Doubly Robust AIPW) ----------
print("\nCalculating ATE (Doubly Robust)...")
# Clip PS to avoid division by zero
ps = np.clip(train_df['ps'], 0.05, 0.95)
T = train_df['T']
Y = train_df['Y']
mu1 = train_df['mu1']
mu0 = train_df['mu0']

# AIPW Formula
dr_mu1 = mu1 + (T / ps) * (Y - mu1)
dr_mu0 = mu0 + ((1 - T) / (1 - ps)) * (Y - mu0)
ate_dr = np.mean(dr_mu1 - dr_mu0)

print(f"Estimated ATE (High vs Low Rate): {ate_dr:.4f}")
if ate_dr > 0:
    print(f"Interpretation: High interest rates INCREASE default risk by {ate_dr*100:.2f} percentage points.")
else:
    print(f"Interpretation: High interest rates DECREASE default risk by {abs(ate_dr)*100:.2f} percentage points.")

# ---------- 7. Estimate CATE (Heterogeneous Effects) ----------
print("\nCalculating CATE (Who is sensitive?)...")
# Simple difference of potential outcomes
train_df['cate'] = train_df['mu1'] - train_df['mu0']
print("CATE Distribution:")
print(train_df['cate'].describe())

# Train a model to predict CATE for new users
# Target = derived CATE from training
cate_model = GradientBoostingRegressor(n_estimators=50, max_depth=3, random_state=42)
cate_model.fit(X_train, train_df['cate'])
joblib.dump(cate_model, ARTIFACT_DIR / "cate_model.joblib")

# Generate CATE for Validation Set
val_df['cate_pred'] = cate_model.predict(X_val)

# ---------- 8. Save Artifacts ----------
# We save these scores to help Step 10 (RL) understand causality
train_df[['cate', 'ps', 'mu1', 'mu0']].to_parquet(ARTIFACT_DIR / "causal_train_scores.parquet", index=False)
val_df[['cate_pred', 'ps']].to_parquet(ARTIFACT_DIR / "causal_val_scores.parquet", index=False)

causal_meta = {
    "ate": float(ate_dr),
    "threshold_value": float(T_THRESHOLD),
    "threshold_col": rate_col
}
with open(ARTIFACT_DIR / "causal_metadata.json", "w") as f:
    json.dump(causal_meta, f)

print("\nSTEP 5 COMPLETE.")
