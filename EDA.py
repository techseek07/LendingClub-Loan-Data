# ---------------------- STEP 3: EXPLORATORY DATA ANALYSIS (EDA) ----------------------
# Goals:
# 1. Validate "Policy Stats" (Do the Step 2 rejection rates make sense?).
# 2. Validate "Vintage Stability" (Monthly default rates across 2007-2015).
# 3. Visualize the "Invisible Boundary" (Accepted vs Rejected distributions).
# 4. Check Interest Rate Sensitivity (LOWESS causal proxy).
# 5. (Optional) 2D Risk Heatmaps (FICO vs DTI).

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.nonparametric.smoothers_lowess import lowess

warnings.filterwarnings("ignore")
pd.set_option("display.max_columns", 50)
sns.set_style("whitegrid")

ARTIFACT_DIR = Path("artifacts")
IMG_DIR = ARTIFACT_DIR / "images"
IMG_DIR.mkdir(exist_ok=True)

print("STEP 3: EDA, Policy Validation, and Causal Intuition Checks")

# -------------------------------------------------------------------
# 1. Load artifacts from Step 2 (train only)
# -------------------------------------------------------------------
acc_train_path      = ARTIFACT_DIR / "accepted_train_aug.parquet"
combined_train_path = ARTIFACT_DIR / "combined_train_raw.parquet"
fico_stats_path     = ARTIFACT_DIR / "fico_stats_train.parquet"
dti_stats_path      = ARTIFACT_DIR / "dti_stats_train.parquet"
state_stats_path    = ARTIFACT_DIR / "state_stats_train.parquet"

required = [acc_train_path, combined_train_path,
            fico_stats_path, dti_stats_path, state_stats_path]
missing = [p for p in required if not p.exists()]
if missing:
    raise FileNotFoundError(
        f"Missing Step 2 artifacts. Ensure Step 2 saved: {', '.join(str(p) for p in missing)}"
    )

print("Loading Step 2 artifacts...")
acc_train      = pd.read_parquet(acc_train_path)
combined_train = pd.read_parquet(combined_train_path)
fico_stats     = pd.read_parquet(fico_stats_path)
dti_stats      = pd.read_parquet(dti_stats_path)
state_stats    = pd.read_parquet(state_stats_path)

print(f"Accepted Train (acc_train): {len(acc_train):,} rows")
print(f"Combined Train (acc+rej):   {len(combined_train):,} rows")

# Ensure basic numeric types for later plots
acc_train["fico_range_low"] = pd.to_numeric(acc_train.get("fico_range_low"), errors="coerce")
acc_train["dti"]            = pd.to_numeric(acc_train.get("dti"), errors="coerce")

# -------------------------------------------------------------------
# 2. Validate rejection-rate engineering (FICO / DTI / State)
# -------------------------------------------------------------------
print("\n[Analysis 1] Validating Rejection-Rate features from Step 2...")

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# A. Rejection rate vs FICO bucket
sns.barplot(
    data=fico_stats,
    x="fico_bucket",
    y="fico_rejection_rate",
    ax=axes[0],
    palette="Reds",
)
axes[0].set_title("Rejection Rate by FICO Bucket", fontsize=12)
axes[0].set_ylabel("Rejection Rate")
axes[0].set_xlabel("FICO Range")
axes[0].tick_params(axis="x", rotation=45)

# B. Rejection rate vs DTI bucket
sns.barplot(
    data=dti_stats,
    x="dti_bucket",
    y="dti_rejection_rate",
    ax=axes[1],
    palette="Oranges",
)
axes[1].set_title("Rejection Rate by DTI Bucket", fontsize=12)
axes[1].set_ylabel("Rejection Rate")
axes[1].set_xlabel("DTI Range")
axes[1].tick_params(axis="x", rotation=45)

# C. Top 10 states by rejection rate
top_rej_states = state_stats.sort_values("state_rejection_rate", ascending=False).head(10)
sns.barplot(
    data=top_rej_states,
    x="addr_state",
    y="state_rejection_rate",
    ax=axes[2],
    palette="Blues_r",
)
axes[2].set_title("Top 10 States by Rejection Rate", fontsize=12)
axes[2].set_ylabel("Rejection Rate")
axes[2].set_xlabel("State")

plt.tight_layout()
plt.savefig(IMG_DIR / "rejection_rate_validation.png")
plt.show()
print("Check: FICO rejection rate should decrease with higher FICO; DTI should increase with higher DTI.")

# -------------------------------------------------------------------
# 3. Vintage stability on train (monthly default rate)
# -------------------------------------------------------------------
print("\n[Analysis 2] Monthly Vintage Stability in Train...")

# Ensure we have a target column; Step 2 did not create it explicitly for EDA
if "target" not in acc_train.columns:
    def _map_target_eda(s):
        s = str(s).lower()
        if "charged off" in s or "default" in s:
            return 1
        if "fully paid" in s:
            return 0
        return np.nan
    if "loan_status" not in acc_train.columns:
        raise KeyError("loan_status not present to derive target for EDA.")
    acc_train["target"] = acc_train["loan_status"].apply(_map_target_eda)

# Convert to period for plotting
acc_train["issue_d_parsed"] = pd.to_datetime(acc_train["issue_d_parsed"], errors="coerce")
acc_train["issue_month_dt"] = acc_train["issue_d_parsed"].dt.to_period("M").astype(str)

vintage_agg = (
    acc_train
    .groupby("issue_month_dt")["target"]
    .agg(count="size", default_rate="mean")
    .sort_index()
)

plt.figure(figsize=(12, 5))
sns.lineplot(
    data=vintage_agg,
    x=vintage_agg.index,
    y="default_rate",
    marker="o",
    linewidth=2,
    color="purple",
)
plt.title("Vintage Drift: Default Rate by Issue Month (Multi-Year)", fontsize=14)
plt.ylabel("Default Rate")
plt.xlabel("Issue Month")
plt.grid(True, alpha=0.3)
plt.xticks(rotation=45)
# If too many ticks, sparse them
if len(vintage_agg) > 20:
    plt.gca().xaxis.set_major_locator(plt.MaxNLocator(20))

plt.tight_layout()
plt.savefig(IMG_DIR / "vintage_drift.png")
plt.show()
print("Inspect: This now shows the trend across ALL years in the training set.")

# -------------------------------------------------------------------
# 4. Policy boundary plots (accepted vs rejected) for FICO & DTI
# -------------------------------------------------------------------
print("\n[Analysis 3] Policy Boundary: Accepted vs Rejected distributions...")

def plot_boundary(df, col, x_label, x_lim=None, annotation_x=None):
    plt.figure(figsize=(10, 6))

    # Numeric conversion for safety
    df_num = df.copy()
    df_num[col] = pd.to_numeric(df_num[col], errors="coerce")

    sns.histplot(
        data=df_num,
        x=col,
        hue="is_approved",
        hue_order=[1, 0],                   # 1 = accepted, 0 = rejected
        stat="density",
        common_norm=False,
        element="step",
        bins=50,
        palette={1: "green", 0: "red"},
    )

    # Fix legend labels: 1 -> Accepted, 0 -> Rejected
    handles, labels = plt.gca().get_legend_handles_labels()
    label_map = {"1": "Accepted", "0": "Rejected"}
    labels = [label_map.get(l, l) for l in labels]
    plt.legend(handles, labels, title="Status")

    plt.title(f"Policy Boundary: {x_label} (Accepted vs Rejected)", fontsize=14)
    plt.xlabel(x_label)
    plt.ylabel("Density")
    if x_lim is not None:
        plt.xlim(x_lim)

    # Optional RL-annotation near the boundary
    if annotation_x is not None:
        ymin, ymax = plt.ylim()
        plt.annotate(
            "RL Agent explores here →",
            xy=(annotation_x, ymax * 0.15),
            xytext=(annotation_x + 20, ymax * 0.30),
            arrowprops=dict(facecolor="black", shrink=0.05),
            fontsize=10,
        )

    plt.tight_layout()
    plt.savefig(IMG_DIR / f"policy_boundary_{col}.png")
    plt.show()

# Ensure combined_train numeric cols
combined_train["fico_range_low"] = pd.to_numeric(
    combined_train.get("fico_range_low"), errors="coerce"
)
combined_train["dti"] = pd.to_numeric(combined_train.get("dti"), errors="coerce")
combined_train["dti_plot"] = combined_train["dti"].clip(lower=0, upper=100)

# FICO boundary
plot_boundary(
    combined_train,
    col="fico_range_low",
    x_label="FICO Score",
    x_lim=(500, 850),
    annotation_x=660,
)

# DTI boundary (no annotation, threshold less obvious)
plot_boundary(
    combined_train,
    col="dti_plot",
    x_label="Debt-to-Income Ratio (DTI)",
    x_lim=(0, 60),
    annotation_x=None,
)

# -------------------------------------------------------------------
# 5. 2D risk heatmap (FICO vs DTI) on train
# -------------------------------------------------------------------
print("\n[Analysis 4] 2D Risk Heatmap: FICO vs DTI...")

acc_train["fico_plot_bin"] = pd.cut(
    acc_train["fico_range_low"],
    bins=[660, 680, 700, 720, 740, 760, 800],
    labels=["660-679", "680-699", "700-719", "720-739", "740-759", "760+"],
)

acc_train["dti_plot_bin"] = pd.cut(
    acc_train["dti"],
    bins=[0, 10, 15, 20, 25, 30, 40],
    labels=["0-10", "10-15", "15-20", "20-25", "25-30", "30+"],
)

risk_df = acc_train.dropna(subset=["fico_plot_bin", "dti_plot_bin", "target"])
pivot_risk = risk_df.pivot_table(
    index="dti_plot_bin",
    columns="fico_plot_bin",
    values="target",
    aggfunc="mean",
)

plt.figure(figsize=(10, 8))
sns.heatmap(
    pivot_risk,
    annot=True,
    fmt=".1%",
    cmap="RdYlGn_r",
    cbar_kws={"label": "Default Rate"},
)
plt.title("Risk Heatmap: Default Rate (DTI vs FICO)", fontsize=14)
plt.tight_layout()
plt.savefig(IMG_DIR / "risk_heatmap_dti_vs_fico.png")
plt.show()

# -------------------------------------------------------------------
# 6. Interest rate sensitivity (LOWESS)
# -------------------------------------------------------------------
print("\n[Analysis 5] Interest Rate vs Default Risk (LOWESS)...")

# Clean interest rate
acc_train["int_rate_val"] = pd.to_numeric(
    acc_train["int_rate"].astype(str).str.replace(r"[%,]", "", regex=True),
    errors="coerce",
)

# Sample to avoid slowness with large multi-year dataset
sample = (
    acc_train
    .dropna(subset=["int_rate_val", "target"])
    .sample(n=min(5000, len(acc_train)), random_state=42)
)

x = sample["int_rate_val"].values
y = sample["target"].values
smooth = lowess(y, x, frac=0.3)

plt.figure(figsize=(10, 6))
plt.scatter(
    x,
    y + np.random.normal(0, 0.02, size=len(y)),
    alpha=0.1,
    color="gray",
    s=10,
    label="Loans (jittered)",
)
plt.plot(smooth[:, 0], smooth[:, 1], color="red", linewidth=3, label="Trend (LOWESS)")
plt.title("Price Sensitivity: Default Probability vs Interest Rate", fontsize=14)
plt.xlabel("Interest Rate (%)")
plt.ylabel("Probability of Default")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(IMG_DIR / "rate_sensitivity_lowess.png")
plt.show()

print("\nSTEP 3 COMPLETE.")
print(f"EDA plots saved under: {IMG_DIR}")
print("Ready for Step 4 (Data Cleaning & Leakage Removal).")
# ---------------------- STEP 4 (FINAL LEAK-PROOF V3): ADVANCED PREPROCESSING ----------------------
# Logic:
# 1. Loads Step 2 data (which is now time-split and leak-free regarding Rejection Rates).
# 2. DROPS all post-origination leakage columns (hardship, deferral, settlement).
# 3. Saves 'feature_names.json' and processed parquets.

import os, warnings, joblib, json
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.preprocessing import RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import KFold

warnings.filterwarnings("ignore")
ARTIFACT_DIR = Path("artifacts")
ARTIFACT_DIR.mkdir(exist_ok=True)

print("STEP 4: Advanced Preprocessing (Leak-Proof V3)")

# ---------- 1. Load Data ----------
train_path = ARTIFACT_DIR / "accepted_train_aug.parquet"
val_path   = ARTIFACT_DIR / "accepted_val_aug.parquet"
test_path  = ARTIFACT_DIR / "accepted_test_aug.parquet"

if not train_path.exists():
    raise FileNotFoundError("Step 2 artifacts not found. Run Step 2 first.")

print("Loading splits...")
train_df = pd.read_parquet(train_path)
val_df   = pd.read_parquet(val_path)
test_df  = pd.read_parquet(test_path)

# Deduplicate columns (Safety check)
train_df = train_df.loc[:, ~train_df.columns.duplicated()]
val_df   = val_df.loc[:, ~val_df.columns.duplicated()]
test_df  = test_df.loc[:, ~test_df.columns.duplicated()]

print(f"Loaded - Train: {len(train_df):,}, Val: {len(val_df):,}, Test: {len(test_df):,}")

# ---------- 2. Define Target & Drop LEAKAGE Columns (EXTENDED) ----------
def get_target(status):
    s = str(status).lower()
    if 'charged off' in s or 'default' in s: return 1
    if 'fully paid' in s: return 0
    return np.nan 

for df in [train_df, val_df, test_df]:
    if 'target' not in df.columns:
        df['target'] = df['loan_status'].apply(get_target)

# Drop rows with NaN target
train_df = train_df.dropna(subset=['target'])
val_df = val_df.dropna(subset=['target'])
test_df = test_df.dropna(subset=['target'])

# --- EXTENDED LEAKAGE LIST (CRITICAL) ---
leakage_cols = [
    # Payment / Recovery Leakage
    'total_pymnt', 'total_pymnt_inv', 'total_rec_prncp', 'total_rec_int', 
    'total_rec_late_fee', 'recoveries', 'collection_recovery_fee', 
    'last_pymnt_d', 'last_pymnt_amnt', 'next_pymnt_d', 'out_prncp', 'out_prncp_inv',
    
    # Credit Score Leakage (Future FICO)
    'last_fico_range_low', 'last_fico_range_high', 'last_credit_pull_d',
    
    # Settlement Leakage (Post-Default)
    'debt_settlement_flag', 'debt_settlement_flag_date', 'settlement_status', 
    'settlement_date', 'settlement_amount', 'settlement_percentage', 'settlement_term',
    
    # Hardship / Deferral Leakage (Post-Origination Distress)
    'hardship_flag', 'hardship_type', 'hardship_reason', 'hardship_status',
    'deferral_term', 'hardship_amount', 'hardship_start_date', 'hardship_end_date',
    'payment_plan_start_date', 'hardship_length', 'hardship_dpd', 
    'hardship_loan_status', 'orig_projected_additional_accrued_interest',
    'hardship_payoff_balance_amount', 'hardship_last_payment_amount',
    
    # Redundant / Administrative
    'funded_amnt', 'funded_amnt_inv', 
    'id', 'member_id', 'url', 'desc', 'emp_title', 'title', 'issue_d', 'issue_month', 
    'pymnt_plan', 'target_raw'
]

print(f"Dropping {len(leakage_cols)} potential leakage columns...")
for df in [train_df, val_df, test_df]:
    # Standard drop
    cols_to_drop = [c for c in leakage_cols if c in df.columns]
    
    # Aggressive Pattern Matching Drop (Case-Insensitive)
    # This catches "Hardship_Amount" even if list has "hardship_amount"
    extra_leaks = [c for c in df.columns if 'hardship' in c.lower() or 'deferral' in c.lower() or 'settlement' in c.lower()]
    
    final_drop_list = list(set(cols_to_drop + extra_leaks))
    df.drop(columns=final_drop_list, inplace=True, errors='ignore')

# ---------- 3. Robust Numeric Conversion & Missing Indicators ----------
print("Creating missing indicators...")
cols_with_na = [c for c in train_df.columns if train_df[c].isna().any()]
for c in cols_with_na:
    ind = c + "_missing"
    train_df[ind] = train_df[c].isna().astype(int)
    val_df[ind] = val_df[c].isna().astype(int)
    test_df[ind] = test_df[c].isna().astype(int)

# Safe Numeric Conversion
num_cands = ['annual_inc', 'loan_amnt', 'dti', 'revol_bal', 'tot_cur_bal', 'tot_coll_amt', 'fico_range_low']
for df in [train_df, val_df, test_df]:
    for c in num_cands:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')

# Clean Percents
def clean_percent(val):
    if pd.isna(val): return np.nan
    try: return float(str(val).replace('%','').strip())
    except: return np.nan

for df in [train_df, val_df, test_df]:
    if 'int_rate' in df.columns:
        df['int_rate_val'] = df['int_rate'].apply(clean_percent) if df['int_rate'].dtype=='O' else df['int_rate']
    if 'revol_util' in df.columns:
        df['revol_util_val'] = df['revol_util'].apply(clean_percent) if df['revol_util'].dtype=='O' else df['revol_util']
    if 'term' in df.columns:
        df['term_val'] = df['term'].astype(str).str.extract(r'(\d+)').astype(float)
    else:
        df['term_val'] = 36.0
    
    if 'zip3' in df.columns:
        df['zip3'] = df['zip3'].fillna('000').astype(str)

# ---------- 4. Feature Engineering ----------
print("Engineering domain features...")

def engineer_features(df):
    df = df.copy()
    # Log Transforms
    for col in ['annual_inc', 'revol_bal', 'tot_cur_bal', 'tot_coll_amt', 'loan_amnt']:
        if col in df.columns:
            df[f'log_{col}'] = np.log1p(df[col].fillna(0).clip(lower=0))
            
    # Ratios
    inc = df['annual_inc'].fillna(0)
    df['loan_frac_income'] = df['loan_amnt'] / (inc + 1.0)
    
    # Credit Age
    if 'earliest_cr_line' in df.columns and 'issue_d_parsed' in df.columns:
        ecl = pd.to_datetime(df['earliest_cr_line'], errors='coerce')
        ecl = ecl.fillna(df['issue_d_parsed'])
        df['credit_age_years'] = (df['issue_d_parsed'] - ecl).dt.days / 365.25
        df['credit_age_years'] = df['credit_age_years'].clip(lower=0)
    else:
        df['credit_age_years'] = 0.0

    # Interactions
    grade_map = {'A':7, 'B':6, 'C':5, 'D':4, 'E':3, 'F':2, 'G':1}
    df['grade_ord'] = df['grade'].map(grade_map).fillna(0).astype(int)
    
    df['fico'] = pd.to_numeric(df['fico_range_low'], errors='coerce').fillna(0)
    df['fico_below_700'] = (df['fico'] < 700).astype(int)
    df['fico_squared'] = df['fico'] ** 2
    df['fico_grade_interaction'] = df['fico'] * df['grade_ord']
    df['fico_distance_660'] = df['fico'] - 660 
    
    df['is_term_60'] = (df['term_val'] == 60).astype(int)
    df['term_60_lowgrade'] = ((df['is_term_60'] == 1) & (df['grade_ord'] <= 5)).astype(int)
    
    df['dti_clean'] = pd.to_numeric(df['dti'], errors='coerce').fillna(0)
    df['dti_high_risk'] = (df['dti_clean'] > 25).astype(int)
    df['dti_20_25'] = ((df['dti_clean'] > 20) & (df['dti_clean'] <= 25)).astype(int)
    
    df['purpose'] = df['purpose'].fillna('MISSING').astype(str).str.lower()
    df['is_small_business'] = df['purpose'].str.contains('small_business|business').astype(int)
    
    df['disbursement_method'] = df['disbursement_method'].fillna('MISSING').astype(str).str.lower()
    df['is_cash_disbursement'] = (df['disbursement_method'] == 'cash').astype(int)
    
    return df

train_df = engineer_features(train_df)
val_df = engineer_features(val_df)
test_df = engineer_features(test_df)

# ---------- 5. Winsorization ----------
print("Winsorizing...")
winsor_cols = [
    'annual_inc', 'loan_amnt', 'dti_clean', 'revol_bal', 'revol_util_val', 'total_acc',
    'tot_cur_bal', 'tot_coll_amt', 'loan_frac_income', 'credit_age_years',
    'fico', 'int_rate_val', 
    'state_rejection_rate', 'zip3_rejection_rate', 'fico_rejection_rate', 'dti_rejection_rate'
]
winsor_cols = [c for c in winsor_cols if c in train_df.columns]

def fit_winsorize_bounds(s, lower_p=0.001, upper_p=0.995):
    low = s.quantile(lower_p)
    up  = s.quantile(upper_p)
    return low, up

def apply_winsorize(s, low, up):
    return s.clip(lower=low, upper=up)

winsor_bounds = {}
for c in winsor_cols:
    low, up = fit_winsorize_bounds(train_df[c], lower_p=0.001, upper_p=0.995)
    winsor_bounds[c] = (low, up)
    for df in [train_df, val_df, test_df]:
        df[c] = apply_winsorize(df[c], low, up)

joblib.dump(winsor_bounds, ARTIFACT_DIR / "winsor_bounds.joblib")

# ---------- 6. Target Encoding ----------
print("Target Encoding...")
cat_cols = ['grade', 'sub_grade', 'home_ownership', 'verification_status', 'purpose', 'addr_state']

valid_zips = train_df['zip3'].value_counts()[lambda x: x >= 200].index
for df in [train_df, val_df, test_df]:
    df['zip3_grouped'] = df['zip3'].apply(lambda x: x if x in valid_zips else 'OTHER')
cat_cols.append('zip3_grouped')

for c in cat_cols:
    if c not in train_df.columns: continue
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    train_df[f'{c}_te'] = np.nan
    for tr_ix, val_ix in kf.split(train_df):
        means = train_df.iloc[tr_ix].groupby(c)['target'].mean()
        train_df.loc[train_df.index[val_ix], f'{c}_te'] = train_df.iloc[val_ix][c].map(means)
    
    global_mean = train_df['target'].mean()
    train_df[f'{c}_te'] = train_df[f'{c}_te'].fillna(global_mean)
    
    global_map = train_df.groupby(c)['target'].mean()
    val_df[f'{c}_te'] = val_df[c].map(global_map).fillna(global_mean)
    test_df[f'{c}_te'] = test_df[c].map(global_map).fillna(global_mean)

# ---------- 7. Final Features (Index Preserved) ----------
print("Finalizing & Saving...")

# Exclude non-numeric & leakage
exclude_cols = set(cat_cols + ['zip3', 'target', 'loan_status', 'target_raw', 'issue_d_parsed'])
all_numeric_cols = [c for c in train_df.columns 
                    if c not in exclude_cols 
                    and np.issubdtype(train_df[c].dtype, np.number)]

all_numeric_cols = sorted(list(set(all_numeric_cols)))

# Drop 100% empty columns
valid_features = []
for c in all_numeric_cols:
    if train_df[c].isna().all():
        print(f"  [WARN] Dropping empty column: {c}")
    else:
        valid_features.append(c)

final_features = valid_features
print(f"Final features: {len(final_features)}")

# Impute & Scale (Fit on Train, Apply to All)
imputer = SimpleImputer(strategy='median')
scaler = RobustScaler()

# Apply transformations IN PLACE to preserve the Index
train_df[final_features] = imputer.fit_transform(train_df[final_features])
val_df[final_features]   = imputer.transform(val_df[final_features])
test_df[final_features]  = imputer.transform(test_df[final_features])

train_df[final_features] = scaler.fit_transform(train_df[final_features])
val_df[final_features]   = scaler.transform(val_df[final_features])
test_df[final_features]  = scaler.transform(test_df[final_features])

# Save Artifacts
joblib.dump(imputer, ARTIFACT_DIR / "imputer.joblib")
joblib.dump(scaler, ARTIFACT_DIR / "scaler.joblib")
with open(ARTIFACT_DIR / "feature_names.json", "w") as f:
    json.dump(final_features, f)

# --- CRITICAL FIX: PRESERVE INDICES ---
train_df.to_parquet(ARTIFACT_DIR / "train_pre.parquet")
val_df.to_parquet(ARTIFACT_DIR / "val_pre.parquet")
test_df.to_parquet(ARTIFACT_DIR / "test_pre.parquet")

print("\nSTEP 4 COMPLETE.")
