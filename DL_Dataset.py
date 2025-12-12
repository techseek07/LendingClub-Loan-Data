# ---------------------- STEP 6 (MULTI-YEAR COMPATIBLE): BUILD DEEP LEARNING DATASET ----------------------
# Logic:
# 1. Load the "Baked" data (train/val/test_pre) from Step 4.
# 2. **CRITICAL:** Check for Hardship/Deferral Leakage in numeric features.
# 3. Define Categorical vs Numeric features.
# 4. Build Integer Index Maps for Categoricals (Fit on Train, Handle Unknowns).
# 5. Save PyTorch-ready arrays (X_num, X_cat, y) and metadata.

import os, warnings, joblib, json, sys
from pathlib import Path
import numpy as np, pandas as pd

warnings.filterwarnings("ignore")
ARTIFACT_DIR = Path("artifacts")
ARTIFACT_DIR.mkdir(exist_ok=True)

print("STEP 6: Preparing Deep Learning Dataset (Numeric + Categorical Indices)")

# ---------- 1. Load The "Refined Product" (from Step 4) ----------
train_path = ARTIFACT_DIR / "train_pre.parquet"
val_path   = ARTIFACT_DIR / "val_pre.parquet"
test_path  = ARTIFACT_DIR / "test_pre.parquet"
feature_meta_path = ARTIFACT_DIR / "feature_names.json"

if not train_path.exists():
    raise FileNotFoundError("Step 4 artifacts not found. Run Step 4 first.")

print("Loading preprocessed data...")
train_df = pd.read_parquet(train_path)
val_df   = pd.read_parquet(val_path)
test_df  = pd.read_parquet(test_path)

# Load the list of Numeric Features defined in Step 4
with open(feature_meta_path, "r") as f:
    numeric_features = json.load(f)

print(f"Data Loaded: Train {len(train_df):,}, Val {len(val_df):,}, Test {len(test_df):,}")

# ---------- 2. CRITICAL SAFETY CHECK (Leakage Prevention) ----------
print("Running Anti-Leakage Diagnostics...")
# These keywords generally indicate post-origination distress signals
leakage_keywords = ['hardship', 'deferral', 'settlement', 'payment_plan']
found_leaks = [c for c in numeric_features if any(k in c.lower() for k in leakage_keywords)]

if found_leaks:
    print("\n" + "!"*50)
    print("CRITICAL ERROR: LEAKAGE DETECTED IN FEATURE LIST")
    print("!"*50)
    print(f"The following post-origination columns are still present:\n{found_leaks}")
    print("-" * 50)
    print("ACTION REQUIRED: You must re-run 'Step 4' to drop these columns.")
    print("Stopping execution to prevent training a cheating model.")
    sys.exit(1) # Stop the script immediately
else:
    print("✅ Leakage Check Passed: No hardship/deferral columns found.")

# ---------- 3. Identify Categorical Features ----------
# These are the columns Step 4 left as Strings.
# We explicitly list them to avoid accidental encoding of ID columns.
potential_cats = [
    'grade', 'sub_grade', 'home_ownership', 'verification_status',
    'purpose', 'addr_state', 'zip3_grouped'
]

# Only keep ones that actually exist in the dataframe
cat_cols = [c for c in potential_cats if c in train_df.columns]
print(f"Categorical Features found for Embeddings: {cat_cols}")

# ---------- 4. Safe Overlap Check ----------
# Ensure no feature appears in both lists (Numeric AND Categorical)
overlap = set(numeric_features).intersection(set(cat_cols))
if overlap:
    print(f"[WARN] Overlap detected: {overlap} are in both lists. Removing from Numeric.")
    numeric_features = [c for c in numeric_features if c not in cat_cols]

print(f"Final Numeric Features count: {len(numeric_features)}")

# ---------- 5. Build Categorical Index Maps (String -> Int) ----------
print("Building Categorical Index Maps...")
cat_index_map = {}

for c in cat_cols:
    # 1. Ensure string format (Handle any residual NaNs safely)
    # Using 'MISSING' ensures NaNs get their own learnable embedding if frequent
    train_df[c] = train_df[c].astype(str).replace('nan', 'MISSING')
    val_df[c]   = val_df[c].astype(str).replace('nan', 'MISSING')
    test_df[c]  = test_df[c].astype(str).replace('nan', 'MISSING')

    # 2. Find unique values in TRAIN only (Strictly correct methodology)
    # New categories in Test/Val will be mapped to 0 (Unknown)
    uniques = sorted(train_df[c].unique().tolist())

    # 3. Create Map: Value -> Int (Start at 1, leave 0 for 'Unknown')
    mapping = {val: i+1 for i, val in enumerate(uniques)}
    cat_index_map[c] = mapping

    # 4. Helper to apply map
    def apply_map(series, mapping_dict):
        # Map values; if not found (e.g. new category in Test), fill with 0 (Unknown)
        return series.map(mapping_dict).fillna(0).astype(int)

    # 5. Apply to all splits
    idx_col = f'{c}_idx'
    train_df[idx_col] = apply_map(train_df[c], mapping)
    val_df[idx_col]   = apply_map(val_df[c], mapping)
    test_df[idx_col]  = apply_map(test_df[c], mapping)

# Save Mappings (Vital for Inference/Deployment)
joblib.dump(cat_index_map, ARTIFACT_DIR / "cat_index_map.joblib")

# ---------- 6. Construct Final Arrays (X_num, X_cat, y) ----------
print("Constructing dense/sparse arrays for PyTorch...")

def get_arrays(df):
    # A. Numeric Matrix (Float32)
    # These are already scaled/imputed from Step 4
    X_num = df[numeric_features].values.astype(np.float32)

    # B. Categorical Matrix (Int64)
    # We use the newly created _idx columns
    cat_idx_cols = [f'{c}_idx' for c in cat_cols]
    X_cat = df[cat_idx_cols].values.astype(np.int64)

    # C. Target (Float32)
    y = df['target'].values.astype(np.float32)

    return X_num, X_cat, y

X_num_train, X_cat_train, y_train = get_arrays(train_df)
X_num_val,   X_cat_val,   y_val   = get_arrays(val_df)
X_num_test,  X_cat_test,  y_test  = get_arrays(test_df)

print(f"Train Shapes -> Num: {X_num_train.shape}, Cat: {X_cat_train.shape}, Y: {y_train.shape}")

# ---------- 7. Save Artifacts for Step 7 (Training) ----------
print("Saving Step 6 Artifacts...")

# Save Arrays efficiently
dl_data = {
    'train': {'X_num': X_num_train, 'X_cat': X_cat_train, 'y': y_train},
    'val':   {'X_num': X_num_val,   'X_cat': X_cat_val,   'y': y_val},
    'test':  {'X_num': X_num_test,  'X_cat': X_cat_test,  'y': y_test}
}
joblib.dump(dl_data, ARTIFACT_DIR / "dl_data.joblib")

# Save Metadata (Cardinality) so the Neural Net knows embedding sizes
# Cardinality = Count of Uniques + 1 (for Unknown/0)
cat_cardinalities = [len(cat_index_map[c]) + 1 for c in cat_cols]

dl_metadata = {
    "num_feat_count": len(numeric_features),
    "cat_feat_names": cat_cols,
    "cat_cardinalities": cat_cardinalities,
    "numeric_names": numeric_features
}
with open(ARTIFACT_DIR / "dl_metadata.json", "w") as f:
    json.dump(dl_metadata, f)

print("\nSTEP 6 COMPLETE.")
print("Files saved: dl_data.joblib, dl_metadata.json, cat_index_map.joblib")
