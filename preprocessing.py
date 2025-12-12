#-------------------- STEP 1 (FIXED & ROBUST): RAM-SAFE MULTI-YEAR SAMPLING --------------------
# Goal: Create a representative dataset covering 2007-2018 that fits in RAM.
import os, sys, warnings, gc
from pathlib import Path
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
ARTIFACT_DIR = Path("artifacts")
ARTIFACT_DIR.mkdir(exist_ok=True)

# ---------- CONFIGURATION ----------
ACCEPTED_PATH = "/content/a.csv"
REJECTED_PATH = "/content/r.csv"
SAMPLE_RATE = 0.15
SEED = 42

print(f"STEP 1: Processing Multi-Year Data (Sampling Rate: {SAMPLE_RATE:.0%})")

# ---------- 1. Process 'Accepted' (a.csv) ----------
if os.path.exists(ACCEPTED_PATH):
    print(f"Streaming {ACCEPTED_PATH}...")

    chunk_size = 200_000
    processed_chunks = []
    total_rows = 0
    years_seen = set()

    # Iterate through the file without loading it all
    for chunk in pd.read_csv(ACCEPTED_PATH, chunksize=chunk_size, low_memory=False):

        if 'id' in chunk.columns:
            chunk['id'] = chunk['id'].astype(str)

        # B. Optimize Numeric Types (Save Memory)
        float_cols = chunk.select_dtypes(include=['float64']).columns
        chunk[float_cols] = chunk[float_cols].astype('float32')

        # C. Check Years (Verification)
        if 'issue_d' in chunk.columns:
            # Extract year roughly to confirm we are seeing multiple years
            chunk_years = chunk['issue_d'].astype(str).str.extract(r'(\d{4})')[0].dropna().unique()
            years_seen.update(chunk_years)

        # This keeps 15% of the data IN THIS TIME PERIOD.
        sample = chunk.sample(frac=SAMPLE_RATE, random_state=SEED)

        processed_chunks.append(sample)
        total_rows += len(sample)
        # print(f"   Processed chunk... Accumulating {len(sample)} rows. (Total: {total_rows})")

        # Force garbage collection
        del chunk

    # E. Combine and Save
    print("Concatenating samples...")
    full_sample = pd.concat(processed_chunks, ignore_index=True)

    # Double check 'id' before save
    if 'id' in full_sample.columns:
        full_sample['id'] = full_sample['id'].astype(str)

    output_path = ARTIFACT_DIR / "raw_accepted.parquet"
    full_sample.to_parquet(output_path, index=False)

    print(f"\n✅ SUCCESS: Saved {len(full_sample):,} rows to {output_path}")
    print(f"   Years covered: {sorted(list(years_seen))}")
    print(f"   Memory Usage: {full_sample.memory_usage(deep=True).sum() / 1024**2:.1f} MB")

    del full_sample, processed_chunks
    gc.collect()

else:
    print(f"❌ Error: {ACCEPTED_PATH} not found.")

# ---------- 2. Process 'Rejected' (r.csv) ----------
# Rejected data is massive (20GB+). We need an even smaller sample rate here.
REJ_SAMPLE_RATE = 0.02 # 2%

if os.path.exists(REJECTED_PATH):
    print(f"\nStreaming {REJECTED_PATH} (Sampling {REJ_SAMPLE_RATE:.0%})...")

    rej_chunks = []
    total_rej = 0

    try:
        for chunk in pd.read_csv(REJECTED_PATH, chunksize=500_000, low_memory=False):
            # Optimize
            float_cols = chunk.select_dtypes(include=['float64']).columns
            chunk[float_cols] = chunk[float_cols].astype('float32')

            # Sample
            sample = chunk.sample(frac=REJ_SAMPLE_RATE, random_state=SEED)
            rej_chunks.append(sample)
            total_rej += len(sample)

            # Limit rejected rows to save RAM (500k is enough for rejection stats)
            if total_rej > 500_000:
                print("   Reached rejected row limit (500k). Stopping early to save RAM.")
                break

        print("Concatenating rejected samples...")
        full_rej = pd.concat(rej_chunks, ignore_index=True)

        output_path_rej = ARTIFACT_DIR / "raw_rejected.parquet"
        full_rej.to_parquet(output_path_rej, index=False)
        print(f"✅ SUCCESS: Saved {len(full_rej):,} rejected rows.")

        del full_rej, rej_chunks
        gc.collect()

    except Exception as e:
        print(f"⚠️ Warning processing rejected data: {e}")
else:
    print("\n(Skipping Rejected: File not found)")

print("\nSTEP 1 COMPLETE. Multi-year data is ready.")
# ---------------------- STEP 2 (MULTI-YEAR + RAM SAFE): Robust Split & Leak-Free Stats ----------------------
# Logic:
# 1. Load Parquet files from Step 1 (chunks are already handled there).
# 2. Sort Accepted data by Time (2007 -> 2018).
# 3. Split 70/15/15 based on Time (Train on Past, Test on Future).
# 4. Filter Rejected data to only include rows from the "Train" timeline (No future leaks).
# 5. Compute Stats (Geo, FICO, DTI) using only valid history.
import os, warnings, json, gc
from pathlib import Path
import numpy as np, pandas as pd

warnings.filterwarnings("ignore")
ARTIFACT_DIR = Path("artifacts")
ARTIFACT_DIR.mkdir(exist_ok=True)

# ---------- User-provided paths ----------
accepted_parquet = ARTIFACT_DIR / "raw_accepted.parquet"
rejected_parquet = ARTIFACT_DIR / "raw_rejected.parquet"

print("STEP 2: Processing Multi-Year Data (Time-Based Split)")

# ---------- 1) Load Accepted Data ----------
if not accepted_parquet.exists():
    raise FileNotFoundError("Run Step 1 first to generate 'raw_accepted.parquet'.")

print(f"Loading {accepted_parquet}...")
acc = pd.read_parquet(accepted_parquet)

# ---------- 2) Load Rejected Data ----------
if rejected_parquet.exists():
    print(f"Loading {rejected_parquet}...")
    rej = pd.read_parquet(rejected_parquet)
else:
    print("No rejected data found.")
    rej = None

# ---------- 3) Standardize Columns ----------
print("Standardizing columns...")

# Map Rejected columns to Accepted names
rej_to_acc_map = {
    "State": "addr_state",
    "Amount Requested": "loan_amnt",
    "Risk_Score": "fico_range_low",
    "Application Date": "issue_d",
    "Debt-To-Income Ratio": "dti",
    "Zip Code": "zip_code",
    "Employment Length": "emp_length"
}

if rej is not None:
    rej.rename(columns={k: v for k, v in rej_to_acc_map.items() if k in rej.columns}, inplace=True)
    
    # Date Parsing (Critical for Time-Aware filtering)
    rej['issue_d_parsed'] = pd.to_datetime(rej['issue_d'], errors='coerce')
    
    # Numeric Cleaning
    if 'dti' in rej.columns:
        rej['dti'] = pd.to_numeric(rej['dti'].astype(str).str.replace(r'[%,]', '', regex=True), errors='coerce')
    if 'zip_code' in rej.columns:
        rej['zip3'] = rej['zip_code'].astype(str).str[:3].str.replace(r'[^0-9A-Za-z]', '', regex=True).fillna('000')
    
    for c in ['fico_range_low', 'loan_amnt']:
        if c in rej.columns:
            rej[c] = pd.to_numeric(rej[c], errors='coerce').astype(np.float32)
            
    rej['is_approved'] = 0

# Standardize Accepted
acc['is_approved'] = 1

# Date Parsing
if 'issue_d' in acc.columns:
    acc['issue_d_parsed'] = pd.to_datetime(acc['issue_d'], errors='coerce')
elif 'issue_month' in acc.columns:
    acc['issue_d_parsed'] = pd.to_datetime(acc['issue_month'].astype(str), errors='coerce')
else:
    acc['issue_d_parsed'] = pd.NaT

# Drop rows with no date (cannot split them safely)
acc = acc.dropna(subset=['issue_d_parsed'])

if 'addr_state' not in acc.columns and 'state' in acc.columns:
    acc = acc.rename(columns={'state': 'addr_state'})

if 'zip_code' in acc.columns:
    acc['zip3'] = acc['zip_code'].astype(str).str[:3].str.replace(r'[^0-9A-Za-z]','', regex=True)

# Float32 Optimization
for c in ['fico_range_low','dti','loan_amnt']:
    if c in acc.columns:
        acc[c] = pd.to_numeric(acc[c], errors='coerce').astype(np.float32)

# ---------- 4) Perform Time-Based Split ----------
print("Sorting by Date for Time-Series Split...")
acc = acc.sort_values('issue_d_parsed').reset_index(drop=True)

n_acc = len(acc)
i1 = int(0.70 * n_acc)
i2 = int(0.85 * n_acc)

train_cutoff_date = acc.iloc[i1]['issue_d_parsed']
val_cutoff_date = acc.iloc[i2]['issue_d_parsed']

print(f"Time Split Configuration:")
print(f" - Train: Start -> {train_cutoff_date.date()} ({i1:,} rows)")
print(f" - Val:   {train_cutoff_date.date()} -> {val_cutoff_date.date()} ({i2-i1:,} rows)")
print(f" - Test:  {val_cutoff_date.date()} -> End ({n_acc-i2:,} rows)")

# Create Splits
acc_train = acc.iloc[:i1].copy()
acc_val = acc.iloc[i1:i2].copy()
acc_test = acc.iloc[i2:].copy()

# CLEANUP: Free RAM
del acc
gc.collect()

# ---------- 5) Prepare Rejection Stats (Strictly Past-Only) ----------
# We must only use Rejected applications that happened BEFORE or DURING the training period.
# Using 2018 rejections to calculate 2012 stats would be data leakage.

if rej is not None:
    print(f"Filtering Rejected data (cutoff: {train_cutoff_date.date()})...")
    # Keep only rejections from the 'Train' timeline
    rej_train = rej[rej['issue_d_parsed'] <= train_cutoff_date].copy()
    print(f" - Raw Rejected: {len(rej):,}")
    print(f" - Train-Valid Rejected: {len(rej_train):,}")
    
    del rej
    gc.collect()
else:
    rej_train = pd.DataFrame()

# Create Combined Train for Stats Calculation
combined_train = pd.concat([acc_train, rej_train], ignore_index=True, sort=False)
print(f"Combined History for Stats: {len(combined_train):,} rows")

# ---------- 6) Compute Stats (Helper) ----------
def get_rej_stats(df, group_col, bucket_col=None):
    col = bucket_col if bucket_col else group_col
    df[col] = df[col].astype(str).replace('nan', 'MISSING')
    
    stats = df.groupby(col).agg(
        total_count=('is_approved', 'count'),
        rejected_count=('is_approved', lambda x: (x==0).sum())
    ).reset_index()
    
    stats['rejection_rate'] = np.where(stats['total_count'] > 0,
                                       stats['rejected_count'] / stats['total_count'], 0.0)
    return stats

# A. State Stats
combined_train['addr_state'] = combined_train['addr_state'].fillna('__MISSING__').astype(str)
state_stats = get_rej_stats(combined_train, 'addr_state')
state_stats.rename(columns={'rejection_rate': 'state_rejection_rate'}, inplace=True)
state_stats.to_parquet(ARTIFACT_DIR / "state_stats_train.parquet", index=False)

# B. Zip3 Stats
combined_train['zip3'] = combined_train['zip3'].fillna('000').astype(str)
zip_stats = get_rej_stats(combined_train, 'zip3')
zip_stats.rename(columns={'rejection_rate': 'zip3_rejection_rate'}, inplace=True)
zip_stats.to_parquet(ARTIFACT_DIR / "zip3_stats_train.parquet", index=False)

# C. FICO Stats
combined_train['fico'] = combined_train['fico_range_low'].fillna(0)
fico_bins = [300,600,650,700,750,800,900]
fico_labels = ['<600','600-649','650-699','700-749','750-799','800+']
combined_train['fico_bucket'] = pd.cut(combined_train['fico'], bins=fico_bins, labels=fico_labels, include_lowest=True)

fico_stats = get_rej_stats(combined_train, 'fico', bucket_col='fico_bucket')
fico_stats.rename(columns={'rejection_rate': 'fico_rejection_rate'}, inplace=True)
fico_stats.to_parquet(ARTIFACT_DIR / "fico_stats_train.parquet", index=False)

# D. DTI Stats
combined_train['dti'] = combined_train['dti'].fillna(-1)
dti_bins = [-1,10,20,25,30,40,100]
dti_labels = ['0-10','10-20','20-25','25-30','30-40','40+']
combined_train['dti_bucket'] = pd.cut(combined_train['dti'], bins=dti_bins, labels=dti_labels, include_lowest=True)

dti_stats = get_rej_stats(combined_train, 'dti', bucket_col='dti_bucket')
dti_stats.rename(columns={'rejection_rate': 'dti_rejection_rate'}, inplace=True)
dti_stats.to_parquet(ARTIFACT_DIR / "dti_stats_train.parquet", index=False)

# CLEANUP: Save raw view for EDA, then delete
combined_train.to_parquet(ARTIFACT_DIR / "combined_train_raw.parquet", index=False)
del combined_train, rej_train
gc.collect()

# ---------- 7) Augment & Save ----------
global_rej_rate = state_stats['rejected_count'].sum() / state_stats['total_count'].sum()

def augment_df(df):
    # State
    df['addr_state'] = df.get('addr_state', '__MISSING__').fillna('__MISSING__').astype(str)
    df = df.merge(state_stats[['addr_state', 'state_rejection_rate']], on='addr_state', how='left')
    
    # Zip3
    df['zip3'] = df.get('zip3', '000').fillna('000').astype(str)
    df = df.merge(zip_stats[['zip3', 'zip3_rejection_rate']], on='zip3', how='left')
    
    # FICO
    df['fico'] = pd.to_numeric(df['fico_range_low'], errors='coerce').fillna(0)
    df['fico_bucket'] = pd.cut(df['fico'], bins=fico_bins, labels=fico_labels, include_lowest=True).astype(str)
    fico_stats['fico_bucket'] = fico_stats['fico_bucket'].astype(str)
    df = df.merge(fico_stats[['fico_bucket', 'fico_rejection_rate']], on='fico_bucket', how='left')
    
    # DTI
    df['dti'] = pd.to_numeric(df['dti'], errors='coerce').fillna(-1)
    df['dti_bucket'] = pd.cut(df['dti'], bins=dti_bins, labels=dti_labels, include_lowest=True).astype(str)
    dti_stats['dti_bucket'] = dti_stats['dti_bucket'].astype(str)
    df = df.merge(dti_stats[['dti_bucket', 'dti_rejection_rate']], on='dti_bucket', how='left')
    
    # Fill NAs with global mean (for new Zips/States in Val/Test)
    for c in ['state_rejection_rate', 'zip3_rejection_rate', 'fico_rejection_rate', 'dti_rejection_rate']:
        df[c] = df[c].fillna(global_rej_rate)
        
    return df

print("Augmenting and Saving Splits...")
for name, df in [("train", acc_train), ("val", acc_val), ("test", acc_test)]:
    aug_df = augment_df(df)
    aug_df.to_parquet(ARTIFACT_DIR / f"accepted_{name}_aug.parquet", index=False)
    print(f" - Saved accepted_{name}_aug.parquet ({len(aug_df):,} rows)")
    del aug_df, df
    gc.collect()

print("\nSTEP 2 COMPLETE.")



