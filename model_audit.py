# ---------------------- STEP 9 (MULTI-YEAR READY): MODEL AUDIT & BASELINE COMPARISON ----------------------
# Goals:
# 1. Evaluate the "Leak-Proof" Deep Learning Model on the UNSEEN TEST set.
# 2. Train a Logic Check Baseline (Logistic Regression on FICO + Grade).
# 3. Confirm that the "Cheat Code" is gone (Gap should be realistic, e.g., 2-5%).

import os, json, joblib, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Dataset

warnings.filterwarnings("ignore")
ARTIFACT_DIR = Path("artifacts")
MODEL_DIR = ARTIFACT_DIR / "models"

print("STEP 9: AUDIT - Final Verification")

# ---------- 1. Load Data (Features & Labels) ----------
print("Loading processed test data...")
# We use the 'pre' parquet files because they contain the raw features needed for Baseline
train_df = pd.read_parquet(ARTIFACT_DIR / "train_pre.parquet")
test_df  = pd.read_parquet(ARTIFACT_DIR / "test_pre.parquet")

# Load DL Arrays for the Advanced Model
dl_data = joblib.load(ARTIFACT_DIR / "dl_data.joblib")
meta = json.load(open(ARTIFACT_DIR / "dl_metadata.json"))

# ---------- 2. Run Baseline (Logistic Regression) ----------
print("\n--- BASELINE: Logistic Regression (FICO + Grade) ---")
# FICO and Grade are the "Industry Standard" basics.
# If a complex model can't beat these, it's useless.

# Handle feature name variations from Step 4
base_feats = ['fico', 'grade_ord'] # Default
if 'grade_ord' not in train_df.columns:
    if 'grade_te' in train_df.columns:
        print(" 'grade_ord' not found, using 'grade_te' instead.")
        base_feats = ['fico', 'grade_te']
    else:
        print(" 'grade' features missing. Using only FICO.")
        base_feats = ['fico']

# Prepare Baseline Inputs
X_base_train = train_df[base_feats].fillna(0)
y_base_train = train_df['target']
X_base_test  = test_df[base_feats].fillna(0)
y_base_test  = test_df['target']

# Train
lr = LogisticRegression(solver='lbfgs')
lr.fit(X_base_train, y_base_train)

# Evaluate
base_probs = lr.predict_proba(X_base_test)[:, 1]
base_auc = roc_auc_score(y_base_test, base_probs)

print(f"Baseline AUC (Industry Standard): {base_auc:.4f}")

# ---------- 3. Run Deep Learning Model on TEST Set ----------
print("\n--- DEEP LEARNING: FT-Transformer on Test Set ---")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Re-define Architecture (Must match Step 7 exactly)
class FeatureTokenizer(nn.Module):
    def __init__(self, num_numeric, cat_cardinalities, d_token):
        super().__init__()
        self.num_weights = nn.Parameter(torch.randn(num_numeric, d_token))
        self.num_bias = nn.Parameter(torch.randn(num_numeric, d_token))
        self.cat_embeddings = nn.ModuleList([nn.Embedding(c, d_token) for c in cat_cardinalities])
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_token))
    def forward(self, x_num, x_cat):
        batch_size = x_num.shape[0]
        x_num_emb = x_num.unsqueeze(-1) * self.num_weights.unsqueeze(0) + self.num_bias.unsqueeze(0)
        x_cat_emb = [emb(x_cat[:, i]) for i, emb in enumerate(self.cat_embeddings)]
        x_cat_emb = torch.stack(x_cat_emb, dim=1)
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        return torch.cat([cls_tokens, x_num_emb, x_cat_emb], dim=1)

class FTTransformer(nn.Module):
    def __init__(self, num_numeric, cat_cardinalities, d_token=192, n_layers=3, n_heads=8, d_ffn_factor=1.33, dropout=0.1):
        super().__init__()
        self.tokenizer = FeatureTokenizer(num_numeric, cat_cardinalities, d_token)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_token, nhead=n_heads, dim_feedforward=int(d_token * d_ffn_factor),
            dropout=dropout, batch_first=True, norm_first=True, activation="gelu"
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.head = nn.Sequential(nn.LayerNorm(d_token), nn.ReLU(), nn.Linear(d_token, 1))
    def forward(self, x_num, x_cat):
        x = self.tokenizer(x_num, x_cat)
        x = self.transformer(x)
        return self.head(x[:, 0, :])

# Load Model
model = FTTransformer(
    num_numeric=meta['num_feat_count'],
    cat_cardinalities=meta['cat_cardinalities'],
    d_token=64, n_layers=2, n_heads=4, dropout=0.2
).to(DEVICE)

model_path = MODEL_DIR / "best_dl_model.pth"
if not model_path.exists():
    raise FileNotFoundError("Best DL model not found. Run Step 7 first.")

model.load_state_dict(torch.load(model_path, map_location=DEVICE))
model.eval()

# Prepare Test Loader (using Test Set arrays from Step 6)
X_num_test_arr, X_cat_test_arr, y_test_arr = dl_data['test']['X_num'], dl_data['test']['X_cat'], dl_data['test']['y']

class LoanDataset(Dataset):
    def __init__(self, X_num, X_cat, y):
        self.X_num = torch.tensor(X_num, dtype=torch.float32)
        self.X_cat = torch.tensor(X_cat, dtype=torch.long)
        self.y = torch.tensor(y, dtype=torch.float32)
    def __len__(self): return len(self.y)
    def __getitem__(self, idx): return self.X_num[idx], self.X_cat[idx], self.y[idx]

test_loader = DataLoader(LoanDataset(X_num_test_arr, X_cat_test_arr, y_test_arr), batch_size=1024, shuffle=False)

# Predict
dl_preds = []
dl_targets = []
with torch.no_grad():
    for x_num, x_cat, y in test_loader:
        x_num, x_cat = x_num.to(DEVICE), x_cat.to(DEVICE)
        logits = model(x_num, x_cat).squeeze()
        probs = torch.sigmoid(logits)
        dl_preds.extend(probs.cpu().numpy())
        dl_targets.extend(y.numpy())

dl_auc = roc_auc_score(dl_targets, dl_preds)
print(f"Deep Learning Test AUC: {dl_auc:.4f}")

# ---------- 4. The Verdict ----------
print("\n" + "="*40)
print("AUDIT VERDICT")
print("="*40)
print(f"Baseline (Logistic Reg): {base_auc:.4f}")
print(f"Deep Learning (FT-Trans): {dl_auc:.4f}")
print("-" * 40)

gap = dl_auc - base_auc

if dl_auc > 0.90:
    print("🚨 SUSPICIOUS: AUC is still too high (>0.90).")
    print("   Check Step 4 again. You likely still have leakage (e.g. 'recoveries' or 'payment_plan').")
elif gap < 0:
    print("❌ ERROR: Deep Learning is WORSE than simple logistic regression.")
    print("   Cause: Overfitting, bad hyperparameters, or incorrect feature scaling.")
elif gap < 0.01:
    print("⚠️  WARNING: Minimal Lift (<1%).")
    print("   The DL model is essentially just learning FICO score. It adds no value.")
else:
    print(f"✅ PASS: Healthy Lift (+{gap*100:.2f}%).")
    print("   The model is capturing complex risk factors beyond just FICO/Grade.")
    print("   Proceed to Step 10 (Reinforcement Learning).")

print("\nSTEP 9 COMPLETE.")
