# ---------------------- STEP 7 (MULTI-YEAR READY): TRAIN DEEP LEARNING MODEL ----------------------
# Goals:
# 1. Load the Honest (Leak-Free) Data from Step 6.
# 2. Train a Transformer (FT-Transformer) to predict Default Risk.
# 3. Save the best model for calibration (Step 8) and RL validation.

import os, json, joblib, time, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score, f1_score

warnings.filterwarnings("ignore")
ARTIFACT_DIR = Path("artifacts")
MODEL_DIR = ARTIFACT_DIR / "models"
MODEL_DIR.mkdir(exist_ok=True)

# Reproducibility
RND_SEED = 42
np.random.seed(RND_SEED)
torch.manual_seed(RND_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RND_SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"STEP 7: Deep Learning Training on {DEVICE}")

# ---------- 1. Load Data (from Step 6) ----------
dl_data_path = ARTIFACT_DIR / "dl_data.joblib"
meta_path = ARTIFACT_DIR / "dl_metadata.json"

if not dl_data_path.exists():
    raise FileNotFoundError("Step 6 artifacts not found. Run Step 6 first.")

print("Loading dataset arrays...")
data = joblib.load(dl_data_path)
with open(meta_path, "r") as f:
    meta = json.load(f)

X_num_train, X_cat_train, y_train = data['train']['X_num'], data['train']['X_cat'], data['train']['y']
X_num_val,   X_cat_val,   y_val   = data['val']['X_num'],   data['val']['X_cat'],   data['val']['y']

# Sanity Check on Shapes
print(f"Training Data: {X_num_train.shape[0]} samples, {X_num_train.shape[1]} numeric features")
print(f"Validation Data: {X_num_val.shape[0]} samples")

# Class Weighting
num_pos = y_train.sum()
num_neg = len(y_train) - num_pos
pos_weight = num_neg / max(num_pos, 1.0)
print(f"Class Balance: {num_pos:.0f} Defaults / {len(y_train)} Total. Using pos_weight={pos_weight:.2f}")

# ---------- 2. PyTorch Dataset & DataLoader ----------
class LoanDataset(Dataset):
    def __init__(self, X_num, X_cat, y):
        self.X_num = torch.tensor(X_num, dtype=torch.float32)
        self.X_cat = torch.tensor(X_cat, dtype=torch.long)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X_num[idx], self.X_cat[idx], self.y[idx]

BATCH_SIZE = 512
train_loader = DataLoader(LoanDataset(X_num_train, X_cat_train, y_train), batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(LoanDataset(X_num_val, X_cat_val, y_val), batch_size=BATCH_SIZE*2, shuffle=False)

# ---------- 3. FT-Transformer Architecture ----------
class FeatureTokenizer(nn.Module):
    def __init__(self, num_numeric, cat_cardinalities, d_token):
        super().__init__()
        self.num_weights = nn.Parameter(torch.randn(num_numeric, d_token))
        self.num_bias = nn.Parameter(torch.randn(num_numeric, d_token))
        self.cat_embeddings = nn.ModuleList([
            nn.Embedding(c, d_token) for c in cat_cardinalities
        ])
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

        # PyTorch 1.12+ safe activation ("gelu")
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_token,
            nhead=n_heads,
            dim_feedforward=int(d_token * d_ffn_factor),
            dropout=dropout,
            batch_first=True,
            norm_first=True,
            activation="gelu"
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.head = nn.Sequential(
            nn.LayerNorm(d_token),
            nn.ReLU(),
            nn.Linear(d_token, 1)
        )

    def forward(self, x_num, x_cat):
        x = self.tokenizer(x_num, x_cat)
        x = self.transformer(x)
        return self.head(x[:, 0, :])

# Initialize
model = FTTransformer(
    num_numeric=meta['num_feat_count'],
    cat_cardinalities=meta['cat_cardinalities'],
    d_token=64, n_layers=2, n_heads=4, dropout=0.2
).to(DEVICE)

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight).to(DEVICE))

# ---------- 4. Helper: Evaluation Loop ----------
def evaluate(loader):
    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for x_num, x_cat, y in loader:
            x_num, x_cat = x_num.to(DEVICE), x_cat.to(DEVICE)
            logits = model(x_num, x_cat).squeeze()
            probs = torch.sigmoid(logits)
            preds.extend(probs.cpu().numpy())
            targets.extend(y.numpy())
    return np.array(targets), np.array(preds)

# ---------- 5. Training Loop ----------
EPOCHS = 15
best_auc = 0.0
patience = 4
patience_counter = 0

print(f"Starting Training for {EPOCHS} epochs...")

for epoch in range(EPOCHS):
    start_time = time.time()
    model.train()
    train_loss = 0

    for x_num, x_cat, y in train_loader:
        x_num, x_cat, y = x_num.to(DEVICE), x_cat.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()
        logits = model(x_num, x_cat).squeeze()
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    train_loss /= len(train_loader)

    # Evaluate on Val
    y_true, y_probs = evaluate(val_loader)
    val_auc = roc_auc_score(y_true, y_probs)

    print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {train_loss:.4f} | Val AUC: {val_auc:.4f} | Time: {time.time()-start_time:.1f}s")

    # Early Stopping
    if val_auc > best_auc:
        best_auc = val_auc
        torch.save(model.state_dict(), MODEL_DIR / "best_dl_model.pth")
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print(f"Early stopping triggered at Epoch {epoch+1}")
            break

# ---------- 6. Final Evaluation ----------
print("\nLoading Best Model for Final Report...")
model.load_state_dict(torch.load(MODEL_DIR / "best_dl_model.pth"))

y_true, y_probs = evaluate(val_loader)
val_preds_bin = (y_probs > 0.5).astype(int)
final_auc = roc_auc_score(y_true, y_probs)
final_f1 = f1_score(y_true, val_preds_bin)

print("-" * 30)
print(f"Final Model Results (Best Epoch):")
print(f"AUC Score: {final_auc:.4f}")
print(f"F1 Score:  {final_f1:.4f}")
print("-" * 30)

metrics = {"val_auc": float(final_auc), "val_f1": float(final_f1)}
with open(ARTIFACT_DIR / "dl_metrics.json", "w") as f:
    json.dump(metrics, f)

print("STEP 7 COMPLETE. Next: Step 8 (Calibration).")
# ---------------------- STEP 8 (MULTI-YEAR READY): CALIBRATION & PROFIT ----------------------
# Goals:
# 1. Generate risk predictions using the trained DL model (Step 7).
# 2. Calibrate probabilities using Isotonic Regression (Crucial for "Fair" Thresholds).
# 3. ALIGN raw financial data (Loan Amount, Int Rate) with predictions via Index.
# 4. Find the Risk Threshold that maximizes Total Portfolio Profit.

import os, json, joblib, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from sklearn.calibration import calibration_curve
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss

warnings.filterwarnings("ignore")
ARTIFACT_DIR = Path("artifacts")
IMG_DIR = ARTIFACT_DIR / "images"
IMG_DIR.mkdir(exist_ok=True)

print("STEP 8: Advanced Calibration & Profit Optimization")

# ---------- 1. Load Resources ----------
dl_data_path = ARTIFACT_DIR / "dl_data.joblib"
model_path   = ARTIFACT_DIR / "models/best_dl_model.pth"
meta_path    = ARTIFACT_DIR / "dl_metadata.json"
raw_val_path = ARTIFACT_DIR / "accepted_val_aug.parquet"
pre_val_path = ARTIFACT_DIR / "val_pre.parquet"

# Load Validation Data (Features & Labels)
if not dl_data_path.exists():
    raise FileNotFoundError("Step 6 data not found.")

data = joblib.load(dl_data_path)
X_num_val, X_cat_val, y_val = data['val']['X_num'], data['val']['X_cat'], data['val']['y']

# Re-define Model Architecture (Must match Step 7 exactly)
with open(meta_path, "r") as f: 
    meta = json.load(f)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

# Load Model Weights
model = FTTransformer(
    num_numeric=meta['num_feat_count'],
    cat_cardinalities=meta['cat_cardinalities'],
    d_token=64, n_layers=2, n_heads=4, dropout=0.2
).to(DEVICE)
model.load_state_dict(torch.load(model_path, map_location=DEVICE))
model.eval()

# ---------- 2. Generate Raw Predictions ----------
print("Generating raw predictions...")
batch_size = 1024
preds_raw = []
with torch.no_grad():
    for i in range(0, len(X_num_val), batch_size):
        x_num = torch.tensor(X_num_val[i:i+batch_size], dtype=torch.float32).to(DEVICE)
        x_cat = torch.tensor(X_cat_val[i:i+batch_size], dtype=torch.long).to(DEVICE)
        logits = model(x_num, x_cat).squeeze()
        preds_raw.extend(torch.sigmoid(logits).cpu().numpy())

preds_raw = np.array(preds_raw)

# ---------- 3. Calibrate Probabilities ----------
print("Calibrating model (Isotonic Regression)...")
iso_reg = IsotonicRegression(out_of_bounds='clip')
iso_reg.fit(preds_raw, y_val)
preds_calibrated = iso_reg.predict(preds_raw)

brier_imp = brier_score_loss(y_val, preds_raw) - brier_score_loss(y_val, preds_calibrated)
print(f"Brier Score Improvement: {brier_imp:.5f}")
joblib.dump(iso_reg, ARTIFACT_DIR / "calibrator.joblib")

# ---------- 4. Safe Data Alignment (CRITICAL) ----------
print("Aligning with raw business data...")

# A. Load the PROCESSED Validation set to get the VALID INDICES
# These indices correspond exactly to X_num_val and y_val
val_pre_df = pd.read_parquet(pre_val_path)
valid_indices = val_pre_df.index

# B. Load the RAW Validation set (Has Loan Amount, Interest Rate)
raw_val_df = pd.read_parquet(raw_val_path)

# C. Filter Raw Data using the Indices
# This ensures row 0 in 'business_df' matches row 0 in 'preds_calibrated'
business_df = raw_val_df.loc[valid_indices].copy()

# D. Verification
if len(business_df) != len(preds_calibrated):
    print(f"⚠️ CRITICAL MISMATCH: Data Rows {len(business_df)} != Predictions {len(preds_calibrated)}")
    # Last resort fallback if indices were reset somewhere (risky)
    business_df = business_df.iloc[:len(preds_calibrated)]
else:
    print(f"✅ Alignment Successful: {len(business_df)} loans.")

# E. Clean Financial Columns
business_df['loan_amnt'] = pd.to_numeric(business_df['loan_amnt'], errors='coerce').fillna(0)

def clean_rate(x):
    # Handle mixed types (string "12%" or float 12.0)
    if isinstance(x, str):
        return float(x.replace('%', '').strip())
    return float(x)

if 'int_rate' in business_df.columns:
    business_df['int_rate_clean'] = business_df['int_rate'].apply(clean_rate).fillna(0)
else:
    # Fallback if int_rate was renamed/processed earlier
    business_df['int_rate_clean'] = business_df.get('int_rate_val', 0)

# ---------- 5. Profit Optimization ----------
print("Optimizing Threshold...")

loan_amnts = business_df['loan_amnt'].values
int_rates = business_df['int_rate_clean'].values / 100.0
actual_outcomes = y_val

# Precompute Financial Outcomes
# Scenario 1: Paid Back (Gain Interest)
gain_if_paid = loan_amnts * int_rates
# Scenario 2: Default (Lose Principal) -> Assumes 100% Loss Given Default for DL baseline
loss_if_default = loan_amnts

# Threshold Search (0.01 to 0.50)
thresholds = np.linspace(0.01, 0.50, 100)
profits = []

for t in thresholds:
    # Approve if Risk < Threshold
    decisions = (preds_calibrated < t)
    
    # Vectorized PnL Calculation
    # If Approved: Check Outcome (1=Default, 0=Paid)
    # If Denied: 0
    pnl = np.where(decisions, 
                   np.where(actual_outcomes == 1, -loss_if_default, gain_if_paid), 
                   0)
    profits.append(pnl.sum())

# Find Best Threshold
best_idx = np.argmax(profits)
best_threshold = thresholds[best_idx]
max_profit = profits[best_idx]

# Plot Profit Curve
plt.figure(figsize=(10, 6))
plt.plot(thresholds, profits, color='green', linewidth=2)
plt.axvline(best_threshold, color='black', linestyle='--', label=f'Optimal: {best_threshold:.2%}')
plt.axhline(0, color='gray', linewidth=0.5)
plt.title(f"Profit vs. Risk Threshold (Max: ${max_profit:,.0f})")
plt.xlabel("Max Allowed Default Probability")
plt.ylabel("Total Portfolio Profit ($)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig(IMG_DIR / "profit_threshold.png")
plt.close()

print("-" * 40)
print(f"Optimal Risk Cutoff: {best_threshold:.2%}")
print(f"Projected Profit:    ${max_profit:,.2f}")
print("-" * 40)

# Save Metadata for RL Comparison
meta = {"optimal_threshold": float(best_threshold), "max_profit": float(max_profit)}
with open(ARTIFACT_DIR / "calibration_metadata.json", "w") as f:
    json.dump(meta, f)

print("STEP 8 COMPLETE.")
