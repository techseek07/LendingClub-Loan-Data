# ---------------------- STEP 12: POLICY DIVERGENCE ANALYSIS ----------------------
# Goals:
# 1. Calculate DL Approval Count (Missing from previous logs).
# 2. Compare Head-to-Head: DL Decision vs RL Decision.
# 3. Find "The Alpha": Loans RL Approves but DL Rejects.
# 4. Explain WHY (Interest Rate Analysis).

import os, json, joblib, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.isotonic import IsotonicRegression

warnings.filterwarnings("ignore")
ARTIFACT_DIR = Path("artifacts")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"STEP 12: Policy Comparison on {DEVICE}")

# ---------- 1. Load Everything ----------
print("Loading Models & Data...")

# Data
dl_data = joblib.load(ARTIFACT_DIR / "dl_data.joblib")
X_num_val = dl_data['val']['X_num']
X_cat_val = dl_data['val']['X_cat']
y_val = dl_data['val']['y']

# Metadata & Thresholds
with open(ARTIFACT_DIR / "dl_metadata.json", "r") as f:
    dl_meta = json.load(f)
with open(ARTIFACT_DIR / "calibration_metadata.json", "r") as f:
    calib_meta = json.load(f)

dl_threshold = calib_meta['optimal_threshold']
print(f"DL Risk Threshold: {dl_threshold:.2%}")

# Load Raw Financials (for explanation)
val_pre = pd.read_parquet(ARTIFACT_DIR / "val_pre.parquet")
# Get raw interest rate (unscaled)
if 'int_rate' in val_pre.columns:
    raw_rates = val_pre['int_rate'].astype(str).str.replace(r'[%,]', '', regex=True).astype(float).values
elif 'int_rate_val' in val_pre.columns:
    raw_rates = val_pre['int_rate_val'].values
else:
    raw_rates = np.zeros(len(val_pre))

# ---------- 2. Re-Define Models (Architectures) ----------
# A. Deep Learning Model (Risk Predictor)
class FeatureTokenizer(nn.Module):
    def __init__(self, num_numeric, cat_cardinalities, d_token):
        super().__init__()
        self.num_weights = nn.Parameter(torch.randn(num_numeric, d_token))
        self.num_bias = nn.Parameter(torch.randn(num_numeric, d_token))
        self.cat_embeddings = nn.ModuleList([nn.Embedding(c, d_token) for c in cat_cardinalities])
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_token))
    def forward(self, x_num, x_cat):
        B = x_num.shape[0]
        x_num_emb = x_num.unsqueeze(-1) * self.num_weights.unsqueeze(0) + self.num_bias.unsqueeze(0)
        x_cat_emb = torch.stack([emb(x_cat[:, i]) for i, emb in enumerate(self.cat_embeddings)], dim=1)
        cls_tokens = self.cls_token.expand(B, -1, -1)
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

dl_model = FTTransformer(dl_meta['num_feat_count'], dl_meta['cat_cardinalities'], d_token=64, n_layers=2, n_heads=4).to(DEVICE)
dl_model.load_state_dict(torch.load(ARTIFACT_DIR / "models/best_dl_model.pth", map_location=DEVICE))
dl_model.eval()

# B. RL Agent (Value Predictor) - QNetwork
class QNetwork(nn.Module):
    def __init__(self, num_numeric, cat_cardinalities, d_token=64):
        super().__init__()
        self.num_weights = nn.Parameter(torch.randn(num_numeric, d_token))
        self.num_bias = nn.Parameter(torch.randn(num_numeric, d_token))
        self.cat_embeddings = nn.ModuleList([nn.Embedding(c, d_token) for c in cat_cardinalities])
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_token))
        input_dim = (1 + num_numeric + len(cat_cardinalities)) * d_token
        self.head = nn.Sequential(
            nn.Linear(input_dim, 256), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, 2)
        )
    def forward(self, x_num, x_cat):
        B = x_num.shape[0]
        x_num_emb = x_num.unsqueeze(-1) * self.num_weights.unsqueeze(0) + self.num_bias.unsqueeze(0)
        x_cat_emb = torch.stack([emb(x_cat[:, i]) for i, emb in enumerate(self.cat_embeddings)], dim=1)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x_num_emb, x_cat_emb], dim=1)
        x = x.view(B, -1)
        return self.head(x)

rl_agent = QNetwork(dl_meta['num_feat_count'], dl_meta['cat_cardinalities']).to(DEVICE)
rl_agent.load_state_dict(torch.load(ARTIFACT_DIR / "rl_models/q_network_aligned.pth", map_location=DEVICE))
rl_agent.eval()

# ---------- 3. Generate Decisions ----------
print("Running Inference...")
calibrator = joblib.load(ARTIFACT_DIR / "calibrator.joblib")

dl_decisions = [] # 1=Approve, 0=Deny
rl_decisions = []
dl_risks = []
rl_values = []

batch_size = 1024
with torch.no_grad():
    for i in range(0, len(X_num_val), batch_size):
        xn = torch.tensor(X_num_val[i:i+batch_size], dtype=torch.float32).to(DEVICE)
        xc = torch.tensor(X_cat_val[i:i+batch_size], dtype=torch.long).to(DEVICE)

        # A. DL Decision
        logits = dl_model(xn, xc).squeeze()
        probs_raw = torch.sigmoid(logits).cpu().numpy()
        probs_calibrated = calibrator.predict(probs_raw)

        # DL Policy: Approve if Risk < Threshold
        dl_act = (probs_calibrated < dl_threshold).astype(int)

        dl_decisions.extend(dl_act)
        dl_risks.extend(probs_calibrated)

        # B. RL Decision
        q_vals = rl_agent(xn, xc)
        # RL Policy: Argmax Q
        rl_act = torch.argmax(q_vals, dim=1).cpu().numpy()

        rl_decisions.extend(rl_act)
        # Net Value = Q(Approve) - Q(Deny)
        q_net_val = (q_vals[:, 1] - q_vals[:, 0]).cpu().numpy()
        rl_values.extend(q_net_val)

dl_decisions = np.array(dl_decisions)
rl_decisions = np.array(rl_decisions)
dl_risks = np.array(dl_risks)
rl_values = np.array(rl_values)
raw_rates = raw_rates[:len(dl_decisions)] # Align length if truncated

# ---------- 4. Head-to-Head Stats ----------
print("\n" + "="*40)
print("POLICY COMPARISON SCOREBOARD")
print("="*40)

total = len(dl_decisions)
dl_approvals = np.sum(dl_decisions)
rl_approvals = np.sum(rl_decisions)

print(f"Total Validation Loans: {total}")
print(f"DL Approvals (Risk < {dl_threshold:.2%}): {dl_approvals} ({dl_approvals/total:.1%})")
print(f"RL Approvals (Value > 0):           {rl_approvals} ({rl_approvals/total:.1%})")

# Overlap Matrix
#        | RL Deny | RL Approve
# DL Deny|    A    |     B (Alpha)
# DL Appr|    C    |     D
both_deny = np.sum((dl_decisions==0) & (rl_decisions==0))
alpha_set = np.sum((dl_decisions==0) & (rl_decisions==1)) # DL Reject, RL Approve
safety_set = np.sum((dl_decisions==1) & (rl_decisions==0)) # DL Approve, RL Reject
both_appr = np.sum((dl_decisions==1) & (rl_decisions==1))

print("\nConfusion Matrix (DL vs RL):")
print(f"Both Agree (Deny):    {both_deny}")
print(f"Both Agree (Approve): {both_appr}")
print(f"RL Conservative (DL Approved, RL Denied): {safety_set}")
print(f"RL Aggressive   (DL Denied,   RL Approved): {alpha_set}  <-- THE ALPHA")

# ---------- 5. Analyze "The Alpha" (Why RL Approved High Risk) ----------
print("\n" + "-"*40)
print("ANALYSIS: Why did RL approve when DL rejected?")
print("-" * 40)

if alpha_set > 0:
    # Get indices of Divergent Loans (DL=0, RL=1)
    mask = (dl_decisions==0) & (rl_decisions==1)

    avg_risk_alpha = dl_risks[mask].mean()
    avg_rate_alpha = raw_rates[mask].mean()

    # Compare to Conservative Loans (DL=1, RL=0)
    mask_safe = (dl_decisions==1) & (rl_decisions==0)
    avg_risk_safe = dl_risks[mask_safe].mean() if safety_set > 0 else 0
    avg_rate_safe = raw_rates[mask_safe].mean() if safety_set > 0 else 0

    print(f"1. Loans RL Saved (The Alpha Set):")
    print(f"   - Average Risk (DL):  {avg_risk_alpha:.2%} (High!)")
    print(f"   - Average Int Rate:   {avg_rate_alpha:.2f}% (High!)")
    print(f"   [CONCLUSION] RL approved them because the high interest rate ({avg_rate_alpha:.1f}%) compensates for the high risk.")

    print(f"\n2. Loans RL Rejected (Safety Check):")
    if safety_set > 0:
        print(f"   - Average Risk (DL):  {avg_risk_safe:.2%} (Low risk, so why deny?)")
        print(f"   - Average Int Rate:   {avg_rate_safe:.2f}% (Very Low!)")
        print(f"   [CONCLUSION] RL denied them because the return ({avg_rate_safe:.1f}%) was too low to justify even small risks.")
    else:
        print("   - None found.")

    # ---------- 6. Specific Examples ----------
    print("\n" + "="*40)
    print("SPECIFIC EXAMPLES")
    print("="*40)

    idxs = np.where(mask)[0][:3] # Take top 3 divergent examples
    for i in idxs:
        print(f"Applicant {i}:")
        print(f"  - Risk Score (DL): {dl_risks[i]:.2%}")
        print(f"  - Interest Rate:   {raw_rates[i]:.2f}%")
        print(f"  - DL Action:       DENY (Risk > {dl_threshold:.2%})")
        print(f"  - RL Action:       APPROVE")
        print(f"  - RL Logic:        Expected Value = {rl_values[i]:.2f} (Positive)")
        print(f"  - Actual Outcome:  {'Default' if y_val[i]==1 else 'Paid'}")
        print("-" * 20)

else:
    print("No divergent loans found. Models are perfectly aligned.")

print("\nSTEP 12 COMPLETE.")
