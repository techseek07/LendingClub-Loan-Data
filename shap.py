# ---------------------- STEP 11 (FIXED): EXPLAINABLE AI (XAI) ----------------------
# Fixes:
# 1. Loads the correct model file ('q_network_aligned.pth').
# 2. Distills the complex Q-Network into an interpretable XGBoost model.
# 3. Uses SHAP to visualize the "Strategy" (e.g., Interest Rate vs Risk).

import os, json, joblib, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import xgboost as xgb
import shap

warnings.filterwarnings("ignore")
ARTIFACT_DIR = Path("artifacts")
IMG_DIR = ARTIFACT_DIR / "images"
IMG_DIR.mkdir(exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"STEP 11: Model Explainability on {DEVICE}")

# ---------- 1. Load Resources ----------
print("Loading Models & Data...")

# Data
dl_data = joblib.load(ARTIFACT_DIR / "dl_data.joblib")
X_num_test = dl_data['test']['X_num']
X_cat_test = dl_data['test']['X_cat']

# Metadata & Feature Names
with open(ARTIFACT_DIR / "dl_metadata.json", "r") as f:
    dl_meta = json.load(f)
with open(ARTIFACT_DIR / "feature_names.json", "r") as f:
    numeric_features = json.load(f)
cat_features = dl_meta['cat_feat_names']
all_features = numeric_features + cat_features

# Re-define RL Architecture (Must match Step 10)
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

# Load Agent Weights (FIXED FILENAME HERE)
q_net = QNetwork(
    num_numeric=dl_meta['num_feat_count'],
    cat_cardinalities=dl_meta['cat_cardinalities']
).to(DEVICE)

# Try loading the correct file from Step 10
model_path = ARTIFACT_DIR / "rl_models/q_network_aligned.pth"
if not model_path.exists():
    # Fallback if you ran an older version
    model_path = ARTIFACT_DIR / "rl_models/q_network.pth"

print(f"Loading weights from: {model_path}")
q_net.load_state_dict(torch.load(model_path, map_location=DEVICE))
q_net.eval()

# ---------- 2. Generate "Teacher" Predictions (Q-Values) ----------
print("Querying RL Agent for Q-Values on Test Set...")

batch_size = 1024
q_approve_preds = []

with torch.no_grad():
    for i in range(0, len(X_num_test), batch_size):
        xn = torch.tensor(X_num_test[i:i+batch_size], dtype=torch.float32).to(DEVICE)
        xc = torch.tensor(X_cat_test[i:i+batch_size], dtype=torch.long).to(DEVICE)

        # Get Q-Values
        q_vals = q_net(xn, xc)
        # Explain the NET VALUE: Q(Approve) - Q(Deny)
        # Positive = Approve, Negative = Deny
        net_value = q_vals[:, 1] - q_vals[:, 0]
        q_approve_preds.extend(net_value.cpu().numpy())

q_approve_preds = np.array(q_approve_preds)

# ---------- 3. Distillation (Train XGBoost Surrogate) ----------
print("Training Surrogate XGBoost to mimic RL Logic...")

X_distill = np.hstack([X_num_test, X_cat_test])
df_distill = pd.DataFrame(X_distill, columns=all_features)

surrogate = xgb.XGBRegressor(n_estimators=100, max_depth=4, learning_rate=0.1, n_jobs=-1)
surrogate.fit(df_distill, q_approve_preds)

r2 = surrogate.score(df_distill, q_approve_preds)
print(f"Surrogate Fidelity (R^2): {r2:.4f}")

# ---------- 4. SHAP Analysis (The "Why") ----------
print("Calculating SHAP Values...")

explainer = shap.TreeExplainer(surrogate)
# Sample 2000 points for speed
sample_idx = np.random.choice(len(df_distill), size=min(2000, len(df_distill)), replace=False)
X_sample = df_distill.iloc[sample_idx]
shap_values = explainer.shap_values(X_sample)

# A. Global Importance Plot
plt.figure(figsize=(10, 8))
shap.summary_plot(shap_values, X_sample, show=False, max_display=15)
plt.title("What drives the RL Agent's Value Calculation?")
plt.savefig(IMG_DIR / "rl_shap_summary.png", bbox_inches='tight')
print("Saved: rl_shap_summary.png")

# ---------- 5. The "Alpha" Analysis (Interest Rate Impact) ----------
print("\nAnalyzing Specific Drivers...")

# Check the impact of Interest Rate specifically
if 'int_rate_val' in X_sample.columns:
    int_rate_idx = X_sample.columns.get_loc('int_rate_val')
    int_rate_shap = shap_values[:, int_rate_idx]

    avg_impact = np.mean(int_rate_shap)
    print(f"Average SHAP Impact of Interest Rate: {avg_impact:.4f}")

    if avg_impact > 0:
        print("[INSIGHT] Positive Impact: Higher Interest Rates generally increase the Agent's willingness to approve.")
        print("This confirms the agent is 'Yield Hunting' (Risk-Adjusted Return).")
    else:
        print("[INSIGHT] Negative/Neutral Impact: The agent is treating Rate purely as a Risk signal, not a Reward signal.")

print("\nSTEP 11 COMPLETE.")
