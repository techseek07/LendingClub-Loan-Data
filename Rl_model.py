# ---------------------- STEP 10 (FINAL ROBUST + OPPORTUNITY COST): OFFLINE RL WITH ALIGNMENT ----------------------
# Goals:
# 1. Load Safe Features (State) from Step 6.
# 2. Load Raw Financials (Reward) and ALIGN them perfectly using indices.
# 3. Define Exact PnL Reward: (Total Payment - Funded Amount).
# 4. Add small opportunity-cost penalty for denying (lazy capital).
# 5. Train a Q-network (contextual bandit) to predict this value.

import os, json, joblib, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

warnings.filterwarnings("ignore")
ARTIFACT_DIR = Path("artifacts")
RL_DIR = ARTIFACT_DIR / "rl_models"
RL_DIR.mkdir(exist_ok=True)

# Reproducibility
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"STEP 10: Advanced Offline RL on {DEVICE}")

# ---------- 1. Load State Space (Safe Inputs) ----------
print("Loading Feature Inputs (Step 6)...")
dl_data = joblib.load(ARTIFACT_DIR / "dl_data.joblib")

X_num_train = dl_data['train']['X_num']
X_cat_train = dl_data['train']['X_cat']
X_num_val   = dl_data['val']['X_num']
X_cat_val   = dl_data['val']['X_cat']

with open(ARTIFACT_DIR / "dl_metadata.json", "r") as f:
    dl_meta = json.load(f)

# ---------- 2. Load & Align Reward Space (CRITICAL) ----------
print("Loading and Aligning Raw Financials...")

# A. Use processed data to get the exact index set used for DL
train_pre = pd.read_parquet(ARTIFACT_DIR / "train_pre.parquet")
val_pre   = pd.read_parquet(ARTIFACT_DIR / "val_pre.parquet")

# B. Load raw augmented data from Step 2
train_raw_full = pd.read_parquet(ARTIFACT_DIR / "accepted_train_aug.parquet")
val_raw_full   = pd.read_parquet(ARTIFACT_DIR / "accepted_val_aug.parquet")

# C. Align by index so rows match X_num/X_cat and labels
train_raw = train_raw_full.loc[train_pre.index].copy()
val_raw   = val_raw_full.loc[val_pre.index].copy()

print(f"Aligned Train: {len(train_raw)} rows (from {len(train_raw_full)} raw)")
print(f"Aligned Val:   {len(val_raw)} rows (from {len(val_raw_full)} raw)")

def calculate_realized_pnl(df: pd.DataFrame) -> np.ndarray:
    # Investment (Money Out)
    if 'funded_amnt' in df.columns:
        inv = pd.to_numeric(df['funded_amnt'], errors='coerce').fillna(0)
    else:
        inv = pd.to_numeric(df['loan_amnt'], errors='coerce').fillna(0)
    # Return (Money In) = total_pymnt (principal + interest + fees + recoveries)
    ret = pd.to_numeric(df['total_pymnt'], errors='coerce').fillna(0)
    return (ret - inv).values

pnl_train = calculate_realized_pnl(train_raw)
pnl_val   = calculate_realized_pnl(val_raw)

# Strict safety checks
assert len(pnl_train) == len(X_num_train), "Train alignment failed!"
assert len(pnl_val)   == len(X_num_val),   "Val alignment failed!"

# ---------- 3. Construct Reward Matrix ----------
print("Constructing RL Training Targets...")

REWARD_SCALE = 10000.0        # Scale PnL to roughly [-1, 1] range
DENY_PENALTY = -0.01          # Small opportunity-cost penalty for denying (scaled units)

# Action 1: Approve -> Reward is realized PnL (scaled)
r_approve_train = pnl_train / REWARD_SCALE

# Action 0: Deny -> Small negative reward (lazy capital penalty)
r_deny_train = np.full_like(r_approve_train, DENY_PENALTY)

print(f"Avg PnL per Loan (Train): ${pnl_train.mean():,.2f}")

# ---------- 4. Define Q-Network ----------
class QNetwork(nn.Module):
    def __init__(self, num_numeric, cat_cardinalities, d_token=64):
        super().__init__()
        self.num_weights = nn.Parameter(torch.randn(num_numeric, d_token))
        self.num_bias = nn.Parameter(torch.randn(num_numeric, d_token))
        self.cat_embeddings = nn.ModuleList(
            [nn.Embedding(c, d_token) for c in cat_cardinalities]
        )
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_token))

        input_dim = (1 + num_numeric + len(cat_cardinalities)) * d_token

        self.head = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 2)  # [Q_deny, Q_approve]
        )

    def forward(self, x_num, x_cat):
        B = x_num.shape[0]
        x_num_emb = x_num.unsqueeze(-1) * self.num_weights.unsqueeze(0) + self.num_bias.unsqueeze(0)
        x_cat_emb = torch.stack(
            [emb(x_cat[:, i]) for i, emb in enumerate(self.cat_embeddings)],
            dim=1
        )
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x_num_emb, x_cat_emb], dim=1)
        x = x.view(B, -1)
        return self.head(x)

q_net = QNetwork(
    num_numeric=dl_meta['num_feat_count'],
    cat_cardinalities=dl_meta['cat_cardinalities']
).to(DEVICE)

optimizer = optim.Adam(q_net.parameters(), lr=1e-4)
# SmoothL1Loss (Huber) is more robust than plain MSE for heavy-tailed PnL
loss_fn = nn.SmoothL1Loss()

# ---------- 5. Training Loop ----------
class OfflineRLDataset(Dataset):
    def __init__(self, x_num, x_cat, r_deny, r_approve):
        self.x_num = torch.tensor(x_num, dtype=torch.float32)
        self.x_cat = torch.tensor(x_cat, dtype=torch.long)
        self.r_deny = torch.tensor(r_deny, dtype=torch.float32)
        self.r_approve = torch.tensor(r_approve, dtype=torch.float32)

    def __len__(self):
        return len(self.r_approve)

    def __getitem__(self, i):
        return self.x_num[i], self.x_cat[i], self.r_deny[i], self.r_approve[i]

train_loader = DataLoader(
    OfflineRLDataset(X_num_train, X_cat_train, r_deny_train, r_approve_train),
    batch_size=256, shuffle=True
)

EPOCHS = 15
print(f"\nTraining Agent for {EPOCHS} epochs...")

for epoch in range(EPOCHS):
    q_net.train()
    total_loss = 0.0

    for xn, xc, r_deny, r_app in train_loader:
        xn, xc = xn.to(DEVICE), xc.to(DEVICE)
        r_deny, r_app = r_deny.to(DEVICE), r_app.to(DEVICE)

        optimizer.zero_grad()
        q_preds = q_net(xn, xc)                  # [batch, 2]
        q_targets = torch.stack([r_deny, r_app], dim=1)
        loss = loss_fn(q_preds, q_targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    print(f"Epoch {epoch+1} | Loss: {total_loss / len(train_loader):.5f}")

torch.save(q_net.state_dict(), RL_DIR / "q_network_aligned.pth")

# ---------- 6. Head-to-Head Evaluation ----------
print("\nComparing Policies on Validation Set...")

# A. Load DL threshold profit from Step 8
with open(ARTIFACT_DIR / "calibration_metadata.json", "r") as f:
    calib_meta = json.load(f)
dl_max_profit = calib_meta['max_profit']

# B. Evaluate RL Policy on Val
q_net.eval()
rl_decisions = []

class ValDataset(Dataset):
    def __init__(self, x_num, x_cat):
        self.x_num = torch.tensor(x_num, dtype=torch.float32)
        self.x_cat = torch.tensor(x_cat, dtype=torch.long)

    def __len__(self):
        return len(self.x_num)

    def __getitem__(self, i):
        return self.x_num[i], self.x_cat[i]

val_loader = DataLoader(
    ValDataset(X_num_val, X_cat_val),
    batch_size=1024, shuffle=False
)

with torch.no_grad():
    for xn, xc in val_loader:
        xn, xc = xn.to(DEVICE), xc.to(DEVICE)
        q_values = q_net(xn, xc)                # [batch, 2]
        # Argmax over [Q_deny, Q_approve].
        # With DENY_PENALTY < 0 and Q_approve ~ expected scaled PnL,
        # this approximates approving whenever expected PnL > DENY_PENALTY.
        actions = torch.argmax(q_values, dim=1).cpu().numpy()
        rl_decisions.extend(actions)

rl_decisions = np.array(rl_decisions)

# C. Compute RL portfolio profit using UNscaled PnL on val
rl_portfolio_value = np.sum(np.where(rl_decisions == 1, pnl_val, 0.0))

print("-" * 40)
print("FINAL RESULTS")
print("-" * 40)
print(f"1. Deep Learning (Risk Threshold {calib_meta['optimal_threshold']:.1%}):")
print(f"   Profit: ${dl_max_profit:,.0f}")
print("-" * 40)
print("2. Reinforcement Learning (Q-Maximization + Opportunity Cost):")
print(f"   Approvals: {np.sum(rl_decisions)} loans ({np.mean(rl_decisions):.1%})")
print(f"   Profit:    ${rl_portfolio_value:,.0f}")
print("-" * 40)

if rl_portfolio_value > dl_max_profit:
    print("🏆 WINNER: Reinforcement Learning")
    print(f"Lift: +${rl_portfolio_value - dl_max_profit:,.0f}")
else:
    print("🏆 WINNER: Deep Learning Threshold")
    print(f"Gap: -${dl_max_profit - rl_portfolio_value:,.0f}")

print("\nSTEP 10 COMPLETE.")
