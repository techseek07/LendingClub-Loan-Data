# LendingClub-Loan-Data

A reproducible end-to-end pipeline for building, calibrating, auditing, and explaining a leak-proof credit risk model on LendingClub data. The project is organized as a stepwise notebook/script pipeline covering sampling, preprocessing, feature engineering, exploratory analysis, training a deep-learning risk predictor (FT-Transformer), calibration and profit optimization, offline RL for decision policy, SHAP explainability, and policy divergence analysis.

---

## Quick summary of pipeline steps (high-level)

1. Step 1 — Sampling & Raw Preprocessing
   - Script: preprocessing.py
   - Purpose: Stream large accepted/rejected CSVs, sample to create a RAM-safe multi-year dataset, save processed parquet artifacts.

2. Step 2 — (Implicit/earlier) Feature engineering & rejection-rate features
   - Artifacts used in later steps: accepted_train_aug.parquet, accepted_val_aug.parquet, combined_train_raw.parquet, and computed stats (fico_stats, dti_stats, state_stats).

3. Step 3 — Exploratory Data Analysis (EDA)
   - Script: EDA.py
   - Purpose: Validate policy statistics, vintage stability, visualize distributions, interest-rate sensitivity (LOWESS), and risk heatmaps.

4. Step 6 — Create leak-free arrays for DL (not shown as a separate file but expected in artifacts)
   - Outputs: artifacts/dl_data.joblib (train/val/test arrays), artifacts/dl_metadata.json, artifacts/feature_names.json, artifacts/*_pre.parquet (train_pre.parquet, val_pre.parquet, test_pre.parquet)

5. Step 7 — Deep Learning model training & Step 8 — Calibration & Profit
   - Script: DL_model_and_callibration.py
   - Model: FT-Transformer-like architecture (FeatureTokenizer + TransformerEncoder)
   - Training: BCEWithLogitsLoss with pos_weight to handle class imbalance
   - Saves model to: artifacts/models/best_dl_model.pth
   - Calibration: Isotonic Regression; saves calibrator to artifacts/calibrator.joblib
   - Profit optimization: aligns predictions with raw business data, searches a threshold to maximize portfolio profit, saves calibration metadata to artifacts/calibration_metadata.json and a profit plot to artifacts/images/profit_threshold.png

6. Step 9 — Model audit & Baseline comparison
   - Script: model_audit.py
   - Purpose: Evaluate the DL model on the test set, compare to a logistic regression baseline (FICO + Grade)

7. Step 10 — Offline RL (aligned)
   - Script: Rl_model.py
   - Purpose: Create aligned reward signal (realized PnL), train a Q-network / contextual policy using the leak-free features and aligned financials
   - Saved RL artifacts: artifacts/rl_models/q_network_aligned.pth (expected)

8. Step 11 — Explainability (SHAP & XGBoost distillation)
   - Script: shap.py
   - Purpose: Distill Q-network strategy into an XGBoost model and produce SHAP visualizations; saves images to artifacts/images

9. Step 12 — Policy divergence & difference analysis
   - Script: difference.py
   - Purpose: Compare DL and RL decisions, find loans RL approves and DL rejects, analyze interest-rate patterns, produce explanations.

---

## Repository files (main scripts)
- preprocessing.py — multi-year sampling & basic streaming preprocessing from large CSVs.
- EDA.py — exploratory visual analyses and validation checks.
- DL_Dataset.py — utilities for building dataset arrays / dataset class used for DL (expected to support dl_data.joblib generation).
- DL_model_and_callibration.py — training FT-Transformer, calibration (isotonic), profit threshold search and saving model/metadata.
- model_audit.py — compares DL model to logistic baseline on test set.
- Rl_model.py — offline RL pipeline, PnL alignment and Q-network training.
- shap.py — XAI: distillation and SHAP visualizations.
- difference.py — policy divergence / head-to-head comparisons.

(If you maintain or rename files, update this list.)

---

## Expected inputs & outputs

Inputs (examples / default paths used by scripts)
- Accepted raw CSV: /content/a.csv
- Rejected raw CSV: /content/r.csv
  These are referenced in preprocessing.py — update paths as needed or supply via environment/drive mounting.

Key artifact outputs (under artifacts/):
- artifacts/raw_accepted.parquet
- artifacts/accepted_train_aug.parquet, accepted_val_aug.parquet
- artifacts/combined_train_raw.parquet
- artifacts/train_pre.parquet, val_pre.parquet, test_pre.parquet
- artifacts/dl_data.joblib — contains train/val/test arrays: X_num, X_cat, y
- artifacts/dl_metadata.json — metadata used to re-instantiate models (num_feat_count, cat_cardinalities, cat_feat_names)
- artifacts/feature_names.json — numeric feature names
- artifacts/models/best_dl_model.pth
- artifacts/calibrator.joblib
- artifacts/calibration_metadata.json
- artifacts/rl_models/q_network_aligned.pth (expected)
- artifacts/images/*.png — plots from EDA, calibration/profit, SHAP visuals

Important note on alignment:
- Several steps critically align processed arrays (dl arrays) with raw business records by using the exact indices (train_pre.parquet / val_pre.parquet). This is essential to avoid label or reward leakage when pairing predictions with financial columns (loan_amnt, int_rate, total_pymnt). Never reorder or reset indices without ensuring index alignment.

---

## Requirements

The code uses (non-exhaustive; pin versions as needed):
- Python 3.8+
- pandas, numpy
- pyarrow or fastparquet (for parquet I/O)
- scikit-learn
- torch (PyTorch) — GPU recommended for training
- joblib
- matplotlib, seaborn
- statsmodels (for LOWESS)
- xgboost, shap (for explainability)
- optionally: sklearn-isotonic (IsotonicRegression is in sklearn), and other common libs

Install example:
pip install -r requirements.txt
(If you don't have requirements.txt, create one with the packages above.)

---

## How to run (recommended order)

1. Prepare raw CSVs and update paths in preprocessing.py (ACCEPTED_PATH, REJECTED_PATH). Run:
   python preprocessing.py

2. Run feature engineering / Step 2 script (if provided — not explicitly named in files but implied). Ensure accepted_train_aug.parquet and other required artifacts are available.

3. Run EDA:
   python EDA.py
   - Produces image outputs under artifacts/images

4. Generate DL arrays (Step 6): run script that creates artifacts/dl_data.joblib and metadata (DL_Dataset.py or another step-equivalent). Example:
   python DL_Dataset.py
   - Ensures dl_data.joblib contains structured arrays for train/val/test.

5. Train DL model & calibrate:
   python DL_model_and_callibration.py
   - Produces best_dl_model.pth, calibrator.joblib, calibration metadata and a profit plot.

6. Audit models:
   python model_audit.py

7. Offline RL:
   python Rl_model.py

8. SHAP explainability:
   python shap.py

9. Policy divergence:
   python difference.py

Note: Each script expects artifacts from previous steps — run in order or ensure artifacts are available.

---

## Model details (summary)

- Deep learning predictor: FT-Transformer-like architecture implemented with a FeatureTokenizer that:
  - Encodes numeric features through learned per-feature linear tokens
  - Embeds categorical features with Embedding layers
  - Prepends a CLS token, passes tokens through PyTorch TransformerEncoder, and uses the CLS output with a small head to predict logit for default probability.
- Training uses BCEWithLogitsLoss and class pos_weight to counter imbalance.
- Calibration uses Isotonic Regression for reliable probability outputs.
- Profit optimization maps calibrated probabilities to an approval decision and computes portfolio profit assuming:
  - If approved and paid: loan_amount * int_rate
  - If approved and default: full loss of principal (baseline 100% LGD assumed)
  - Threshold search across risk cutoffs to maximize total profit.

- RL agent: Q-network based on the same tokenization pattern, trained with aligned realized PnL as reward and small opportunity-cost penalties for denying.

- Explainability: Distillation into XGBoost + SHAP to interpret which features drive approvals and where RL differs from DL.

---

## Reproducibility & device

- Many scripts set random seeds (e.g., 42) for numpy and torch.
- Training will use GPU when available; device detection performed in scripts. For CPU-only runs, training will be slower.


---

## Contact / Author

Repository owner: techseek07 (see the repository for contact details). For questions about assumptions (e.g., LGD, interest handling, or index alignment), open an issue with details and sample traces of the artifact files.

---

## Acknowledgements & references

- Uses typical ML & DL libraries (PyTorch, scikit-learn, XGBoost, SHAP).
- The FT-Transformer architecture is inspired by tabular-transformer approaches (tokenizing per-feature and using a transformer encoder).

---


```
