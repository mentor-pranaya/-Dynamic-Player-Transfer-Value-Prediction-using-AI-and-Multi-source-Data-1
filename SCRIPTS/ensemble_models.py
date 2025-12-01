"""
Week-6 Ensemble Models

Goal:
1. Load main dataset + LSTM predictions
2. Train:
   - Plain XGBoost (no LSTM feature)
   - Ensemble XGBoost (with LSTM prediction as extra feature)
3. Compare RMSE/MAE of:
   - LSTM alone
   - XGBoost
   - Ensemble (LSTM + XGBoost)
4. Save comparison report.
"""

import os
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
import matplotlib.pyplot as plt

# ===========================
# 1. Load data + LSTM preds
# ===========================
CSV_PATH = r"D:\INFOSYS\DATA\processed\final_afe_dataset.csv"
LSTM_PRED_PATH = r"D:\INFOSYS\scripts\lstm_multivariate_preds.csv"  # change if file name different
TARGET_COL = "Market Value"

if not os.path.exists(CSV_PATH):
    print("Dataset not found. Check CSV_PATH.")
    raise SystemExit

if not os.path.exists(LSTM_PRED_PATH):
    print("LSTM prediction file not found. Check LSTM_PRED_PATH.")
    raise SystemExit

df = pd.read_csv(CSV_PATH)
lstm_df = pd.read_csv(LSTM_PRED_PATH)

print("Main data shape:", df.shape)
print("LSTM preds shape:", lstm_df.shape)

# Add row_index to main df for merging
df["row_index"] = df.index

# We only need row_index + LSTM predicted value
if "y_pred" not in lstm_df.columns:
    print("Column 'y_pred' not in LSTM file. Check file content.")
    raise SystemExit

lstm_df = lstm_df[["row_index", "y_pred"]].rename(columns={"y_pred": "lstm_pred"})

# Merge LSTM predictions into main dataframe
merged = df.merge(lstm_df, on="row_index", how="inner")

print("Merged shape (only rows with LSTM preds):", merged.shape)

if TARGET_COL not in merged.columns:
    print("Target column missing after merge.")
    raise SystemExit

# ===========================
# 2. Build features (with & without LSTM)
# ===========================

# Columns to drop from features
drop_cols = []
for c in ["player_name", "player_id", "date", "Unnamed: 0"]:
    if c in merged.columns:
        drop_cols.append(c)

# Base numeric features from merged df
numeric_cols = merged.select_dtypes(include=[np.number]).columns.tolist()

# Remove target and row_index from feature list
for c in [TARGET_COL, "row_index"]:
    if c in numeric_cols:
        numeric_cols.remove(c)

# X with LSTM feature (ensemble)
X_with_lstm = merged[numeric_cols].copy()  # this already includes 'lstm_pred' because it's numeric
y = merged[TARGET_COL].values

print("\nTotal numeric features (including lstm_pred):", X_with_lstm.shape[1])

# X without LSTM feature (plain XGBoost baseline)
if "lstm_pred" in X_with_lstm.columns:
    X_without_lstm = X_with_lstm.drop(columns=["lstm_pred"])
else:
    print("Warning: 'lstm_pred' not in feature matrix; ensemble may not be using LSTM properly.")
    X_without_lstm = X_with_lstm.copy()

# ===========================
# 3. Train-test split (no shuffle)
# ===========================
X_train_plain, X_test_plain, y_train, y_test = train_test_split(
    X_without_lstm, y, test_size=0.2, shuffle=False
)

X_train_ens, X_test_ens, _, _ = train_test_split(
    X_with_lstm, y, test_size=0.2, shuffle=False
)

print("\nPlain XGBoost shapes:")
print("X_train_plain:", X_train_plain.shape, "X_test_plain:", X_test_plain.shape)

print("\nEnsemble XGBoost shapes:")
print("X_train_ens:", X_train_ens.shape, "X_test_ens:", X_test_ens.shape)

# ===========================
# 4. Plain XGBoost (no LSTM feature)
# ===========================
xgb_plain = XGBRegressor(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)

print("\nTraining plain XGBoost...")
xgb_plain.fit(X_train_plain, y_train)
y_pred_plain = xgb_plain.predict(X_test_plain)

rmse_plain = np.sqrt(mean_squared_error(y_test, y_pred_plain))
mae_plain = mean_absolute_error(y_test, y_pred_plain)

print("Plain XGBoost RMSE:", rmse_plain)
print("Plain XGBoost MAE :", mae_plain)

# ===========================
# 5. Ensemble XGBoost (with LSTM pred feature)
# ===========================
xgb_ens = XGBRegressor(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)

print("\nTraining Ensemble XGBoost (with LSTM feature)...")
xgb_ens.fit(X_train_ens, y_train)
y_pred_ens = xgb_ens.predict(X_test_ens)

rmse_ens = np.sqrt(mean_squared_error(y_test, y_pred_ens))
mae_ens = mean_absolute_error(y_test, y_pred_ens)

print("Ensemble XGBoost RMSE:", rmse_ens)
print("Ensemble XGBoost MAE :", mae_ens)

# ===========================
# 6. LSTM-only RMSE on same test rows
# ===========================
# For LSTM, we use 'lstm_pred' from X_test_ens as prediction
if "lstm_pred" in X_test_ens.columns:
    lstm_test_pred = X_test_ens["lstm_pred"].values
    rmse_lstm = np.sqrt(mean_squared_error(y_test, lstm_test_pred))
    mae_lstm = mean_absolute_error(y_test, lstm_test_pred)
else:
    rmse_lstm = np.nan
    mae_lstm = np.nan
    print("Warning: lstm_pred missing; cannot compute LSTM-only metrics correctly.")

print("\nLSTM-only RMSE:", rmse_lstm)
print("LSTM-only MAE :", mae_lstm)

# ===========================
# 7. Save comparison report
# ===========================
results = pd.DataFrame({
    "Model": ["LSTM-only", "XGBoost", "Ensemble (LSTM + XGBoost)"],
    "RMSE": [rmse_lstm, rmse_plain, rmse_ens],
    "MAE": [mae_lstm, mae_plain, mae_ens],
})

results.to_csv("week6_ensemble_results.csv", index=False)
print("\nSaved: week6_ensemble_results.csv")

# ===========================
# 8. Simple RMSE bar plot
# ===========================
plt.figure(figsize=(6, 4))
plt.bar(results["Model"], results["RMSE"])
plt.title("Model RMSE Comparison")
plt.ylabel("RMSE (lower is better)")
plt.xticks(rotation=15)
plt.tight_layout()
plt.savefig("plot_week6_rmse_comparison.png")
plt.show()

print("Saved: plot_week6_rmse_comparison.png")
print("\n Week-6 Ensemble modeling completed.")
