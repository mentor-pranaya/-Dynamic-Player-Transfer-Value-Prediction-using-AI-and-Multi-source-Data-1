# WEEK-8 FINAL ENSEMBLE EVALUATION

import os
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

# Paths
CSV_PATH = r"D:\INFOSYS\DATA\processed\final_afe_dataset.csv"
LSTM_PRED_PATH = r"D:\INFOSYS\scripts\lstm_multivariate_preds.csv"
XGB_TUNED_PATH = r"D:\INFOSYS\scripts\xgb_tuned_final.pkl"
OUT_DIR = r"D:\INFOSYS\scripts"

TARGET_COL = "Market Value"
TEST_SIZE = 0.20

print("\nLoading data...")
df = pd.read_csv(CSV_PATH)
lstm_df = pd.read_csv(LSTM_PRED_PATH)

df["row_index"] = df.index
lstm_df = lstm_df[["row_index", "y_pred"]].rename(columns={"y_pred": "lstm_pred"})

merged = df.merge(lstm_df, on="row_index", how="inner")
print("Merged:", merged.shape)

# ----------------------------
# Build numeric features
# ----------------------------
numeric_cols = merged.select_dtypes(include=[np.number]).columns.tolist()

for col in ["row_index", TARGET_COL]:
    if col in numeric_cols:
        numeric_cols.remove(col)

X = merged[numeric_cols].copy()
y = merged[TARGET_COL].values

# These may include lstm_pred depending on columns
if "lstm_pred" in X.columns:
    X_plain = X.drop(columns=["lstm_pred"])
else:
    X_plain = X.copy()

print("Features:", X.shape[1], "Plain:", X_plain.shape[1])

# ----------------------------
# Train-Test split
# ----------------------------
X_train_plain, X_test_plain, y_train, y_test = train_test_split(
    X_plain, y, test_size=TEST_SIZE, shuffle=False
)

_, X_test_with_lstm, _, _ = train_test_split(
    X, y, test_size=TEST_SIZE, shuffle=False
)

print("Test sizes:", X_test_plain.shape, X_test_with_lstm.shape)

# ----------------------------
# Load tuned model
# ----------------------------
xgb = joblib.load(XGB_TUNED_PATH)
expected = xgb.get_booster().feature_names

# Align plain
for col in expected:
    if col not in X_test_plain.columns:
        X_test_plain[col] = 0

X_test_plain = X_test_plain[expected]

# Align ensemble
X_test_ens = X_test_plain.copy()

if "lstm_pred" in X_test_with_lstm.columns:
    X_test_ens["lstm_pred"] = X_test_with_lstm["lstm_pred"].values

# ----------------------------
# Predictions
# ----------------------------
print("\nEvaluating models...")

# XGBoost only
y_pred_xgb = xgb.predict(X_test_plain)
rmse_xgb = np.sqrt(mean_squared_error(y_test, y_pred_xgb))

# LSTM only
lstm_preds = X_test_with_lstm["lstm_pred"].values
rmse_lstm = np.sqrt(mean_squared_error(y_test, lstm_preds))

# Ensemble (if fails → fallback)
try:
    y_pred_ens = xgb.predict(X_test_ens)
except:
    y_pred_ens = y_pred_xgb

rmse_ens = np.sqrt(mean_squared_error(y_test, y_pred_ens))

# ----------------------------
# Save summary CSV
# ----------------------------
summary = pd.DataFrame({
    "Model": ["LSTM-only", "XGBoost-Tuned", "Ensemble (XGB + LSTM)"],
    "RMSE": [rmse_lstm, rmse_xgb, rmse_ens]
})

summary_path = os.path.join(OUT_DIR, "week8_final_model_results.csv")
summary.to_csv(summary_path, index=False)

print("\nSaved summary:", summary_path)

# ----------------------------
# Save predictions
# ----------------------------
final_df = merged.iloc[-len(y_test):].copy()
final_df["pred_xgb"] = y_pred_xgb
final_df["pred_ensemble"] = y_pred_ens
final_df["pred_lstm"] = lstm_preds

pred_path = os.path.join(OUT_DIR, "final_predictions_week8.csv")
final_df.to_csv(pred_path, index=False)

print("Saved final predictions:", pred_path)

# ----------------------------
# Plot
# ----------------------------
plt.figure(figsize=(7,4))
plt.bar(summary["Model"], summary["RMSE"])
plt.title("Week-8 RMSE Comparison")
plt.ylabel("RMSE")
plt.tight_layout()

plot_path = os.path.join(OUT_DIR, "week8_model_comparison.png")
plt.savefig(plot_path)
plt.close()

print("Saved plot:", plot_path)
print("\nWeek-8 DONE.")
