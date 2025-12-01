"""
week7_xgb_tuning.py
Small randomized hyperparameter tuning for XGBoost using TimeSeriesSplit.

"""

import os
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit, train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor
from scipy.stats import uniform, randint

# Paths
CSV_PATH = r"D:\INFOSYS\DATA\processed\final_afe_dataset.csv"
OUT_DIR = r"D:\INFOSYS\scripts"
os.makedirs(OUT_DIR, exist_ok=True)
TARGET_COL = "Market Value"

# Load dataset
df = pd.read_csv(CSV_PATH)
df["row_index"] = df.index

# Build numeric feature matrix (same way used earlier)
drop_cols = []
for c in ["player_name", "player_id", "date", "Unnamed: 0"]:
    if c in df.columns:
        drop_cols.append(c)
X_all = df.drop(columns=drop_cols + [TARGET_COL], errors="ignore").select_dtypes(include=[np.number])
y_all = df[TARGET_COL].values

# Train/test split (time-order)
TEST_SIZE = 0.20
X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, test_size=TEST_SIZE, shuffle=False)

print("X_train:", X_train.shape, "X_test:", X_test.shape)

# Baseline model (i have already trained before, but i need to show baseline again)
base = XGBRegressor(objective="reg:squarederror", random_state=42, n_estimators=300, learning_rate=0.05, max_depth=6)
base.fit(X_train, y_train)
y_base = base.predict(X_test)
rmse_base = np.sqrt(mean_squared_error(y_test, y_base))
mae_base = mean_absolute_error(y_test, y_base)
r2_base = r2_score(y_test, y_base)
print(f"Baseline XGB -> RMSE: {rmse_base:.6f}, MAE: {mae_base:.6f}, R2: {r2_base:.6f}")

# Randomized search (small)
param_dist = {
    "n_estimators": [100, 200, 300],
    "max_depth": [3, 5, 6],
    "learning_rate": [0.03, 0.05, 0.1],
    "subsample": [0.7, 0.8, 1.0],
    "colsample_bytree": [0.6, 0.8, 1.0]
}

tscv = TimeSeriesSplit(n_splits=3)

rnd = RandomizedSearchCV(
    estimator=XGBRegressor(objective="reg:squarederror", random_state=42),
    param_distributions=param_dist,
    n_iter=6,                # small number to be fast
    cv=tscv,
    scoring="neg_mean_squared_error",
    verbose=1,
    n_jobs=-1,
    random_state=42
)

print("Starting RandomizedSearchCV (small)...")
rnd.fit(X_train, y_train)

best_params = rnd.best_params_
print("Best params:", best_params)

# Save best model
best_xgb = rnd.best_estimator_
joblib.dump(best_xgb, os.path.join(OUT_DIR, "xgb_tuned.pkl"))
print("Saved tuned model: xgb_tuned.pkl")

# Evaluate tuned model on test set
y_tuned = best_xgb.predict(X_test)
rmse_tuned = np.sqrt(mean_squared_error(y_test, y_tuned))
mae_tuned = mean_absolute_error(y_test, y_tuned)
r2_tuned = r2_score(y_test, y_tuned)
print(f"Tuned XGB -> RMSE: {rmse_tuned:.6f}, MAE: {mae_tuned:.6f}, R2: {r2_tuned:.6f}")

# Save results summary
cv_results = pd.DataFrame(rnd.cv_results_)[[
    "params", "mean_test_score", "std_test_score", "rank_test_score"
]]
cv_results["mean_rmse"] = np.sqrt(-cv_results["mean_test_score"])
cv_results.to_csv(os.path.join(OUT_DIR, "week7_xgb_cv_results.csv"), index=False)
print("Saved:", os.path.join(OUT_DIR, "week7_xgb_cv_results.csv"))

# Quick plot of mean RMSE for tested candidates
plt.figure(figsize=(8,4))
vals = cv_results["mean_rmse"].values
labels = [f"p{idx+1}" for idx in range(len(vals))]
plt.bar(labels, vals)
plt.title("XGBoost RandomSearch mean RMSE (cv)")
plt.ylabel("RMSE")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "plot_xgb_search_rmse.png"))
plt.show()
print("Saved plot_xgb_search_rmse.png")

# Save final metrics CSV
summary = pd.DataFrame({
    "Model": ["XGB-base", "XGB-tuned"],
    "RMSE": [rmse_base, rmse_tuned],
    "MAE": [mae_base, mae_tuned],
    "R2": [r2_base, r2_tuned]
})
summary.to_csv(os.path.join(OUT_DIR, "week7_xgb_summary.csv"), index=False)
print("Saved week7_xgb_summary.csv")

print("Week-7 XGBoost tuning completed.")
