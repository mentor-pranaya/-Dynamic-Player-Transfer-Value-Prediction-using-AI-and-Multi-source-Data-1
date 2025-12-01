"""
XGBoost model for predicting Market Value

Steps:
1. Load final dataset
2. Use numeric columns as features
3. Target = Market Value
4. Time-based train-test split (no shuffle)
5. Train XGBoost regressor
6. Evaluate using RMSE and MAE
7. Save predictions and plot
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor

# 1. Load data
CSV_PATH = r"D:\INFOSYS\DATA\processed\final_afe_dataset.csv"
TARGET_COL = "Market Value"

if not os.path.exists(CSV_PATH):
    print("Dataset not found. Check CSV_PATH.")
    raise SystemExit

df = pd.read_csv(CSV_PATH)
print("Loaded dataset:", CSV_PATH)
print("Shape:", df.shape)

if TARGET_COL not in df.columns:
    print("Target column not found. Check TARGET_COL.")
    raise SystemExit

# 2. Build X and y
drop_cols = []
for c in ["player_name", "player_id", "date", "Unnamed: 0"]:
    if c in df.columns:
        drop_cols.append(c)

X_full = df.drop(columns=drop_cols + [TARGET_COL], errors="ignore")
X_full = X_full.select_dtypes(include=[np.number])

y_full = df[TARGET_COL].values

print("\nFeature matrix shape:", X_full.shape)
print("Target shape:", y_full.shape)

# 3. Train-test split 
X_train, X_test, y_train, y_test = train_test_split(
    X_full,
    y_full,
    test_size=0.2,
    shuffle=False
)

print("\nTrain shapes:", X_train.shape, y_train.shape)
print("Test shapes :", X_test.shape, y_test.shape)

# 4. Build XGBoost model
xgb_model = XGBRegressor(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)

print("\nTraining XGBoost model...")
xgb_model.fit(X_train, y_train)

# 5. Predictions & metrics
y_pred = xgb_model.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)

print("\nXGBoost Test RMSE:", rmse)
print("XGBoost Test MAE :", mae)

# 6. Save prediction file
results_df = pd.DataFrame({
    "y_true": y_test,
    "y_pred": y_pred
})
results_df.to_csv("xgb_preds.csv", index=False)
print("\nSaved: xgb_preds.csv")

# 7. Plot predictions vs true
plt.figure(figsize=(8, 4))
plt.plot(y_test, label="True Market Value")
plt.plot(y_pred, label="Predicted Market Value")
plt.title("XGBoost Predictions vs True Values")
plt.xlabel("Test sample index")
plt.ylabel("Market Value")
plt.legend()
plt.tight_layout()
plt.savefig("plot_xgb_pred_vs_true.png")
plt.show()

print("Saved plot: plot_xgb_pred_vs_true.png")
print("\n XGBoost model done.")
