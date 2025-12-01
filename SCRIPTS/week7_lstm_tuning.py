"""
week7_lstm_tuning.py
Small grid search for LSTM hyperparameters.
"""

import os
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

# Paths
CSV_PATH = r"D:\INFOSYS\DATA\processed\final_afe_dataset.csv"
OUT_DIR = r"D:\INFOSYS\scripts"
os.makedirs(OUT_DIR, exist_ok=True)
TARGET_COL = "Market Value"

# Load dataset
df = pd.read_csv(CSV_PATH)

# Build numeric features and target
drop_cols = []
for c in ["player_name", "player_id", "date", "Unnamed: 0"]:
    if c in df.columns:
        drop_cols.append(c)

X_full = df.drop(columns=drop_cols + [TARGET_COL], errors="ignore").select_dtypes(include=[np.number])
y_full = df[TARGET_COL].values.reshape(-1, 1)

# scale
scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()
X_scaled = scaler_X.fit_transform(X_full)
y_scaled = scaler_y.fit_transform(y_full).squeeze()

# sequence maker (same as earlier)
SEQ_LEN = 8
def make_sequences(X, y, seq_len):
    X_list, y_list = [], []
    for i in range(len(X) - seq_len):
        X_list.append(X[i:i+seq_len])
        y_list.append(y[i+seq_len])
    return np.array(X_list), np.array(y_list)

X_seq, y_seq = make_sequences(X_scaled, y_scaled, SEQ_LEN)
print("Built sequences:", X_seq.shape, y_seq.shape)

# train/test split (time order)
n_samples = len(X_seq)
n_test = int(n_samples * 0.2)
n_train = n_samples - n_test

X_train = X_seq[:n_train]
X_test = X_seq[n_train:]
y_train = y_seq[:n_train]
y_test = y_seq[n_train:]

print("Train:", X_train.shape, "Test:", X_test.shape)

# small grid
units_list = [32, 64]
drop_list = [0.1, 0.2]
lr_list = [0.001, 0.005]
batch_size = 32
epochs = 30   # early stopping will stop earlier often

results = []
best_rmse = 1e9
best_model = None
best_config = None

def build_model(seq_len, n_feat, units=64, dropout=0.2, lr=0.001):
    model = Sequential()
    model.add(Input(shape=(seq_len, n_feat)))
    model.add(LSTM(units, return_sequences=True))
    model.add(Dropout(dropout))
    model.add(LSTM(units//2, return_sequences=False))
    model.add(Dropout(dropout))
    model.add(Dense(16, activation="relu"))
    model.add(Dense(1, activation="linear"))
    opt = tf.keras.optimizers.Adam(learning_rate=lr)
    model.compile(optimizer=opt, loss="mse", metrics=["mae"])
    return model

n_feat = X_train.shape[2]
for units in units_list:
    for dropout in drop_list:
        for lr in lr_list:
            print(f"\nTrying config units={units}, dropout={dropout}, lr={lr}")
            model = build_model(SEQ_LEN, n_feat, units=units, dropout=dropout, lr=lr)
            es = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
            history = model.fit(
                X_train, y_train,
                validation_data=(X_test, y_test),
                epochs=epochs,
                batch_size=batch_size,
                callbacks=[es],
                verbose=0
            )
            # predict & inverse scale
            y_pred_scaled = model.predict(X_test).squeeze()
            y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1,1)).squeeze()
            y_true = scaler_y.inverse_transform(y_test.reshape(-1,1)).squeeze()

            rmse = np.sqrt(mean_squared_error(y_true, y_pred))
            mae = mean_absolute_error(y_true, y_pred)
            r2 = r2_score(y_true, y_pred)
            print(f"--> RMSE: {rmse:.6f}, MAE: {mae:.6f}, R2: {r2:.6f}")

            results.append({
                "units": units, "dropout": dropout, "lr": lr, "rmse": rmse, "mae": mae, "r2": r2
            })

            # save best
            if rmse < best_rmse:
                best_rmse = rmse
                best_model = model
                best_config = {"units": units, "dropout": dropout, "lr": lr}

# Save results dataframe
res_df = pd.DataFrame(results)
res_df.to_csv(os.path.join(OUT_DIR, "week7_lstm_tuning_results.csv"), index=False)
print("Saved week7_lstm_tuning_results.csv")

# Save best model and scaler objects
if best_model is not None:
    best_model.save(os.path.join(OUT_DIR, "best_lstm_model.keras"))
    joblib.dump(scaler_X, os.path.join(OUT_DIR, "scaler_X_for_lstm.pkl"))
    joblib.dump(scaler_y, os.path.join(OUT_DIR, "scaler_y_for_lstm.pkl"))
    print("Saved best LSTM model and scalers. Best config:", best_config)

# Quick plot RMSE across tried configs
plt.figure(figsize=(8,4))
plt.plot(res_df["rmse"].values, marker="o")
plt.title("LSTM tuning: RMSE for tried configs")
plt.xlabel("config index")
plt.ylabel("RMSE")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "plot_lstm_tuning_rmse.png"))
plt.show()
print("Saved plot_lstm_tuning_rmse.png")

print("Week-7 LSTM tuning completed.")
