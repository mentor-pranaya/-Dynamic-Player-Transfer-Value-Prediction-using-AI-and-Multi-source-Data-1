"""
Multivariate Encoder-Decoder LSTM
Multi-step forecasting of 'Market Value'
using multiple numeric features (performance, sentiment, etc.)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, RepeatVector, TimeDistributed, Dense
from sklearn.metrics import mean_squared_error
import os

# 1. Load data
CSV_PATH = r"D:\INFOSYS\DATA\processed\final_afe_dataset.csv"
TARGET_COL = "Market Value"

if not os.path.exists(CSV_PATH):
    print("Dataset not found. Check CSV_PATH.")
    raise SystemExit

df = pd.read_csv(CSV_PATH)
print("Loaded:", CSV_PATH, "shape:", df.shape)

if TARGET_COL not in df.columns:
    print("Target column not found. Check TARGET_COL.")
    raise SystemExit

# 2. Build X (features) and y (target)
drop_cols = []
for c in ["player_name", "player_id", "date", "Unnamed: 0"]:
    if c in df.columns:
        drop_cols.append(c)

X_full = df.drop(columns=drop_cols + [TARGET_COL], errors="ignore")
X_full = X_full.select_dtypes(include=[np.number])   # all numeric features

y_full = df[TARGET_COL].values.reshape(-1, 1)

print("X_full shape:", X_full.shape)
print("y_full shape:", y_full.shape)

# 3. Scale data
scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()

X_scaled = scaler_X.fit_transform(X_full)
y_scaled = scaler_y.fit_transform(y_full).squeeze()

# 4. Create sequences for multi-step forecasting
INPUT_LEN = 8    # past 8 steps
OUTPUT_LEN = 5   # predict next 5 Market Value points

def make_multi_step_sequences(X_data, y_data, in_len, out_len):
    X_list, y_list = [], []
    for i in range(len(X_data) - in_len - out_len):
        X_list.append(X_data[i:i+in_len])                         # (in_len, n_features)
        y_list.append(y_data[i+in_len:i+in_len+out_len])          # (out_len,)
    X_arr = np.array(X_list)
    y_arr = np.array(y_list).reshape(-1, out_len, 1)              # (samples, out_len, 1)
    return X_arr, y_arr

X_seq, y_seq = make_multi_step_sequences(X_scaled, y_scaled, INPUT_LEN, OUTPUT_LEN)

print("X_seq shape:", X_seq.shape)
print("y_seq shape:", y_seq.shape)

# 5. Train-test split
n_samples = len(X_seq)
n_test = int(0.2 * n_samples)
n_train = n_samples - n_test

X_train = X_seq[:n_train]
X_test  = X_seq[n_train:]
y_train = y_seq[:n_train]
y_test  = y_seq[n_train:]

print("Train:", X_train.shape, y_train.shape)
print("Test :", X_test.shape, y_test.shape)

n_features = X_train.shape[2]

# 6. Build encoder-decoder model
model = Sequential()
model.add(LSTM(64, activation="relu", input_shape=(INPUT_LEN, n_features)))
model.add(RepeatVector(OUTPUT_LEN))
model.add(LSTM(64, activation="relu", return_sequences=True))
model.add(TimeDistributed(Dense(1)))
model.compile(optimizer="adam", loss="mse")

model.summary()

# 7. Train
history = model.fit(
    X_train, y_train,
    epochs=30,
    batch_size=32,
    validation_data=(X_test, y_test),
    verbose=2
)

print(" Encoder-Decoder multivariate LSTM training done.")

# 8. Predict and evaluate (flatten all steps)
y_pred_scaled = model.predict(X_test)

y_test_flat = scaler_y.inverse_transform(y_test.reshape(-1, 1)).squeeze()
y_pred_flat = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).squeeze()

rmse_all = np.sqrt(mean_squared_error(y_test_flat, y_pred_flat))
print("Multi-step Forecasting RMSE (all future steps):", rmse_all)

# ===================== PLOTS =====================

# 1) Loss curve (train vs val)
plt.figure(figsize=(8, 4))
plt.plot(history.history["loss"], label="Train Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.title("Encoder-Decoder LSTM Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss (MSE)")
plt.legend()
plt.tight_layout()
plt.savefig("plot_encoder_decoder_loss.png")
plt.show()

# 2) Example forecast vs true (for one sample)
# Take first test sample
sample_true_scaled = y_test[0].reshape(-1, 1)          # (OUTPUT_LEN, 1)
sample_pred_scaled = y_pred_scaled[0].reshape(-1, 1)   # (OUTPUT_LEN, 1)

sample_true = scaler_y.inverse_transform(sample_true_scaled).squeeze()
sample_pred = scaler_y.inverse_transform(sample_pred_scaled).squeeze()

steps = range(1, OUTPUT_LEN + 1)

plt.figure(figsize=(8, 4))
plt.plot(steps, sample_true, marker="o", label="True Future Values")
plt.plot(steps, sample_pred, marker="x", label="Predicted Future Values")
plt.title("Encoder-Decoder Multi-step Forecast (Example Sample)")
plt.xlabel("Future Step")
plt.ylabel("Market Value")
plt.legend()
plt.tight_layout()
plt.savefig("plot_encoder_decoder_sample_forecast.png")
plt.show()

print("Saved plots:")
print("  plot_encoder_decoder_loss.png")
print("  plot_encoder_decoder_sample_forecast.png")
