"""
Univariate LSTM with plotting
Uses only Market Value to predict next Market Value
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# 1. Load data
CSV_PATH = r"D:\INFOSYS\DATA\processed\final_afe_dataset.csv"
df = pd.read_csv(CSV_PATH)

values = df["Market Value"].values.reshape(-1, 1)

# 2. Scale data
scaler = MinMaxScaler()
values_scaled = scaler.fit_transform(values)

# 3. Create sequences
SEQ_LEN = 8

def make_sequences(data, seq_len=8):
    X, y = [], []
    for i in range(len(data) - seq_len):
        X.append(data[i:i+seq_len])
        y.append(data[i+seq_len])
    return np.array(X), np.array(y)

X, y = make_sequences(values_scaled, SEQ_LEN)

# 4. Train-test split
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

print("Shapes:")
print("X_train:", X_train.shape, "y_train:", y_train.shape)
print("X_test :", X_test.shape, "y_test :", y_test.shape)

# 5. Build model
model = Sequential()
model.add(LSTM(50, input_shape=(SEQ_LEN, 1)))
model.add(Dense(1))
model.compile(optimizer="adam", loss="mse")

model.summary()

# 6. Train
history = model.fit(
    X_train, y_train,
    epochs=30,
    batch_size=32,
    validation_data=(X_test, y_test),
    verbose=2
)

# 7. Predict
y_pred = model.predict(X_test)

# inverse scaling
y_test_inv = scaler.inverse_transform(y_test.reshape(-1, 1))
y_pred_inv = scaler.inverse_transform(y_pred)

# ================= PLOTS =================

# Loss plot
plt.figure(figsize=(8,4))
plt.plot(history.history["loss"], label="Train Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.title("Univariate LSTM Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.tight_layout()
plt.savefig("plot_univariate_lstm_loss.png")
plt.show()

# Prediction vs actual plot
plt.figure(figsize=(8,4))
plt.plot(y_test_inv, label="Actual Market Value")
plt.plot(y_pred_inv, label="Predicted Market Value")
plt.title("Univariate LSTM Prediction vs Actual")
plt.xlabel("Time Step")
plt.ylabel("Market Value")
plt.legend()
plt.tight_layout()
plt.savefig("plot_univariate_lstm_pred_vs_true.png")
plt.show()

print(" Univariate LSTM with plots completed")
print("Saved plots:")
print("plot_univariate_lstm_loss.png")
print("plot_univariate_lstm_pred_vs_true.png")
