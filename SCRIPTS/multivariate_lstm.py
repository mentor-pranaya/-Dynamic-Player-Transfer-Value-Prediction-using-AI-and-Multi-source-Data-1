"""
Multivariate LSTM
Uses multiple numeric features (performance + sentiment + etc.)
to predict 'Market Value' for each time step.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import joblib

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# ===========================
# 1. PATH
# ===========================
CSV_PATH = r"D:\INFOSYS\DATA\processed\final_afe_dataset.csv"
TARGET_COL = "Market Value"
SEQ_LEN = 8
TEST_SIZE = 0.2
BATCH_SIZE = 32
EPOCHS = 50

# ===========================
# 2. LOAD DATA
# ===========================
if not os.path.exists(CSV_PATH):
    print("Dataset file not found. Check CSV_PATH.")
    raise SystemExit

df = pd.read_csv(CSV_PATH)
print("Loaded dataset:", CSV_PATH)
print("Shape:", df.shape)

if TARGET_COL not in df.columns:
    print("Target column not found. Check TARGET_COL.")
    raise SystemExit

# ===========================
# 3. FEATURES (X) & TARGET (y)
# ===========================
# Drop only IDs / names / date – anni numeric features tho multidata
drop_cols = []
for c in ["player_name", "player_id", "date", "Unnamed: 0"]:
    if c in df.columns:
        drop_cols.append(c)

# X = ALL numeric features except target & drop_cols
X_full = df.drop(columns=drop_cols + [TARGET_COL], errors="ignore")
X_full = X_full.select_dtypes(include=[np.number])

# y = Market Value
y_full = df[TARGET_COL].values

print("\nFeature matrix shape (before sequences):", X_full.shape)
print("Number of samples:", len(y_full))

# ===========================
# 4. SCALE DATA
# ===========================
scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()

X_scaled = scaler_X.fit_transform(X_full)
y_scaled = scaler_y.fit_transform(y_full.reshape(-1, 1)).squeeze()

joblib.dump(scaler_X, "scaler_X_multi.pkl")
joblib.dump(scaler_y, "scaler_y_multi.pkl")
print("\nSaved scalers: scaler_X_multi.pkl, scaler_y_multi.pkl")

# ===========================
# 5. CREATE SEQUENCES (MULTIVARIATE)
# ===========================
def make_sequences(X, y, seq_len):
    X_list, y_list = [], []
    for i in range(len(X) - seq_len):
        X_list.append(X[i:i+seq_len])      # past seq_len rows of ALL features
        y_list.append(y[i+seq_len])        # next step Market Value only
    return np.array(X_list), np.array(y_list)

X_seq, y_seq = make_sequences(X_scaled, y_scaled, SEQ_LEN)
print("\nBuilt sequences:")
print("X_seq shape:", X_seq.shape)  # (samples, seq_len, n_features)
print("y_seq shape:", y_seq.shape)  # (samples,)

# ===========================
# 6. TRAIN / TEST SPLIT 
# ===========================
n_samples = len(X_seq)
n_test = int(n_samples * TEST_SIZE)
n_train = n_samples - n_test

X_train = X_seq[:n_train]
X_test  = X_seq[n_train:]
y_train = y_seq[:n_train]
y_test  = y_seq[n_train:]

print("\nTrain shapes:", X_train.shape, y_train.shape)
print("Test shapes :", X_test.shape, y_test.shape)

# ===========================
# 7. BUILD MULTIVARIATE LSTM
# ===========================
n_features = X_train.shape[2]

def build_lstm(seq_len, n_feat):
    model = Sequential()
    model.add(LSTM(64, return_sequences=True, input_shape=(seq_len, n_feat)))
    model.add(Dropout(0.2))
    model.add(LSTM(32, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(16, activation="relu"))
    model.add(Dense(1, activation="linear"))
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    return model

model = build_lstm(SEQ_LEN, n_features)
model.summary()

# ===========================
# 8. TRAIN MODEL
# ===========================
es = EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)
mc = ModelCheckpoint("lstm_multivariate.keras", monitor="val_loss", save_best_only=True)

history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=[es, mc],
    verbose=2
)

# ===========================
# 9. EVALUATE & SAVE PREDICTIONS
# ===========================
best_model = tf.keras.models.load_model("lstm_multivariate.keras", compile=False)
y_pred_scaled = best_model.predict(X_test).squeeze()

y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).squeeze()
y_true = scaler_y.inverse_transform(y_test.reshape(-1, 1)).squeeze()

rmse = np.sqrt(mean_squared_error(y_true, y_pred))
mae = mean_absolute_error(y_true, y_pred)

print("\nMultivariate LSTM RMSE:", rmse)
print("Multivariate LSTM MAE :", mae)

# Row index alignment for ensemble / reports
start_index = SEQ_LEN + n_train
row_idx = list(range(start_index, start_index + len(y_pred)))

pred_df = pd.DataFrame({
    "row_index": row_idx,
    "y_true": y_true,
    "y_pred": y_pred
})
pred_df.to_csv("lstm_multivariate_preds.csv", index=False)

print("\nSaved: lstm_multivariate_preds.csv")

# ===========================
# 10. PLOTS
# ===========================
plt.figure(figsize=(8, 4))
plt.plot(history.history["loss"], label="train_loss")
plt.plot(history.history["val_loss"], label="val_loss")
plt.title("Multivariate LSTM Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss (MSE)")
plt.legend()
plt.tight_layout()
plt.savefig("plot_lstm_multivariate_loss.png")
plt.show()

plt.figure(figsize=(8, 4))
plt.plot(y_true, label="True value")
plt.plot(y_pred, label="Predicted value")
plt.title("Multivariate LSTM Predictions vs True Values")
plt.xlabel("Test sample index")
plt.ylabel("Market Value")
plt.legend()
plt.tight_layout()
plt.savefig("plot_lstm_multivariate_pred_vs_true.png")
plt.show()

print("\nSaved plots: plot_lstm_multivariate_loss.png, plot_lstm_multivariate_pred_vs_true.png")
print(" Multivariate LSTM done.")
