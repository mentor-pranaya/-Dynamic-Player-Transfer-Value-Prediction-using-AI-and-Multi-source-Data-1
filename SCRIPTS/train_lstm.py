"""
LSTM training 

steps: 
1. Load the final dataset
2. Use numeric columns as input features
3. Use one chosen column as target (market value)
4. Make sequences for time-series (LSTM)
5. Train LSTM model
6. Save the  predictions 
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# ===========================
# 1. CSV_PATH 
# ===========================

# Path of my dataset
CSV_PATH = r"D:\INFOSYS\DATA\processed\final_afe_dataset.csv"

# IMPORTANT: name of the column that has the player value / transfer value
TARGET_COL = "Market Value"

# how many past steps to look at
SEQ_LEN = 8

# test size (last 20% of data will be test set)
TEST_SIZE = 0.20

BATCH_SIZE = 32
EPOCHS = 50

# ===========================
# 2. LOAD DATA
# ===========================

if not os.path.exists(CSV_PATH):
    print("Dataset file not found. Check CSV_PATH in the script.")
    raise SystemExit

df = pd.read_csv(CSV_PATH)
print("Loaded dataset:", CSV_PATH)
print("Shape:", df.shape)

print("\nAll columns in dataset:")
print(df.columns.tolist())

# ===========================
# 3. TARGET COLUMN
# ===========================

if TARGET_COL not in df.columns:
    print("\nThe target column name is NOT in the dataframe.")
    print("Please open train_lstm.py and set TARGET_COL correctly.")
    raise SystemExit

target_col = TARGET_COL
print("\nUsing target column:", target_col)

# ===========================
# 4. BUILD FEATURES (X) AND TARGET (y)
# ===========================

# remove some columns we don't want as features or if we don't use
drop_cols = []
for c in ["player_name", "player_id", "date", "Unnamed: 0"]:
    if c in df.columns:
        drop_cols.append(c)

# X_full = all numeric features except target + drop_cols
X_full = df.drop(columns=drop_cols + [target_col], errors="ignore")
X_full = X_full.select_dtypes(include=[np.number])

y_full = df[target_col].values

print("\nFeature matrix shape (before sequences):", X_full.shape)
print("Number of samples:", len(y_full))

# ===========================
# 5. SCALE DATA
# ===========================

from sklearn.preprocessing import MinMaxScaler
import joblib

scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()

X_scaled = scaler_X.fit_transform(X_full)
y_scaled = scaler_y.fit_transform(y_full.reshape(-1, 1)).squeeze()

joblib.dump(scaler_X, "scaler_X.pkl")
joblib.dump(scaler_y, "scaler_y.pkl")
print("\nSaved scalers: scaler_X.pkl, scaler_y.pkl")

# ===========================
# 6. CREATE SEQUENCES FOR LSTM
# ===========================

def make_sequences(X, y, seq_len):
    X_list = []
    y_list = []
    for i in range(len(X) - seq_len):
        X_list.append(X[i:i+seq_len])
        y_list.append(y[i+seq_len])
    return np.array(X_list), np.array(y_list)

X_seq, y_seq = make_sequences(X_scaled, y_scaled, SEQ_LEN)
print("\nBuilt sequences:")
print("X_seq shape:", X_seq.shape)   # (samples, seq_len, features)
print("y_seq shape:", y_seq.shape)   # (samples,)

# ===========================
# 7. TRAIN / TEST SPLIT (TIME SERIES)
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
# 8. BUILD LSTM MODEL
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
# 9. TRAIN MODEL
# ===========================

# early stopping: stop if val_loss not improving
es = EarlyStopping(
    monitor="val_loss",
    patience=10,
    restore_best_weights=True
)

# save the best model to a file
mc = ModelCheckpoint(
    "lstm_model.keras",
    monitor="val_loss",
    save_best_only=True
)

history = model.fit(
    X_train,
    y_train,
    validation_data=(X_test, y_test),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=[es, mc],
    verbose=2
)

# ===========================
# 10. EVALUATE AND SAVE PREDICTIONS
# ===========================

# load the best model
best_model = tf.keras.models.load_model("lstm_model.keras", compile=False)

# predict on test set
y_pred_scaled = best_model.predict(X_test).squeeze()

# inverse scale back to original values
y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).squeeze()
y_true = scaler_y.inverse_transform(y_test.reshape(-1, 1)).squeeze()

rmse = np.sqrt(mean_squared_error(y_true, y_pred))
mae = mean_absolute_error(y_true, y_pred)

print("\nLSTM Test RMSE:", rmse)
print("LSTM Test MAE :", mae)

# map predictions back to original row indices for later merging
start_index = SEQ_LEN + n_train
row_idx = list(range(start_index, start_index + len(y_pred)))

pred_df = pd.DataFrame({
    "row_index": row_idx,
    "y_true": y_true,
    "y_pred": y_pred
})

pred_df.to_csv("lstm_preds.csv", index=False)
np.save("lstm_preds.npy", y_pred)

print("\nSaved prediction files:")
print("  lstm_preds.csv")
print("  lstm_preds.npy")

# ===========================
# 11. PLOTS FOR REPORT / PPT
# ===========================

plt.figure(figsize=(8, 4))
plt.plot(history.history["loss"], label="train_loss")
plt.plot(history.history["val_loss"], label="val_loss")
plt.title("LSTM loss")
plt.xlabel("Epoch")
plt.ylabel("Loss (MSE)")
plt.legend()
plt.tight_layout()
plt.savefig("plot_lstm_loss.png")
plt.show()

plt.figure(figsize=(8, 4))
plt.plot(y_true, label="True value")
plt.plot(y_pred, label="Predicted value")
plt.title("LSTM predictions vs True values")
plt.xlabel("Test sample index")
plt.ylabel("Transfer value")
plt.legend()
plt.tight_layout()
plt.savefig("plot_lstm_pred_vs_true.png")
plt.show()

print("\nSaved plots:")
print("  plot_lstm_loss.png")
print("  plot_lstm_pred_vs_true.png")
print("\nDone.")
