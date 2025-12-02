# lstm_model.py
# type: ignore
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint


def build_lstm_model(input_shape, lstm_units=64, dropout_rate=0.2):
    """
    Builds an LSTM regression model for transfer value prediction.

    Parameters
    ----------
    input_shape : tuple
        Shape of the input data (timesteps, features)
    lstm_units : int
        Number of LSTM units
    dropout_rate : float
        Dropout percentage

    Returns
    -------
    model : tf.keras.Model
    """
    model = Sequential([
        LSTM(lstm_units, return_sequences=True, input_shape=input_shape),
        Dropout(dropout_rate),

        LSTM(lstm_units),
        Dropout(dropout_rate),

        Dense(32, activation='relu'),
        Dense(1)  # Regression output
    ])

    model.compile(
        optimizer='adam',
        loss='mse',
        metrics=['mae']
    )
    return model


def train_lstm(model, X_train, y_train, X_val, y_val, save_path="models/lstm_model.keras"):
    """
    Trains the LSTM model with early stopping.
    """

    callbacks = [
        EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
        ModelCheckpoint(filepath=save_path, monitor="val_loss", save_best_only=True)
    ]

    history = model.fit(
        X_train, y_train,
        epochs=50,
        batch_size=32,
        validation_data=(X_val, y_val),
        callbacks=callbacks,
        verbose=1
    )

    return history


def predict_lstm(model, X_test):
    return model.predict(X_test)
