# utils.py

import joblib
import numpy as np
from sklearn.preprocessing import MinMaxScaler


def scale_data(train, val, test):
    """
    Scales numerical data using MinMaxScaler.
    Returns scaled datasets and the scaler.
    """
    scaler = MinMaxScaler()
    scaler.fit(train)

    return (
        scaler.transform(train),
        scaler.transform(val),
        scaler.transform(test),
        scaler
    )


def save_scaler(scaler, path="models/scaler.pkl"):
    """Save scaler object."""
    joblib.dump(scaler, path)


def load_scaler(path="models/scaler.pkl"):
    """Load scaler object."""
    return joblib.load(path)


def reshape_for_lstm(X, timesteps):
    """
    Converts flat features → LSTM 3D format
    (samples, timesteps, features).
    """
    return np.reshape(X, (X.shape[0], timesteps, X.shape[1] // timesteps))
