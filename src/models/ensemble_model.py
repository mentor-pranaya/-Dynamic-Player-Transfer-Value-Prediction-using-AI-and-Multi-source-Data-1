# ensemble_model.py

import numpy as np
from xgboost import XGBRegressor


class EnsembleModel:
    """
    Combines XGBoost + LSTM predictions using a meta-model.
    """

    def __init__(self):
        self.meta_model = XGBRegressor(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=4,
            subsample=0.8,
            colsample_bytree=0.9
        )

    def train(self, lstm_preds_train, xgb_preds_train, y_train):
        """
        Trains ensemble model using predictions from two base models.
        """

        stacked_inputs = np.column_stack((lstm_preds_train, xgb_preds_train))
        self.meta_model.fit(stacked_inputs, y_train)

    def predict(self, lstm_preds_test, xgb_preds_test):
        stacked_inputs = np.column_stack((lstm_preds_test, xgb_preds_test))
        return self.meta_model.predict(stacked_inputs)
