from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error 
import sys
print(sys.executable)

lstm_train_pred = model.predict(X_train)
lstm_test_pred = model.predict(X_test)

X_train_flat = X_train[:, -1, :]
X_test_flat = X_test[:, -1, :]

X_train_ensemble = np.hstack([X_train_flat, lstm_train_pred])
X_test_ensemble = np.hstack([X_test_flat, lstm_test_pred])

xgb = XGBRegressor(n_estimators=100, random_state=42)
xgb.fit(X_train_ensemble, y_train)
ensemble_pred = xgb.predict(X_test_ensemble)

print("LSTM RMSE:", np.sqrt(mean_squared_error(y_test, lstm_test_pred)))
print("Ensemble RMSE:", np.sqrt(mean_squared_error(y_test, ensemble_pred)))