import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, explained_variance_score
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from xgboost import XGBRegressor
import shap
from statsmodels.tsa.arima.model import ARIMA
import warnings

warnings.filterwarnings('ignore')

# Load the data
data = pd.read_csv('HistoricalQuotes.csv')  # Replace with your dataset
data['Date'] = pd.to_datetime(data['Date'])
data.set_index('Date', inplace=True)

# Remove leading/trailing spaces from column names
data.columns = data.columns.str.strip()

# Remove dollar signs and convert columns to float
for col in ['Close/Last', 'Open', 'High', 'Low']:
    data[col] = data[col].replace(r'[\$,]', '', regex=True).astype(float)

# Forward fill to handle missing values
data = data.fillna(method='ffill')

# Split the data into train and test sets
train_size = int(len(data) * 0.8)
train, test = data.iloc[:train_size], data.iloc[train_size:]

# ARIMA Model
def evaluate_arima_model(train, test, arima_order):
    model = ARIMA(train, order=arima_order)
    model_fit = model.fit()
    predictions = model_fit.forecast(steps=len(test))
    rmse = np.sqrt(mean_squared_error(test, predictions))
    return rmse, predictions

# Grid search for ARIMA parameters
p_values = range(0, 7)
d_values = range(0, 3)
q_values = range(0, 7)

best_rmse, best_order = float("inf"), None
for p in p_values:
    for d in d_values:
        for q in q_values:
            order = (p, d, q)
            try:
                rmse, _ = evaluate_arima_model(train['Close/Last'], test['Close/Last'], order)
                if rmse < best_rmse:
                    best_rmse, best_order = rmse, order
            except:
                continue

print(f'Best ARIMA parameters: {best_order}')
print(f'Best ARIMA RMSE: {best_rmse}')

# Build and fit the best ARIMA model
arima_model = ARIMA(train['Close/Last'], order=best_order)
arima_fit = arima_model.fit()

# Predictions
# Use integer-based indexing for ARIMA predictions
arima_train_pred = arima_fit.predict(start=0, end=len(train)-1)
arima_test_pred = arima_fit.predict(start=len(train), end=len(train)+len(test)-1)

# Preprocess data for LSTM
def create_dataset(data, time_step=1):
    X, y = [], []
    for i in range(len(data) - time_step - 1):
        X.append(data[i:(i + time_step), 0])
        y.append(data[i + time_step, 0])
    return np.array(X), np.array(y)

scaler = MinMaxScaler(feature_range=(0, 1))
data_scaled = scaler.fit_transform(data['Close/Last'].values.reshape(-1, 1))

X, y = create_dataset(data_scaled, time_step=10)
X = X.reshape(X.shape[0], X.shape[1], 1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build and fit LSTM model
lstm_model = Sequential()
lstm_model.add(LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
lstm_model.add(Dropout(0.2))
lstm_model.add(LSTM(50, return_sequences=False))
lstm_model.add(Dropout(0.2))
lstm_model.add(Dense(25))
lstm_model.add(Dense(1))
lstm_model.compile(optimizer='adam', loss='mean_squared_error')
lstm_model.fit(X_train, y_train, epochs=100, batch_size=32)

# Make predictions
train_pred = lstm_model.predict(X_train)
test_pred = lstm_model.predict(X_test)

# Inverse transform predictions
train_pred = scaler.inverse_transform(train_pred)
test_pred = scaler.inverse_transform(test_pred)

# Prepare features for XGBoost
train_features = np.concatenate((X_train.reshape(X_train.shape[0], -1), train_pred), axis=1)
test_features = np.concatenate((X_test.reshape(X_test.shape[0], -1), test_pred), axis=1)

# Print shapes for debugging
print(f"Shape of train_features before ARIMA: {train_features.shape}")
print(f"Shape of arima_train_pred: {arima_train_pred.shape}")

# Ensure ARIMA predictions are aligned with LSTM predictions
arima_train_pred = arima_train_pred[:train_features.shape[0]]
arima_test_pred = arima_test_pred[:test_features.shape[0]]

# Combine ARIMA predictions
train_features = np.concatenate((train_features, arima_train_pred.values.reshape(-1, 1)), axis=1)
test_features = np.concatenate((test_features, arima_test_pred.values.reshape(-1, 1)), axis=1)

# Train XGBoost model
xgb_model = XGBRegressor()
xgb_model.fit(train_features, y_train)

# Make final predictions
final_train_pred = xgb_model.predict(train_features)
final_test_pred = xgb_model.predict(test_features)

# Calculate metrics
train_mse = mean_squared_error(y_train, final_train_pred)
test_mse = mean_squared_error(y_test, final_test_pred)
train_mae = mean_absolute_error(y_train, final_train_pred)
test_mae = mean_absolute_error(y_test, final_test_pred)
train_r2 = r2_score(y_train, final_train_pred)
test_r2 = r2_score(y_test, final_test_pred)
train_explained_var = explained_variance_score(y_train, final_train_pred)
test_explained_var = explained_variance_score(y_test, final_test_pred)

# Print metrics
print(f'Train MSE: {train_mse}')
print(f'Test MSE: {test_mse}')
print(f'Train MAE: {train_mae}')
print(f'Test MAE: {test_mae}')
print(f'Train R²: {train_r2}')
print(f'Test R²: {test_r2}')
print(f'Train Explained Variance: {train_explained_var}')
print(f'Test Explained Variance: {test_explained_var}')

# SHAP Analysis
explainer = shap.Explainer(xgb_model)
shap_values = explainer.shap_values(test_features)
shap.summary_plot(shap_values, test_features)

# Plot results
plt.figure(figsize=(12, 6))
plt.plot(y_test, label='Actual')
plt.plot(final_test_pred, label='Predicted', color='r')
plt.title('Actual vs Predicted')
plt.legend()
plt.show()
