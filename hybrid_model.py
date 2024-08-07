import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVR
from xgboost import XGBRegressor
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt

# Load the CSV file into a DataFrame
data = pd.read_csv("HistoricalQuotes.csv")

# Remove leading spaces from column names
data.columns = data.columns.str.strip()

# Remove dollar signs and leading/trailing whitespace from relevant columns
for col in ['Close/Last', 'Open', 'High', 'Low']:
    data[col] = data[col].astype(str).str.replace(r'[$, ]', '', regex=True)

# Convert relevant columns to numeric
for col in ['Close/Last', 'Open', 'High', 'Low']:
    data[col] = pd.to_numeric(data[col])

# Define feature columns
features = ['Close/Last', 'Volume', 'Open', 'High', 'Low']

# Normalize the feature data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data[features])

# Separate features and target variable
X = scaled_data[:, 1:]  # Features (excluding 'Close/Last')
y = scaled_data[:, 0]   # Target variable (assuming 'Close/Last' is the first column in scaled_data)

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train SVM model
svm_model = SVR(kernel='rbf')
svm_model.fit(X_train, y_train)
svm_predictions = svm_model.predict(X_test)

# Train XGBoost model
xgb_model = XGBRegressor(objective='reg:squarederror', random_state=42)
xgb_model.fit(X_train, y_train)
xgb_predictions = xgb_model.predict(X_test)

# Prepare data for LSTM model
seq_length = 20

def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data)-seq_length-1):
        X.append(data[i:(i+seq_length)])
        y.append(data[i + seq_length, 0])  # Predict the 'Close/Last' price
    return np.array(X), np.array(y)

# Create sequences for LSTM
X_lstm_train, y_lstm_train = create_sequences(scaled_data[:int(len(scaled_data) * 0.8)], seq_length)
X_lstm_test, y_lstm_test = create_sequences(scaled_data[int(len(scaled_data) * 0.8):], seq_length)

# Define LSTM model
lstm_model = Sequential()
lstm_model.add(LSTM(units=50, return_sequences=True, input_shape=(X_lstm_train.shape[1], X_lstm_train.shape[2])))
lstm_model.add(Dropout(0.2))
lstm_model.add(LSTM(units=50, return_sequences=False))
lstm_model.add(Dropout(0.2))
lstm_model.add(Dense(units=1))

# Compile the LSTM model
lstm_model.compile(optimizer='adam', loss='mean_squared_error')

# Train the LSTM model
lstm_model.fit(X_lstm_train, y_lstm_train, epochs=100, batch_size=32)

# Make predictions with LSTM model
lstm_predictions = lstm_model.predict(X_lstm_test)

# Combine predictions (example: simple averaging)
svm_predictions_resized = svm_predictions[-len(lstm_predictions):]
xgb_predictions_resized = xgb_predictions[-len(lstm_predictions):]
hybrid_predictions = (svm_predictions_resized + xgb_predictions_resized + lstm_predictions.flatten()) / 3

# Inverse scaling for hybrid predictions
hybrid_predictions = scaler.inverse_transform(np.hstack([hybrid_predictions.reshape(-1, 1), np.zeros((hybrid_predictions.shape[0], 4))]))[:, 0]
y_test_resized = y_test[-len(hybrid_predictions):]
y_test_resized = scaler.inverse_transform(np.hstack([y_test_resized.reshape(-1, 1), np.zeros((y_test_resized.shape[0], 4))]))[:, 0]

# Evaluate the hybrid model performance
hybrid_loss = mean_squared_error(y_test_resized, hybrid_predictions)
print(f'Hybrid Model Testing Loss: {hybrid_loss}')

# Calculate metrics for Hybrid model
hybrid_mse = mean_squared_error(y_test_resized, hybrid_predictions)
hybrid_rmse = np.sqrt(hybrid_mse)
hybrid_mae = mean_absolute_error(y_test_resized, hybrid_predictions)
hybrid_r2 = r2_score(y_test_resized, hybrid_predictions)

print(f'Hybrid Model Metrics:')
print(f'Mean Squared Error (MSE): {hybrid_mse}')
print(f'Root Mean Squared Error (RMSE): {hybrid_rmse}')
print(f'Mean Absolute Error (MAE): {hybrid_mae}')
print(f'R-squared Score (R2): {hybrid_r2}')

# Optionally, plot predictions
plt.figure(figsize=(12, 6))
plt.plot(y_test_resized, color='blue', label='Actual Stock Price')
plt.plot(hybrid_predictions, color='red', label='Hybrid Predicted Stock Price')
plt.title('Stock Price Prediction (Hybrid Model)')
plt.xlabel('Time')
plt.ylabel('Stock Price')
plt.legend()
plt.show()
