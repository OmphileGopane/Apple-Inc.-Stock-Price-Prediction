import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
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
for col in ['Open', 'High', 'Low']:
    data[col] = pd.to_numeric(data[col])

# Define feature columns
features = ['Close/Last', 'Volume', 'Open', 'High', 'Low']

# Normalize the feature data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data[features])

# Split the data into training and testing sets
train_size = int(len(scaled_data) * 0.8)
test_size = len(scaled_data) - train_size
train_data, test_data = scaled_data[0:train_size,:], scaled_data[train_size:len(scaled_data),:]

# Define function to create sequences for LSTM model
def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data)-seq_length-1):
        X.append(data[i:(i+seq_length)])
        y.append(data[i + seq_length, 0])  # Predict the 'Close/Last' price
    return np.array(X), np.array(y)

# Define sequence length
seq_length = 20

# Create sequences for training and testing data
X_train, y_train = create_sequences(train_data, seq_length)
X_test, y_test = create_sequences(test_data, seq_length)

# Define LSTM model
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(Dropout(0.2))
model.add(LSTM(units=50, return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(units=1))

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
history = model.fit(X_train, y_train, epochs=100, batch_size=32, verbose=1)

# Evaluate the model
train_loss = model.evaluate(X_train, y_train, verbose=0)
test_loss = model.evaluate(X_test, y_test, verbose=0)
print(f'Training Loss: {train_loss:.4f}, Testing Loss: {test_loss:.4f}')

# Make predictions
predictions = model.predict(X_test)

# Inverse scaling (only for the 'Close/Last' price)
predictions = scaler.inverse_transform(np.hstack([predictions, np.zeros((predictions.shape[0], 4))]))[:,0]
y_test = scaler.inverse_transform(np.hstack([y_test.reshape(-1, 1), np.zeros((y_test.shape[0], 4))]))[:,0]

# Calculate additional evaluation metrics
mae = mean_absolute_error(y_test, predictions)
r2 = r2_score(y_test, predictions)
print(f'Mean Absolute Error: {mae:.4f}')
print(f'R^2 Score: {r2:.4f}')

# Plot the predictions against the actual values
plt.figure(figsize=(12, 6))
plt.plot(y_test, color='blue', label='Actual Stock Price')
plt.plot(predictions, color='red', label='Predicted Stock Price')
plt.title('Stock Price Prediction (LSTM)')
plt.xlabel('Time')
plt.ylabel('Stock Price')
plt.legend()
plt.show()
