import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBRegressor
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
X = scaled_data[:, 1:]  # Features (excluding 'Close/Last' if it's in features)
y = scaled_data[:, 0]   # Target variable (assuming 'Close/Last' is the first column in scaled_data)

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train XGBoost model
xgb_model = XGBRegressor(objective='reg:squarederror', random_state=42)
xgb_model.fit(X_train, y_train)

# Get feature importances from XGBoost
feature_importances = xgb_model.feature_importances_

# Map feature importance values to feature names
feature_names = features[1:]  # Exclude 'Close/Last'
importance_dict = dict(zip(feature_names, feature_importances))

# Sort features by importance (descending order)
sorted_importance = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)

# Print feature importances
print("Feature Importances:")
for feature, importance in sorted_importance:
    print(f"{feature}: {importance:.4f}")

# Make predictions
predictions_train = xgb_model.predict(X_train)
predictions_test = xgb_model.predict(X_test)

# Evaluate the model
train_loss = mean_squared_error(y_train, predictions_train)
test_loss = mean_squared_error(y_test, predictions_test)
print(f'Training Loss (MSE): {train_loss:.4f}')
print(f'Testing Loss (MSE): {test_loss:.4f}')

# Calculate additional evaluation metrics
mae = mean_absolute_error(y_test, predictions_test)
r2 = r2_score(y_test, predictions_test)
print(f'Mean Absolute Error (MAE): {mae:.4f}')
print(f'R^2 Score: {r2:.4f}')

# Optionally, plot predictions
plt.figure(figsize=(12, 6))
plt.plot(y_test, color='blue', label='Actual Stock Price')
plt.plot(predictions_test, color='red', label='Predicted Stock Price')
plt.title('Stock Price Prediction (XGBoost(with XAI))')
plt.xlabel('Time')
plt.ylabel('Stock Price')
plt.legend()
plt.show()
