import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from sklearn.model_selection import train_test_split
import xgboost as xgb
import shap

# Load and preprocess the data
data = pd.read_csv('HistoricalQuotes.csv')  # Replace with your dataset

# Inspect the column names
print("Column names in the dataset:", data.columns)

data['Date'] = pd.to_datetime(data['Date'])
data.set_index('Date', inplace=True)

# Assuming the column might have extra spaces, let's strip any leading/trailing spaces
data.columns = data.columns.str.strip()

# Print the column names again to verify
print("Column names after stripping spaces:", data.columns)

# Replace dollar signs and convert to float
if 'Close/Last' in data.columns:
    data['Close/Last'] = data['Close/Last'].replace('[\$,]', '', regex=True).astype(float)
else:
    print("Column 'Close/Last' not found in the dataset. Available columns:", data.columns)
    exit()

# Split the data into training and testing sets
train_size = int(len(data) * 0.8)
train, test = data[:train_size], data[train_size:]

# ARIMA Model
y_train = train['Close/Last']
y_test = test['Close/Last']
arima_order = (5, 1, 0)  # Example order, tune this according to your data
arima_model = ARIMA(y_train, order=arima_order)
arima_fit = arima_model.fit()
arima_pred = arima_fit.forecast(steps=len(y_test))

# Residuals from ARIMA model
train['ARIMA_Pred'] = arima_fit.predict(start=0, end=len(y_train)-1)
train['Residuals'] = train['Close/Last'] - train['ARIMA_Pred']

# XGBoost Model
X_train = np.array(train.index).reshape(-1, 1)  # Example feature, you can add more features
y_train_residuals = train['Residuals']
X_test = np.array(test.index).reshape(-1, 1)

xgb_model = xgb.XGBRegressor()
xgb_model.fit(X_train, y_train_residuals)

# XGBoost predictions
xgb_pred = xgb_model.predict(X_test)

# Hybrid Model Prediction
hybrid_pred = arima_pred + xgb_pred

# Evaluate the Hybrid Model
test['Hybrid_Pred'] = hybrid_pred
print(test[['Close/Last', 'Hybrid_Pred']])

# Explainability with SHAP
explainer = shap.Explainer(xgb_model)
shap_values = explainer(X_test)

# Plot SHAP values
shap.summary_plot(shap_values, X_test)
