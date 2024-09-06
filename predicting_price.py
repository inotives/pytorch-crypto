import numpy as np
import torch
from sklearn.preprocessing import StandardScaler

import utils.data_preprocessing as dp
from models.price_predictors.model_LSTM import LSTMModel


# Function to prepare input data for prediction
def prepare_input_data(recent_data, scaler, n_steps):
    # Convert recent_data DataFrame to a NumPy array
    recent_data_values = recent_data.values

    # Scale the data using the same scaler used during training
    data_scaled = scaler.transform(recent_data)

    # Create the sequence (we only need one sequence for prediction)
    X = []
    X.append(data_scaled[-n_steps:])  # Last n_steps rows
    return torch.tensor(np.array(X), dtype=torch.float32)


# Function to inverse transform the predicted price
def inverse_transform_prediction(predicted_price, scaler):
    # Since we used scaler on the entire dataset with multiple features, we need to inverse transform
    # only the first feature (e.g., 'close') assuming that's the target
    dummy_data = np.zeros((predicted_price.shape[0], scaler.scale_.shape[0]))
    dummy_data[:, 0] = predicted_price[:, 0]  # Replace only the 'close' column
    return scaler.inverse_transform(dummy_data)[:, 0]


# Load the saved LSTM model
input_size = 8  # Number of features used during training
hidden_size = 50
num_layers = 2
output_size = 1

model = LSTMModel(input_size, hidden_size, num_layers, output_size)
model.load_state_dict(torch.load('lstm_model.pth'))
model.eval()  # Set the model to evaluation mode

# Load your full dataset
csvdata='ohlcv_bitcoin_20240821'
data = dp.data_load(f"data/raw/{csvdata}.csv")
df = dp.adding_features(data)


# Load the recent data (e.g., the last 60 days) with the feature used for training
features = ['close', 'tr', 'atr_ma14', 'ema_20', 'wma_20', 'cumulative_return', 'vwap', 'rsi']
recent_data = df[features][-60:]

# Initialize and fit the scaler on the entire dataset (do this before using prepare_input_data)
scaler = StandardScaler()
scaler.fit(df[features].values)  # Fit scaler on the original training data

# Prepare the input data for the model
n_steps = 60  # Number of time steps used in the LSTM model
X_input = prepare_input_data(recent_data, scaler, n_steps)


# Make the prediction
with torch.no_grad():
    prediction = model(X_input)

# Convert the prediction to a numpy array
predicted_price = prediction.numpy()


# Inverse transform to get the actual predicted price
next_day_price = inverse_transform_prediction(predicted_price, scaler)

print(f"Predicted next day's price: {next_day_price[0]:.4f}")
