import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_squared_error
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import plotly.graph_objs as go
import matplotlib.pyplot as plt

import utils.data_preprocessing as dp


class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

        self.data_df = None

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

def preparing_data(csvdata):
    """Load Data, Preprocessed the features and data clean up... """

    # load_data from csv
    data = dp.data_load(f"data/raw/{csvdata}.csv")

    # add features to the ohlcv data
    data_with_features = dp.adding_features(data)

    # make sure to eliminate all NA.
    data_with_features.dropna(inplace=True)

    return data_with_features

def create_sequences(data, n_steps, n_out=7):
    """Create sequence for LSTM used."""
    X, y = [], []
    for i in range(len(data) - n_steps - n_out):
        X.append(data[i:i+n_steps])
        y.append(data[i+n_steps:i+n_steps+n_out, 0])  # Assuming the first feature is the target
    return np.array(X), np.array(y)

def plot_predictions(y_true, y_pred, scaler, n_features):
    """
    Plot the actual vs predicted values after inverse scaling.

    Parameters:
    - y_true: Actual target values
    - y_pred: Predicted target values
    - scaler: Scaler used for inverse transformation
    - n_features: Number of features used for scaling
    """
    # Reshape y_true and y_pred for inverse_transform
    y_true_reshaped = y_true.reshape(-1, 1)
    y_pred_reshaped = y_pred.reshape(-1, 1)

    # Create dummy feature columns to match the scaler's expected input
    y_true_reshaped_full = np.hstack([y_true_reshaped] + [np.zeros((y_true_reshaped.shape[0], n_features - 1))])
    y_pred_reshaped_full = np.hstack([y_pred_reshaped] + [np.zeros((y_pred_reshaped.shape[0], n_features - 1))])

    # Inverse transform
    y_true_inv = scaler.inverse_transform(y_true_reshaped_full)[:, 0]
    y_pred_inv = scaler.inverse_transform(y_pred_reshaped_full)[:, 0]

    plt.figure(figsize=(12, 6))
    plt.plot(y_true_inv, label='Actual Prices')
    plt.plot(y_pred_inv, label='Predicted Prices', linestyle='--')
    plt.title('Crypto Price Prediction')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend()
    plt.show()


def train_model(n_steps, hidden_size, num_layers, output_size, num_epochs, batch_size, learning_rate, features, input_df, save_model=False, plot_graph=False): 
    """ Train LSTM Model by using the parameters define """
    # Load and PreProcess the data
    data = input_df[features].values

    # Scaling data
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)

    # Create sequences
    X, y = create_sequences(data_scaled, n_steps, n_out=output_size)

    # Convert to PyTorch tensors
    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32)

    # Train-test split
    dataset = TensorDataset(X, y)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Initialize model, criterion, and optimizer
    input_size = len(features)  # Update input_size based on selected features
    model = LSTMModel(input_size, hidden_size, num_layers, output_size)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Train the model
    print(f"START MODEL TRAINING ... TOTAL_EPOCH: {num_epochs}\n===================")
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)

            # Reshape outputs and targets to be 1D tensors
            outputs = outputs.view(-1)
            targets = targets.view(-1)

            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * inputs.size(0)

        avg_epoch_loss = epoch_loss / len(train_dataset)
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Average Loss: {avg_epoch_loss:.4f}')

    # Evaluate the model
    model.eval()
    all_predictions = []
    all_targets = []
    with torch.no_grad():
        for inputs, targets in test_loader:
            outputs = model(inputs)
            all_predictions.append(outputs.numpy())
            all_targets.append(targets.numpy())

    # Flatten predictions and targets
    y_pred = np.concatenate(all_predictions, axis=0)
    y_test = np.concatenate(all_targets, axis=0)

    # Calculate RMSE
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    print(f'Root Mean Squared Error: {rmse:.4f}')

    # Plot predictions
    if plot_graph: plot_predictions(y_test, y_pred, scaler, len(features))

    # Optionally, save the model
    if save_model: torch.save(model.state_dict(), 'lstm_model.pth')

    return model

def run_predictor(data_src): 
    # Parameters
    n_steps = 60
    hidden_size = 50
    num_layers = 2
    output_size = 7
    num_epochs = 100
    batch_size = 64
    learning_rate = 0.001
    features = ['close', 'tr', 'atr_ma14', 'ema_20', 'wma_20', 'cumulative_return', 'vwap', 'rsi']

    # dataset for training the model
    df = preparing_data(data_src)

    # Train LSTM model base on the inputs 
    lstm_model = train_model(
        n_steps, 
        hidden_size, 
        num_layers, 
        output_size, 
        num_epochs,
        batch_size,
        learning_rate,
        features,
        input_df=df
    )

    # Use the model to run the predictor
    recent_data = df[features][-n_steps:]
    scaler = StandardScaler()
    recent_data_scaled = scaler.fit_transform(recent_data.values)

    # Convert recent data to tensor and reshape it to match the model input
    recent_data_tensor = torch.tensor(recent_data_scaled, dtype=torch.float32).unsqueeze(0)

    # Make predictions
    lstm_model.eval()
    with torch.no_grad():
        predicted_scaled = lstm_model(recent_data_tensor).numpy().flatten()

    # Inverse transform the predicted prices
    recent_data_unscaled = scaler.inverse_transform(recent_data.values)
    last_close_price = recent_data_unscaled[-1, 0]  # Assuming 'close' is the first feature
    predicted_prices = scaler.inverse_transform(
        np.hstack((predicted_scaled.reshape(-1, 1), np.zeros((output_size, len(features) - 1))))
    )[:, 0]

    # Display the predicted prices
    print("Predicted Prices for the Next 7 Days:")
    for i, price in enumerate(predicted_prices, 1):
        print(f"Day {i}: {price:.2f}")

    return predicted_prices    


