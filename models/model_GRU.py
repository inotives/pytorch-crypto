import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.metrics import mean_squared_error

# Define the GRU model
class GRUModel(nn.Module):
    """ GRU (Gated Recurrent Unit)
    a type of recurrent neural network (RNN) that is designed to address the vanishing gradient problem that can occur in traditional RNNs. 
    GRUs use gating mechanisms to control the flow of information, allowing them to learn long-term dependencies more effectively.

    How GRUs Work
    -------------------------
    Unlike LSTMs, GRUs have only two gates:
    - Update gate: Controls how much information from the previous state is passed to the current state.
    - Reset gate: Controls how much information from the previous input is forgotten.
    This simplified architecture makes GRUs computationally efficient and can sometimes outperform LSTMs for certain tasks.

    Using GRUs for Price Forecasting with OHLCV Data
    -----------------------
    1. Data Preparation:
    - Gather historical data: Collect OHLCV (Open, High, Low, Close, Volume) data for the asset you want to forecast.
    - Feature engineering: Create additional features like moving averages, RSI, MACD, etc., to provide more context.
    - Normalize data: Scale the features to a common range (e.g., 0-1) to improve model convergence.

    2. Model Architecture:
    - Define GRU layers: Create a sequence of GRU layers to capture long-term dependencies.
    - Add dense layers: Use dense layers to process the output from the GRU layers and make predictions.

    3. Training:
    - Split data: Divide the data into training and testing sets.
    - Compile the model: Specify the loss function (e.g., mean squared error) and optimizer (e.g., Adam).
    - Train the model: Fit the model to the training data.

    4. Evaluation:
    - Evaluate on test set: Use the trained model to make predictions on the testing set and evaluate its performance using metrics 
      like MSE, MAE, or R-squared.
    """

    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(GRUModel, self.__init__())
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.gru(x, h0)
        out = self.fc(out[:, -1, :])
        return out
    
# Function to create sequences
def create_sequences(data, n_steps):
    X, y = [], []
    for i in range(len(data) - n_steps - 7):
        X.append(data[i:i + n_steps])
        y.append(data[i + n_steps:i + n_steps + 7])
    return np.array(X), np.array(y)

# Training function
def train_model(n_steps, hidden_size, num_layers, output_size, num_epochs, batch_size, learning_rate, features, input_df, save_model=False, plot_graph=False): 
    # Load and preprocess the data
    data = input_df[features].values

    # Scaling data
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)

    # Create sequences
    X, y = create_sequences(data_scaled, n_steps)

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
    input_size = len(features)
    model = GRUModel(input_size, hidden_size, num_layers, output_size)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Train the model
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)

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

    y_pred = np.concatenate(all_predictions, axis=0)
    y_test = np.concatenate(all_targets, axis=0)

    # Calculate RMSE
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    print(f'Root Mean Squared Error: {rmse:.4f}')

    if save_model:
        torch.save(model.state_dict(), 'gru_model.pth')

    return model, scaler

# Function to predict the next 7 days
def predict_next_7_days(model, scaler, recent_data, n_steps, features):
    model.eval()
    with torch.no_grad():
        scaled_recent_data = scaler.transform(recent_data[-n_steps:])
        X_recent = torch.tensor(scaled_recent_data, dtype=torch.float32).unsqueeze(0)
        prediction = model(X_recent)
        predicted_prices = scaler.inverse_transform(prediction.numpy())
        return predicted_prices

# Function to run the predictor
def run_predictor(): 
    # Parameters
    n_steps = 60
    hidden_size = 50
    num_layers = 2
    output_size = 7  # Predicting the next 7 days
    num_epochs = 100
    batch_size = 64
    learning_rate = 0.001
    features = ['close', 'tr', 'atr_ma14', 'ema_20', 'wma_20', 'cumulative_return', 'vwap', 'rsi']

    # Dataset for training the model
    df = preparing_data('ohlcv_bitcoin_20240821')

    # Train GRU model
    gru_model, scaler = train_model(
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

    # Predict the next 7 days
    recent_data = df[features].values[-n_steps:]
    predicted_prices = predict_next_7_days(gru_model, scaler, recent_data, n_steps, features)
    print(f'Predicted Prices for the next 7 days: {predicted_prices}')