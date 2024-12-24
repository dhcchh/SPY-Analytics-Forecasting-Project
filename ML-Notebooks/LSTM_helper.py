import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import random as random
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class SPYLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1, dropout=0.2):
        super(SPYLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.dropout = nn.Dropout(p=dropout)  # Monte Carlo Dropout
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        # LSTM output
        out, _ = self.lstm(x)  # out: (batch_size, seq_length, hidden_size)
        # Only take the last time step for prediction
        out = self.dropout(out[:, -1, :])  # Apply dropout during training and inference
        out = self.fc(out)  # out: (batch_size, output_size)
        return out

def create_sequences(data, seq_length):
    """
    Creates sequences with the correct shape for LSTM input.

    Parameters:
    - data: Array of log returns.
    - seq_length: Number of timesteps in each sequence.

    Returns:
    - X: Sequences with shape (num_sequences, seq_length, 1).
    - y: Targets with shape (num_sequences,).
    """
    sequences = []
    targets = []
    for i in range(len(data) - seq_length):
        sequences.append(data[i:i + seq_length].reshape(-1, 1))  # Add feature dimension
        targets.append(data[i + seq_length])
    return np.array(sequences), np.array(targets)

def forecast_with_ci(model, initial_sequence, steps, num_simulations=100, ci=0.95, seed=42):
    """
    Forecasts log returns for the next `steps` timesteps and computes confidence intervals.

    Parameters:
    - model: Trained LSTM model.
    - initial_sequence: Initial sequence of log returns (length = seq_length).
    - steps: Number of timesteps to forecast.
    - num_simulations: Number of Monte Carlo simulations for confidence intervals.
    - ci: Confidence level for intervals (default = 95%).
    - seed: Random seed for reproducibility.

    Returns:
    - mean_log_returns: Mean forecasted log returns.
    - lower_log_returns: Lower bound of forecasted log returns.
    - upper_log_returns: Upper bound of forecasted log returns.
    """
    # Set the random seed for reproducibility
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # Enable Monte Carlo Dropout
    model.train()  # Keep dropout active during inference for Monte Carlo Dropout

    forecasted_simulations = []
    input_seq = torch.tensor(initial_sequence, dtype=torch.float32).unsqueeze(0).unsqueeze(2).to(device)

    for _ in range(num_simulations):
        forecasted = []
        seq = input_seq.clone()
        for _ in range(steps):
            with torch.no_grad():
                pred = model(seq)  # Predict next log return
                pred = pred.squeeze(1)  # Ensure pred is 2D: (batch_size, input_size)
                forecasted.append(pred.item())
                # Append prediction and remove the oldest value
                seq = torch.cat((seq[:, 1:, :], pred.unsqueeze(0).unsqueeze(2)), dim=1)
        forecasted_simulations.append(forecasted)

    # Convert simulations to tensor
    forecasted_simulations = torch.tensor(forecasted_simulations, device=device)

    # Compute mean and standard deviation
    mean_log_returns = forecasted_simulations.mean(dim=0)
    std_log_returns = forecasted_simulations.std(dim=0)

    # Compute confidence intervals
    z_value = 1.96  # For 95% confidence interval
    lower_log_returns = mean_log_returns - z_value * std_log_returns
    upper_log_returns = mean_log_returns + z_value * std_log_returns

    return mean_log_returns, lower_log_returns, upper_log_returns

    
# Convert log returns to prices
def log_returns_to_prices(log_returns, initial_price):
    """
    Converts a sequence of log returns to prices.

    Parameters:
    - log_returns: Predicted log returns (torch.Tensor).
    - initial_price: Last known price to start price computation.

    Returns:
    - prices: Reconstructed prices from log returns.
    """
    prices = torch.exp(log_returns.cumsum(dim=0)) * initial_price
    return prices

def plot_forecast(mean_prices, lower_prices, upper_prices, steps):
    """
    Plots the forecasted prices along with their confidence intervals.

    Parameters:
    - mean_prices: Tensor of mean forecasted prices.
    - lower_prices: Tensor of lower bound prices (confidence interval).
    - upper_prices: Tensor of upper bound prices (confidence interval).
    - steps: Number of timesteps in the forecast.
    """
    # Convert tensors to NumPy arrays
    mean_prices = mean_prices.cpu().numpy()
    lower_prices = lower_prices.cpu().numpy()
    upper_prices = upper_prices.cpu().numpy()

    # Generate time steps for plotting
    time_steps = range(1, steps + 1)

    # Plot mean prices and confidence intervals
    plt.figure(figsize=(12, 6))
    plt.plot(time_steps, mean_prices, label=f'Mean Forecast (Last: US$ {mean_prices[-1]:.2f})', color='blue', linewidth=2)
    plt.plot(time_steps, lower_prices, label=f'Lower Bound (Last: US$ {lower_prices[-1]:.2f})', color='red', linestyle='dotted', linewidth=1.5)
    plt.plot(time_steps, upper_prices, label=f'Upper Bound (Last: US$ {upper_prices[-1]:.2f})', color='green', linestyle='dotted', linewidth=1.5)
    plt.fill_between(time_steps, lower_prices, upper_prices, color='blue', alpha=0.2, label='95% Confidence Interval')
    plt.title('Forecasted Prices with Confidence Intervals', fontsize=16)
    plt.xlabel('Months into the Future', fontsize=14)
    plt.ylabel('Price', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(alpha=0.3)
    plt.show()

def plot_mean_forecast(mean_prices, steps):
    """
    Plots the forecasted mean prices.

    Parameters:
    - mean_prices: Tensor of mean forecasted prices.
    - steps: Number of timesteps in the forecast.
    """
    # Convert tensor to NumPy array
    mean_prices = mean_prices.cpu().numpy()

    # Generate time steps for plotting
    time_steps = range(1, steps + 1)

    # Plot mean prices
    plt.figure(figsize=(12, 6))
    plt.plot(time_steps, mean_prices, label=f'Mean Forecast (Last: US$ {mean_prices[-1]:.2f})', color='blue', linewidth=2)
    plt.title('Forecasted Mean Prices', fontsize=16)
    plt.xlabel('Months into the Future', fontsize=14)
    plt.ylabel('Price', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(alpha=0.3)
    plt.show()