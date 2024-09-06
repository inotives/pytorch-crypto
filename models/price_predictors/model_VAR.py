import pandas as pd
from statsmodels.tsa.api import VAR
from sklearn.metrics import mean_squared_error
import numpy as np

import utils.data_preprocessing as dp 

class VARModel():
    def __init__(self, data_src): 
        self.data_src = data_src
        self.data_df = self.data_prep()

    def data_prep(self): 
        data_fullpath = f"data/raw/{self.data_src}.csv"
        data = dp.data_load(data_fullpath)
        data = dp.adding_features(data)

        data.dropna(inplace=True)

        return data 

    def run_predictor(self, forecast_days=3): 
        # define features X and target Y 
        df = self.data_df

        df = df.asfreq('D')
        
        # Select features for the VAR model (OHLCV + any other derived features)
        df_var = df[['open', 'high', 'low', 'close', 'volume', 'marketCap', 'vwap', 'rsi']]

        # Handle missing values if necessary (e.g., forward fill)
        df_var = df_var.ffill()

        # Split the data into training and testing sets
        train_size = int(0.8 * len(df_var))
        train, test = df_var[:train_size], df_var[train_size:]

        # Fit the VAR model
        model = VAR(train)
        model_fitted = model.fit(maxlags=15, ic='aic')

        # Print the lag order
        lag_order = model_fitted.k_ar
        print(f'Lag Order: {lag_order}')

        # Make predictions on the test set
        forecast_input = train.values[-lag_order:]
        forecast = model_fitted.forecast(y=forecast_input, steps=len(test))

        # Convert forecast to DataFrame for easy handling
        forecast_df = pd.DataFrame(forecast, index=test.index, columns=test.columns)

        # Evaluate the model using RMSE on the test set
        rmse = np.sqrt(mean_squared_error(test['close'], forecast_df['close']))
        print(f'Root Mean Squared Error on Test Set: {rmse:.4f}')

        # Use the model to forecast the next 7 days
        forecast_input_next = df_var.values[-lag_order:]
        forecast_next = model_fitted.forecast(y=forecast_input_next, steps=forecast_days)

        # Convert to DataFrame
        forecast_next_df = pd.DataFrame(forecast_next, columns=df_var.columns)
        forecast_next_df.index = pd.date_range(start=df.index[-1], periods=8, freq='D')[1:]
        print(f"Forecast for the next 7 days:\n{forecast_next_df[['close']]}")


