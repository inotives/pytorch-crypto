import pandas as pd
import numpy as np
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

import utils.data_preprocessing as dp 

class SVRModel():
    def __init__(self, data_src): 
        self.data_src = data_src
        self.data_df = self.data_prep()

    def data_prep(self): 
        data_fullpath = f"data/raw/{self.data_src}.csv"
        data = dp.data_load(data_fullpath)
        data = dp.adding_features(data)

        # Feature engineering: We'll use previous closing prices as features
        data['shifted_close'] = data['close'].shift(-1)

        data.dropna(inplace=True)

        return data 

    def run_predictor(self):

        df = self.data_df

        # Define features (X) and target (y)
        X = df[['close']].values  # Feature: current close price
        y = df['shifted_close'].values  # Target: next day's close price

        # Standardize the data
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Split the data into training and test sets
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, shuffle=False)

        # Initialize the SVR model
        svr = SVR(kernel='rbf', C=100, gamma=0.1)

        # Train the SVR model
        svr.fit(X_train, y_train)

        # Predict on the test set to evaluate performance
        y_pred = svr.predict(X_test)

        # Forecast for the next 7 days
        last_close = X_scaled[-1]  # Last close price from the dataset

        predicted_prices = []
        for i in range(7):
            next_price_scaled = svr.predict(last_close.reshape(1, -1))
            predicted_prices.append(next_price_scaled[0])
            
            # Update last_close for the next prediction (recursive forecasting)
            last_close = scaler.transform([[next_price_scaled[0]]])

        # Inverse transform the predicted prices to original scale
        predicted_prices = scaler.inverse_transform(np.array(predicted_prices).reshape(-1, 1))

        print("SVR forecast for the next 7 days:", predicted_prices.flatten())
