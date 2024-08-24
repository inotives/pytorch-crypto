import pandas as pd 
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

import utils.data_preprocessing as dp 

class RandomForestModel():
    def __init__(self, data_src): 
        self.data_src = data_src
        self.data_df = self.data_prep()

    def data_prep(self): 
        data_fullpath = f"data/raw/{self.data_src}.csv"
        data = dp.data_load(data_fullpath)
        data = dp.adding_features(data)
        
        # Shifted close as the target for next day prediction
        data['shifted_close'] = data['close'].shift(-1)

        data.dropna(inplace=True)

        return data 

    def run_predictor(self): 
        # define features X and target Y 
        df = self.data_df
        X = df[['close', 'volume', 'ema_20', 'ma_20', 'rsi']].values
        y = df['shifted_close'].values 

         # Standardize the entire dataset, but only use the price scaler for inverse transformation
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
    
        # Separate scaler for close prices
        price_scaler = StandardScaler()
        y_scaled = price_scaler.fit_transform(y.reshape(-1, 1))

        # Split the data into training and test sets
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.2, shuffle=False)


        # Initialize the Random Forest model
        rf = RandomForestRegressor(n_estimators=100, random_state=42)

        # Train the Random Forest model
        rf.fit(X_train, y_train.ravel())

        # Predict on the test set to evaluate performance
        y_pred = rf.predict(X_test)


         # Forecast for the next 7 days
        last_features = X_scaled[-1]  # Last feature set from the dataset
        predicted_prices = []

        for i in range(7):
            next_price_scaled = rf.predict(last_features.reshape(1, -1))
            
            # Update the feature set for the next prediction (recursive forecasting)
            next_features = np.array([next_price_scaled[0], df['volume'].iloc[-1], df['ema_20'].iloc[-1], df['ma_20'].iloc[-1], df['rsi'].iloc[-1]])
            last_features = np.hstack([next_price_scaled[0], last_features[1:]])  # Update only the price feature

            predicted_prices.append(next_price_scaled[0])
            
        
        # Inverse transform only the predicted price column to original scale
        predicted_prices = price_scaler.inverse_transform(np.array(predicted_prices).reshape(-1, 1))


        print("Random Forest forecast for the next 7 days:", predicted_prices.flatten())


        
        return 